import torch
import torch.nn.functional as F
import tqdm

from mi_optimize.memory import  clear_mem
from mi_optimize.quantization import Precision, PRECISION_TO_BIT

from .utils import track_hessian_hook_to_cpu, track_hessian_hook_to_cuda
from .base import BaseQuantizer
from mi_optimize.quantization import PRECISION_TO_STR


class LinearSpqrQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, w_groupsize, outlier_relative_threshold=0.2, qq_scale_bits=3, qq_zero_bits=3, qq_zero_sym=False, qq_groupsize=16, wbit=Precision.FP16, offload='cpu', device='cuda',**kwarg):
        super().__init__(quant_hub_linear, wbit=wbit, offload=offload, device=device)
        self.nsamples = 0
        self.H = torch.zeros_like((self.quant_hub_linear.core.weight), device=self.offload)
        self.permutation_order = "identity"
        self.keep_H = True
        self.percdamp = 1e-2
        self.keep_last_columns = 0
        self.blocksize = 128
        self.simplified_outliers = False
        self.verbose = False
        self.perchannel = True
        self.bits = wbit
        self.sym = False

        self.groupsize = w_groupsize
        self.outlier_relative_threshold = outlier_relative_threshold
        self.round_zero = False

        self.qq_scale_bits = qq_scale_bits
        self.qq_zero_bits = qq_zero_bits
        self.qq_zero_sym = qq_zero_sym
        self.qq_groupsize = qq_groupsize

    def add_hook(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if track_hessian_hook_to_cpu not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_hessian_hook_to_cpu)

    @torch.jit.script
    def find_greedy_nearest_indices(self, weight: torch.Tensor, use_abs: bool = False):
        ordered_unit_weight_t = weight.detach().t().clone()

        ordered_unit_weight_t /= ordered_unit_weight_t.norm(p=2, dim=-1, keepdim=True)
        distance_matrix = ordered_unit_weight_t @ ordered_unit_weight_t.T

        if use_abs:
            distance_matrix = abs(distance_matrix)
        permutation = torch.arange(len(ordered_unit_weight_t), device=weight.device)
        for dim_i in range(len(ordered_unit_weight_t) - 2):
            nearest_dim_i = (dim_i + 1) + distance_matrix[dim_i, dim_i + 1:].argmax()
            next_dim_i = torch.full_like(nearest_dim_i, dim_i + 1)
            index_pair = torch.stack([next_dim_i, nearest_dim_i])
            swapped_index_pair = torch.stack([nearest_dim_i, next_dim_i])
            ordered_unit_weight_t[index_pair] = ordered_unit_weight_t[swapped_index_pair]
            distance_matrix[index_pair] = distance_matrix[swapped_index_pair]
            distance_matrix[:, index_pair] = distance_matrix[:, swapped_index_pair]
            permutation[index_pair] = permutation[swapped_index_pair]
        return permutation

    def get_permutation_order(self, H: torch.Tensor, W: torch.Tensor, permutation_order: str = "identity", use_abs: bool = False):
        if permutation_order == "spearman":
            w_rank = W.argsort(dim=0).argsort(dim=0).float()
            w_rank = w_rank - w_rank.mean(dim=0, keepdim=True)
            perm = self.find_greedy_nearest_indices(w_rank, use_abs)
        elif permutation_order == "act_order":
            perm = torch.argsort(torch.diag(H), descending=True)
        elif permutation_order == "identity":
            perm = torch.arange(H.shape[0], device=H.device)
        elif isinstance(permutation_order, torch.Tensor):
            return permutation_order
        else:
            raise ValueError(f"Unknown permutation order name: {permutation_order}")
        return perm

    def quant_tensor(self, x, scale, zero, bits, eps=1e-9):
        maxq = torch.tensor(2 ** bits - 1)
        q = torch.clamp(torch.round(x / scale.clamp_min(eps) + zero), 0, maxq)
        return scale * (q - zero)

    def find_params(self, x, bits, perchannel, sym, weight=False):
        maxq = torch.tensor(2**bits - 1)
        shape = x.shape
        if perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        xmin = x.min(1).values
        xmax = x.max(1).values

        if sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = xmin == xmax
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / maxq
        if sym:
            zero = torch.full_like(scale, (maxq + 1) / 2)
        else:
            zero = -xmin / scale

        if not perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            scale = scale.reshape(shape)
            zero = zero.reshape(shape)
            return scale, zero
        if len(shape) == 4:
            scale = scale.reshape((1, -1, 1, 1))
            zero = zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            scale = scale.reshape((1, 1, -1))
            zero = zero.reshape((1, 1, -1))
        if len(shape) == 2:
            scale = scale.unsqueeze(0)
            zero = zero.unsqueeze(0)
        return scale, zero

    def get_leave_one_out_error(self, group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, *, bits, sym):
        assert group_weight.ndim == 2
        loo_indices = torch.arange(
            group_weight.shape[1], device=group_weight.device)
        loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype) 
        groupwise_loo_data = group_weight[:, loo_indices] # [num_groups, num_loo = groupsize, groupsize - 1]
        scale, zero_point = self.find_params(groupwise_loo_data.flatten(0, 1), perchannel=True, sym=sym, bits=bits, weight=True)

        loo_groupwise_reconstructed_weights = self.quant_tensor(groupwise_loo_data.flatten(
            0, 1), scale, zero_point, bits).reshape_as(groupwise_loo_data)
        loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]   # [num_loo = groupsize, groupsize - 1]
        assert group_diag_hessian_inv_cho.ndim == 1

        loo_errors_sq = ((loo_groupwise_reconstructed_weights - groupwise_loo_data) /
                         loo_group_diag_hessian_inv_cho).square().sum(-1)
        assert loo_errors_sq.shape == group_weight.shape # [num_groups, num_loo = groupsize]

        scale, zero_point = self.find_params(group_weight, bits=bits, perchannel=True, sym=sym, weight=True)
        baseline_reconstructed_weights = self.quant_tensor(group_weight, scale, zero_point, bits)
        baseline_errors_sq = (
            ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
        )

        reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
        return reduction_in_squared_error

    @torch.no_grad()
    def quantize(self):
        # quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            H = self.quant_hub_linear.core.H.to(self.device)
            weight = self.quant_hub_linear.core.weight.detach().to(self.device).float()
            perm = self.get_permutation_order(H, weight, permutation_order=self.permutation_order)
            weight = weight[:, perm]
            if self.keep_H:
                H = H.clone()
            else:
                H = None

            H = H[perm][:, perm]
            self.dead = torch.diag(H) == 0
            if self.percdamp > 0:
                ix = torch.arange(len(H), device=weight.device)
                H[ix, ix] += self.percdamp * abs(torch.diag(H)).mean()
                del ix
            H[self.dead, self.dead] = 1
            weight[:, self.dead] = 0

            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
            H_inv_cho_diag = torch.diag(H_inv_cho)
            del H

            assert H_inv_cho.shape[0] == H_inv_cho.shape[1] == weight.shape[1], "weight must be [out_features, in_features]"
            out_dim, in_dim = weight.shape

            if self.groupsize is None:
                self.groupsize = in_dim

            outlier_column_indices = torch.empty(0, dtype=torch.int64, device=weight.device)
            del H_inv

            outlier_scale = (weight.var(dim=0) / torch.diag(H_inv_cho).square()).mean().item()
            unstructured_outlier_threshold = self.outlier_relative_threshold * outlier_scale
            in_group_index = -1

            quantization_errors = torch.zeros_like(weight)
            unstructured_outlier_mask = torch.zeros_like(weight, dtype=torch.bool)

            block_start_iter = range(0, in_dim - self.keep_last_columns, self.blocksize)
            block_start_iter = tqdm(block_start_iter, leave=False) if self.verbose else block_start_iter

            w_scale = torch.empty(0).to(self.device)
            w_zero_point = torch.empty(0).to(self.device)
            qs_scale = torch.empty(0).to(self.device)
            qs_zero_point = torch.empty(0).to(self.device)
            qz_scale = torch.empty(0).to(self.device)
            qz_zero_point = torch.empty(0).to(self.device)

            for block_start in block_start_iter:
                block_end = min(block_start + self.blocksize, in_dim)
                for column_index in range(block_start, block_end):
                    if column_index % self.groupsize == 0:
                        in_group_index += 1
                        group_weight = weight[:, column_index: column_index + self.groupsize]
                        if self.simplified_outliers or (unstructured_outlier_threshold == float("inf")):
                            scale, zero_point = self.find_params(group_weight, bits=self.wbit, perchannel=self.perchannel, sym=self.sym, weight=True)
                        else:
                            assert self.perchannel, "refitting quantizer is only implemented for perchannel=True"
                            group_diag_hessian_inv_cho = H_inv_cho_diag[column_index: column_index + self.groupsize]
                            loo_quantization_error_sq = self.get_leave_one_out_error(
                                group_weight, group_diag_hessian_inv_cho, bits=PRECISION_TO_BIT[self.bits], sym=self.sym
                            )

                            likely_unstructured_outlier_mask = (loo_quantization_error_sq > unstructured_outlier_threshold).float()

                            non_outlier_mask = 1 - likely_unstructured_outlier_mask
                            mean_over_non_outliers = torch.sum(group_weight * non_outlier_mask, dim=1, keepdim=True) / torch.sum(
                                non_outlier_mask, dim=1, keepdim=True
                                ).clamp_min(1)
                            group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                                1 - non_outlier_mask
                            )
                            scale, zero_point = self.find_params(group_weight_without_outliers, bits=self.wbit, perchannel=self.perchannel, sym=self.sym, weight=True)
                            del group_diag_hessian_inv_cho, loo_quantization_error_sq
                            del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                        w_scale = torch.cat((w_scale, scale), dim=1)
                        w_zero_point = torch.cat((w_zero_point, zero_point), dim=1)

                        origin_scale_shape = scale.shape
                        origin_zero_point_shape = zero_point.shape
                        scale_groups = scale.reshape(-1, self.qq_groupsize)
                        zero_point_groups = zero_point.reshape(-1, self.qq_groupsize)
                        scale_cur, zero_point_cur = self.find_params(scale_groups, bits=self.qq_scale_bits, perchannel=True, sym=False, weight=True)
                        q_scale = self.quant_tensor(
                            scale_groups, scale_cur, zero_point_cur, bits=self.qq_scale_bits
                            )

                        qs_scale = torch.cat((qs_scale, scale_cur), dim=1)
                        qs_zero_point = torch.cat((qs_zero_point, zero_point_cur), dim=1)

                        q_scale = q_scale.reshape(origin_scale_shape)
                        scale_cur, zero_point_cur = self.find_params(zero_point_groups, bits=self.qq_scale_bits, perchannel=True, sym=False, weight=True)
                        q_zero_point = self.quant_tensor(
                            zero_point_groups, scale_cur, zero_point_cur, bits=self.qq_scale_bits
                        )
                        q_zero_point = q_zero_point.reshape(origin_zero_point_shape)

                        qz_scale = torch.cat((qz_scale, scale_cur), dim=1)
                        qz_zero_point = torch.cat((qz_zero_point, zero_point_cur), dim=1)

                        del group_weight

                    weight_i_quantized = self.quant_tensor(
                        weight[:, column_index].unsqueeze(1), q_scale, q_zero_point, bits=self.wbit
                    ).reshape_as(weight[:, column_index])

                    delta_weight_i = weight[:, column_index] - weight_i_quantized
                    quantization_errors[:, column_index] = delta_weight_i / H_inv_cho[column_index, column_index]  # [out_dim]

                    if unstructured_outlier_threshold != float("inf"):
                        unstructured_outlier_mask[:, column_index] = (
                            quantization_errors[:, column_index].square() > unstructured_outlier_threshold
                        )

                        is_outlier = unstructured_outlier_mask[:, column_index].float()
                        weight_i_quantized_wo_outliers = self.quant_tensor(
                            (weight[:, column_index] * (1 - is_outlier)).unsqueeze(1), q_scale, q_zero_point, bits=self.wbit
                        ).reshape_as(weight[:, column_index])
                        weight_i_quantized = (
                            weight_i_quantized_wo_outliers * (1 - is_outlier) + weight[:, column_index] * is_outlier
                        )
                        del weight_i_quantized_wo_outliers

                        delta_weight_i = weight[:, column_index] - weight_i_quantized
                        quantization_errors[:, column_index] = delta_weight_i / H_inv_cho[column_index, column_index]  # [out_dim]

                    weight[:, column_index] = weight_i_quantized
                    weight[:, column_index + 1: block_end].addr_(
                        quantization_errors[:, column_index],
                        H_inv_cho[column_index, column_index + 1: block_end],
                        alpha=-1,
                    )

                weight[:, block_end:].addmm_(
                    quantization_errors[:, block_start:block_end],
                    H_inv_cho[block_start:block_end, block_end:],
                    alpha=-1,
                )

            if self.permutation_order != "identity":
                invperm = torch.argsort(perm)
                weight = weight[:, invperm]
                
            Q = weight.reshape(self.quant_hub_linear.core.weight.shape).to(self.quant_hub_linear.core.weight.data.dtype)
            self.w_scale = w_scale
            self.w_zero_point = w_zero_point
            self.qs_scale = qs_scale
            self.qs_zero_point = qs_zero_point
            self.qz_scale = qz_scale
            self.qz_zero_point = qz_zero_point
            self.Q = Q
            # self.w_scale = MEMORY_BANK.add_value('{id}_w_scale'.format(id=id(self)), w_scale, self.offload)
            # self.w_zero_point = MEMORY_BANK.add_value('{id}_w_zero_point'.format(id=id(self)), w_zero_point, self.offload)
            # self.qs_scale = MEMORY_BANK.add_value('{id}_qs_scale'.format(id=id(self)), qs_scale, self.offload)
            # self.qs_zero_point = MEMORY_BANK.add_value('{id}_qs_zero_point'.format(id=id(self)), qs_zero_point, self.offload)
            # self.qz_scale = MEMORY_BANK.add_value('{id}_qz_scale'.format(id=id(self)), qz_scale, self.offload)
            # self.qz_zero_point = MEMORY_BANK.add_value('{id}_qz_zero_point'.format(id=id(self)), qz_zero_point, self.offload)
            # self.Q = MEMORY_BANK.add_value('{id}_Q'.format(id=id(self)), Q, self.offload)
            
            self.w_count = weight.numel()
            self.normal_outliers_count = unstructured_outlier_mask.to(
                torch.int32).sum().item()
        if self.abit not in [Precision.FP16, Precision.FP32]:
            raise RuntimeError('spqr quantizer cannot support quantization of activations to {} bit'.format(PRECISION_TO_STR[self.abit]))

    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32 or origin_dtype == torch.float():
            x = x.float()
        else:
            raise RuntimeError('spqr quantizer cannot support quantization of activations to {} bit'.format(PRECISION_TO_STR[self.abit]))

        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            w = self.Q.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(origin_dtype)

    def get_average_number_of_bits(
        self,
        wbits: int = 3,
        qq_scale_bits: int = 3,
        qq_zero_bits: int = 3,
        qqq_scale_bits: int = 16,
        qqq_zero_bits: int = 16,
        groupsize: int = 16,
        qq_groupsize: int = 16,
        round_zero: bool = False,
        global_ol_n_share: float = 0.00,
    ):
        # if not quantized stats are in full precision
        qq_scale_bits = qq_scale_bits or 16
        qq_zero_bits = qq_zero_bits or 16
        groupsize = groupsize or float('inf')
        qq_groupsize = qq_groupsize or float('inf')

        if groupsize is None:
            wbits_avg = wbits
        elif round_zero:
            wbits_avg = wbits + (qq_scale_bits + wbits) / groupsize \
                + (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)
        else:
            wbits_avg = wbits + (qq_scale_bits + qq_zero_bits) / groupsize \
                + 2 * (qqq_scale_bits + qqq_zero_bits) / (groupsize * qq_groupsize)

        # correct accounting for outliers
        if global_ol_n_share > 0:
            wbits_avg += 32 * global_ol_n_share

        return round(wbits_avg, 2)

    def to(self, desc):
        if hasattr(self, 'Q'):
            self.Q = self.Q.to(desc)
        if hasattr(self, 'w_scale'):
            self.w_scale = self.w_scale.to(desc)
        if hasattr(self, 'w_zero_point'):
            self.w_zero_point = self.w_zero_point.to(desc)
        if hasattr(self, 'qs_scale'):
            self.qs_scale = self.qs_scale.to(desc)
        if hasattr(self, 'qs_zero_point'):
            self.qs_zero_point = self.qs_zero_point.to(desc)
        if hasattr(self, 'qz_scale'):
            self.qz_scale = self.qz_scale.to(desc)
        if hasattr(self, 'qz_zero_point'):
            self.qz_zero_point = self.qz_zero_point.to(desc)
        return self
