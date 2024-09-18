import torch
import torch.nn.functional as F

from mi_optimize.memory import MEMORY_BANK, clear_mem
from mi_optimize.quantization import Precision, PRECISION_TO_BIT

from .utils import track_input_hook_to_cpu, track_input_hook_to_cuda
from .base import BaseQuantizer


class LinearAwqQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, w_has_zero=True, w_unsign=True, w_groupsize=128,  auto_scale=True, auto_clip=True, w_qtype='per_group', wbit=Precision.FP16, abit=Precision.FP16, offload='cpu', device='cuda',**kwarg) -> None:
        super().__init__(quant_hub_linear = quant_hub_linear, w_unsign=w_unsign, w_bits=wbit, w_has_zero=w_has_zero, offload=offload, device=device)
        self.auto_scale = auto_scale
        self.auto_clip = auto_clip
        self.groupsize = w_groupsize
        self.zero_point = w_has_zero
        self.w_qtype = w_qtype

    def add_hook(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if track_input_hook_to_cpu not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_input_hook_to_cpu)

    @torch.no_grad()
    def get_weight_scale(self, weight, q_groupsize=-1):
        org_shape = weight.shape
        if q_groupsize > 0:
            weight = weight.view(-1, q_groupsize)
        scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
        scale = scale.view(org_shape)
        scale = scale.mean(0)
        return scale

    @torch.no_grad()
    def get_act_scale(self, x):
        return x.abs().view(-1, x.shape[-1]).mean(0)

    def pseudo_quantize_tensor(self, w, n_bit=8,
                               zero_point=True, q_groupsize=-1,
                               inplace=False,
                               get_scale_zp=False
                               ):
        org_w_shape = w.shape
        if q_groupsize > 0:
            assert org_w_shape[-1] % q_groupsize == 0
            w = w.reshape(-1, q_groupsize)
        else:
            w = w.reshape(-1, w.shape[-1])
        # print(w.size())
        assert w.dim() == 2
        min_val = None
        if zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2 ** n_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:  # we actually never used this
            assert min_val is None
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bit - 1) - 1
            min_int = - 2 ** (n_bit - 1)
            scales = max_val / max_int
            zeros = 0

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        if inplace:
            ((w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)).mul_(scales)
        else:
            w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales

        assert torch.isnan(w).sum() == 0

        w = w.reshape(org_w_shape)

        if get_scale_zp:
            return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
        else:
            return w


    @torch.no_grad()
    def auto_scale_layer(self, layer, n_bit, x):
        # w: co, ci
        # x: n, ci
        weight = layer.weight.detach()
        x = x.to(weight.device)
        w_max = self.get_weight_scale(weight, q_groupsize=self.groupsize)

        with torch.no_grad():
            org_out = layer(x)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = self.get_act_scale(x)

        best_error = float('inf')
        best_ratio = -1
        best_scales = None

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = (x_max.pow(ratio) / w_max.pow(1-ratio)).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            # layer.weight.data = layer.weight.clone().mul_(scales.view(1, -1))
            weight_s = weight.mul(scales.view(1, -1))
            layer.weight.data = self.pseudo_quantize_tensor(weight_s, n_bit=n_bit, zero_point=self.zero_point, q_groupsize=self.groupsize) / (scales.view(1, -1))
            out = layer(x)
            if isinstance(out, tuple):
                out = out[0]
            
            # loss = (org_out - out).abs().mean().item()
            loss = F.mse_loss(org_out, out).item()
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            layer.load_state_dict(org_sd)
            del loss, out, weight_s, scales
            clear_mem()

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        clear_mem()
        return best_scales.detach()

    def apply_scale(self, w, scales):
        scales = scales.to(w.device)
        w.data = w.data.mul_(scales.view(1, -1))

    def auto_clip_layer(self, w, input_feat, n_bit, n_grid=20,  max_shrink=0.5, n_sample_token=1):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        groupsize = self.groupsize if self.groupsize > 0 else w.shape[1]

        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, groupsize)
        input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, groupsize)

        oc_batch_size = 256
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = - max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(
                    cur_w, n_bit=n_bit, zero_point=self.zero_point, q_groupsize=self.groupsize)
                cur_out = (input_feat * q_w).sum(dim=-1)

                err = (cur_out - org_out).abs().mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        del input_feat, err, best_max_val_all
        del org_out
        clear_mem()
        return best_max_val.squeeze(1)

    def apply_clip(self, w, clip):
        max_val = clip.to(w.device)
        org_shape = w.shape
        w.data = w.data.reshape(*max_val.shape[:2], -1)
        w.data = torch.clamp(w.data, -max_val, max_val)
        w.data = w.data.reshape(org_shape)

    @torch.no_grad()
    def quantize(self):
        # quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            input_list = [x[0] for x in self.quant_hub_linear.core.input_tracks]
            input_feat = torch.cat(input_list, dim=1)
            input_feat = input_feat.to(self.device)
            Q = self.quant_hub_linear.core.weight.detach().to(self.device)

            if self.auto_scale:
                layer = self.quant_hub_linear.core.to(self.device)
                scales_list = self.auto_scale_layer(layer, n_bit=PRECISION_TO_BIT[self.wbit], x=input_feat)
                self.apply_scale(Q, scales_list)
                input_feat = input_feat.div_(scales_list.view(1, -1).to(input_feat.device))
                del layer
            if self.auto_clip:
                clip_list = self.auto_clip_layer(Q, input_feat, n_bit=PRECISION_TO_BIT[self.wbit])
                self.apply_clip(Q, clip_list)

            quant_Q, scale, zero_point = self.pseudo_quantize_tensor(Q, n_bit=PRECISION_TO_BIT[self.wbit], zero_point=self.zero_point, q_groupsize=self.groupsize, get_scale_zp=True)
            
            self.w_scale = scale
            self.w_zero_point = zero_point
            self.smooth_factor = scales_list
            self.fake_w = quant_Q
            del Q, quant_Q, scale, zero_point, input_feat, scales_list, clip_list, self.quant_hub_linear.core.input_tracks
            clear_mem()

        # quantize activation
        if self.abit not in [Precision.FP16, Precision.FP32]:
            raise RuntimeError('awq quantizer cannot support quantization of activations to {} bit'.format(PRECISION_TO_STR[self.abit]))

    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32 and origin_dtype == torch.float32:
            x = x.float()
        else:
            raise RuntimeError('awq quantizer cannot support quantization of activations to {} bit'.format(PRECISION_TO_STR[self.abit]))

        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            scales = self.smooth_factor
            x = x.div(scales.view(1, -1).to(x.device))
            w = self.fake_w.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(origin_dtype)

    def to(self, desc):
        if hasattr(self, 'w_scale'):
            self.w_scale = self.w_scale.to(desc)
        if hasattr(self, 'w_zero_point'):
            self.w_zero_point = self.w_zero_point.to(desc)
        if hasattr(self, 'smooth_factor'):
            self.smooth_factor = self.smooth_factor.to(desc)
        if hasattr(self, 'fake_w'):
            self.fake_w = self.fake_w.to(desc)
        return self

