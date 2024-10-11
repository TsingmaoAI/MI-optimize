import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver

from mi_optimize.memory import MEMORY_BANK, clear_mem
from mi_optimize.quantization import Precision, PRECISION_TO_BIT

from .utils import track_hessian_hook_to_cpu, track_hessian_hook_to_cuda
from .base import BaseQuantizer


class LinearGPTQQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, blocksize=128, w_groupsize=-1, percdamp=.01, actorder=True, wbit=Precision.FP16, abit=Precision.FP16, w_qscheme=torch.per_channel_affine, w_qtype='per_channel', offload='cpu', device='cuda', **kwarg) -> None:
        super().__init__(quant_hub_linear, wbit, abit, offload, device)
        self.blocksize = blocksize
        self.groupsize = w_groupsize
        self.nsamples = 0
        self.ready = False
        self.percdamp = percdamp
        self.actorder = actorder
        self.w_qtype = w_qtype

        self.rows = self.quant_hub_linear.core.out_features
        self.columns = self.quant_hub_linear.core.in_features 
        self.w_qscheme = w_qscheme

    def add_hook(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if track_hessian_hook_to_cuda not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_hessian_hook_to_cuda)

    @torch.no_grad()
    def tensor_quant(self, x, scale, zero_point, bits, qscheme=torch.per_channel_symmetric):
        if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
            x = torch.fake_quantize_per_channel_affine(
                x,
                scale.to(x.device),
                zero_point.to(x.device),
                0, 0, 2 ** PRECISION_TO_BIT[bits] - 1
            ).to(x)
        else:
            x = torch.fake_quantize_per_tensor_affine(
                x,
                scale.to(x.device),
                zero_point.to(x.device),
                0, 2 ** PRECISION_TO_BIT[bits] - 1
            ).to(x)

        return x

    def find_params(self, w, bits, qscheme=torch.per_channel_symmetric):
        if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
            observer = PerChannelMinMaxObserver(
                qscheme=qscheme,
                quant_min=0,
                quant_max=2 ** PRECISION_TO_BIT[bits] - 1
            )
        else:
            observer = MinMaxObserver(
                qscheme=qscheme,
                quant_min=0,
                quant_max=2**PRECISION_TO_BIT[bits] - 1
            )
        for i in w:
            observer(i)
        scale, zero_point = observer.calculate_qparams()
        return scale, zero_point

    @torch.no_grad()
    def quantize(self):
        # quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            W = self.quant_hub_linear.core.weight.data.detach().to(self.device)
            H = self.quant_hub_linear.core.H.to(self.device)
            W = W.float()

            scale, zero_point = self.find_params(
                [W], self.wbit, self.w_qscheme)

            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            if self.actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            damp = self.percdamp*torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.device)
            H[diag, diag] += damp

            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
            
            w_scale = torch.empty(0)
            w_zero_point = torch.empty(0)
            
            for i1 in range(0, self.columns, self.blocksize):
                i2 = min(i1 + self.blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if self.groupsize != -1:
                        if (i1 + i) % self.groupsize == 0:
                            scale, zero_point = self.find_params(
                                [W[:, (i1 + i):(i1 + i + self.groupsize)]], self.wbit, self.w_qscheme)
                            w_scale = torch.cat((w_scale, scale.unsqueeze(0)), dim=1)
                            w_zero_point = torch.cat((w_zero_point, zero_point.unsqueeze(0)), dim=1)
                    q = self.tensor_quant(
                        w, scale, zero_point, self.wbit, qscheme=self.w_qscheme
                    ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            torch.cuda.synchronize()

            if self.actorder:
                invperm = torch.argsort(perm)
                Q = Q[:, invperm]
            if self.groupsize == -1:
                w_scale = scale.unsqueeze(1)
                w_zero_point = zero_point.unsqueeze(1)
            Q = Q.reshape(self.quant_hub_linear.core.weight.shape).to(self.quant_hub_linear.core.weight.data.dtype)
            self.w_scale = w_scale
            self.w_zero_point = w_zero_point
            self.fake_w = Q
            
            del W, H, Losses, damp, diag, Hinv, W1, Q1, Err1, Losses1, Hinv1, self.quant_hub_linear.core.H
            clear_mem()

        # quantize activation
        if self.abit not in [Precision.FP16, Precision.FP32]:
            raise RuntimeError('gptq quantizer cannot support quantization of activations to {} bit'.format(PRECISION_TO_STR[self.abit]))

    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
            raise RuntimeError('gptq quantizer cannot support quantization of activations to {} bit'.format(PRECISION_TO_STR[self.abit]))

        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            w = self.fake_w.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(origin_dtype)

    def to(self, desc):
        if hasattr(self, 'w_scale'):
            self.w_scale = self.w_scale.to(desc)
        if hasattr(self, 'w_zero_point'):
            self.w_zero_point = self.w_zero_point.to(desc)
        if hasattr(self, 'fake_w'):
            self.fake_w = self.fake_w.to(desc)
        return self
