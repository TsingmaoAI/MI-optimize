import torch
import torch.nn.functional as F

from mi_optimize.memory import MEMORY_BANK, clear_mem
from mi_optimize.quantization import Precision, PRECISION_TO_BIT
from .base import BaseQuantizer

class LinearZeroquantQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, w_groupsize=128, wbit=Precision.FP16, abit=Precision.FP16, offload='cpu', device='cuda',**kwarg):
        super().__init__(quant_hub_linear, w_bits=wbit, a_bits=abit, offload=offload, device=device)
        self.groupsize = w_groupsize
    
    @torch.no_grad()
    def quantize_weight_per_group_absmax(self, t, n_bits=8, q_group_size=128):
        t_shape = t.shape
        assert t_shape[-1] % q_group_size == 0
        t = t.reshape(-1, q_group_size)
        scales = t.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)
        t = t.reshape(t_shape)
        return t, scales

    @torch.no_grad()
    def quantize_activation_per_token_absmax(self, t, n_bits=8):
        t_shape = t.shape
        t.view(-1, t_shape[-1])
        scales = t.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2**(n_bits-1)-1
        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)
        return t

    @torch.no_grad()
    def quantize(self):
        # quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            Q, scale = self.quantize_weight_per_group_absmax(self.quant_hub_linear.core.weight, n_bits=self.wbit, q_group_size=self.groupsize)
            self.fake_w = Q
            self.w_scale = scale

    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32 and origin_dtype == torch.float32:
            x = x.float()
        else:
            x = self.quantize_activation_per_token_absmax(x, n_bits=self.abit)

        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            w = self.Q.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(origin_dtype)

    def to(self, desc):
        if hasattr(self, 'fake_w'):
            self.fake_w = self.fake_w.to(desc)
        if hasattr(self, 'w_scale'):
            self.w_scale = self.w_scale.to(desc)
        return self