import torch
import torch.nn.functional as F

from functools import partial

from mi_optimize.memory import clear_mem
from mi_optimize.quantization import Precision, PRECISION_TO_BIT

from .utils import track_input_hook_to_cpu, track_input_hook_to_cuda
from .base import BaseQuantizer

class LinearSmoothQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, alpha=0.5, w_qtype='per_channel', a_qtype='per_tensor', w_groupsize=128, quant_out=False, quantization_type='dynamic', wbit=Precision.INT8, abit=Precision.INT8, offload='cpu', device='cuda', **kwargs):
        super().__init__(quant_hub_linear, wbit, abit, offload, device)
        self.alpha = alpha
        self.quant_out = quant_out
        self.weight_quant = w_qtype
        self.groupsize = w_groupsize
        self.w_qtype = w_qtype
        self.act_quant = a_qtype
        self.quantization_type = quantization_type

    def enable_quant_out(self):
        self.quant_out = True

    def disable_quant_out(self):
        self.quant_out = False

    def add_hook(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if track_input_hook_to_cpu not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_input_hook_to_cpu)

    def get_act_scale(self, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        return comming_max

    def get_smooth_scale(self, w, act_scales, alpha=0.5):
        device, dtype = w.device, w.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)
        weight_scales = w.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
        scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)).clamp(min=1e-5).to(device).to(dtype)
        return scales

    @torch.no_grad()
    def quantize_weight_per_group_absmax(self, w, n_bits=8, q_group_size=128):
        org_w_shape = w.shape
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
        scales = w.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2**(n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_().mul_(scales)
        w = w.reshape(org_w_shape)
        return w, scales.view(w.shape[0], -1)

    @torch.no_grad()
    def quantize_weight_per_channel_absmax(self, w, n_bits=8):
        scales = w.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2**(n_bits-1)-1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_().mul_(scales)
        return w, scales

    @torch.no_grad()
    def quantize_weight_per_tensor_absmax(self, w, n_bits=8):
        scales = w.abs().max()
        q_max = 2**(n_bits-1)-1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_().mul_(scales)
        return w, scales

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
    def quantize_activation_per_tensor_absmax(self, t, n_bits=8):
        t_shape = t.shape
        t.view(-1, t_shape[-1])
        scales = t.abs().max()
        q_max = 2**(n_bits-1)-1
        scales.clamp_(min=1e-5).div_(q_max)
        t.div_(scales).round_().mul_(scales)
        return t

    @torch.no_grad()
    def quantize(self):
        # quantize activation
        if self.abit not in [Precision.FP16, Precision.FP32]:
            inputs = [x[0].view(-1, x[0].shape[-1]) for x in self.quant_hub_linear.core.input_tracks]
            input_feat = torch.cat(inputs, dim=0)
            w = self.quant_hub_linear.core.weight.clone()
            act_scales = self.get_act_scale(input_feat)
            scales = self.get_smooth_scale(w, act_scales, self.alpha)
            if self.act_quant == 'per_token':
                self.quant_act = partial(self.quantize_activation_per_token_absmax, n_bits=self.abit)
            elif self.act_quant == 'per_tensor':
                self.quant_act = partial(self.quantize_activation_per_tensor_absmax, n_bits=self.abit)
            else:
                raise ValueError(f"Invalid act_quant")

            # self.scales = MEMORY_BANK.add_value('{id}_scales'.format(id=id(self)), scales, self.offload)
            self.smooth_factor = scales
            del input_feat, act_scales, scales, self.quant_hub_linear.core.input_tracks, w
        
        #quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if self.abit not in [Precision.FP16, Precision.FP32]:
                scales = self.smooth_factor.to(self.device)
                s_w = self.quant_hub_linear.core.weight.mul(scales)
                del scales
            else:
                s_w = self.quant_hub_linear.core.weight

            if self.weight_quant == 'per_group':
                Q, scale = self.quantize_weight_per_group_absmax(s_w, n_bits=self.wbit, q_group_size=self.groupsize)
            elif self.weight_quant == 'per_channel':
                Q, scale = self.quantize_weight_per_channel_absmax(s_w, n_bits=self.wbit)
                self.groupsize = -1
            elif self.weight_quant == 'per_tensor':
                Q, scale = self.quantize_weight_per_tensor_absmax(s_w, n_bits=self.wbit)
                self.groupsize = -1
            else:
                raise ValueError(f'Invalid weight_quant: {self.weight_quant}')

            self.fake_w = Q
            self.w_scale = scale
            self.w_zero_point = torch.zeros_like(scale)
            del Q, scale

        clear_mem()
    
    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
            scales = self.smooth_factor
            x = x.div(scales.view(1, -1).to(x.device))
            x = self.quant_act(x, n_bits=self.abit)

        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            w = self.fake_w.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        y = F.linear(x, w, bias).to(origin_dtype)
        if self.quant_out:
            y = self.quant_act(y)
        return y

    def to(self, desc):
        if hasattr(self, 'smooth_factor'):
            self.smooth_factor = self.smooth_factor.to(desc)
        if hasattr(self, 'fake_w'):
            self.fake_w = self.fake_w.to(desc)
        if hasattr(self, 'w_scale'):
            self.w_scale = self.w_scale.to(desc)
        if hasattr(self, 'w_zero_point'):
            self.w_zero_point = self.w_zero_point.to(desc)
        return self
