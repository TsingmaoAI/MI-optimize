import torch
import torch.nn.functional as F

from mi_optimize.memory import MEMORY_BANK, clear_mem
from mi_optimize.quantization import Precision, PRECISION_TO_BIT

from .base import BaseQuantizer

class LinearFP8Quantizer(BaseQuantizer):
    def __init__(self, quant_hub_layer, act_quant='E4M3', weight_quant='E4M3', wbit=Precision.INT8, abit=Precision.INT8, offload='cpu', device='cuda',**kwarg):
        super().__init__(quant_hub_layer, wbit, abit, offload, device)
        self.E4M3_bound = 240
        self.E5M2_bound = 57344 
        self.act_quant = act_quant
        self.weight_quant = weight_quant
    
    def quanz_fix_E4M3(self, input, S, is_scale, is_channel=False, channel_dim=0):
        S = S if is_scale else torch.ones_like(S)
        if is_channel:
            view_s = [1 if i != channel_dim else input.size(channel_dim) \
                    for i in range(len(input.size()))]
            S = S.view(view_s)
        sign = torch.sign(input)
        ab = torch.abs(input)
        ab = ab * S
        ab = torch.where(ab>2**7*1.875, torch.ones_like(ab)*2**7*1.875, ab)
        ab = torch.where(ab<=2.0**(-10), torch.zeros_like(ab), ab)
        E = torch.where(ab < 2**(-6), torch.ones_like(ab) * (-6), torch.floor(torch.log2(ab)))
        M = torch.round(ab * 2**(-E) * 8) * 0.125
        input_ = M * 2**E * sign / S
        input_ = torch.where(torch.isnan(input) + torch.isinf(input) + input==0.0, input, input_)
        return input_
    
    def quanz_fix_E5M2(self, input, S, is_scale, is_channel=False, channel_dim=0):
        S = S if is_scale else torch.ones_like(S)
        if is_channel:
            view_s = [1 if i != channel_dim else input.size(channel_dim) \
                    for i in range(len(input.size()))]
            S = S.view(view_s)
        sign = torch.sign(input)
        ab = torch.abs(input)
        ab = ab * S
        ab = torch.where(ab>2**15*1.75, torch.ones_like(ab)*2**15*1.75, ab)
        ab = torch.where(ab<=2.0**(-17), torch.zeros_like(ab), ab)
        E = torch.where(ab < 2**(-14), torch.ones_like(ab) * (-14), torch.floor(torch.log2(ab)))
        M = torch.round(ab * 2**(-E) * 4) * 0.25
        input_ = M * 2**E * sign / S
        input_ = torch.where(torch.isnan(input) + torch.isinf(input) + input==0.0, input, input_)
        return input_

    def quantize(self):
        #quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            w = self.quant_hub_linear.core.weight
            if self.weight_quant == 'E4M3':
                S = self.E4M3_bound / w.abs().max(-1).values 
                Q = self.quanz_fix_E4M3(w, S=S, is_scale=True, is_channel=True, channel_dim=0)
            elif self.weight_quant == 'E5M2':
                S = w.abs().max(-1).values / self.E5M2_bound
                Q = self.quanz_fix_E5M2(w, S=S, is_scale=True, is_channel=True, channel_dim=0)
            else:
                raise RuntimeError('not support weight quant to {}'.format(self.weight_quant))

            self.w_scale = MEMORY_BANK.add_value('{id}_w_scale'.format(id=id(self)), S, self.offload)
            self.Q = MEMORY_BANK.add_value('{id}_Q'.format(id=id(self)), Q, self.offload)
        
    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
            amax = torch.max(x)
            if self.act_quant == 'E4M3':
                S = self.E4M3_bound / amax
                x = self.quanz_fix_E4M3(x, S=S, is_scale=True, is_channel=False, channel_dim=0)
            elif self.act_quant == 'E5M2':
                S = amax / self.E5M2_bound
                x = self.quanz_fix_E5M2(x, S=S, is_scale=True, is_channel=False, channel_dim=0)
            else:
                raise RuntimeError('not support weight quant to {}'.format(self.weight_quant))
        
        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            w = self.Q.value.to(x)
    
        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(origin_dtype)
    
    def to(self, desc):
        self.w_scale.to(desc)
        self.Q.to(desc)
        return self