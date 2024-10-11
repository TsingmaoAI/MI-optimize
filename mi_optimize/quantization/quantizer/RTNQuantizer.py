import logging
import torch
import torch.nn.functional as F
from .utils  import Quantizer

from mi_optimize.memory import MEMORY_BANK, clear_mem
from mi_optimize.quantization import Precision, PRECISION_TO_BIT

from .utils import track_input_hook_to_cpu
from .base import BaseQuantizer


class LinearRTNQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, w_groupsize=-1, a_groupsize=-1, a_qtype="per_tensor", w_qtype="per_group", w_has_zero:bool=False, a_has_zero: bool=False, w_unsign:bool=True, a_unsign:bool=True, quantization_type='static', wbit=Precision.FP16, abit=Precision.FP16, offload='cpu', device='cuda', **kwargs):
        super().__init__(quant_hub_linear, wbit, abit, offload, device, w_unsign=w_unsign, a_unsign=a_unsign)
        self.w_groupsize = w_groupsize
        self.a_groupsize = a_groupsize
        self.a_qtype = a_qtype
        self.w_qtype = w_qtype
        self.w_has_zero = w_has_zero
        self.a_method = a_has_zero
        self.quantization_type = quantization_type
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            self.w_quantizer = Quantizer(bits=PRECISION_TO_BIT[self.wbit], has_zero=w_has_zero, qtype=self.w_qtype, groupsize=self.w_groupsize, unsign=w_unsign) 
        if self.abit not in [Precision.FP16, Precision.FP32]:
            self.a_quantizer = Quantizer(bits=PRECISION_TO_BIT[self.abit], has_zero=a_has_zero, qtype=self.a_qtype, groupsize=self.a_groupsize, unsign=a_unsign)

    def add_hook(self):
        if self.abit not in [Precision.FP16, Precision.FP32]:
            if track_input_hook_to_cpu not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_input_hook_to_cpu)
    
    @torch.no_grad()
    def quantize(self):
        #quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            w = self.quant_hub_linear.core.weight.to(self.device)
            fake_w, scales, zero_points = self.w_quantizer.quantize_dequantize(data=w)
            self.fake_w = fake_w
            self.w_scale = scales
            self.w_zero_point = zero_points
            del scales, zero_points, fake_w

        #quantize activation
        if self.abit not in [Precision.FP16, Precision.FP32]:
            if self.quantization_type == 'static':
                if self.a_qtype == 'per_tensor':
                    max_seq_len = max(x[0].shape[1] for x in self.quant_hub_linear.core.input_tracks)
                    x = torch.cat([F.pad(x[0], (0, 0, 0, max_seq_len - x[0].shape[1])) for x in self.quant_hub_linear.core.input_tracks], dim=0)
                    x = x.to(self.device)
                    scales, zero_points = self.a_quantizer.find_params(x_min=x.min(), x_max=x.max())
                    self.a_scale = scales
                    self.a_zero_point = zero_points
                    del scales, zero_points, x, self.quant_hub_linear.core.input_tracks
                else:
                    logging.info('just a_qtype is per_tensor support static quantize')
            elif self.quantization_type == 'dynamic':
                pass
            else:
                raise ValueError('quantization type support static and dynamic')
        clear_mem()

    def forward(self, x):
        orgin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
            if self.quantization_type == 'static':
                quantized_x = self.a_quantizer.quantize(x, self.a_scale, self.a_zero_point)
                x = self.a_quantizer.dequantize(quantized_x, self.a_scale, self.a_zero_point)
            elif self.quantization_type == 'dynamic':
                x, scales, zero_points = self.a_quantizer.quantize_dequantize(data=x)
            else:
                raise ValueError('quantization type support static and dynamic')
            
        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float().to(x)
        else:
            w = self.fake_w.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(orgin_dtype)
    
    def to(self, desc):
        if hasattr(self, 'w_scale'):
            self.w_scale = self.w_scale.to(desc)
        if hasattr(self, 'w_zero_point'):
            self.w_zero_point = self.w_zero_point.to(desc)
        if hasattr(self, 'fake_w'):
            self.fake_w = self.fake_w.to(desc)
        if hasattr(self, 'a_scale'):
            self.a_scale = self.a_scale.to(desc)
        if hasattr(self, 'a_zero_point'):
            self.a_zero_point = self.a_zero_point.to(desc)
        return self
 