from abc import ABC
import torch
from torch.nn import functional as F
import sys
sys.path.append('../..')

from mi_optimize.quantization import Precision, PRECISION_TO_BIT
from mi_optimize.quantization.quantizer import LinearRTNQuantizer, LinearGPTQQuantizer, LinearAwqQuantizer, LinearSmoothQuantizer
from mi_optimize.quantization.quantizer import Quantizer
import numpy as np

class QModule(torch.nn.Module):
    pass

BITMASK = [
    0x1,
    0x3,
    0x7,
    0xf,
    0x1f,
    0x3f,
    0x7f,
    0xff
]


class QLinear(QModule):
    def __init__(self, in_channels, out_channels, bias=None, w_bits=4, a_bits=16, w_groupsize=128, a_groupsize=None, a_has_zero=False, a_qtype='per_token', w_has_zero=False, w_qtype='per_channel', quantization_type='dynamic', a_unsign=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.smooth_factor = None
        self.w_groupsize = w_groupsize
        self.a_groupsize = a_groupsize
        self.a_has_zero = a_has_zero
        self.w_has_zero = w_has_zero
        self.a_qtype = a_qtype
        self.w_qtype = w_qtype
        self.quantization_type = quantization_type
        self.a_unsign = a_unsign

        if bias is not None:
            self.register_buffer('bias', torch.empty(out_channels))
        else:
            self.register_buffer('bias', None)
        if w_bits <= 8:
            if w_qtype=='per_channel':
                self.register_buffer('w_scale', torch.empty(out_channels, 1))
                self.register_buffer('w_zero_point', torch.empty(out_channels, 1))
            elif w_qtype=='per_group':
                self.register_buffer('w_scale', torch.empty([out_channels, in_channels//w_groupsize]))
                self.register_buffer('w_zero_point', torch.empty([out_channels, in_channels//w_groupsize]))
            elif w_qtype == 'per_tensor':
                self.register_buffer('w_scale', torch.empty([1]))
                self.register_buffer('w_zero_point', torch.empty([1]))
            else:
                raise ValueError('not support weight qtype:{}'.format(w_qtype))
            self.register_buffer('pack_weight', torch.empty(in_channels * w_bits // 32, out_channels, dtype=torch.int32))
        else:
            self.register_buffer('weight', torch.empty(out_channels, in_channels))
            self.register_buffer('w_scale', None)
            self.register_buffer('w_zero_point', None)

        if a_bits <= 8:
            if a_qtype == 'per_channel':
                self.register_buffer('a_scale', torch.empty(out_channels))
                self.register_buffer('a_zero_point', torch.empty(out_channels))
            elif a_qtype == 'per_tensor':
                self.register_buffer('a_scale', torch.empty([1]))
                self.register_buffer('a_zero_point', torch.empty([1]))
            elif a_qtype == 'per_token':
                assert quantization_type =='dynamic', 'per token quantization only support dynamic'
            else:
                raise ValueError('not support activate qtype:{}'.format(a_qtype))
            self.a_quantizer = Quantizer(bits=PRECISION_TO_BIT[a_bits], has_zero=a_has_zero, qtype=a_qtype, groupsize=a_groupsize, unsign=self.a_unsign)
        else:
            self.register_buffer('a_scale', None)
            self.register_buffer('a_zero_point', None)
    
    def unpack_weight(self, qweight, wbit):
        rows, cols = qweight.shape
        intweight_rows = rows * (32 // wbit)
        
        qweight = qweight.to('cuda')
        
        intweight = torch.zeros((intweight_rows, cols), dtype=torch.int32, device='cuda')
        
        idx_weight = (torch.arange(intweight_rows, device='cuda') * wbit) // 32
        off_weight = (torch.arange(intweight_rows, device='cuda') * wbit) % 32
        
        simple_mask = torch.tensor(BITMASK[wbit - 1], dtype=torch.int32, device='cuda')
        
        mask_simple = (wbit + off_weight <= 32)
        if mask_simple.any():
            shifts_simple = (32 - off_weight[mask_simple] - wbit).to(torch.int32)
            intweight[mask_simple] = torch.bitwise_and(
                torch.bitwise_right_shift(qweight[idx_weight[mask_simple]].to(torch.int32), shifts_simple[:, None]), 
                simple_mask
            )

        mask_complex = (wbit + off_weight > 32)
        if mask_complex.any():
            complex_mask1 = torch.tensor([BITMASK[32 - off - 1] for off in off_weight[mask_complex]], dtype=torch.int32, device='cuda')
            complex_mask2 = torch.tensor([BITMASK[wbit + off - 32 - 1] for off in off_weight[mask_complex]], dtype=torch.int32, device='cuda')
            
            shifts_complex1 = (wbit + off_weight[mask_complex] - 32).to(torch.int32)
            shifts_complex2 = (64 - wbit - off_weight[mask_complex]).to(torch.int32)
            
            idx_weight_complex = idx_weight[mask_complex].to(torch.int64)
            
            part1 = torch.bitwise_and(qweight[idx_weight_complex].to(torch.int32), complex_mask1[:, None])
            part1 = torch.bitwise_left_shift(part1, shifts_complex1[:, None])
            
            part2 = torch.bitwise_right_shift(qweight[idx_weight_complex + 1].to(torch.int32), shifts_complex2[:, None])
            part2 = torch.bitwise_and(part2, complex_mask2[:, None])
            
            intweight[mask_complex] = torch.bitwise_or(part1, part2)

        return intweight

    @torch.no_grad()
    def forward(self, x):
        if self.w_bits<=8:
            w = self.pack_weight
            w = self.unpack_weight(qweight=w, wbit=self.w_bits)
            w = w.t().to(x)
            out_channel, in_channel = w.shape
            if self.w_qtype == 'per_group' and self.w_groupsize > 0:   
                w = w.reshape(-1, self.w_groupsize)
            scale = self.w_scale.reshape(-1, 1).to(w)
            zero = self.w_zero_point.reshape(-1, 1).to(w)
            w = (w - zero) * scale
            w = w.reshape(out_channel, in_channel)
        else:
            w = self.weight.to(x)
        if self.smooth_factor is not None:
            x = x.div(self.smooth_factor.view(1, -1).to(x.device))
        if self.a_bits <= 8:
            if self.quantization_type == 'static':
                scales = self.a_scale.to(x)
                zero_points = self.a_zero_point.to(x)
                intx = self.a_quantizer.quantize(x, scale=scales, zero_point=zero_points)
                x = self.a_quantizer.dequantize(intx, scale=scales, zero_point=zero_points)
            elif self.quantization_type == 'dynamic':
                # print("xxx origin", x[0][:1][:10])
                x, scales, zero_points = self.a_quantizer.quantize_dequantize(x)
                # print('scales', scales[:1][:])
                # print('zero_points', zero_points[:1][:])
                # print('xxx', x[0][:1][:10])
                # exit()
            else:
                raise ValueError('quantization_type: {} is not support', self.quantization_type)
        if self.bias is not None:
            self.bias = self.bias.to(x)
        return F.linear(x, w, self.bias)

    @classmethod
    def pack_from_rtn_quantizer(cls, module: LinearRTNQuantizer):
        qlinear = cls(
            in_channels=module.quant_hub_linear.core.in_features,
            out_channels=module.quant_hub_linear.core.out_features,
            bias=module.quant_hub_linear.core.bias is not None, 
            w_bits=PRECISION_TO_BIT[module.wbit],
            a_bits=PRECISION_TO_BIT[module.abit],
            w_groupsize=module.w_groupsize,
            a_groupsize = module.a_groupsize,
            a_qtype= module.a_qtype,
            w_qtype=module.w_qtype,
            quantization_type = module.quantization_type,
            a_unsign = module.a_unsign
            
        )
        
        bias   = module.quant_hub_linear.core.bias
        
        if module.abit <= Precision.INT8 and module.quantization_type=='static':
            qlinear.a_scale.data.copy_(module.a_scale)
            qlinear.a_zero_point.data.copy_(module.a_zero_point)

        fake_w = module.fake_w
        if module.wbit <= Precision.INT8:
            fake_w = module.fake_w
            if module.w_qtype=='per_group' and module.w_groupsize != -1:
                fake_w = fake_w.reshape(-1, module.w_groupsize)
            w_scale = module.w_scale
            zero_point = module.w_zero_point
            
            wbit = PRECISION_TO_BIT[module.wbit]
            intweight = ((fake_w.data / w_scale.reshape(-1, 1)) + zero_point.reshape(-1, 1)).float().round().int()
            
            if module.w_qtype=='per_group' and module.w_groupsize != -1:
                intweight = intweight.reshape_as(module.fake_w)
            intweight = intweight.t().cpu().contiguous().numpy().astype(np.uint32)
            qweight   = np.zeros((intweight.shape[0] * wbit // 32, intweight.shape[1]), dtype=np.uint32)
            
            for i in range(intweight.shape[0]):
                idx_weight = (i * wbit) // 32
                off_weight = (i * wbit) %  32
                if wbit + off_weight <= 32:
                    qweight[idx_weight] = qweight[idx_weight]<< wbit
                    qweight[idx_weight] |= intweight[i]     
                else:
                    qweight[idx_weight] = qweight[idx_weight] << (32 - off_weight)
                    qweight[idx_weight] |= (intweight[i] >> (wbit - 32 + off_weight) )
                    qweight[idx_weight + 1] |= (intweight[i]&BITMASK[wbit - 32 + off_weight-1])

            qlinear.pack_weight.data.copy_(torch.from_numpy(qweight.astype(np.int32)))
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None  
            
            qlinear.w_scale.data.copy_(w_scale)
            
            qlinear.w_zero_point.data.copy_(zero_point)
        else:
            qlinear.weight.data.copy_(fake_w)
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None
        return qlinear

    @classmethod
    def pack_from_gptq_quantizer(cls, module: LinearGPTQQuantizer):
        qlinear = cls(
            in_channels=module.quant_hub_linear.core.in_features,
            out_channels=module.quant_hub_linear.core.out_features,
            bias=module.quant_hub_linear.core.bias is not None, 
            w_bits=PRECISION_TO_BIT[module.wbit],
            a_bits=PRECISION_TO_BIT[module.abit],
            w_groupsize=module.groupsize,
            a_qtype= module.a_qtype,
            w_qtype=module.w_qtype
        )
        bias = module.quant_hub_linear.core.bias
        
        if module.abit <= Precision.INT8:
            qlinear.a_scale.data.copy_(module.a_scale)
            qlinear.a_zero_point.data.copy_(module.a_zero_point)

        fake_w = module.fake_w
        if module.wbit <= Precision.INT8:
            fake_w = module.fake_w
            if module.w_qtype=='per_group' and module.w_groupsize != -1:
                fake_w = fake_w.reshape(-1, module.w_groupsize)
            w_scale = module.w_scale
            zero_point = module.w_zero_point
            
            wbit = PRECISION_TO_BIT[module.wbit]
            intweight = ((fake_w.data / w_scale.reshape(-1, 1)) + zero_point.reshape(-1, 1)).float().round().int()
            
            if module.w_qtype=='per_group' and module.w_groupsize != -1:
                intweight = intweight.reshape_as(module.fake_w)
            intweight = intweight.t().cpu().contiguous().numpy().astype(np.uint32)
            qweight   = np.zeros((intweight.shape[0] * wbit // 32, intweight.shape[1]), dtype=np.uint32)
            
            for i in range(intweight.shape[0]):
                idx_weight = (i * wbit) // 32
                off_weight = (i * wbit) %  32
                if wbit + off_weight <= 32:
                    qweight[idx_weight] = qweight[idx_weight]<< wbit
                    qweight[idx_weight] |= intweight[i]     
                else:
                    qweight[idx_weight] = qweight[idx_weight] << (32 - off_weight)
                    qweight[idx_weight] |= (intweight[i] >> (wbit - 32 + off_weight) )
                    qweight[idx_weight + 1] |= (intweight[i]&BITMASK[wbit - 32 + off_weight-1])

            qlinear.pack_weight.data.copy_(torch.from_numpy(qweight.astype(np.int32)))
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None
            

            qlinear.w_scale.data.copy_(w_scale)
            
            qlinear.w_zero_point.data.copy_(zero_point)
        else:
            qlinear.weight.data.copy_(fake_w)
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None
        return qlinear
    
    @classmethod
    def pack_from_awq_quantizer(cls, module:  LinearAwqQuantizer):
        qlinear = cls(
            in_channels=module.quant_hub_linear.core.in_features,
            out_channels=module.quant_hub_linear.core.out_features,
            bias=module.quant_hub_linear.core.bias is not None, 
            w_bits=PRECISION_TO_BIT[module.wbit],
            a_bits=PRECISION_TO_BIT[module.abit],
            w_groupsize=module.groupsize,
            w_qtype=module.w_qtype
        )
        
        bias = module.quant_hub_linear.core.bias
        
        if module.wbit <= Precision.INT8:
            qlinear.smooth_factor = module.smooth_factor
            fake_w = module.fake_w
            if module.w_qtype=='per_group' and module.groupsize != -1:
                fake_w = fake_w.reshape(-1, module.groupsize)
            w_scale = module.w_scale
            zero_point = module.w_zero_point
            
            wbit = PRECISION_TO_BIT[module.wbit]
            intweight = ((fake_w.data / w_scale.reshape(-1, 1)) + zero_point.reshape(-1, 1)).float().round().int()
            
            if module.w_qtype=='per_group' and module.groupsize != -1:
                intweight = intweight.reshape_as(module.fake_w)
            intweight = intweight.t().cpu().contiguous().numpy().astype(np.uint32)
            qweight   = np.zeros((intweight.shape[0] * wbit // 32, intweight.shape[1]), dtype=np.uint32)
            
            for i in range(intweight.shape[0]):
                idx_weight = (i * wbit) // 32
                off_weight = (i * wbit) %  32
                if wbit + off_weight <= 32:
                    qweight[idx_weight] = qweight[idx_weight]<< wbit
                    qweight[idx_weight] |= intweight[i]     
                else:
                    qweight[idx_weight] = qweight[idx_weight] << (32 - off_weight)
                    qweight[idx_weight] |= (intweight[i] >> (wbit - 32 + off_weight) )
                    qweight[idx_weight + 1] |= (intweight[i]&BITMASK[wbit - 32 + off_weight-1])

            qlinear.pack_weight.data.copy_(torch.from_numpy(qweight.astype(np.int32)))
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None
            
            qlinear.w_scale.data.copy_(w_scale)
            
            qlinear.w_zero_point.data.copy_(zero_point)
        else:
            qlinear.weight.data.copy_(fake_w)
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None
        return qlinear
    
    
    @classmethod
    def pack_from_smooth_quantizer(cls, module: LinearSmoothQuantizer):
        qlinear = cls(
            in_channels=module.quant_hub_linear.core.in_features,
            out_channels=module.quant_hub_linear.core.out_features,
            bias=module.quant_hub_linear.core.bias is not None, 
            w_bits=PRECISION_TO_BIT[module.wbit],
            a_bits=PRECISION_TO_BIT[module.abit],
            w_groupsize=module.groupsize,
            a_qtype= module.a_qtype,
            w_qtype=module.w_qtype,
            quantization_type=module.quantization_type
        )
        
        bias = module.quant_hub_linear.core.bias
        
        if module.abit <= Precision.INT8:
            qlinear.smooth_factor = module.smooth_factor
            
        if module.wbit <= Precision.INT8:
            fake_w = module.fake_w
            if module.w_qtype=='per_group' and module.groupsize != -1:
                fake_w = fake_w.reshape(-1, module.groupsize)
            w_scale = module.w_scale
            zero_point = module.w_zero_point
            
            wbit = PRECISION_TO_BIT[module.wbit]
            intweight = ((fake_w.data / w_scale.reshape(-1, 1)) + zero_point.reshape(-1, 1)).float().round().int()
            
            if module.w_qtype=='per_group' and module.groupsize != -1:
                intweight = intweight.reshape_as(module.fake_w)
            intweight = intweight.t().cpu().contiguous().numpy().astype(np.uint32)
            qweight   = np.zeros((intweight.shape[0] * wbit // 32, intweight.shape[1]), dtype=np.uint32)
            
            for i in range(intweight.shape[0]):
                idx_weight = (i * wbit) // 32
                off_weight = (i * wbit) %  32
                if wbit + off_weight <= 32:
                    qweight[idx_weight] = qweight[idx_weight]<< wbit
                    qweight[idx_weight] |= intweight[i]     
                else:
                    qweight[idx_weight] = qweight[idx_weight] << (32 - off_weight)
                    qweight[idx_weight] |= (intweight[i] >> (wbit - 32 + off_weight) )
                    qweight[idx_weight + 1] |= (intweight[i]&BITMASK[wbit - 32 + off_weight-1])

            qlinear.pack_weight.data.copy_(torch.from_numpy(qweight.astype(np.int32)))
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None
            
            qlinear.w_scale.data.copy_(w_scale)
            
            qlinear.w_zero_point.data.copy_(zero_point)
        else:
            qlinear.weight.data.copy_(fake_w)
            if bias is not None:
                qlinear.bias.data.copy_(bias)
            else:
                qlinear.bias = None
        return qlinear

    
    
    