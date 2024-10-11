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

DEVICE = torch.device('cpu')
DEBUG = True
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
            self.register_buffer('weight', torch.empty(out_channels, in_channels * w_bits // 32, dtype=torch.int32))
            ####
            # 按out_channels维度拼接
            #self.register_buffer('weight', torch.empty(out_channels * w_bits // 32, in_channels, dtype=torch.int32))
            ####
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
    
    def unpack_weight_hight(self, qweight, wbit):
        rows, cols = qweight.shape
        intweight_rows = rows * (32 // wbit)
        
        qweight = qweight.to(DEVICE)
        
        intweight = torch.zeros((intweight_rows, cols), dtype=torch.int32).to(DEVICE)
        
        idx_weight = (torch.arange(intweight_rows).to(DEVICE) * wbit) // 32
        off_weight = (torch.arange(intweight_rows).to(DEVICE) * wbit) % 32
        
        simple_mask = torch.tensor(BITMASK[wbit - 1], dtype=torch.int32).to(DEVICE)
        
        mask_simple = (wbit + off_weight <= 32)
        if mask_simple.any():
            shifts_simple = (32 - off_weight[mask_simple] - wbit).to(torch.int32)
            intweight[mask_simple] = torch.bitwise_and(
                torch.bitwise_right_shift(qweight[idx_weight[mask_simple]].to(torch.int32), shifts_simple[:, None]), 
                simple_mask
            )

        mask_complex = (wbit + off_weight > 32)
        if mask_complex.any():
            complex_mask1 = torch.tensor([BITMASK[32 - off - 1] for off in off_weight[mask_complex]], dtype=torch.int32).to(DEVICE)
            complex_mask2 = torch.tensor([BITMASK[wbit + off - 32 - 1] for off in off_weight[mask_complex]], dtype=torch.int32).to(DEVICE)
            
            shifts_complex1 = (wbit + off_weight[mask_complex] - 32).to(torch.int32)
            shifts_complex2 = (64 - wbit - off_weight[mask_complex]).to(torch.int32)
            
            idx_weight_complex = idx_weight[mask_complex].to(torch.int64)
            
            part1 = torch.bitwise_and(qweight[idx_weight_complex].to(torch.int32), complex_mask1[:, None])
            part1 = torch.bitwise_left_shift(part1, shifts_complex1[:, None])
            
            part2 = torch.bitwise_right_shift(qweight[idx_weight_complex + 1].to(torch.int32), shifts_complex2[:, None])
            part2 = torch.bitwise_and(part2, complex_mask2[:, None])
            
            intweight[mask_complex] = torch.bitwise_or(part1, part2)

        return intweight
    
    def unpack_weight_low(self, qweight, wbit):
        rows, cols = qweight.shape
        intweight_rows = rows * (32 // wbit)

        qweight = qweight.to(DEVICE)

        intweight = torch.zeros((intweight_rows, cols), dtype=torch.int32).to(DEVICE)

        idx_weight = (torch.arange(intweight_rows).to(DEVICE) * wbit) // 32
        off_weight = (torch.arange(intweight_rows).to(DEVICE) * wbit) % 32

        simple_mask = torch.tensor(BITMASK[wbit - 1], dtype=torch.int32).to(DEVICE)

        mask_simple = (wbit + off_weight <= 32)
        if mask_simple.any():
            shifts_simple = off_weight[mask_simple].to(torch.int32)
            intweight[mask_simple] = torch.bitwise_and(
                torch.bitwise_right_shift(qweight[idx_weight[mask_simple]].to(torch.int32), shifts_simple[:, None]),
                simple_mask
            )

        mask_complex = (wbit + off_weight > 32)
        if mask_complex.any():
            complex_mask1 = torch.tensor([BITMASK[off] for off in off_weight[mask_complex]], dtype=torch.int32).to(DEVICE)
            complex_mask2 = torch.tensor([BITMASK[wbit + off - 32 - 1] for off in off_weight[mask_complex]], dtype=torch.int32).to(DEVICE)

            shifts_complex1 = (wbit + off_weight[mask_complex] - 32).to(torch.int32)
            shifts_complex2 = (64 - wbit - off_weight[mask_complex]).to(torch.int32)

            idx_weight_complex = idx_weight[mask_complex].to(torch.int64)

            part1 = torch.bitwise_and(qweight[idx_weight_complex].to(torch.int32), complex_mask1[:, None])
            part1 = torch.bitwise_left_shift(part1, shifts_complex1[:, None])

            part2 = torch.bitwise_and(qweight[idx_weight_complex + 1].to(torch.int32), complex_mask2[:, None])
            part2 = torch.bitwise_right_shift(part2, shifts_complex2[:, None])

            intweight[mask_complex] = torch.bitwise_or(part1, part2)
        # 检查符号位，修正负数
        sign_bit_mask = 1 << (wbit - 1)
        negative_mask = intweight & sign_bit_mask != 0  # 检查符号位
        intweight[negative_mask] -= (1 << wbit)  # 转换为负数
        return intweight
    
    def unpack_weight(self, qweight, wbit):
        # 将 qweight 转换为 numpy 的 uint32 格式
        qweight = qweight.cpu().contiguous().numpy().astype(np.uint32)
        
        # 计算原始 intweight 的行数
        intweight_rows = qweight.shape[0] * 32 // wbit
        
        # 初始化解压后的 intweight，类型为 np.uint32
        intweight = np.zeros((intweight_rows, qweight.shape[1]), dtype=np.int32)
        
        # 计算符号位的掩码，用来判断是否为负数
        sign_bit_mask = 1 << (wbit - 1)
        
        # 从低位开始解压
        for i in range(intweight.shape[0]):
            idx_weight = (i * wbit) // 32
            off_weight = (i * wbit) % 32
            
            # 先从 qweight 中取出低位部分
            extracted_value = (qweight[idx_weight] >> off_weight) & BITMASK[wbit - 1]
            
            # 如果该部分跨越了32位边界，则需要从下一段中取出高位部分
            if wbit + off_weight > 32:
                extracted_value |= (qweight[idx_weight + 1] << (32 - off_weight)) & BITMASK[wbit - 1]
            
            # 调试信息
            # print(f"Row {i}, idx_weight: {idx_weight}, off_weight: {off_weight}, extracted_value (before sign check): {extracted_value}")
        
            # 检查符号位并转换为负数，逐个元素处理
            for j in range(extracted_value.shape[0]):
                if extracted_value[j] & sign_bit_mask:
                    # 将补码的负数转换为int32的负数形式
                    extracted_value[j] -= (1 << wbit)
            # 调试信息
            intweight[i] = extracted_value
        
        # 将 intweight 返回为 PyTorch 的 Tensor，并转换回 int32 类型
        return torch.from_numpy(intweight).to(torch.int32).to(DEVICE)

    @torch.no_grad()
    def forward(self, x):
        # print('x_shape', x.shape)
        if DEBUG:
            if self.w_bits==8 and self.a_bits==8:
                assert self.w_qtype != 'per_group',"weight type not support per group" 
                bsz = x.shape[0]
                w = self.weight.t()
                # w = self.unpack_weight_low(qweight=w, wbit=self.w_bits)
                w = self.unpack_weight_low(qweight=w, wbit=self.w_bits)
                w = w.t()
                
                # print('unpack_weight', w)
                # exit()
                w_scale = self.w_scale.reshape(-1, 1)
                w_zero = self.w_zero_point.reshape(-1, 1)
                # origin_shape = data.shape
                x = x.reshape(-1, x.shape[-1])
             
                x_min = x.amin(dim=1, keepdim=True)
                x_max = x.amax(dim=1, keepdim=True)
                # print('x_min_max', x_min, x_max)
                x_scale, x_zero = self.a_quantizer.find_params(x_min=x_min, x_max=x_max)
                int_x = self.a_quantizer.quantize(x, x_scale, x_zero)
                int_x = int_x.to(torch.int32)
                # for i in range(3):
                #     print('x_scale {:.8f}'.format(x_scale[i][0]))
                #     print('x_zero {: 8f}'.format(x_zero[i][0]))
                #     # print('int_x {:.6f}'.format(int_x[0, :10]))
                
                # for i in range(10):
                #     print('x, {:.8f}'.format(x[0][i]))
                #     print('qx, {:.8f}'.format(int_x[0][i]))
                w_zero_inter = w_zero.repeat_interleave(int_x.shape[-1], dim=1)

                output_int = ((torch.matmul(int_x, w.t())).to(torch.float32) - torch.matmul(int_x.to(torch.float32), w_zero_inter.t())) 
                output = output_int * torch.matmul(x_scale, w_scale.t())
                # out_int32 = torch.zeros(8, dtype = torch.int32)
                # out_int16 = torch.zeros(16, dtype=torch.int32)
                # out_int32_tmp = torch.zeros(8, dtype = torch.int32)
                # for i in range(0, w.shape[1], 32):
                #     count = 0
                #     for j in range(i, i+32, 2):
                #         x_tmp1 = int_x[0][j]
                #         w_tmp1 = w[51][j]
                #         x_tmp2 = int_x[0][j+1]
                #         w_tmp2 = w[51][j+1]
                #         mul_tmp = x_tmp1 * w_tmp1 + x_tmp2 * w_tmp2
                #         out_int16[count] = mul_tmp
                #         count+=1
                #     for k in range(8):
                #         out_int32_tmp[k] = out_int16[2*k] + out_int16[2*k+1]
                #     out_int32 += out_int32_tmp
                    # print('wwww', w[51][i:i+32])
                    # print('xxx_index', i, int_x[0][i:i+32])
                    # print("int16", out_int16)
                    # print("int32", out_int32_tmp)
                    # print("int32", out_int32)
                    
                # for i in range(51, 58):
                #     for j in range(3):
                #         print('tmpout, {:.8f}'.format((torch.matmul(int_x, w.t()))[j][i]))
                #         print('zerosum, {:.8f}'.format((torch.matmul(int_x.to(torch.float32), w_zero_inter.t())[j][i])))
                #         print('intsum, {:.8f}'.format(output_int[j][i]))
                #         print('scalesmu, {:.8f}'.format(torch.matmul(x_scale, w_scale.t())[j][i]))
                        
                #         print('output, {:.8f}'.format(output[j][i]))
                if self.bias is not None:
                    output += self.bias.view(1, -1)
                    # print('output_bias', output[:, 0])
                # exit()
                return output.reshape(bsz, -1, output.shape[-1])
            
        else:    
            if self.w_bits<=8:
                w = self.weight.t()
                w = self.unpack_weight_low(qweight=w, wbit=self.w_bits)
                w = w.t().to(x)
                # print('w_1', w[0][:32])
                out_channel, in_channel = w.shape
                if self.w_qtype == 'per_group' and self.w_groupsize > 0:   
                    w = w.reshape(-1, self.w_groupsize)
                scale = self.w_scale.reshape(-1, 1).to(w)
                zero = self.w_zero_point.reshape(-1, 1).to(w)
                w = (w - zero) * scale
                w = w.reshape(out_channel, in_channel)
                # print("w_scale", scale[:10])
                # print("w_zero", zero[:10])
                # print("w_q", w[0][:32])
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
                    
                    # print("x", x[0][0][:32])
                    x, scales, zero_points = self.a_quantizer.quantize_dequantize(x)
                    # print('x_scales', scales[0][0])
                    # print('x_zero_points', zero_points[0][0])
                    # print('qx', x[0][0][:32])
                    # exit()
                else:
                    raise ValueError('quantization_type: {} is not support', self.quantization_type)
            if self.bias is not None:
                self.bias = self.bias.to(x)
            y = F.linear(x, w, self.bias)
            # print('origin', y.shape, y)
            # exit()
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
            
            # print('w_pack', intweight)
            # exit()
            intweight = intweight.t().cpu().contiguous().numpy().astype(np.uint32)
           
            ###
            #按out_channels维度拼接
            #intweight = intweight.cpu().contiguous().numpy().astype(np.uint32)
            ###
            qweight   = np.zeros((intweight.shape[0] * wbit // 32, intweight.shape[1]), dtype=np.uint32)
            
            #从低位开始拼接
            for i in range(intweight.shape[0]):
                idx_weight = (i * wbit) // 32
                off_weight = (i * wbit) %  32
                
                low_bits = intweight[i] & BITMASK[wbit - 1]
                
                qweight[idx_weight] |= (low_bits << off_weight)
                if wbit + off_weight > 32:
                    qweight[idx_weight + 1] |= (low_bits >> (32 - off_weight)) & BITMASK[wbit - 32 + off_weight]
                    
            #从高位开始拼接
            # for i in range(intweight.shape[0]):
            #     idx_weight = (i * wbit) // 32
            #     off_weight = (i * wbit) %  32
            #     if wbit + off_weight <= 32:
            #         qweight[idx_weight] = qweight[idx_weight]<< wbit
            #         qweight[idx_weight] |= intweight[i]     
            #     else:
            #         qweight[idx_weight] = qweight[idx_weight] << (32 - off_weight)
            #         qweight[idx_weight] |= (intweight[i] >> (wbit - 32 + off_weight) )
            #         qweight[idx_weight + 1] |= (intweight[i]&BITMASK[wbit - 32 + off_weight-1])

            qlinear.weight.data.copy_(torch.from_numpy(qweight.T.astype(np.int32)))
            ###
            #按out_channels维度拼接
            #qlinear.weight.data.copy_(torch.from_numpy(qweight.astype(np.int32)))
            ###
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
                qweight[idx_weight] |= (intweight[i] << off_weight)
                if wbit + off_weight > 32:
                    qweight[idx_weight + 1] |= (intweight[i] >> (32 - off_weight)) & BITMASK[wbit - 32 + off_weight]
            # for i in range(intweight.shape[0]):
            #     idx_weight = (i * wbit) // 32
            #     off_weight = (i * wbit) %  32
            #     if wbit + off_weight <= 32:
            #         qweight[idx_weight] = qweight[idx_weight]<< wbit
            #         qweight[idx_weight] |= intweight[i]     
            #     else:
            #         qweight[idx_weight] = qweight[idx_weight] << (32 - off_weight)
            #         qweight[idx_weight] |= (intweight[i] >> (wbit - 32 + off_weight) )
            #         qweight[idx_weight + 1] |= (intweight[i]&BITMASK[wbit - 32 + off_weight-1])

            qlinear.weight.data.copy_(torch.from_numpy(qweight.T.astype(np.int32)))
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

            qlinear.weight.data.copy_(torch.from_numpy(qweight.T.astype(np.int32)))
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

            qlinear.weight.data.copy_(torch.from_numpy(qweight.T.astype(np.int32)))
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

    
    
    