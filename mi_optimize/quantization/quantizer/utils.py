import torch
import math
import logging
from torch import nn

from mi_optimize.memory import clear_mem

def generate_track_input_output_hook(offload_device='cpu'):
    def track_input_output_hook(module, inputs, outputs):
        if not hasattr(module, 'input_output_tracks'):
            module.input_output_tracks = []
        if not torch.is_tensor(inputs):
            inputs = [inp.to(offload_device) for inp in inputs]
        else:
            inputs = [inputs.to(offload_device)]
        if not torch.is_tensor(outputs):
            outputs = [out.to(offload_device) for out in outputs]
        else:
            outputs = [outputs.to(offload_device)]
        module.input_output_tracks.append([inputs, outputs])
    return track_input_output_hook

track_input_output_hook_to_cpu = generate_track_input_output_hook('cpu')
track_input_output_hook_to_cuda = generate_track_input_output_hook('cuda')

def generate_track_input_hook(offload_device='cpu'):
    def track_input_hook(module, inputs, outputs):
        if not hasattr(module, 'input_tracks'):
            module.input_tracks = []
        if not torch.is_tensor(inputs):
            inputs = [inp.to(offload_device) for inp in inputs]
        else:
            inputs = [inputs.to(offload_device)]
        module.input_tracks.append(inputs)
    return track_input_hook

track_input_hook_to_cpu = generate_track_input_hook('cpu')
track_input_hook_to_cuda = generate_track_input_hook('cuda')

def generate_track_hessian_hook(offload_device):
    def track_hessian_hook(module, inputs, outputs):
        if not torch.is_tensor(inputs):
            inputs = [inp for inp in inputs]
        else:
            inputs = [inputs]

        if not hasattr(module, 'H'):
            module.H = torch.zeros(
                (module.in_features, module.in_features), device=offload_device)
        if not hasattr(module, 'nsamples'):
            module.nsamples = 0
        
        device = module.H.device
        inp = inputs[0].to(device)
        H = module.H
        
        if len(inp) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.size(0)
        inp = inp.reshape(-1, inp.size(-1))
        inp = inp.t()
        H *= module.nsamples / (module.nsamples + tmp)
        module.nsamples += tmp
        inp = math.sqrt(2 / module.nsamples) * inp.float()
        H += inp.matmul(inp.t())
        module.H = H.to(offload_device)
        del H, inp, tmp
        clear_mem()
    return track_hessian_hook

track_hessian_hook_to_cpu = generate_track_hessian_hook('cpu')
track_hessian_hook_to_cuda = generate_track_hessian_hook('cuda')

def generate_track_quip_hessian_hook(offload_device):
    def track_hessian_hook(module, inputs, outputs):
        if not torch.is_tensor(inputs):
            inputs = [inp for inp in inputs]
        else:
            inputs = [inputs]

        if not hasattr(module, 'H'):
            module.H = torch.zeros((module.in_features, module.in_features), dtype=torch.float64, device=offload_device)
        if not hasattr(module, 'nsamples'):
            module.nsamples = 0
        
        inp = inputs[0].to('cuda')
        H = module.H.to('cuda')
        if len(inp) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.size(0)
        inp = inp.reshape(-1, inp.size(-1))
        inp = inp.t()
        module.nsamples += tmp
        inp = inp.to(torch.float64)
        H += inp.matmul(inp.t())
        module.H = H.to(offload_device)
        del H, inp, tmp, inputs
        clear_mem()
    return track_hessian_hook

track_quip_hessian_hook_to_cpu = generate_track_quip_hessian_hook('cpu')
track_quip_hessian_hook_to_cuda = generate_track_quip_hessian_hook('cuda')


class Quantizer(torch.nn.Module):
    def __init__(self, bits=8, has_zero=False, qtype='per_tensor', groupsize=-1, unsign=True):
        super(Quantizer, self).__init__()
        self.bits = bits
        self.has_zero = has_zero
        self.qtype = qtype
        self.groupsize = groupsize
        if unsign:
            self.qmin = 0
            self.qmax = (1<<bits) - 1
        else:
            self.qmin = -(1<<(bits-1))
            self.qmax = (1<<(bits-1)) - 1
    
    def find_params(self, x_min, x_max):
        if not self.has_zero:
            max_abs_value = torch.max(abs(x_max), abs(x_min))
            scale = max_abs_value / ((self.qmax - self.qmin) // 2)
            # zero_point = 0 if self.qmin<0 else (self.qmax + self.qmin) // 2
            zero_point = 0 if self.qmin<0 else 1<<(self.bits-1)
            zero_point = zero_point * torch.ones_like(scale)
        else:
            scale = (x_max - x_min) / (self.qmax - self.qmin)
            zero_point = self.qmin - torch.round(x_min / scale)
        return scale, zero_point

    def quantize(self, data, scale, zero_point):
        quantized_data = torch.round(data / scale) + zero_point
        quantized_data = torch.clamp(quantized_data, self.qmin, self.qmax)
        return quantized_data
    
    def dequantize(self, quantized_data, scale, zero_point):
        dequantized_data = scale * (quantized_data - zero_point)
        return dequantized_data
        
    def quantize_dequantize(self, data):
        if self.qtype == 'per_tensor':
            x_min = data.min()
            x_max = data.max()
            scales, zero_points = self.find_params(x_min=x_min, x_max=x_max)
            quantized_data = self.quantize(data, scales, zero_points)
            dequantized_data = self.dequantize(quantized_data, scales, zero_points)
        elif self.qtype == 'per_channel':
            origin_shape = data.shape
            data.reshape(-1, origin_shape[-1])
            x_min = data.amin(dim=1, keepdim=True)
            x_max = data.amax(dim=1, keepdim=True)
            scales, zero_points = self.find_params(x_min=x_min, x_max=x_max)
            quantized_data = self.quantize(data, scales, zero_points)
            dequantized_data = self.dequantize(quantized_data, scales, zero_points)
            dequantized_data = dequantized_data.reshape(origin_shape)
        elif self.qtype == 'per_group':
            origin_shape = data.shape
            if self.groupsize == -1:
                logging.info("qtype is per_group, but groupsize==-1")
            elif self.groupsize > 0:
                assert origin_shape[-1] % self.groupsize == 0
            else:
                raise ValueError('groupsize is erro number')
            data = data.reshape(-1, self.groupsize)
            x_min = data.amin(dim=1, keepdim=True)
            x_max = data.amax(dim=1, keepdim=True)
            scales, zero_points = self.find_params(x_min=x_min, x_max=x_max)
            quantized_data = self.quantize(data, scales, zero_points)
            dequantized_data = self.dequantize(quantized_data, scales, zero_points)
            dequantized_data = dequantized_data.reshape(origin_shape)
            scales = scales.reshape(-1, origin_shape[-1] // self.groupsize)
            zero_points = zero_points.reshape(-1, origin_shape[-1] // self.groupsize)
        elif self.qtype == 'per_dimension':
            assert data.dim()==3, f'per_dimension just support x dim is 3, now x dim is {data.dim()}'
            origin_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            x_min = data.amin(dim=0, keepdim=True)
            x_max = data.amax(dim=0, keepdim=True)
            scales, zero_points = self.find_params(x_min=x_min, x_max=x_max)
            quantized_data = self.quantize(data, scale=scales, zero_point=zero_points)
            dequantized_data = self.dequantize(quantized_data, scales, zero_points)
        elif self.qtype == 'per_token':
            origin_shape = data.shape
            data = data.reshape(-1, origin_shape[-1])
            x_min = data.amin(dim=1, keepdim=True)
            x_max = data.amax(dim=1, keepdim=True)
            scales, zero_points = self.find_params(x_min=x_min, x_max=x_max)
            quantized_data = self.quantize(data, scales, zero_points)
            dequantized_data = self.dequantize(quantized_data, scales, zero_points)
            dequantized_data = dequantized_data.reshape(origin_shape)
        else:
            raise ValueError("Unsupported quantization type. Use 'per_tensor', 'per_channel', or 'per_group'. or 'per_dimension' or 'per_token'")
        
        return dequantized_data, scales, zero_points

