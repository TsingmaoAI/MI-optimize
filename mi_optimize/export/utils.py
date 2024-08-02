import torch

from mi_optimize.quantization.utils import replace_module
from mi_optimize.export.qnn import QLinear
from mi_optimize.quantization.layers import LinearQuantHub
from mi_optimize.quantization.quantizer import LinearRTNQuantizer, LinearGPTQQuantizer, LinearSmoothQuantizer, LinearAwqQuantizer

def transform_layers(module):
    if isinstance(module, LinearQuantHub):
        if isinstance(module.default_quantizer, LinearRTNQuantizer):
            return QLinear.pack_from_rtn_quantizer(module.default_quantizer)
        if isinstance(module.default_quantizer, LinearGPTQQuantizer):
            return QLinear.pack_from_gptq_quantizer(module.default_quantizer)
        if isinstance(module.default_quantizer, LinearSmoothQuantizer):
            return QLinear.pack_from_smooth_quantizer(module.default_quantizer)
        if isinstance(module.default_quantizer, LinearAwqQuantizer):
            return QLinear.pack_from_awq_quantizer(module.default_quantizer)
    return module


def export_module(model: torch.nn.Module):
    return replace_module(model, LinearQuantHub, transform_layers, display=True)

    

