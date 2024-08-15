import torch
from torch import nn
import logging
from tqdm import tqdm

from ..utils import replace_module, find_layers
from ..layers import LinearQuantHub
from mi_optimize.quantization.quantizer import *
from mi_optimize.memory import clear_mem

@torch.no_grad()
def qwen_sequential(model, algo, data, **kwargs):
    device = kwargs.get('device', 'cuda')
    offload = kwargs.get('offload', 'cpu')
    
    model = model.to(device)
    logging.info('\n==== replace linear modules ====')
    if kwargs['abit'] <=8 or kwargs['wbit'] <=8:
        replace_module(model, exclude_layers=kwargs.get('skip_layers'), include_layers=['.*'])
        layers_linear = find_layers(model, (LinearQuantHub, ))
        for name, layer in tqdm(layers_linear.items()):
            layer: LinearQuantHub
            if algo=='awq':
                layer.register_quantizer(LinearAwqQuantizer(layer, **kwargs))
            elif algo=='gptq':
                layer.register_quantizer(LinearGPTQQuantizer(layer, **kwargs))
            elif algo=='rtn':
                layer.register_quantizer(LinearRTNQuantizer(layer, **kwargs))
            elif algo=='spqr':
                layer.register_quantizer(LinearSpqrQuantizer(layer, **kwargs))
            elif algo=='zeroquant':
                layer.register_quantizer(LinearZeroquantQuantizer(layer, **kwargs))
            elif algo=='smoothquant':
                layer.register_quantizer(LinearSmoothQuantizer(layer, **kwargs))
            elif algo=='quip':
                layer.register_quantizer(QuIPQuantizer(layer, **kwargs))
            elif algo=='awq+gptq':
                layer.register_quantizer(LinearAwqQuantizer(layer, **kwargs))
                layer.register_quantizer(LinearGPTQQuantizer(layer, **kwargs))
            elif algo=='smoothquant+gptq':
                layer.register_quantizer(LinearSmoothQuantizer(layer, **kwargs))
                layer.register_quantizer(LinearGPTQQuantizer(layer, **kwargs))
            else:
                raise RuntimeError(f'No {algo} Quantizer!')
            layer.prepare_hook()
            # iterate through some data
        for input in data:
            outputs = model(input.to(model.device))
    else:
        layers_linear = {}

    all_quant_layers = {**layers_linear}
    print('\n==== quantizing layers ====')
    for name, layer in tqdm(all_quant_layers.items()):
        if algo=='awq+gptq':
            layer.remove_hook()
            layer.quantizer[0].quantize()
            smooth_factor = layer.quantizer[0].smooth_factor
            smooth_weight = layer.core.weight.data.mul(smooth_factor)
            layer.core.weight.data = smooth_weight.to(layer.core.weight.device)
            layer.quantizer[1].quantize()
            Q = layer.quantizer[1].Q.value
            layer.quantizer[1].to(offload)
            layer.quantizer[0].Q = Q
            layer.set_default_quantizer(0)
            del layer.quantizer[1], layer.core.weight
            layer.to(offload)
            clear_mem()
        elif algo=='smoothquant+gptq':
            layer.remove_hook()
            layer.quantizer[0].quantize()
            smooth_factors = layer.quantizer[0].smooth_factor
            smooth_weight = layer.core.weight.data.mul(smooth_factors.view(1, -1))
            layer.core.weight.data = smooth_weight.to(layer.core.weight.data)
            layer.quantizer[1].quantize()
            Q = layer.quantizer[1].Q.value
            layer.quantizer[0].Q = Q
            layer.set_default_quantizer(0)
            del layer.quantizer[1], layer.core.weight
            layer.to(offload)
            clear_mem()
        else:
            layer.remove_hook() 
            layer.quantize()  
            layer.set_default_quantizer(0)
            del layer.core.weight
            layer.to(offload)
    return model