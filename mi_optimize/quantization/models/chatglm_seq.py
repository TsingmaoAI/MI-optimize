import torch
from torch import nn
import logging
import tqdm

from ..utils import replace_module, find_layers
from ..layers import LinearQuantHub
from mi_optimize.quantization.quantizer import *
from mi_optimize.memory import clear_mem

def chatglm_sequential(model, algo, data, **kwargs):
    device = kwargs.get('device', 'cuda')
    offload = kwargs.get('offload', 'cpu')
    block_sequential = kwargs.get('block_sequential', False)
    layer_sequential = kwargs.get('layer_sequential', False) 
    
    with torch.no_grad() :
        replace_module(model, exclude_layers=kwargs.get('skip_layers'), include_layers=['.*'])
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers
        
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        layers[0] = layers[0].to(device)

        dtype = next(iter(model.parameters())).dtype

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.inputs = []
                self.attention_mask = []
                self.rotary_pos_emb = []
            def forward(self, input, **kwargs):
                self.inputs.append(input)
                self.attention_mask.append(kwargs['attention_mask'])
                self.rotary_pos_emb.append(kwargs['rotary_pos_emb'])
                raise ValueError
            
        layers[0] = Catcher(layers[0])
        for batch in data:
            try:
                model(batch.to(device))
            except ValueError:
                pass

        inputs = layers[0].inputs
        attention_mask = layers[0].attention_mask
        rotary_pos_emb = layers[0].rotary_pos_emb
        layers[0] = layers[0].module

        layers[0] = layers[0].to(offload)
        model.model.embed_tokens = model.model.embed_tokens.to(offload)
        model.model.norm = model.model.norm.to(offload)
        torch.cuda.empty_cache()
        
        quant_outputs = [None] * len(inputs)
        fp_outputs = [None] * len(inputs) 

        for i in range(len(layers)):
            block = layers[i].to(device)
            if not block_sequential:
                for j in range(len(data)):
                    fp_outputs[j] = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device), rotary_pos_emb=rotary_pos_emb[j].to(device))[0].to(offload) 
            layer_linear = find_layers(block, (LinearQuantHub))
            if layer_sequential:
                sequential = [
                    ['self_attention.query_key_value'],
                    ['self_attention.dense'],
                    ['mlp.dense_h_to_4h'],
                    ['mlp.dense_4h_to_h']
                ]
            else:
                sequential = [list(layer_linear.keys())]


            for names in sequential:
                subset = {n: layer_linear[n] for n in names}
                for name, layer in subset.items():
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

                for j in range(len(data)):
                    _ = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device), rotary_pos_emb=rotary_pos_emb[j].to(device))[0].to(offload)
                for name, layer in tqdm(subset.items()):
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
                del subset
            if block_sequential:
                for j in range(len(data)):
                    quant_outputs[j] = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device), rotary_pos_emb=rotary_pos_emb[j].to(device))[0].to(offload) 

            # layers[i] = block.to(offload)
            del block
            clear_mem()
            if block_sequential:
                inputs, quant_outputs = quant_outputs, inputs
            else:
                inputs, fp_outputs = fp_outputs, inputs 

        del fp_outputs, inputs, quant_outputs
        clear_mem()
        model.config.use_cache = use_cache
    
    return model

