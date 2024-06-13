import re
from statistics import mean
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from mi_optimize.quantization.layers import LinearQuantHub

def cal_div_loss(result_1, result_2):
    return mean(F.kl_div(F.log_softmax(x_e.float(), dim=-1), F.log_softmax(y_e.float(), dim=-1), reduction='batchmean', log_target=True).item() for x_e, y_e in zip(result_1, result_2))

def _to_device(tensors, device='cuda'):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    if isinstance(tensors, list):
        return [_to_device(t, device) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple([_to_device(t, device) for t in tensors])
    if isinstance(tensors, dict):
        return {k: _to_device(v, device) for k, v in tensors.items()}
    raise ValueError("Unknown type of tensors: {}".format(type(tensors)))

def _to_cpu(tensors):
    return _to_device(tensors, 'cpu')


def _to_gpu(tensors):
    return _to_device(tensors, 'cuda')


def replace_module(model, module_type=torch.nn.Linear, new_module_type=LinearQuantHub, exclude_layers=[], include_layers=['.*'], display=False):
    if display:
        def count_children(module, name=''):
            count = 0
            for child_name, mod in list(module.named_children()):
                if any(re.fullmatch(pat, name + child_name) for pat in include_layers):
                    if any(re.fullmatch(pat, name + child_name) for pat in exclude_layers):
                        continue
                    if isinstance(mod, module_type):
                        count += 1
                    else:
                        count += count_children(mod, name + child_name + '.')
            return count
        count = count_children(model, name='')
        bar = tqdm(total=count)

    # transform in-place
    def transform_children(module, name=''):
        for child_name, mod in list(module.named_children()):
                if any(re.fullmatch(pat, name + child_name) for pat in include_layers):
                    if any(re.fullmatch(pat, name + child_name) for pat in exclude_layers):
                        continue
                if isinstance(mod, module_type):
                    if display:
                        bar.update(1)
                    try:
                        setattr(module, child_name, new_module_type(mod, name=child_name))
                    except:
                        setattr(module, child_name, new_module_type(mod))
                else:
                    transform_children(mod, name + child_name + '.')

    transform_children(model, name='')
    return model


def find_layers(module, layers, name=''):
    if isinstance(layers, list):
        layers = tuple(layers)
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
