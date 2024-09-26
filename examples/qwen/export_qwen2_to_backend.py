import numpy as np
import torch, gc, os
import yaml
from transformers import  AutoTokenizer

# import sys
# sys.path.append("/home/wufang/MI-optimize/examples/qwen")
from mi_optimize.quantization.models.qwen_modeling import Qwen2ForCausalLM

from mi_optimize import quantize
from mi_optimize.export import export_module
from mi_optimize import QLinear

torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
torch.nn.init.xavier_normal_ = lambda *args, **kwargs: None
torch.nn.init.xavier_uniform_ = lambda *args, **kwargs: None
torch.nn.init.normal_ = lambda *args, **kwargs: None
torch.nn.init.uniform_ = lambda *args, **kwargs: None

torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
torch.nn.init.xavier_normal_ = lambda *args, **kwargs: None
torch.nn.init.xavier_uniform_ = lambda *args, **kwargs: None
torch.nn.init.normal_ = lambda *args, **kwargs: None
torch.nn.init.uniform_ = lambda *args, **kwargs: None

torch.manual_seed(100)
np.random.seed(100)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/workspace/models/Qwen2-0.5B-Instruct")
parser.add_argument('--quant-config', type=str, default='./configs/rtn_quant_config.yaml')
args = parser.parse_args()

results_json = {}

# Define quantization configuration
with open(args.quant_config, 'r') as file:
    config = yaml.safe_load(file)

# Load the pre-trained Hugging Face model
model = Qwen2ForCausalLM.from_pretrained(args.model).float()
model: Qwen2ForCausalLM
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Quantize the model
model.disable_save()
quant_model = quantize(model, tokenizer=tokenizer, quant_config=config['quant_config'])

#save model 
quant_model.to('cpu')
quant_model = export_module(quant_model)
quant_model.save_pretrained('mi_model')
    
for i in range(len(model.model.layers)):
    os.makedirs(f"output/layer_{i}", exist_ok=True)
    
nhead = 32
dhead = 128
maxlen = 1024

for params in model.parameters():
    params.requires_grad_(False)

inputs = [6870, 92347, 7246]
#inputs = [6870]
input_id = np.array(inputs)[None].astype(np.int64)
position_id = np.arange(len(inputs))[None]

model.enable_save()

with torch.inference_mode():
    with torch.no_grad():
        out = model.forward(
            torch.tensor(input_id, dtype=torch.long),
            None, torch.tensor(position_id, dtype=torch.long),
            layer_forward=24
        )