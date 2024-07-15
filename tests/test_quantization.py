import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize.export import export_module

# Define paths for the pre-trained model and quantized model
model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = 'llama-2-7b-quant.pth'

# Define quantization configuration
quant_config = {
    "algo": "rtn",
    "kwargs": {
        "w_dtype": "int4",           
        "a_dtype": "float16",        
        "device": "cuda",
        "offload": "cpu",
        "w_qtype": "per_channel",
        "w_has_zero": False,
        "w_unsign": True,
        "quantization_type": "static",
        "layer_sequential": True,
        "skip_layers": [             
            "lm_head"
        ]
    },
    "calibrate_config": {
        "name": "wikitext2",
        "split": "train",
        "nsamples": 1,
        "seqlen": 2048
    }}

# Load the pre-trained Hugging Face model
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()  
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Quantize the model
quantize_model = quantize(model=model, tokenizer=tokenizer, quant_config=quant_config)

# print('model device', model.device)
quantize_model.to('cuda')
print(quantize_model)
input_text = "Llama is a large language model"

input_ids = tokenizer.encode(input_text, return_tensors="pt").to(quantize_model.device)

output = model.generate(input_ids, max_length=20, num_return_sequences=1, do_sample= False)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

# Save the quantized model
model = export_module(model)
torch.save(model, quant_path)
