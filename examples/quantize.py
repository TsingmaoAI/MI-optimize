# Import necessary libraries
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
    "kwargs": {'w_dtype': "int8", 'a_dtype': "int8"},
    "calibrate_name": "ptb"  # select from  ['wikitext2', 'c4', 'ptb', 'cmmlu', 'cmmlu_hm', 'cmmlu_st', 'cmmlu_ss', 'NaturalLanguageInference_mnli']
 }

# Load the pre-trained Hugging Face model
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()  
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Quantize the model
model = quantize(model=model, tokenizer=tokenizer, quant_config=quant_config)

print(model)
model = model.cuda()

input_text = "Llama is a large language model"

input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

output = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample= False)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

# Save the quantized model
model = export_module(model)
torch.save(model, quant_path)