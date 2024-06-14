import torch
import time
from transformers import LlamaTokenizer, TextGenerationPipeline
from mi_optimize.export import qnn

# Path to the quantized model
quant_path = 'llama-2-7b-quant.pth'

# Path to the tokenizer
tokenizer_path = 'meta-llama/Llama-2-7b-hf'

# Load the quantized model
model = torch.load(quant_path)

model = model.cuda()

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# # Input prompt
prompt = "Llama is a large language model"

# # Tokenize the input prompt
inputs_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

# Choose the backend for inference ('naive', 'vllm', 'tensorrt')
backend = 'naive'   

if backend == 'naive':
    start_time = time.time()
    output = model.generate(inputs_ids, max_length=100, num_return_sequences=1, do_sample= False)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    print(f'quantize time {time.time() - start_time}')

elif backend == 'vllm':
    pass  # This will be added soon
