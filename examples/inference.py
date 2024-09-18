import torch
import time
from transformers import LlamaTokenizer, TextGenerationPipeline
from mi_optimize.export import qnn

def main(args):
    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Load the quantized model
    model = torch.load(args.quant_model)

    model.cuda()

    # # Input prompt
    prompt = "Llama is a large language model"

    # # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

    # Choose the backend for inference ('naive', 'vllm', 'tensorrt')
    backend = 'naive'   

    if backend == 'naive':
        start_time = time.time()
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, do_sample=False)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(decoded_output)
        print(f'quantize time {time.time() - start_time}')

    elif backend == 'vllm':
        pass  # This will be added soon

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--quant-model', type=str, default='llama-2-7b-quant.pth')
    parser.add_argument('--backend', type=str, default='naive')
    args = parser.parse_args()
    main(args)