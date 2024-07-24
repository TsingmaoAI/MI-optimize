import torch
import time
import yaml
from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize.export import export_module

def main(args):
    # Define quantization configuration
    with open(args.quant_config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load the pre-trained Hugging Face model
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).half().cuda()  
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    
    input_text = "Llama is a large language model"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    output = model.generate(input_ids, max_length=20, num_return_sequences=1, do_sample= False)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print('full model outputs:', decoded_output)
    
    # Quantize the model
    trick = time.time()
    quant_model = quantize(model=model, tokenizer=tokenizer, quant_config=config['quant_config'])
    print(f'quantize time is {time.time()-trick}')

    # Save the quantized model
    quant_model = export_module(quant_model)
    torch.save(quant_model, args.save)

    output = quant_model.generate(input_ids, max_length=20, num_return_sequences=1, do_sample= False)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print('quantization model outputs:', decoded_output)

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--save', type=str, default='llama-2-7b-quant.pth')
    parser.add_argument('--quant-config', type=str, default='../rtn_quant_config.yaml')
    args = parser.parse_args()
    main(args)
