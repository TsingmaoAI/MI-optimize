# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize.export import export_module

def main(args):

    # Define paths for the pre-trained model and quantized model
    model_path = args.model_path
    quant_path = args.quant_path
    # Define quantization configuration
    quant_config = {
        "algo": "rtn",
        "kwargs": {'w_dtype': "int8", 'a_dtype': "int8"},
        "calibrate_name": "ceval_all"  # select from  ['wikitext2', 'c4', 'ptb', 'cmmlu_all', 'cmmlu_hm', 'cmmlu_st', 'cmmlu_ss']
    }

    # Load the pre-trained Hugging Face model
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()  
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Quantize the model
    model = quantize(model=model, tokenizer=tokenizer, quant_config=quant_config)

    input_text = "Llama is a large language model"

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    output = model.generate(input_ids)

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)

    # Save the quantized model
    model = export_module(model)
    torch.save(model, quant_path)

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--quant-path', type=str, default='llama-2-7b-quant.pth')
    args = parser.parse_args()
    main(args)
