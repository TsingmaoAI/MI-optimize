from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize import Benchmark

# Define paths for the pre-trained model and quantized model
model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = 'llama-2-7b-quant.pth'

# Define quantization configuration
quant_config = {
    "algo": "rtn",
    "kwargs": {'w_dtype': "int4", 'a_type': "float16"},
    "calibrate_data": "wikitext2"  # select from  ['wikitext2', 'c4', 'ptb', 'cmmlu', 'cmmlu_hm', 'cmmlu_st', 'cmmlu_ss', 'NaturalLanguageInference_mnli']
 }

# Load the pre-trained Hugging Face model
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()  
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Quantize the model
model = quantize(model, quant_config=quant_config)

benchmark = Benchmark()
# Evaluate Perplexity (PPL) on various datasets
test_dataset = ['wikitext2']  
results_ppl = benchmark.eval_ppl(model, tokenizer, test_dataset)
print(results_ppl)