import torch
from mi_optimize import Benchmark
from transformers import LlamaTokenizer, AutoModelForCausalLM

# model_path = 'meta-llama/Llama-2-7b-hf'
model_path = 'meta-llama/Llama-2-7b-hf'
quantize_model_path = 'llama-2-7b-quant.pth'
# Load Benchmark
benchmark = Benchmark()

# Load Model && tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = torch.load(quant_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()

# Evaluate Perplexity (PPL) on various datasets
test_dataset = ['wikitext2']  
results_ppl = benchmark.eval_ppl(model, tokenizer, test_dataset)
print(results_ppl)

# Evaluate the model on the ceval_benchmark
results_ceval = benchmark.eval_ceval(model, tokenizer, model_type='baichuan', subject='all', num_shot=0)
print(results_ceval)

# Evaluate the model on the mmlu benchmark
results_cmmlu = benchmark.eval_cmmlu(model, tokenizer, model_type='baichuan', subject='all', num_shot=0)
print(results_cmmlu)

# Evaluate the model on the BOSS benchmark
results_boss = benchmark.eval_boss(model, tokenizer, test_dataset='QuestionAnswering_advqa', split='test', ICL_split='test', num_shot=0)
print(results_boss)

# Evaluate using lm-evaluation-harness
eval_tasks = [
    "winogrande",       
    "piqa",             
    "hellaswag",       
]
results_lm_evaluation = benchmark.eval_lmeval(model, tokenizer, eval_tasks, num_shot=5)
print(results_lm_evaluation)