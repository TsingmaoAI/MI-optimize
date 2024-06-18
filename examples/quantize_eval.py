import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize import Benchmark
from mi_optimize.export import export_module

def main(args):
    # Define paths for the pre-trained model and quantized model
    model_path = args.model_path

    # Define quantization configuration
    quant_config = {
        "algo": "rtn",
        "kwargs": {'w_dtype': "int4", 'a_type': "float16"},
        "calibrate_name": "wikitext2"  # select from  ['wikitext2', 'c4', 'ptb', 'cmmlu', 'cmmlu_hm', 'cmmlu_st', 'cmmlu_ss']
    }

    # Load the pre-trained Hugging Face model
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()  
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Quantize the model
    model = quantize(model, tokenizer=tokenizer, quant_config=quant_config)
    
    model = model.eval()
    model.to('cuda')
    
    benchmark = Benchmark()
    # Evaluate Perplexity (PPL) on various datasets
    if args.eval_ppl:
        test_dataset = ['wikitext2']  
        results = benchmark.eval_ppl(model, tokenizer, test_dataset)
        print(results)
    
    # Evaluate the model on the ceval_benchmark
    if args.eval_ceval:
        results = benchmark.eval_ceval(model, tokenizer, model_type='llama', subject='all', num_shot=0)
        print(results)

    # Evaluate the model on the mmlu benchmark
    if args.eval_cmmlu:
        results = benchmark.eval_cmmlu(model, tokenizer, model_type='llama', subject='all', data_set='test-source', num_shot=0)
        print(results)

    # Evaluate the model on the BOSS benchmark
    if args.eval_boss:
        results = benchmark.eval_boss(model, tokenizer, test_dataset='QuestionAnswering_advqa', split='test', ICL_split='test', num_shot=0)
        print(results)

    # Evaluate using lm-evaluation-harness
    if args.eval_lmeval:
        eval_tasks = [
            "winogrande",       
            "piqa",             
            "hellaswag",       
        ]
        results = benchmark.eval_lmeval(model, tokenizer, eval_tasks, num_shot=5)
        print(results)
    quant_model = export_module(quant_model)
    torch.save(quant_model, args.quant_model_path)
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--quant-model-path', type=str, default='llama-2-7b-quant.pth')
    parser.add_argument('--eval-ppl', action='store_true', help='')
    parser.add_argument('--eval-ceval', action='store_true', help='')
    parser.add_argument('--eval-cmmlu', action='store_true', help='')
    parser.add_argument('--eval-boss', action='store_true', help='')
    parser.add_argument('--eval-lmeval', action='store_true', help='')
    args = parser.parse_args()
    main(args)