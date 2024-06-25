import torch
from mi_optimize import Benchmark
from transformers import LlamaTokenizer, AutoModelForCausalLM

def main(args):
    # Load Benchmark
    benchmark = Benchmark()

    # Load Model && tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model = torch.load(args.quant_model)
    print(model)
    model = model.eval()

    model.cuda()

    # Evaluate Perplexity (PPL) on various datasets
    if args.eval_ppl:
        results = benchmark.eval_ppl(model, tokenizer)
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

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--quant-model', type=str, default='llama-2-7b-quant.pth')
    parser.add_argument('--eval-ppl', action='store_true', help='')
    parser.add_argument('--eval-ceval', action='store_true', help='')
    parser.add_argument('--eval-cmmlu', action='store_true', help='')
    parser.add_argument('--eval-boss', action='store_true', help='')
    parser.add_argument('--eval-lmeval', action='store_true', help='')
    args = parser.parse_args()
    main(args)