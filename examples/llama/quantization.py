import torch
import time
import logging
from transformers import LlamaTokenizer, LlamaForCausalLM
import argparse

from mi_optimize.quantization.models.llama_seq import llama_sequential
from mi_optimize import Benchmark
from mi_optimize.datasets.data_loader import get_calibrate_loader
import datetime

def load_model(model_name_or_path):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto')
    return model


def print_args(args):
    logging.info(f"--model: {args.model_path}")
    logging.info(f"--algo: {args.algo}")
    logging.info(f"--wbit: {args.wbit}")
    logging.info(f"--device: {args.device}")
    logging.info(f"当前时间是: {datetime.datetime.now()}")


if __name__=='__main__':
    import argparse    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--algo', type=str, default='None', choices=['rtn', 'gptq', 'awq', 'spqr', 'zeroquant', 'smoothquant', 'quip', 'awq+gptq', 'smoothquant+gptq'])
    parser.add_argument('--wbit', type=int, default=4) 
    parser.add_argument('--abit', type=int, default=16)
    parser.add_argument('--w-groupsize', type=int, default=128)
    parser.add_argument('--w-qtype', type=str, default='per_group')
    parser.add_argument('--benchmark', type=str, default='')
    parser.add_argument('--num-calibrate', type=int, default=1)
    parser.add_argument('--num-shot', type=int, default=0, help='')
    parser.add_argument('--calibrate-name', type=str, default='wikitext2', choices=['wikitext2', 'c4', 'ptb', 'cmmlu', 'cmmlu_hm', 'cmmlu_st', 'cmmlu_ss', 'NaturalLanguageInference_mnli'])
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--offload', type=str, default='cpu')
    parser.add_argument('--skip-layers', type=list, default=['lm_head'])
    parser.add_argument('--block-sequential', action='store_true', help='')
    parser.add_argument('--layer-sequential', action='store_true', help='')
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()
    args_dict = vars(args)
    
    print_args(args)
    
    model = load_model(args.model_path)
        
    model.eval()
    
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, legacy=False)
    

    calibrate_config ={'name':args.calibrate_name ,'nsamples':args.num_calibrate,'seqlen':args.seqlen}
    calibrate = get_calibrate_loader(tokenizer=tokenizer,calibrate_config=calibrate_config)
    tick = time.time()
    
    model = llama_sequential(model=model, data=calibrate, **args_dict)
    logging.info(f'quantize time {time.time() - tick}')
    
    model = model.to(args.device)
    benchmark = Benchmark()
    if args.benchmark == 'ceval':
        # Evaluate the model on the ceval benchmark
        results_ceval = benchmark.eval_ceval(model=model, tokenizer=tokenizer, model_type='llama', num_shot=args.num_shot)
        logging.info("\nCeval Benchmark Evaluation Results:")
        logging.info(results_ceval)
        
    if args.benchmark == 'mmlu':
        # Evaluate the model on the mmlu benchmark
        results_mmlu = benchmark.eval_cmmlu(model, tokenizer, model_type='llama', num_shot=args.num_shot)
        logging.info("\nMMLU Benchmark Evaluation Results:")
        logging.info(results_mmlu)
        
    if args.benchmark == 'boss':
        # Evaluate the model on the BOSS benchmark
        results_boss = benchmark.eval_boss(model, tokenizer, num_shot=args.num_shot)
        logging.info("\nBOSS Benchmark Evaluation Results:")
        logging.info(results_boss)
        
    if args.benchmark == 'lmeval':
        # Evaluate using lm-evaluation-harness
        eval_tasks = [
            "lambada_openai",  # Evaluating language model completion
            "piqa",            # Evaluating Physical Interaction QA
            "hellaswag",       # Evaluating Common Sense Natural Language Inference
        ]
        results_lm_evaluation = benchmark.eval_lmeval(model, num_shot=args.num_shot, eval_tasks=eval_tasks)
        logging.info("\nLM Evaluation Harness Evaluation Results:")
        logging.info(results_lm_evaluation)

        
    if args.save:
        from mi_optimize.export.utils import export_module
        model = export_module(model)
        torch.save(model, args.save)
        