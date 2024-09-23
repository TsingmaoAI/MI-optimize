import torch
import yaml
import logging
import json
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer
from mi_optimize import quantize
from mi_optimize import Benchmark
from mi_optimize.export import export_module
logging.basicConfig(level=logging.INFO)

def main(args):

    results_json = {}

    # Define quantization configuration
    with open(args.quant_config, 'r') as file:
        config = yaml.safe_load(file)

    # Load the pre-trained Hugging Face model
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).half()  
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Quantize the model
    quant_model = quantize(model, tokenizer=tokenizer, quant_config=config['quant_config'])

    quant_model = quant_model.eval()
    quant_model.to(config['quant_config']['kwargs']['device'])
    
    benchmark = Benchmark()
    # Evaluate Perplexity (PPL) on various datasets
    if args.eval_ppl:
        results = benchmark.eval_ppl(model, tokenizer)
        logging.info("Evaluating Perplexity (PPL) === SUCCESS ===")
        print(results)
        results_json['ppl'] = results
        
    # Evaluate the model on the ceval_benchmark
    if args.eval_ceval:
        results = benchmark.eval_ceval(model, tokenizer, model_type='llama', subject='all', num_shot=0)
        logging.info("Evaluating C-EVAL === SUCCESS ===")
        print(results)
        results_json['eval_ceval'] = results

    # Evaluate the model on the mmlu benchmark
    if args.eval_cmmlu:
        results = benchmark.eval_cmmlu(model, tokenizer, model_type=model.config.model_type, subject='all', split='test-source', num_shot=0)
        logging.info("Evaluating CMMLU === SUCCESS ===")
        print(results)
        results_json['eval_cmmlu'] = results

    # Evaluate the model on the BOSS benchmark
    if args.eval_boss:
        results = benchmark.eval_boss(model, tokenizer, test_dataset='QuestionAnswering_advqa', split='test', ICL_split='test', num_shot=0)
        logging.info("Evaluating BOSS === SUCCESS ===")
        print(results)
        results_json['eval_boss'] = results

    # Evaluate using lm-evaluation-harness
    if args.eval_lmeval:
        eval_tasks = [
            "winogrande",       
            "piqa",             
            "hellaswag",       
        ]
        results = benchmark.eval_lmeval(model, tokenizer, eval_tasks, num_shot=5)
        logging.info("Evaluating lm-eval === SUCCESS ===")
        print(results)
        results_json['eval_lmeval'] = results

    #save result 
    output_path = "result.json"
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=4)

    logging.info("Process Finished!")
    
    #save model 
    quant_model.to('cpu')
    quant_model = export_module(quant_model)
    torch.save(quant_model, args.save)
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--save', type=str, default='llama-2-7b-quant.pth')
    parser.add_argument('--quant-config', type=str, default='./configs/rtn_quant_config.yaml')
    parser.add_argument('--eval-ppl', action='store_true', help='')
    parser.add_argument('--eval-ceval', action='store_true', help='')
    parser.add_argument('--eval-cmmlu', action='store_true', help='')
    parser.add_argument('--eval-boss', action='store_true', help='')
    parser.add_argument('--eval-lmeval', action='store_true', help='')
    args = parser.parse_args()
    main(args)