import argparse
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from mi_optimize import Benchmark 


def main(args: argparse.Namespace):
    logging.info(args)

    # Initialize the model and tokenizer
    if args.quantized_model:
        model = torch.load(args.model)
        model.eval()
        tokenizer = None  # Assuming quantized models don't require tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Benchmark ceval
    logging.info("\nEvaluating the model on the ppl_benchmark...")
    benchmark = Benchmark()
    results = benchmark.eval_ppl(model=model, tokenizer=tokenizer, test_datasets=args.eval_tasks)
    logging.info(results)

    # Output ceval results if specified
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model on ceval_benchmark.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file or Huggingface model identifier.')
    parser.add_argument('--quantized-model', action='store_true', help='Whether to use a quantized model.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--eval-tasks', nargs='+', default=["wikitext2"])
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the ceval results in JSON format.')
    args = parser.parse_args()
    main(args)