import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args: argparse.Namespace):
    print(args)

    # Initialize the model
    if args.quantized_model:
        model = torch.load(args.model_path)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model.to(args.device)

    def run_to_completion():
        start_time = time.perf_counter()
        with torch.no_grad():
            if args.quantized_model:
                inputs = torch.randn(args.batch_size, args.input_len).to(args.device)
                model(inputs)
            else:
                tokenizer.pad_token = tokenizer.eos_token
                inputs = tokenizer(["Hello, world!"] * args.batch_size, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                model(**inputs)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion()

    # Benchmark
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion())
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of requests.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model file or Huggingface model identifier.')
    parser.add_argument('--quantized-model', action='store_true', help='Whether to use a quantized model.')
    parser.add_argument('--input-len', type=int, default=32, help='Length of the input sequence.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--num-iters-warmup', type=int, default=10, help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters', type=int, default=30, help='Number of iterations to run for benchmarking.')
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the latency results in JSON format.')
    args = parser.parse_args()
    main(args)