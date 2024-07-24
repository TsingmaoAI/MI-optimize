import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from mi_optimize.export import qnn

def main(args: argparse.Namespace):
    print(args)

    # Initialize the model
    if args.quantized_model:
        model = torch.load(args.model_path)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

    model.to(args.device)

    def run_to_completion():
        start_time = time.perf_counter()
        with torch.no_grad():
            input_ids = torch.randint(10000, size=(args.batch_size, args.input_len)).to(args.device)
            model.generate(input_ids, max_length=args.max_gen, do_sample=False)
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
    print(f'total latency: {np.mean(latencies)} seconds')
    print(f'per token latency: {np.mean(latencies)/(args.max_gen - args.input_len)}')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of requests.')
    parser.add_argument('--model-path', type=str, help='Path to the model file or Huggingface model identifier.')
    parser.add_argument('--quantized-model', action='store_true', help='Whether to use a quantized model.')
    parser.add_argument('--input-len', type=int, default=4, help='Length of the input sequence.')
    parser.add_argument('--max-gen', type=int, default=128, help='max length of generate tokens')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--num-iters-warmup', type=int, default=10, help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters', type=int, default=30, help='Number of iterations to run for benchmarking.')
    args = parser.parse_args()
    main(args)