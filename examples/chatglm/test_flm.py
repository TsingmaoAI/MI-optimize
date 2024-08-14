from fastllm_pytools import llm
import time
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer

def test_latency(args) :
    # Initialize the model
    if args.quantized_model:
        model = llm.model(args.model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code = True)
        model.cuda()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code = True)


    def run_to_completion():
        start_time = time.perf_counter()
        inputs = "Hello, world!"
        with torch.no_grad():
            if args.quantized_model:
                inputs = 'Hello ,world'
                model.response(inputs)
            else:
                inputs = tokenizer(["Hello, world!"], return_tensors="pt", padding=True, truncation=True)
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





if __name__ == '__main__' :
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

    tokenizer = AutoTokenizer.from_pretrained('/home/wf/models/chatglm3-6b',trust_remote_code = True)
    input = '晚上睡不着怎么办'
    model = llm.model('/home/wf/nx/MI-optimize/examples/chatglm/chatglm3-6b-int4.flm')
    cnt = 0
    s = time.time()
    output = []
    for i  in range(1):
        output.append(model.response('晚上睡不着怎么办'))
    e = time.time()
    for i in range(1):
        cnt += len(tokenizer.encode(output[i]))

    print(cnt/(e-s))


    model = AutoModelForCausalLM.from_pretrained('/home/wf/models/chatglm3-6b',trust_remote_code = True)
    model.cuda()
    model.eval()
    cnt = 0
    s =time.time()
    output = []
    for i in range(1):
        response , history= model.chat(tokenizer, input, history=[])
        output.append(response)
    e = time.time()

    for i in range(1):
        cnt += len(tokenizer.encode(output[i]))
    print(cnt/(e-s))



    
    
