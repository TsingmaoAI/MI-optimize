#!/bin/bash
python examples/baichuan/quantization.py \
    --model-path "/models/Baichuan2-7B-Base" \
    --calibrate-name "c4" \
    --num-calibrate "128" \
    --algo "gptq" \
    --benchmark "ceval" \
    --num-shot "5" \
    --wbit "2" \


python examples/baichuan/quantization.py \
    --model-path "/models/Baichuan2-7B-Base" \
    --calibrate-name "c4" \
    --num-calibrate "128" \
    --algo "gptq" \
    --benchmark "ceval" \
    --num-shot "5" \
    --wbit "3" \


python examples/baichuan/quantization.py \
    --model-path "/models/Baichuan2-7B-Base" \
    --calibrate-name "c4" \
    --num-calibrate "128" \
    --algo "gptq" \
    --benchmark "ceval" \
    --num-shot "5" \
    --wbit "4" \