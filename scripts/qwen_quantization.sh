#!/bin/bash
python examples/baichuan/quantization.py \
    --model-path "/models/qwen2-7b-instruct" \
    --algo "rtn" \
    --benchmark "ceval" \
