#!/bin/bash
python examples/baichuan/quantization.py \
    --model-path "/models/Baichuan2-7B-Base" \
    --algo "rtn" \
    --benchmark "ceval" \
