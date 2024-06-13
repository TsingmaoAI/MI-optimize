#!/bin/bash

# 设置默认参数
MODEL_PATH=""
ALGO="rtn"
WBIT=4
ABIT=16
W_GROUPSIZE=128
W_QTYPE="per_group"
BENCHMARK="ceval"
NUM_CALIBRATE=1
NUM_SHOT=0
CALIBRATE_NAME="c4"
SEQLEN=2048
DEVICE="cuda"
OFFLOAD="cpu"
SKIP_LAYERS="lm_head"
BLOCK_SEQUENTIAL=""
LAYER_SEQUENTIAL=""
SAVE=""

# 运行 Python 脚本并传递参数
python examples/baichuan/quantization.py \
    --model-path "$MODEL_PATH" \
    --algo "$ALGO" \
    --wbit "$WBIT" \
    --abit "$ABIT" \
    --w-groupsize "$W_GROUPSIZE" \
    --w-qtype "$W_QTYPE" \
    --benchmark "$BENCHMARK" \
    --num-calibrate "$NUM_CALIBRATE" \
    --num-shot "$NUM_SHOT" \
    --calibrate-name "$CALIBRATE_NAME" \
    --seqlen "$SEQLEN" \
    --device "$DEVICE" \
    --offload "$OFFLOAD" \
    --skip-layers "$SKIP_LAYERS" \
    $BLOCK_SEQUENTIAL \
    $LAYER_SEQUENTIAL \
    --save "$SAVE"
