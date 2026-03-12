#!/bin/bash

# sd-img2img 执行脚本
# 用法: ./run.sh [输入图片]

INPUT="${1:-/opt/image/test_input.png}"
OUTPUT="/opt/image/test_output.png"

MODEL_DIR="/opt/gguf/image"

./bin/sd-img2img \
  --diffusion-model $MODEL_DIR/z_image_turbo-Q6_K.gguf \
  --vae $MODEL_DIR/ae.safetensors \
  --llm $MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --prompt "high quality" \
  --strength 0.35 \
  --steps 2 \
  --seed 42 \
  > sd-img2img.log 2>&1

echo "Output: $OUTPUT"
