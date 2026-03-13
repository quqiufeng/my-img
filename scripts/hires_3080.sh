#!/bin/bash

# Hi-Res Fix 流程脚本
# 1. 生成低分辨率草稿图
# 2. img2img 重绘增强
# 3. upscale 放大
#
# 用法:
#   ./hires.sh [提示词] [输出文件]
#   ./hires.sh "a beautiful young woman..." /opt/image/portrait.png

set -e

export LD_LIBRARY_PATH=~/stable-diffusion.cpp/bin:$LD_LIBRARY_PATH
cd /home/dministrator/my-img

PROMPT="${1:-Swiss Alps, majestic mountain peaks, snow-capped mountains, crystal clear lake, green valleys, scenic landscape, dramatic clouds, golden sunlight, travel destination, photorealistic, high detail, 8K quality}"
OUTPUT="${2:-hires_final}"
STEP1="/home/dministrator/${OUTPUT}_step1.png"
STEP2="/home/dministrator/${OUTPUT}_step2.png"
FINAL="/home/dministrator/${OUTPUT}.png"

MODEL_DIR="/opt/image"
IMG2IMG="./bin/sd-img2img"
UPSCALE="./bin/sd-upscale"

echo "=== Step 1: Generate base image ==="
/home/dministrator/my-img/scripts/sdxl.sh "$PROMPT" "$STEP1" 1280 720

# echo ""
# echo "=== Step 2: Upscale ==="
# $UPSCALE \
#   --model $MODEL_DIR/2x_ESRGAN.gguf \
#   --input "$STEP1" \
#   --output "$STEP2" \
#   --scale 2

echo ""
echo "=== Step 3: img2img ==="
# 直接处理 STEP1 原图 (1280x720)
$IMG2IMG \
  --diffusion-model $MODEL_DIR/z_image_turbo-Q6_K.gguf \
  --vae $MODEL_DIR/ae.safetensors \
  --llm $MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --input "$STEP1" \
  --output "$FINAL" \
  --prompt "high quality, detailed, photorealistic" \
  --strength 0.35 \
  --steps 8 \
  --seed 42

echo ""
echo "=== Done ==="
echo "Final image: $FINAL"
ls -lh "$FINAL"
