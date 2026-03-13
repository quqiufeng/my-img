#!/bin/bash

# Hi-Res Fix 流程脚本
# 1. 生成低分辨率草稿图
# 2. img2img 重绘增强
# 3. upscale 放大
#
# 用法:
#   ./hires.sh [提示词] [输出文件]
#   ./hires.sh "a beautiful young woman..." /opt/image/portrait.png

export LD_LIBRARY_PATH=~/stable-diffusion.cpp/bin:$LD_LIBRARY_PATH

PROMPT="${1:-Swiss Alps, majestic mountain peaks, snow-capped mountains, crystal clear lake, green valleys, scenic landscape, dramatic clouds, golden sunlight, travel destination, photorealistic, high detail, 8K quality}"
OUTPUT="${2:-hires_final}"

STEP1="$HOME/${OUTPUT}_step1.png"
STEP2="$HOME/${OUTPUT}_step2.png"
FINAL="$HOME/${OUTPUT}.png"

MODEL_DIR="/opt/image"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMG2IMG="$SCRIPT_DIR/../build/sd-img2img"
UPSCALE="$SCRIPT_DIR/../build/sd-upscale"

echo "=== Step 1: Generate base image ==="
# 用 img.sh 生成第一步
/home/dministrator/my-img/scripts/sdxl.sh "$PROMPT" "$STEP1" 1280 720

# echo ""
# echo "=== Step 2: Upscale ==="
# $UPSCALE \
#   --model $MODEL_DIR/2x_ESRGAN.gguf \
#   --input "$STEP1" \
#   --output "$STEP2" \
#   --scale 2


echo ""

echo "=== Step 3: txt2img (skip img2img - memory issue) ==="
/home/dministrator/my-img/scripts/sdxl.sh "$PROMPT, masterpiece, ultra-detailed, sharp focus, 8k" "$FINAL" 1280 720

echo ""
echo "=== Done ==="
