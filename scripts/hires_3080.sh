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

PROMPT="${1:-Swiss Alps, majestic mountain peaks, snow-capped mountains, crystal clear lake, green valleys, scenic landscape, dramatic clouds, golden sunlight, travel destination, photorealistic, high detail, 8K quality}"
OUTPUT="${2:-hires_final}"
STEP1="~/${OUTPUT}_step1.png"
STEP2="~/${OUTPUT}_step2.png"
FINAL="~/${OUTPUT}.png"

MODEL_DIR="/opt/image"
IMG2IMG="./bin/sd-img2img"
UPSCALE="./bin/sd-upscale"

echo "=== Step 1: Generate base image ==="
/opt/my-img/scripts/img_3080.sh "$PROMPT" "$STEP1" 1280 720

echo ""
echo "=== Step 2: img2img enhance ==="
# SDXL 模型可能不支持 img2img，暂时用回 Turbo
$IMG2IMG \
  --diffusion-model $MODEL_DIR/z_image_turbo-Q8_0.gguf \
  --vae $MODEL_DIR/ae.safetensors \
  --llm $MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --input "$STEP1" \
  --output "$STEP2" \
  --prompt "high quality, detailed, photorealistic, perfect skin, sharp features" \
  --strength 0.45 \
  --steps 8 \
  --seed 42

echo ""
echo "=== Step 3: Upscale ==="
$UPSCALE \
  --model $MODEL_DIR/2x_ESRGAN.gguf \
  --input "$STEP2" \
  --output "$FINAL" \
  --scale 2

echo ""
echo "=== Done ==="
echo "Final image: $FINAL"
ls -lh "$FINAL"
