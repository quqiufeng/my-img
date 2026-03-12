#!/bin/bash
# =============================================================================
# 2x 高清放大脚本 - 先放大再细修
# =============================================================================
# 流程：
#   Step 1: 用 Lanczos 算法物理放大2倍
#   Step 2: img2img 丰富细节
#
# 参数:
#   $1 输入图片
#   $2 输出图片(可选)
#   $3 strength(可选，默认0.3)
#   $4 steps(可选，默认20)
#
# 示例:
#   ./upscale_hires.sh input.png output.png 0.3 20
#   ./upscale_hires.sh /tmp/girl.png /tmp/girl_hires.png 0.3 20

INPUT_FILE="$1"
OUTPUT_FILE="$2"
STRENGTH="${3:-0.35}"
STEPS="${4:-20}"

if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 <input_image> [output] [strength] [steps]"
  echo "示例: $0 1.png 1_hires.png 0.3 20"
  exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file not found: $INPUT_FILE"
  exit 1
fi

# 检测输入尺寸
IMG_SIZE=$(file "$INPUT_FILE" | grep -oP '\d+ x \d+' | head -1)
INPUT_WIDTH=$(echo $IMG_SIZE | cut -d' ' -f1)
INPUT_HEIGHT=$(echo $IMG_SIZE | cut -d' ' -f3)

if [ -z "$INPUT_WIDTH" ] || [ -z "$INPUT_HEIGHT" ]; then
  echo "Error: Cannot detect image size"
  exit 1
fi

# 目标尺寸 = 输入尺寸 × 2
TARGET_WIDTH=$((INPUT_WIDTH * 2))
TARGET_HEIGHT=$((INPUT_HEIGHT * 2))

echo "=============================================="
echo "2x 高清放大 - 先放大再细修"
echo "=============================================="
echo "Input:  ${INPUT_FILE} (${INPUT_WIDTH}x${INPUT_HEIGHT})"
echo "Output: ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "Params: strength=$STRENGTH, steps=$STEPS"

# 默认输出文件名
if [ -z "$OUTPUT_FILE" ]; then
  BASE=$(basename "$INPUT_FILE" .png)
  BASE=$(basename "$BASE" .jpg)
  OUTPUT_FILE="${BASE}_hires_2x.png"
fi

SD_CLI="$HOME/stable-diffusion.cpp/bin/sd-cli"
MODEL_DIR="/opt/image"
DIFFUSION_MODEL="$MODEL_DIR/z_image_turbo-Q6_K.gguf"
TEMP_UPSCALE="/tmp/upscale_hires_temp.png"

echo ""
echo ">>> Step 1/2: 物理放大 (Lanczos) ..."

/usr/bin/convert "$INPUT_FILE" -filter Lanczos -resize ${TARGET_WIDTH}x${TARGET_HEIGHT} "$TEMP_UPSCALE" 2>/dev/null

if [ ! -s "$TEMP_UPSCALE" ]; then
  echo "Error: Step 1 failed - file not created"
  exit 1
fi

echo "  放大完成: $(ls -lh "$TEMP_UPSCALE" | awk '{print $5}')"

echo ""
echo ">>> Step 2/2: img2img 细节重绘 ..."

# 优化后的Prompt
DETAIL_PROMPT="masterpiece, ultra-high definition, sharp focus, highly detailed, 8k, photorealistic, sharp textures, high resolution, film grain"
NEGATIVE_PROMPT="blurry, low quality, deformed, worst quality, smooth, plastic skin, artifacts, ghosting"

nohup $SD_CLI \
  --diffusion-model $DIFFUSION_MODEL \
  --vae $MODEL_DIR/ae.safetensors \
  --llm $MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "$DETAIL_PROMPT" \
  -n "$NEGATIVE_PROMPT" \
  --cfg-scale 2.0 \
  --diffusion-fa \
  --scheduler karras \
  --vae-tiling \
  -i "$TEMP_UPSCALE" \
  --strength "$STRENGTH" \
  -H "$TARGET_HEIGHT" \
  -W "$TARGET_WIDTH" \
  --steps "$STEPS" \
  -o "$OUTPUT_FILE" > /dev/null 2>&1 &

PID=$!
while kill -0 $PID 2>/dev/null; do
  sleep 5
done
wait $PID

# 清理临时文件
rm -f "$TEMP_UPSCALE"

if [ $? -ne 0 ] || [ ! -f "$OUTPUT_FILE" ]; then
  echo "Error: Step 2 failed"
  exit 1
fi

echo ""
echo "=============================================="
echo "Done: $OUTPUT_FILE"
echo "=============================================="
