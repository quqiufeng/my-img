#!/bin/bash
# =============================================================================
# img2img 细节增强脚本 - 不放大尺寸，只丰富细节
# =============================================================================
# 流程：
#   直接用原图进行 img2img，保持原尺寸，只修复细节
#
# 参数:
#   $1 输入图片
#   $2 输出图片(可选)
#   $3 strength(可选，默认0.35)
#   $4 steps(可选，默认20)
#
# 示例:
#   ./hires.sh input.png output.png 0.35 20
#   ./hires.sh /tmp/girl.png /tmp/girl_hires.png 0.35 20

INPUT_FILE="$1"
OUTPUT_FILE="$2"
STRENGTH="${3:-0.35}"
STEPS="${4:-20}"

if [ -z "$INPUT_FILE" ]; then
  echo "Usage: $0 <input_image> [output] [strength] [steps]"
  echo "示例: $0 1.png 1_hires.png 0.35 20"
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

echo "=============================================="
echo "img2img 细节增强"
echo "=============================================="
echo "Input:  ${INPUT_FILE} (${INPUT_WIDTH}x${INPUT_HEIGHT})"
echo "Params: strength=$STRENGTH, steps=$STEPS"

# 默认输出文件名
if [ -z "$OUTPUT_FILE" ]; then
  BASE=$(basename "$INPUT_FILE" .png)
  BASE=$(basename "$BASE" .jpg)
  OUTPUT_FILE="${BASE}_hires.png"
fi

SD_CLI="~/stable-diffusion.cpp/bin/sd-cli"
MODEL_DIR="/opt/image"
DIFFUSION_MODEL="$MODEL_DIR/z_image_turbo-Q6_K.gguf"

echo ""
echo ">>> img2img 细节重绘 ..."

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
  -i "$INPUT_FILE" \
  --strength "$STRENGTH" \
  -H "$INPUT_HEIGHT" \
  -W "$INPUT_WIDTH" \
  --steps "$STEPS" \
  -o "$OUTPUT_FILE" > /dev/null 2>&1 &

PID=$!
while kill -0 $PID 2>/dev/null; do
  sleep 5
done
wait $PID

if [ $? -ne 0 ] || [ ! -f "$OUTPUT_FILE" ]; then
  echo "Error: img2img failed"
  exit 1
fi

echo ""
echo "=============================================="
echo "Done: $OUTPUT_FILE"
echo "=============================================="
