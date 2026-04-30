#!/bin/bash
# =============================================================================
# 图像生成脚本 - RTX 3080 10G 专用版 (低显存优化)
# =============================================================================
#
# 【出图原理】
# 两阶段出图策略：
#   1. HiRes Fix 生成 1280x720（低分辨率基础图，避免 OOM）
#   2. ESRGAN 放大 2x 到 2560x1440（后处理，不消耗 diffusion 显存）
#
# 【3080 10G 显存限制】
# 10G 显存无法直接生成 2560x1440，因此：
#   - 先用 HiRes Fix 出 1280x720（基础 640x360 -> 1280x720）
#   - 再用 ESRGAN 2x 放大到 2560x1440
#   - ESRGAN 放大是后处理，不消耗 diffusion 显存
#
# 【参考提示词示例 (人像)】
# ./img1.sh "half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, clean white background, medium close up" "~/portrait_2560x1440.png"
#
# 【参数说明】
#   $1 - 提示词 (Prompt)
#   $2 - 输出文件路径（可选，默认 ~/YYYYMMDD_HHMMSS_<md5>.png）
# =============================================================================
set -euo pipefail

# 设置库路径
export LD_LIBRARY_PATH=/home/dministrator/onnxruntime-linux-x64-1.20.1/lib:$LD_LIBRARY_PATH

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

MODEL_DIR="${MODEL_DIR:-/opt/image/model}"
SD_CLI="${SD_CLI:-/home/dministrator/my-img/build/myimg-cli}"
DIFFUSION_MODEL="$MODEL_DIR/z_image_turbo-Q5_K_M.gguf"
VAE_MODEL="$MODEL_DIR/ae.safetensors"
LLM_MODEL="$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
UPSCALE_MODEL="$MODEL_DIR/2x_ESRGAN.gguf"

ARGS=()
for arg in "$@"; do
    ARGS+=("$arg")
done

PROMPT="${ARGS[0]:-A beautiful landscape}"
OUTPUT_FILE="${ARGS[1]:-}"

# 固定参数：HiRes Fix 出 1280x720，然后 ESRGAN 放大到 2560x1440
WIDTH=2560
HEIGHT=1440
HIRES_WIDTH=1280
HIRES_HEIGHT=720

echo -e "${BLUE}[INFO] 生成策略: HiRes Fix 1280x720 -> ESRGAN 2x -> 2560x1440${NC}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

echo "========================================"
echo "  Pre-check"
echo "========================================"

if [ ! -f "$SD_CLI" ]; then echo -e "${RED}Error: sd-cli not found: $SD_CLI${NC}"; exit 1; fi
if [ ! -x "$SD_CLI" ]; then echo -e "${RED}Error: sd-cli not executable: $SD_CLI${NC}"; exit 1; fi

for model in "$DIFFUSION_MODEL" "$VAE_MODEL" "$LLM_MODEL"; do
    if [ ! -f "$model" ]; then echo -e "${RED}Error: model not found: $model${NC}"; exit 1; fi
done

# 检查放大模型（必需）
if [ ! -f "$UPSCALE_MODEL" ]; then
    echo -e "${RED}Error: upscale model not found: $UPSCALE_MODEL${NC}"
    exit 1
fi
echo -e "${CYAN}✓ ESRGAN upscale enabled (2x -> 2560x1440)${NC}"

echo -e "${GREEN}✓ All checks passed${NC}"

# HD optimized parameters - sharp and clear
SAMPLING_METHOD="${SAMPLING_METHOD:-euler_a}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-2.5}"
STEPS="${STEPS:-40}"
HIRES_STEPS="${HIRES_STEPS:-30}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.30}"

echo -e "${BLUE}[INFO] HD Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"

# Add quality keywords - sharp and detailed
QUALITY_PREFIX="sharp focus, crisp details, realistic skin texture, natural lighting, professional portrait"
if [[ "$PROMPT" != *"sharp focus"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

# Negative prompt: avoid blur and low quality
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, out of focus, soft focus, hazy, unclear, low quality, worst quality, jpeg artifacts, noise, grain, bad anatomy, deformed, asymmetrical face, watermark, text, logo}"

if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
        OUTPUT="$(basename "$OUTPUT_FILE")"
    else
        OUTPUT_DIR="$HOME"
        OUTPUT="$OUTPUT_FILE"
    fi
else
    OUTPUT_DIR="$HOME"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
    OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT"

# 固定分辨率策略
# 基础生成: 640x360 -> HiRes Fix: 1280x720 -> ESRGAN 2x: 2560x1440
LOW_W=640
LOW_H=360
TARGET_W=$WIDTH
TARGET_H=$HEIGHT

echo ""
echo "========================================"
echo "  HD Image Generation"
echo "========================================"
echo -e "Final Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Stage 1: ${GREEN}${LOW_W}x${LOW_H} -> HiRes Fix -> ${HIRES_WIDTH}x${HIRES_HEIGHT}${NC}"
echo -e "Stage 2: ${GREEN}${HIRES_WIDTH}x${HIRES_HEIGHT} -> ESRGAN 2x -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
echo -e "Upscale: ${CYAN}2x ESRGAN${NC}"
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$RANDOM}"
echo "Generating..."

SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --vae "$VAE_MODEL"
  --llm "$LLM_MODEL"
  -p "$PROMPT"
  -n "$NEGATIVE_PROMPT"
  --cfg-scale "$CFG_SCALE"
  --sampling-method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size 256x256
  --vae-tile-overlap 0.8
  -W "$LOW_W" -H "$LOW_H"
  --steps "$STEPS"
  --hires
  --hires-width "$HIRES_WIDTH"
  --hires-height "$HIRES_HEIGHT"
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  --upscale-model "$UPSCALE_MODEL"
  -s "$SEED"
  -o "$OUTPUT_PATH"
)

"${SD_CMD[@]}"

if [ -f "$OUTPUT_PATH" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File: ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size: ${BLUE}$FILE_SIZE${NC}"
    echo -e "Seed: ${YELLOW}$SEED${NC}"
    echo -e "CFG: ${CYAN}$CFG_SCALE${NC}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo -e "${RED}✗ Generation failed! Output file not found${NC}"
    echo "========================================"
    exit 1
fi
