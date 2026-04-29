#!/bin/bash
# =============================================================================
# 图像生成脚本 - RTX 3080 10G 专用版 (低显存优化)
# =============================================================================
#
# 【出图原理 / Why HiRes Fix】
# 直接生成高分辨率会导致"多人症"、畸形五官等问题，因为扩散模型训练时
# 未见过高分辨率。HiRes Fix 分两阶段：
#   1. 先生成低分辨率图像（构图、骨架正确）
#   2. 在 latent 空间放大后 refine（保留结构 + 补充细节）
#   3. 最后用 ESRGAN 放大到目标分辨率（后处理，不进入 diffusion 流程）
#
# 【3080 10G 显存限制】
# 10G 显存无法直接以高分辨率作为基础（会 OOM 爆显存），因此：
#   - 基础分辨率只能到 1280x720（latent 160x90，刚好在显存极限）
#   - 再通过 HiRes Fix 放大到目标 2560x1440（2x 放大）
#   - ESRGAN 放大是后处理，不消耗 diffusion 显存
#
# 【参考提示词示例 (人像)】
# ./img1.sh "half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, clean white background, medium close up" "~/portrait_2560x1440.png" 2560 1440
#
# 【参数说明】
#   $1 - 提示词 (Prompt)
#   $2 - 输出文件路径
#   $3 - 宽度 (Width)
#   $4 - 高度 (Height)
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
WIDTH="${ARGS[2]:-1280}"
HEIGHT="${ARGS[3]:-720}"

# 2560x1440 模式：参考 quqiufeng，基础 1280x720，HiRes Fix 直接到 2560x1440
if [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    echo -e "${BLUE}[INFO] 2560x1440 Mode: 1280x720 base -> HiRes Fix 2560x1440${NC}"
fi

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

if [ "${FORCE_UPSCALE:-0}" -eq 1 ]; then
    if [ ! -f "$UPSCALE_MODEL" ]; then echo -e "${RED}Error: upscale model not found: $UPSCALE_MODEL${NC}"; exit 1; fi
    echo -e "${CYAN}✓ Upscale mode enabled (2x ESRGAN)${NC}"
fi

echo -e "${GREEN}✓ All checks passed${NC}"

if ! [[ "$WIDTH" =~ ^[0-9]+$ ]] || [ "$WIDTH" -le 0 ]; then echo -e "${RED}Error: width must be positive integer${NC}"; exit 1; fi
if ! [[ "$HEIGHT" =~ ^[0-9]+$ ]] || [ "$HEIGHT" -le 0 ]; then echo -e "${RED}Error: height must be positive integer${NC}"; exit 1; fi

# HD optimized parameters - sharp and clear
# Updated after testing: euler_a + higher cfg + more steps for better clarity
SAMPLING_METHOD="${SAMPLING_METHOD:-euler_a}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-2.5}"
STEPS="${STEPS:-40}"
HIRES_STEPS="${HIRES_STEPS:-30}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.30}"

if [ "$WIDTH" -ge 1920 ] && [ "$HEIGHT" -ge 1080 ]; then
    echo -e "${BLUE}[INFO] Ultra HD Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
else
    echo -e "${BLUE}[INFO] HD Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
fi

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

TARGET_LATENT_W=$((WIDTH / 8))
TARGET_LATENT_H=$((HEIGHT / 8))
TARGET_W=$WIDTH
TARGET_H=$HEIGHT

# quqiufeng 逻辑：目标分辨率直接作为基础，不再折半
# 2560x1440 -> 基础 1280x720 (latent 160x90) -> HiRes 2560x1440
# 1920x1080 -> 基础 1024x576 (latent 128x72) -> HiRes 1920x1080
# 1280x720  -> 基础 640x360 (latent 80x45) -> HiRes 1280x720
if [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    LOW_W=1280
    LOW_H=720
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    LOW_W=1024
    LOW_H=576
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    LOW_W=640
    LOW_H=360
else
    # 通用计算
    LOW_LATENT_W=$(((TARGET_LATENT_W / 2 + 7) / 8 * 8))
    LOW_LATENT_H=$(((TARGET_LATENT_H / 2 + 7) / 8 * 8))
    LOW_W=$((LOW_LATENT_W * 8))
    LOW_H=$((LOW_LATENT_H * 8))
fi

echo ""
echo "========================================"
echo "  HD Image Generation"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
if [ "${FORCE_UPSCALE:-0}" -eq 1 ]; then
    echo -e "Upscale: ${CYAN}2x ESRGAN -> ${TARGET_W}x${TARGET_H}${NC}"
fi
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
  --hires-width "$WIDTH"
  --hires-height "$HEIGHT"
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  -s "$SEED"
  -o "$OUTPUT_PATH"
)

# ESRGAN 放大
if [ "${FORCE_UPSCALE:-0}" -eq 1 ]; then
    SD_CMD+=(--upscale-model "$UPSCALE_MODEL")
fi

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
