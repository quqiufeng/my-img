#!/bin/bash
# =============================================================================
# libTorch HiRes Fix 脚本
# =============================================================================
# 使用 my-img 的 libTorch 版 HiRes Fix 进行高清出图
#
# 用法: ./libTorchHiresfix.sh "prompt" ~/output.png
# =============================================================================

set -euo pipefail

export LD_LIBRARY_PATH=/home/dministrator/onnxruntime-linux-x64-1.20.1/lib:$LD_LIBRARY_PATH

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

PROMPT="${1:-half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, clean white background, medium close up}"
OUTPUT_FILE="${2:-~/libtorch_hires_2560x1440.png}"

# 模型路径
MODEL_DIR="${MODEL_DIR:-/opt/image/model}"
SD_CLI="${SD_CLI:-/home/dministrator/my-img/build/myimg-cli}"
DIFFUSION_MODEL="$MODEL_DIR/z_image_turbo-Q5_K_M.gguf"
VAE_MODEL="$MODEL_DIR/ae.safetensors"
LLM_MODEL="$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# 检查
if [ ! -f "$SD_CLI" ]; then
    echo -e "${RED}Error: myimg-cli not found at $SD_CLI${NC}"
    echo "Please build first: cd build && cmake --build ."
    exit 1
fi

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
OUTPUT_NAME="$(basename "$OUTPUT_FILE" .png)"
mkdir -p "$OUTPUT_DIR"

# 参数
BASE_W=1280
BASE_H=720
TARGET_W=2560
TARGET_H=1440
STRENGTH=0.28
STEPS=55
CFG_SCALE=2.8
SAMPLER="euler"
SCHEDULER="discrete"
SEED="${SEED:-$RANDOM}"

# 质量提示词
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, complete face, full head, professional portrait"
FULL_PROMPT="$QUALITY_PREFIX, $PROMPT"

NEGATIVE_PROMPT="blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders"

echo ""
echo "========================================"
echo "  libTorch HiRes Fix"
echo "========================================"
echo ""
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_FILE${NC}"
echo -e "Seed: ${CYAN}$SEED${NC}"
echo ""
echo "Mode: libTorch (latent upscale in GPU)"
echo "Base: ${BASE_W}x${BASE_H}"
echo "Target: ${TARGET_W}x${TARGET_H} (strength=$STRENGTH)"
echo ""
echo "========================================"
echo ""

# 使用 --hires-mode libtorch 进行单阶段 HiRes Fix
echo -e "${BLUE}[Generating] libTorch HiRes Fix: ${BASE_W}x${BASE_H} → ${TARGET_W}x${TARGET_H}${NC}"
echo ""

"$SD_CLI" \
  --diffusion-model "$DIFFUSION_MODEL" \
  --vae "$VAE_MODEL" \
  --llm "$LLM_MODEL" \
  -p "$FULL_PROMPT" \
  -n "$NEGATIVE_PROMPT" \
  -W "$BASE_W" -H "$BASE_H" \
  --hires \
  --hires-mode libtorch \
  --hires-width "$TARGET_W" \
  --hires-height "$TARGET_H" \
  --hires-strength "$STRENGTH" \
  --hires-steps "$STEPS" \
  --hires-upscaler latent \
  --steps "$STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --sampling-method "$SAMPLER" \
  --scheduler "$SCHEDULER" \
  --diffusion-fa \
  --vae-tiling \
  --vae-tile-size 256x256 \
  --vae-tile-overlap 0.8 \
  -s "$SEED" \
  -o "$OUTPUT_FILE"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo -e "${RED}Error: Generation failed!${NC}"
    exit 1
fi

FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "========================================"
echo -e "${GREEN}✓ libTorch HiRes Fix completed!${NC}"
echo -e "File: ${GREEN}$OUTPUT_FILE${NC}"
echo -e "Size: ${BLUE}$FILE_SIZE${NC}"
echo -e "Resolution: ${CYAN}${TARGET_W}x${TARGET_H}${NC}"
echo -e "Seed: ${YELLOW}$SEED${NC}"
echo "========================================"
echo ""
echo "Features:"
echo "  ✓ Latent upscaling via libTorch (GPU)"
echo "  ✓ Customizable upscale algorithm (bilinear/bicubic/nearest)"
echo "  ✓ Stage-level conditioning injection ready"
echo ""
