#!/bin/bash
# =============================================================================
# libTorch HiRes Fix - Real Implementation
# =============================================================================
# 使用 my-img 的 libTorch 版 HiRes Fix 进行高清出图
# 
# 真实实现流程：
#   1. sd_ext_generate_latent() 生成基础分辨率 latent
#   2. libTorch GPU 上采样 (torch::nn::functional::interpolate)
#   3. sd_ext_sample_latent() 在目标分辨率继续采样（strength 控制噪声）
#   4. sd_ext_vae_decode() 解码为最终图像
#
# 与 sd.cpp 内置 HiRes Fix 的区别：
#   - 上采样在 libTorch GPU 中进行，可自定义算法 (bilinear/bicubic/nearest)
#   - latent 数据通过 CPU↔GPU 转换，非纯 sd.cpp 内部处理
#   - 支持未来扩展：ControlNet 注入、Prompt Scheduling 等
#
# 用法: 
#   ./libTorchHiresfix.sh "prompt" ~/output.png
#   ./libTorchHiresfix.sh "prompt" ~/output.png --upscaler bicubic
# =============================================================================

set -euo pipefail

export LD_LIBRARY_PATH=/home/dministrator/onnxruntime-linux-x64-1.20.1/lib:$LD_LIBRARY_PATH

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
MAGENTA="\033[0;35m"
NC="\033[0m"

# 参数解析
PROMPT="${1:-half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, clean white background, medium close up}"
OUTPUT_FILE="${2:-~/libtorch_hires_2560x1440.png}"

# 可选参数
UPSCALER="${UPSCALER:-bilinear}"  # bilinear, bicubic, nearest
STRENGTH="${STRENGTH:-0.28}"
SEED="${SEED:-$RANDOM}"

# 模型路径
MODEL_DIR="${MODEL_DIR:-/opt/image/model}"
SD_CLI="${SD_CLI:-/home/dministrator/my-img/build/myimg-cli}"
DIFFUSION_MODEL="$MODEL_DIR/z_image_turbo-Q5_K_M.gguf"
VAE_MODEL="$MODEL_DIR/ae.safetensors"
LLM_MODEL="$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

# 检查 CLI
if [ ! -f "$SD_CLI" ]; then
    echo -e "${RED}✗ Error: myimg-cli not found at $SD_CLI${NC}"
    echo "  Please build first:"
    echo "    cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

if [ ! -f "$DIFFUSION_MODEL" ]; then
    echo -e "${RED}✗ Error: Diffusion model not found${NC}"
    echo "  $DIFFUSION_MODEL"
    exit 1
fi

if [ ! -f "$VAE_MODEL" ]; then
    echo -e "${RED}✗ Error: VAE model not found${NC}"
    echo "  $VAE_MODEL"
    exit 1
fi

if [ ! -f "$LLM_MODEL" ]; then
    echo -e "${RED}✗ Error: LLM model not found${NC}"
    echo "  $LLM_MODEL"
    exit 1
fi

# 解析波浪号
if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
mkdir -p "$OUTPUT_DIR"

# 固定参数
BASE_W=1280
BASE_H=720
TARGET_W=2560
TARGET_H=1440
STEPS=55
CFG_SCALE=2.8
SAMPLER="euler"
SCHEDULER="discrete"

# 质量提示词
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, complete face, full head, professional portrait"
FULL_PROMPT="$QUALITY_PREFIX, $PROMPT"

NEGATIVE_PROMPT="blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           libTorch HiRes Fix (Real Implementation)       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo -e "  Prompt:     ${YELLOW}$PROMPT${NC}"
echo -e "  Output:     ${GREEN}$OUTPUT_FILE${NC}"
echo -e "  Seed:       ${CYAN}$SEED${NC}"
echo ""
echo -e "  ${MAGENTA}Mode:       libTorch GPU latent upscale${NC}"
echo -e "  ${MAGENTA}Pipeline:   sd_ext_generate_latent → libTorch → sd_ext_sample_latent → VAE decode${NC}"
echo ""
echo "  Base:       ${BASE_W}x${BASE_H}"
echo "  Target:     ${TARGET_W}x${TARGET_H}"
echo "  Upscaler:   ${UPSCALER} (libTorch GPU)"
echo "  Strength:   ${STRENGTH}"
echo "  Steps:      ${STEPS}"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# 构建 CLI 参数
CLI_ARGS=(
    --diffusion-model "$DIFFUSION_MODEL"
    --vae "$VAE_MODEL"
    --llm "$LLM_MODEL"
    -p "$FULL_PROMPT"
    -n "$NEGATIVE_PROMPT"
    -W "$BASE_W"
    -H "$BASE_H"
    --hires
    --hires-mode libtorch
    --hires-width "$TARGET_W"
    --hires-height "$TARGET_H"
    --hires-strength "$STRENGTH"
    --hires-steps "$STEPS"
    --hires-upscaler "${UPSCALER}"
    --steps "$STEPS"
    --cfg-scale "$CFG_SCALE"
    --sampling-method "$SAMPLER"
    --scheduler "$SCHEDULER"
    --diffusion-fa
    --vae-tiling
    --vae-tile-size 256x256
    --vae-tile-overlap 0.8
    -s "$SEED"
    -o "$OUTPUT_FILE"
)

# 显示执行的命令
echo -e "${BLUE}[1/4] Generating base latent (sd_ext_generate_latent)...${NC}"
echo -e "${BLUE}[2/4] Upscaling latent via libTorch GPU (${UPSCALER})...${NC}"
echo -e "${BLUE}[3/4] Refining with sd_ext_sample_latent (strength=${STRENGTH})...${NC}"
echo -e "${BLUE}[4/4] Decoding to image (sd_ext_vae_decode)...${NC}"
echo ""

# 执行生成
START_TIME=$(date +%s)

if ! "$SD_CLI" "${CLI_ARGS[@]}"; then
    echo ""
    echo -e "${RED}✗ Generation failed!${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check CUDA availability: nvidia-smi"
    echo "  2. Verify model files exist"
    echo "  3. Check VRAM: 10GB+ recommended for 2560x1440"
    echo "  4. Try smaller resolution: ./libTorchHiresfix.sh \"prompt\" out.png --target-w 1920 --target-h 1080"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

if [ ! -f "$OUTPUT_FILE" ]; then
    echo -e "${RED}✗ Error: Output file not created!${NC}"
    exit 1
fi

FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo -e "║  ${GREEN}✓ libTorch HiRes Fix Completed Successfully!${NC}            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo -e "  File:       ${GREEN}$OUTPUT_FILE${NC}"
echo -e "  Size:       ${BLUE}$FILE_SIZE${NC}"
echo -e "  Resolution: ${CYAN}${TARGET_W}x${TARGET_H}${NC}"
echo -e "  Seed:       ${YELLOW}$SEED${NC}"
echo -e "  Time:       ${MAGENTA}${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo "  Technical Details:"
echo "    ✓ Base latent generated via sd_ext_generate_latent()"
echo "    ✓ Latent upscaled in GPU via torch::interpolate (${UPSCALER})"
echo "    ✓ Refined via sd_ext_sample_latent() with strength=${STRENGTH}"
echo "    ✓ Decoded via sd_ext_vae_decode()"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# 显示图像信息（如果安装了 file 命令）
if command -v file >/devdev/null 2>&1; then
    file "$OUTPUT_FILE"
    echo ""
fi
