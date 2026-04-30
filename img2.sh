#!/bin/bash
# =============================================================================
# 图像生成脚本 - RTX 4090D 24G 专用优化版
# 用途: my-img 项目开发完成后的验证工具
# 说明: 此脚本用于验证 my-img 编译后的 sd-workflow 二进制功能
# =============================================================================
#
# 【出图原理 / Why HiRes Fix】
# 直接生成高分辨率会导致"多人症"、畸形五官等问题，因为扩散模型训练时
# 未见过高分辨率。HiRes Fix 分两阶段：
#   1. 先生成低分辨率图像（构图、骨架正确）
#   2. 在 latent 空间放大后 refine（保留结构 + 补充细节）
#
# 【4090D 优化策略】
# 3080 10G 显存只能以 1280x720 为基础，再放大到 2560x1440（2x放大，画质损失大）
# 4090D 24G 显存充足，基础分辨率可以大幅提升：
#   - 2560x1440 目标 → 2048x1152 基础（1.25x放大，latent 损失极小）
#   - 3840x2160 目标 → 2560x1440 基础（1.5x放大，4K 出图）
#   - 基础分辨率更高 → 初始构图和五官更清晰
#   - 放大倍数更小 → latent 插值损失更少
#   - HiRes refine 只需微调纹理 → 不易破坏原有结构
#   - 最终出图质量显著优于低显存方案
#
# 【参考提示词示例 (人像)】
# 2K: ./img2.sh "half body portrait..." "~/portrait_2560x1440.png" 2560 1440
# 4K: ./img2.sh "half body portrait..." "~/portrait_3840x2160.png" 3840 2160
#
# 【参数说明】
#   $1 - 提示词 (Prompt)
#   $2 - 输出文件路径
#   $3 - 宽度 (Width)
#   $4 - 高度 (Height)
#   --upscale - 可选：使用 2x ESRGAN 进一步放大（通常不需要）
# =============================================================================
set -euo pipefail

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

MODEL_DIR="${MODEL_DIR:-/opt/image/model}"
# 使用 my-img 编译后的二进制（开发完成后验证用）
SD_CLI="${SD_CLI:-$HOME/my-img/build/myimg-cli}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/z_image_turbo-Q8_0.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/ae.safetensors}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
UPSCALE_MODEL="${UPSCALE_MODEL:-$MODEL_DIR/2x_ESRGAN.gguf}"

UPSCALE_FLAG=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--upscale" ]; then
        UPSCALE_FLAG=1
    else
        ARGS+=("$arg")
    fi
done

PROMPT="${ARGS[0]:-A beautiful landscape}"
OUTPUT_FILE="${ARGS[1]:-}"
WIDTH="${ARGS[2]:-1280}"
HEIGHT="${ARGS[3]:-720}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

echo "========================================"
echo "  Pre-check"
echo "========================================"

if [ ! -f "$SD_CLI" ]; then echo -e "${RED}Error: sd-workflow not found: $SD_CLI${NC}"; exit 1; fi
if [ ! -x "$SD_CLI" ]; then echo -e "${RED}Error: sd-workflow not executable: $SD_CLI${NC}"; exit 1; fi

for model in "$DIFFUSION_MODEL" "$VAE_MODEL" "$LLM_MODEL"; do
    if [ ! -f "$model" ]; then echo -e "${RED}Error: model not found: $model${NC}"; exit 1; fi
done

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    if [ ! -f "$UPSCALE_MODEL" ]; then echo -e "${RED}Error: upscale model not found: $UPSCALE_MODEL${NC}"; exit 1; fi
    echo -e "${CYAN}✓ Upscale mode enabled (2x ESRGAN)${NC}"
fi

echo -e "${GREEN}✓ All checks passed${NC}"

if ! [[ "$WIDTH" =~ ^[0-9]+$ ]] || [ "$WIDTH" -le 0 ]; then echo -e "${RED}Error: width must be positive integer${NC}"; exit 1; fi
if ! [[ "$HEIGHT" =~ ^[0-9]+$ ]] || [ "$HEIGHT" -le 0 ]; then echo -e "${RED}Error: height must be positive integer${NC}"; exit 1; fi

# HD optimized parameters (4090D 24G 实测调优)
# 基础分辨率大幅提升 → 放大倍数更小 → 画质更好
# 人像推荐: euler + discrete + cfg 3.2 + strength 0.30
# 风景推荐: dpm++2m + karras + cfg 1.5 + strength 0.35
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-3.2}"
STEPS="${STEPS:-30}"
HIRES_STEPS="${HIRES_STEPS:-50}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.30}"

if [ "$WIDTH" -ge 1920 ] && [ "$HEIGHT" -ge 1080 ]; then
    echo -e "${BLUE}[INFO] Ultra HD Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
else
    echo -e "${BLUE}[INFO] HD Mode: steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
fi

# Add quality keywords - enhanced for realism and edge stability
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, complete face, full head, professional portrait"
if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders}"}

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

# =============================================================================
# 低分辨率计算 - 4090D 24G 优化版
# =============================================================================
# HiRes Fix 需要两阶段生成来提升画质。低分辨率的选择原则：
# 1. latent 宽高比必须与目标严格一致（避免变形）
# 2. 基础分辨率越高，放大倍数越小，画质越好
# 3. 4090D 24G 显存可以承受更高的基础分辨率
#
# 对于 2560x1440 (latent 320x180, ratio=1.778)：
#   2048x1152 -> latent 256x144 (ratio=1.778) ✓ 推荐，放大1.25x，画质最佳
#   1920x1080 -> latent 240x135 (ratio=1.778) ✓ 备选，放大1.33x
#   1536x864  -> latent 192x108 (ratio=1.778) ✓ 备选，放大1.67x
#
# 对于 3840x2160 (latent 480x270, ratio=1.778)：
#   2560x1440 -> latent 320x180 (ratio=1.778) ✓ 推荐，放大1.5x，4K出图
#   2304x1296 -> latent 288x162 (ratio=1.778) ✓ 备选，放大1.67x
#
# 对于 1920x1080 (latent 240x135, ratio=1.778)：
#   1536x864  -> latent 192x108 (ratio=1.778) ✓ 推荐，放大1.25x
# =============================================================================

if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    # 4K: 2560x1440 基础 → 1.5x 放大
    LOW_W=2560
    LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    # 2K: 2048x1152 基础 → 1.25x 放大（画质最佳）
    LOW_W=2048
    LOW_H=1152
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    # 1080p: 1536x864 基础 → 1.25x 放大
    LOW_W=1536
    LOW_H=864
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    # 720p: 1024x576 基础 → 1.25x 放大
    LOW_W=1024
    LOW_H=576
else
    # 通用计算：使用目标分辨率的 80% 作为基础（4090D优化，更小放大倍数）
    LOW_LATENT_W=$((TARGET_LATENT_W * 4 / 5))
    LOW_LATENT_H=$((TARGET_LATENT_H * 4 / 5))
    
    # 对齐到 8 的倍数
    LOW_LATENT_W=$(((LOW_LATENT_W + 7) / 8 * 8))
    LOW_LATENT_H=$(((LOW_LATENT_H + 7) / 8 * 8))
    
    LOW_W=$((LOW_LATENT_W * 8))
    LOW_H=$((LOW_LATENT_H * 8))
fi

# 保持比例的最小限制：只在单边小于512时按比例放大
if [ "$LOW_W" -lt 512 ] || [ "$LOW_H" -lt 512 ]; then
    TARGET_RATIO=$(echo "scale=6; $WIDTH / $HEIGHT" | bc)
    if [ "$LOW_W" -lt "$LOW_H" ]; then
        LOW_W=512
        LOW_H=$(echo "scale=0; $LOW_W / $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_H" -lt 512 ]; then LOW_H=512; fi
    else
        LOW_H=512
        LOW_W=$(echo "scale=0; $LOW_H * $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_W" -lt 512 ]; then LOW_W=512; fi
    fi
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
if [ "$UPSCALE_FLAG" -eq 1 ]; then
    UPSCALED_W=$((WIDTH * 2))
    UPSCALED_H=$((HEIGHT * 2))
    echo -e "Upscale: ${CYAN}2x ESRGAN -> ${UPSCALED_W}x${UPSCALED_H}${NC}"
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
  --freeu
  --sag
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

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    SD_CMD+=(--upscale-model "$UPSCALE_MODEL")
    SD_CMD+=(--upscale-repeats 1)
    SD_CMD+=(--upscale-tile-size 1440)
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
