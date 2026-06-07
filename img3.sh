#!/bin/bash
# =============================================================================
# img3.sh - SDXL Base 1.0 高清大图生成脚本 (2560x1440)
# =============================================================================
#
# 用途: 使用 SDXL Base 1.0 (UNet) 生成 2K 高清图像，集成完整增强管线
# 设备: RTX 3080 20GB / RTX 4090 24GB
# 目标分辨率: 2560x1440 (16:9)
#
# 【与 img2.sh (Z-Image DiT) 的差异】
# - 模型: SDXL Base 1.0 safetensors (完整 checkpoint，内置 VAE)
# - 文本编码器: CLIP-L + CLIP-G (不需要 LLM)
# - CFG: 7.0 (SDXL 推荐，高于 DiT 的 3.2)
# - 采样器: euler + discrete (SDXL 稳定)
# - FreeU/SAG/HiRes Fix/VAE Tiling 全部可用
# - 支持 UNet IPAdapter (可选)
#
# 【当前出图基准 (2026-06-07)】
# 扩散模型: sd_xl_base_1.0.safetensors (6.5GB VRAM)
# 文本编码: clip_l.safetensors + clip_g.safetensors (~2.8GB VRAM)
# VAE:      使用 checkpoint 内置 VAE (无需 --vae)
# 分辨率:   1280x720 → 2560x1440 (2x HiRes)
# 步数:     25 → 45 (HiRes)
# HiRes strength: 0.30
# CFG:      7.0 | Sampler: euler | Scheduler: discrete
# FreeU:    b1=1.3, b2=1.4, s1=0.9, s2=0.2
# SAG:      开启 (scale=1.0)
# 增强:     clarity 0.4, sharpen 0.8, smart-sharpen 0.5, edge-sharpen 1.5
# VAE tiling: 128x128, overlap 0.5 (20GB 安全)
# 出图时间: ~5-8 分钟 (RTX 3080 20GB)
#
# 【参考提示词示例 (人像)】
# ./img3.sh "professional portrait of a young woman, soft studio lighting, elegant pose, sharp eyes, clean background" "~/portrait.png" 2560 1440
#
# 【参数说明】
#   $1 - 提示词 (Prompt)
#   $2 - 输出文件路径
#   $3 - 宽度 (Width, 默认 2560)
#   $4 - 高度 (Height, 默认 1440)
#   --upscale - 可选：使用 2x ESRGAN 进一步放大
#   --ipadapter - 可选：启用 IPAdapter UNet 图像提示词
#   --ipadapter-unet-weights PATH - UNet IPAdapter 权重
#   --ipadapter-model PATH - IPAdapter MLP (SDXL Plus v3)
#   --ipadapter-clip-vision PATH - CLIP Vision hidden states
#   --ipadapter-image PATH - 参考图像
# =============================================================================
set -euo pipefail

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

MODEL_DIR="${MODEL_DIR:-/data/models/image}"
SD_CLI="${SD_CLI:-/opt/my-img/build/myimg-cli}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/sd_xl_base_1.0.safetensors}"
CLIP_L_MODEL="${CLIP_L_MODEL:-$MODEL_DIR/clip_l.safetensors}"
CLIP_G_MODEL="${CLIP_G_MODEL:-$MODEL_DIR/clip_g.safetensors}"
# VAE 可选：SDXL Base 内置 VAE。如果外部 VAE 有问题，留空使用内置 VAE
VAE_MODEL="${VAE_MODEL:-}"
UPSCALE_MODEL="${UPSCALE_MODEL:-$MODEL_DIR/2x_ESRGAN.gguf}"

# VAE Tiling 配置（20G显存优化：默认128x128）
VAE_TILE_SIZE="${VAE_TILE_SIZE:-128x128}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"

UPSCALE_FLAG=0
IPADAPTER_FLAG=0
IPADAPTER_UNET_WEIGHTS=""
IPADAPTER_MODEL=""
IPADAPTER_CLIP_VISION=""
IPADAPTER_IMAGE=""
IPADAPTER_WEIGHT="${IPADAPTER_WEIGHT:-0.8}"

ARGS=()
i=0
while [ $i -lt $# ]; do
    arg="${@:$((i+1)):1}"
    if [ "$arg" = "--upscale" ]; then
        UPSCALE_FLAG=1
    elif [ "$arg" = "--ipadapter" ]; then
        IPADAPTER_FLAG=1
    elif [ "$arg" = "--ipadapter-unet-weights" ]; then
        i=$((i+1))
        IPADAPTER_UNET_WEIGHTS="${@:$((i+1)):1}"
    elif [ "$arg" = "--ipadapter-model" ]; then
        i=$((i+1))
        IPADAPTER_MODEL="${@:$((i+1)):1}"
    elif [ "$arg" = "--ipadapter-clip-vision" ]; then
        i=$((i+1))
        IPADAPTER_CLIP_VISION="${@:$((i+1)):1}"
    elif [ "$arg" = "--ipadapter-image" ]; then
        i=$((i+1))
        IPADAPTER_IMAGE="${@:$((i+1)):1}"
    elif [ "$arg" = "--ipadapter-weight" ]; then
        i=$((i+1))
        IPADAPTER_WEIGHT="${@:$((i+1)):1}"
    else
        ARGS+=("$arg")
    fi
    i=$((i+1))
done

PROMPT="${ARGS[0]:-A beautiful high quality landscape}"
OUTPUT_FILE="${ARGS[1]:-}"
WIDTH="${ARGS[2]:-2560}"
HEIGHT="${ARGS[3]:-1440}"

if [[ "$OUTPUT_FILE" == ~* ]]; then
    OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"
fi

echo "========================================"
echo "  Pre-check"
echo "========================================"

if [ ! -f "$SD_CLI" ]; then echo -e "${RED}Error: myimg-cli not found: $SD_CLI${NC}"; exit 1; fi
if [ ! -x "$SD_CLI" ]; then echo -e "${RED}Error: myimg-cli not executable: $SD_CLI${NC}"; exit 1; fi

for model in "$DIFFUSION_MODEL" "$CLIP_L_MODEL" "$CLIP_G_MODEL"; do
    if [ ! -f "$model" ]; then echo -e "${RED}Error: model not found: $model${NC}"; exit 1; fi
done

if [ -n "$VAE_MODEL" ] && [ ! -f "$VAE_MODEL" ]; then
    echo -e "${RED}Error: VAE model not found: $VAE_MODEL${NC}"; exit 1
fi

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    if [ ! -f "$UPSCALE_MODEL" ]; then echo -e "${RED}Error: upscale model not found: $UPSCALE_MODEL${NC}"; exit 1; fi
    echo -e "${CYAN}✓ Upscale mode enabled (2x ESRGAN)${NC}"
fi

if [ "$IPADAPTER_FLAG" -eq 1 ]; then
    if [ -z "$IPADAPTER_UNET_WEIGHTS" ] || [ ! -f "$IPADAPTER_UNET_WEIGHTS" ]; then
        echo -e "${RED}Error: --ipadapter-unet-weights is required when using --ipadapter${NC}"; exit 1
    fi
    if [ -z "$IPADAPTER_MODEL" ] || [ ! -f "$IPADAPTER_MODEL" ]; then
        echo -e "${RED}Error: --ipadapter-model is required when using --ipadapter${NC}"; exit 1
    fi
    if [ -z "$IPADAPTER_CLIP_VISION" ] || [ ! -f "$IPADAPTER_CLIP_VISION" ]; then
        echo -e "${RED}Error: --ipadapter-clip-vision is required when using --ipadapter${NC}"; exit 1
    fi
    if [ -z "$IPADAPTER_IMAGE" ] || [ ! -f "$IPADAPTER_IMAGE" ]; then
        echo -e "${RED}Error: --ipadapter-image is required when using --ipadapter${NC}"; exit 1
    fi
    echo -e "${CYAN}✓ IPAdapter UNet mode enabled (weight=$IPADAPTER_WEIGHT)${NC}"
fi

echo -e "${GREEN}✓ All checks passed${NC}"

if ! [[ "$WIDTH" =~ ^[0-9]+$ ]] || [ "$WIDTH" -le 0 ]; then echo -e "${RED}Error: width must be positive integer${NC}"; exit 1; fi
if ! [[ "$HEIGHT" =~ ^[0-9]+$ ]] || [ "$HEIGHT" -le 0 ]; then echo -e "${RED}Error: height must be positive integer${NC}"; exit 1; fi

# SDXL 优化参数
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-7.0}"
STEPS="${STEPS:-25}"
HIRES_STEPS="${HIRES_STEPS:-45}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.30}"

# FreeU / SAG 默认值 (SDXL UNet 推荐)
FREEU_B1="${FREEU_B1:-1.3}"
FREEU_B2="${FREEU_B2:-1.4}"
FREEU_S1="${FREEU_S1:-0.9}"
FREEU_S2="${FREEU_S2:-0.2}"
SAG_SCALE="${SAG_SCALE:-1.0}"

if [ "$WIDTH" -ge 2560 ] && [ "$HEIGHT" -ge 1440 ]; then
    echo -e "${BLUE}[INFO] Ultra HD Mode (SDXL): steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
else
    echo -e "${BLUE}[INFO] HD Mode (SDXL): steps=$STEPS, cfg=$CFG_SCALE, sampler=$SAMPLING_METHOD${NC}"
fi

# SDXL 质量前缀 (更强调摄影级真实感)
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, professional photography"
if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, embedding:EasyNegative, embedding:bad-hands-5}"

if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
        OUTPUT="$(basename "$OUTPUT_FILE")"
    else
        OUTPUT_DIR="$HOME"
        OUTPUT="$OUTPUT_FILE"
    fi
    BASE="${OUTPUT%.png}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT="${BASE}_${TIMESTAMP}.png"
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

# 低分辨率计算 - SDXL 优化
# 保持 16:9 比例，latent 尺寸对齐到 8 的倍数
# 默认 1280x720 -> 2560x1440 (2x，20GB 安全)
# 可覆盖: BASE_W/BASE_H 环境变量
if [ -n "${BASE_W:-}" ] && [ -n "${BASE_H:-}" ]; then
    LOW_W=$BASE_W
    LOW_H=$BASE_H
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    LOW_W=1280
    LOW_H=720
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    LOW_W=1024
    LOW_H=576
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    LOW_W=1024
    LOW_H=576
else
    LOW_LATENT_W=$((TARGET_LATENT_W * 4 / 5))
    LOW_LATENT_H=$((TARGET_LATENT_H * 4 / 5))
    LOW_LATENT_W=$(((LOW_LATENT_W + 7) / 8 * 8))
    LOW_LATENT_H=$(((LOW_LATENT_H + 7) / 8 * 8))
    LOW_W=$((LOW_LATENT_W * 8))
    LOW_H=$((LOW_LATENT_H * 8))
fi

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
echo "  SDXL HD Image Generation"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
echo -e "FreeU: b1=$FREEU_B1 b2=$FREEU_B2 s1=$FREEU_S1 s2=$FREEU_S2"
echo -e "SAG: scale=$SAG_SCALE"
if [ "$UPSCALE_FLAG" -eq 1 ]; then
    UPSCALED_W=$((WIDTH * 2))
    UPSCALED_H=$((HEIGHT * 2))
    echo -e "Upscale: ${CYAN}2x ESRGAN -> ${UPSCALED_W}x${UPSCALED_H}${NC}"
fi
if [ "$IPADAPTER_FLAG" -eq 1 ]; then
    echo -e "IPAdapter: ${CYAN}UNet mode, weight=$IPADAPTER_WEIGHT${NC}"
fi
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$RANDOM}"
echo "Generating...  $(date '+%H:%M:%S')"

SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --clip-l "$CLIP_L_MODEL"
  --clip-g "$CLIP_G_MODEL"
  -p "$PROMPT"
  -n "$NEGATIVE_PROMPT"
  --cfg-scale "$CFG_SCALE"
  --sampling-method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size "$VAE_TILE_SIZE"
  --vae-tile-overlap "$VAE_TILE_OVERLAP"
  --freeu
  --freeu-b1 "$FREEU_B1"
  --freeu-b2 "$FREEU_B2"
  --freeu-s1 "$FREEU_S1"
  --freeu-s2 "$FREEU_S2"
  --sag
  --sag-scale "$SAG_SCALE"
  --clarity 0.4
  --sharpen 0.8
  --sharpen-radius 2
  --smart-sharpen 0.5
  --smart-sharpen-radius 2
  --edge-sharpen 1.5
  --edge-sharpen-radius 2
  --edge-sharpen-threshold 0.3
  $( [ -d "$MODEL_DIR/embeddings" ] && echo "--embd-dir $MODEL_DIR/embeddings" || true )
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

# 可选：外部 VAE
if [ -n "$VAE_MODEL" ]; then
    SD_CMD+=(--vae "$VAE_MODEL")
fi

# 可选：IPAdapter UNet
if [ "$IPADAPTER_FLAG" -eq 1 ]; then
    SD_CMD+=(
        --ipadapter
        --ipadapter-unet-weights "$IPADAPTER_UNET_WEIGHTS"
        --ipadapter-model "$IPADAPTER_MODEL"
        --ipadapter-clip-vision "$IPADAPTER_CLIP_VISION"
        --ipadapter-image "$IPADAPTER_IMAGE"
        --ipadapter-weight "$IPADAPTER_WEIGHT"
    )
fi

if [ "$UPSCALE_FLAG" -eq 1 ]; then
    SD_CMD+=(--upscale-model "$UPSCALE_MODEL")
    SD_CMD+=(--upscale-repeats 1)
    SD_CMD+=(--upscale-tile-size 1440)
fi

START_TIME=$(date +%s)
"${SD_CMD[@]}"
END_TIME=$(date +%s)
GEN_DURATION=$((END_TIME - START_TIME))

if [ -f "$OUTPUT_PATH" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)

    if [ $GEN_DURATION -ge 60 ]; then
        DURATION_MIN=$((GEN_DURATION / 60))
        DURATION_SEC=$((GEN_DURATION % 60))
        DURATION_STR="${DURATION_MIN}m ${DURATION_SEC}s"
    else
        DURATION_STR="${GEN_DURATION}s"
    fi

    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File:   ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size:   ${BLUE}$FILE_SIZE${NC}"
    echo -e "Time:   ${YELLOW}$DURATION_STR${NC}"
    echo -e "Seed:   ${YELLOW}$SEED${NC}"
    echo -e "CFG:    ${CYAN}$CFG_SCALE${NC}"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo -e "${RED}✗ Generation failed! Output file not found${NC}"
    echo "========================================"
    exit 1
fi
