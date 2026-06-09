#!/bin/bash
# 诊断2560x1440模糊问题 - 控制变量测试
set -euo pipefail

cd /opt/my-img

BUILD_DIR="build"
[ ! -d "$BUILD_DIR" ] && { echo "错误: 找不到构建目录 $BUILD_DIR"; exit 1; }

cd "$BUILD_DIR"
MODEL_DIR="/data/models/image"
CLIP_DIR="$MODEL_DIR"
OUTPUT_BASE="/opt/my-img/test_blur_diagnosis"
mkdir -p "$OUTPUT_BASE"

# 使用相同种子确保可比较
SEED=555888
PROMPT="a girl, oil painting, rich colors"

# 基础参数 (SDXL Base full checkpoint)
BASE_ARGS=(
  --diffusion-model "$MODEL_DIR/sd_xl_base_1.0.safetensors"
  --clip-l "$CLIP_DIR/clip_l.safetensors"
  --clip-g "$CLIP_DIR/clip_g.safetensors"
  --vae-auto
  --diffusion-fa
  --scheduler euler
  -p "$PROMPT"
  -n ""
  --cfg-scale 6.0
  --steps 20
  -s "$SEED"
)

echo "=== 测试1: 1280x720 直接生成 (无 HiRes, 基准) ==="
./myimg-cli "${BASE_ARGS[@]}" \
  -W 1280 -H 720 \
  -o "$OUTPUT_BASE/01_base_1280x720.png" 2>> "$OUTPUT_BASE/test.log"

echo "=== 测试2: 2560x1440 HiRes, 无增强 ==="
./myimg-cli "${BASE_ARGS[@]}" \
  -W 1280 -H 720 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.30 --hires-steps 45 \
  -o "$OUTPUT_BASE/02_hires_only.png" 2>> "$OUTPUT_BASE/test.log"

echo "=== 测试3: 2560x1440 HiRes + FreeU (无 SAG) ==="
./myimg-cli "${BASE_ARGS[@]}" \
  -W 1280 -H 720 \
  --freeu --freeu-b1 1.3 --freeu-b2 1.4 --freeu-s1 0.9 --freeu-s2 0.2 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.30 --hires-steps 45 \
  -o "$OUTPUT_BASE/03_hires_freeu.png" 2>> "$OUTPUT_BASE/test.log"

echo "=== 测试4: 2560x1440 HiRes + FreeU + SAG (无后处理) ==="
./myimg-cli "${BASE_ARGS[@]}" \
  -W 1280 -H 720 \
  --freeu --freeu-b1 1.3 --freeu-b2 1.4 --freeu-s1 0.9 --freeu-s2 0.2 \
  --sag --sag-scale 1.0 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.30 --hires-steps 45 \
  -o "$OUTPUT_BASE/04_hires_freeu_sag.png" 2>> "$OUTPUT_BASE/test.log"

echo "=== 测试5: 2560x1440 HiRes + FreeU + SAG + 低强度后处理 ==="
./myimg-cli "${BASE_ARGS[@]}" \
  -W 1280 -H 720 \
  --freeu --freeu-b1 1.3 --freeu-b2 1.4 --freeu-s1 0.9 --freeu-s2 0.2 \
  --sag --sag-scale 1.0 \
  --clarity 0.2 --sharpen 0.4 --sharpen-radius 1 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.30 --hires-steps 45 \
  -o "$OUTPUT_BASE/05_hires_freeu_sag_post.png" 2>> "$OUTPUT_BASE/test.log"

echo "=== 测试6: 2560x1440 HiRes强度0.15 (FreeU+SAG) ==="
./myimg-cli "${BASE_ARGS[@]}" \
  -W 1280 -H 720 \
  --freeu --freeu-b1 1.3 --freeu-b2 1.4 --freeu-s1 0.9 --freeu-s2 0.2 \
  --sag --sag-scale 1.0 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.15 --hires-steps 30 \
  -o "$OUTPUT_BASE/06_hires_weak.png" 2>> "$OUTPUT_BASE/test.log"

echo "=== 测试7: 2560x1440 直接生成 (无 HiRes) ==="
./myimg-cli "${BASE_ARGS[@]}" \
  -W 2560 -H 1440 \
  -o "$OUTPUT_BASE/07_direct_2560x1440.png" 2>> "$OUTPUT_BASE/test.log"

echo ""
echo "=== 所有测试完成 ==="
echo "输出目录: $OUTPUT_BASE"
echo ""
echo "对比建议:"
echo "1. 01 vs 02: HiRes是否导致模糊"
echo "2. 02 vs 03: FreeU是否导致模糊"
echo "3. 03 vs 04: SAG是否导致模糊"
echo "4. 04 vs 05: 后处理是否导致模糊"
echo "5. 04 vs 06: HiRes强度影响"
echo "6. 04 vs 07: HiRes vs 直接生成"
