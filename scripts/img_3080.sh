#!/bin/bash

# 用法:
#   ./img.sh [提示词] [输出文件] [宽度] [高度]
# 
# 参数说明:
#   提示词     - 要生成的图像描述 (默认: "A beautiful landscape")
#   输出文件    - 输出文件路径 (默认: /opt/时间戳_md5.png)
#                如果包含目录路径，则保存到指定目录
#   宽度       - 图像宽度 (默认: 1920)
#   高度       - 图像高度 (默认: 1080)
#
# 示例:
#   ./img.sh "A cat on the table"
#   ./img.sh "A sunset" /opt/sunset.png 2560 1440
#   ./img.sh "Mountain" mountain.png 1920 1080

# 模型文件路径 (位于 /opt/gguf/image/ 目录)
# Stable Diffusion GGUF 模型 (stable-diffusion.cpp 作者维护)
# https://huggingface.co/leejet/Z-Image-Turbo-GGUF
MODEL_DIR="/opt/image"

PROMPT="${1:-A beautiful landscape}"
OUTPUT_FILE="$2"
WIDTH="${3:-1920}"
HEIGHT="${4:-1080}"

# 默认保存到 /opt 目录
OUTPUT_DIR="$HOME"

# 如果用户指定的输出路径包含目录，则使用用户指定的目录
if [[ "$OUTPUT_FILE" == *"/"* ]]; then
  OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
  OUTPUT="$(basename "$OUTPUT_FILE")"
elif [ -n "$OUTPUT_FILE" ]; then
  OUTPUT="$OUTPUT_FILE"
else
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
  OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

echo "Generating image..."
echo "Prompt: $PROMPT"
echo "Size: ${WIDTH}x${HEIGHT}"
echo "Output: $OUTPUT_DIR/$OUTPUT"

$HOME/stable-diffusion.cpp/bin/sd-cli \
  --diffusion-model $MODEL_DIR/z_image_turbo-Q6_K.gguf \
  --vae $MODEL_DIR/ae.safetensors \
  --llm $MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "$PROMPT" \
  --cfg-scale 1.0 \
  --diffusion-fa \
  --sampling-method euler \
  -H $HEIGHT -W $WIDTH \
  --steps 4 \
  -s $RANDOM \
  -o "$OUTPUT_DIR/$OUTPUT" > /dev/null 2>&1
# 参数说明:
# --scheduler karras: 使用 Karras 调度器，图像更清晰
# --steps 25: 步数越多，细节越好（默认20，推荐25）
# --diffusion-fa: Flash Attention 加速
# --cache-mode easycache: 缓存加速，跳过部分步骤

echo "Image saved to: $OUTPUT_DIR/$OUTPUT"

# =============================================================================
# 风景壁纸生成参考命令 (1280x720 小图)
# =============================================================================
#
# # 1. 马丘比丘
# ./img.sh "Machu Picchu, ancient Incan citadel perched on mountain ridge, misty clouds, lush green mountains, stone ruins, dramatic landscape, golden hour lighting, breathtaking view, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_01.png 1280 720
#
# # 2. 瑞士阿尔卑斯山
# ./img.sh "Swiss Alps, majestic mountain peaks, snow-capped mountains, crystal clear lake, green valleys, scenic landscape, dramatic clouds, golden sunlight, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_02.png 1280 720
#
# # 3. 美国大峡谷
# ./img.sh "Grand Canyon USA, massive red rock canyon, layered rock formations, Colorado River winding through, dramatic desert landscape, golden hour, breathtaking vista, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_03.png 1280 720
#
# # 4. 圣托里尼岛
# ./img.sh "Santorini Greece, iconic blue-domed churches, white-washed buildings, cliffside village, Aegean Sea, sunset sky, romantic atmosphere, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_04.png 1280 720
#
# # 5. 挪威峡湾
# ./img.sh "Norway Fjords, majestic steep cliffs, crystal clear water, mountains reflected in fjord, green vegetation, dramatic landscape, misty atmosphere, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_05.png 1280 720
#
# # 6. 日本富士山
# ./img.sh "Mount Fuji Japan, iconic snow-capped mountain, cherry blossoms in foreground, peaceful lake reflection, traditional Japanese temple, dramatic landscape, serene atmosphere, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_06.png 1280 720
#
# # 7. 布拉格老城广场
# ./img.sh "Prague Old Town Square, historic Gothic architecture, Astronomical Clock, colorful baroque buildings, cobblestone streets, Charles Bridge in distance, golden hour lighting, European charm, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_07.png 1280 720
#
# # 8. 纳米比亚索苏斯盐沼
# ./img.sh "Namibia Sossusvlei, iconic red sand dunes, dead tree silhouettes, Deadvlei pan, dramatic desert landscape, golden hour, clear blue sky, surreal atmosphere, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_08.png 1280 720
#
# # 9. 威尼斯
# ./img.sh "Venice Italy, iconic Grand Canal, historic palazzos, gondola on water, Rialto Bridge, golden sunset, romantic atmosphere, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_09.png 1280 720
#
# # 10. 巴厘岛梯田
# ./img.sh "Bali Rice Terraces, Tegallalang terraced rice fields, lush green tropical landscape, palm trees, traditional Balinese temple, misty mountains in background, scenic beauty, travel destination, photorealistic, high detail, 8K quality" /opt/wallpaper_10.png 1280 720
