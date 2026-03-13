#!/bin/bash

MODEL_DIR="/opt/image"

PROMPT="${1:-A beautiful landscape}"
OUTPUT_FILE="$2"
#1216  832 官方推荐
#1280  720 放大两倍刚好 2560  1440

#WIDTH="${3:-1216 }"
#HEIGHT="${4:-832}"

WIDTH="${3:-1280 }"
HEIGHT="${4:-720}"

OUTPUT_DIR="$HOME"
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

echo "Generating image with RealVisXL_V4..."

$HOME/stable-diffusion.cpp/bin/sd-cli \
  -m "$MODEL_DIR/RealVisXL_V4.0.safetensors" \
  -p "RAW photo, (realistic skin texture:1.2), (visible pores:1.1), $PROMPT, 8k, cinematic lighting, masterpiece" \
  -n "(plastic:1.4), (shiny skin:1.4), (oily skin:1.4), (makeup:1.2), (airbrushed:1.2), cartoon, anime, blurry, distorted" \
  --cfg-scale 4.0 \
  --sampling-method "dpm++2m" \
  -H $HEIGHT -W $WIDTH \
  --steps 35 \
  -s $RANDOM \
  -o "$OUTPUT_DIR/$OUTPUT"

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
