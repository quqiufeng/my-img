#!/bin/bash

MODEL_DIR="/opt/image"

PROMPT="${1:-A beautiful landscape}"
OUTPUT_FILE="$2"
WIDTH="${3:-1024}"
HEIGHT="${4:-1024}"

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

echo "Generating image with Juggernaut-XL..."

$HOME/stable-diffusion.cpp/bin/sd-cli \
  -m "$MODEL_DIR/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" \
  -p "cinematic wide shot, $PROMPT, high quality" \
  -n "low quality, blurry, distorted, duplicate, dark" \
  --cfg-scale 6.0 \
  --sampling-method "dpm++2m" \
  -H $HEIGHT -W $WIDTH \
  --steps 30 \
  -s $RANDOM \
  -o "$OUTPUT_DIR/$OUTPUT"

echo "Image saved to: $OUTPUT_DIR/$OUTPUT"
