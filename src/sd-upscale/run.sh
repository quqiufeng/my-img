#!/bin/bash

INPUT="${1:-/opt/image/test_input.png}"
OUTPUT="/opt/image/test_output_upscale.png"
MODEL="/opt/gguf/image/2x_ESRGAN.gguf"

cd /opt/my-img

nohup ./bin/sd-upscale \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --scale 2 \
  > sd-upscale.log 2>&1 &

echo "Started upscale: $INPUT -> $OUTPUT"
echo "PID: $!"
