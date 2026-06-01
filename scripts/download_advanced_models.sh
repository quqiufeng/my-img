#!/bin/bash
# =============================================================================
# 模型下载脚本 - 下载 my-img 高级功能所需的模型
# =============================================================================

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models/image}"
mkdir -p "$MODEL_DIR"

echo "========================================"
echo "  Downloading Models for my-img"
echo "  Target: $MODEL_DIR"
echo "========================================"

cd "$MODEL_DIR"

# 颜色
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

download_file() {
    local url="$1"
    local output="$2"
    local desc="$3"
    
    if [ -f "$output" ]; then
        echo -e "${GREEN}✓${NC} $desc already exists"
        return 0
    fi
    
    echo -e "${YELLOW}↓${NC} Downloading $desc..."
    if wget -q --show-progress "$url" -O "$output" 2>/dev/null || \
       curl -L --progress-bar "$url" -o "$output" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $desc downloaded"
        return 0
    else
        echo -e "${RED}✗${NC} Failed to download $desc"
        rm -f "$output"
        return 1
    fi
}

# =============================================================================
# 1. Face Restoration Models
# =============================================================================
echo ""
echo "[1/5] Face Restoration Models"
echo "----------------------------------------"

# GFPGAN v1.4 (PyTorch)
download_file \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
    "GFPGANv1.4.pth" \
    "GFPGAN v1.4"

# CodeFormer (PyTorch)
download_file \
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" \
    "codeformer.pth" \
    "CodeFormer"

# =============================================================================
# 2. IPAdapter Models
# =============================================================================
echo ""
echo "[2/5] IPAdapter Models"
echo "----------------------------------------"

# IPAdapter SD1.5
download_file \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin" \
    "ip-adapter_sd15.bin" \
    "IPAdapter SD1.5"

# IPAdapter Plus SD1.5
download_file \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors" \
    "ip-adapter-plus_sd15.safetensors" \
    "IPAdapter Plus SD1.5"

# CLIP Vision Model (for IPAdapter)
download_file \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    "clip_vision_sd15.safetensors" \
    "CLIP Vision Encoder"

# =============================================================================
# 3. T2I-Adapter Models
# =============================================================================
echo ""
echo "[3/5] T2I-Adapter Models"
echo "----------------------------------------"

# T2I-Adapter Sketch
download_file \
    "https://github.com/TencentARC/T2I-Adapter/releases/download/v0.1.0/t2iadapter_sketch_sd15v2.pth" \
    "t2iadapter_sketch_sd15v2.pth" \
    "T2I-Adapter Sketch"

# T2I-Adapter Canny
download_file \
    "https://github.com/TencentARC/T2I-Adapter/releases/download/v0.1.0/t2iadapter_canny_sd15v2.pth" \
    "t2iadapter_canny_sd15v2.pth" \
    "T2I-Adapter Canny"

# T2I-Adapter Keypose
download_file \
    "https://github.com/TencentARC/T2I-Adapter/releases/download/v0.1.0/t2iadapter_keypose_sd15v2.pth" \
    "t2iadapter_keypose_sd15v2.pth" \
    "T2I-Adapter Keypose"

# T2I-Adapter Depth
download_file \
    "https://github.com/TencentARC/T2I-Adapter/releases/download/v0.1.0/t2iadapter_depth_sd15v2.pth" \
    "t2iadapter_depth_sd15v2.pth" \
    "T2I-Adapter Depth"

# =============================================================================
# 4. Face Swap Models
# =============================================================================
echo ""
echo "[4/5] Face Swap Models"
echo "----------------------------------------"

# Face Detection - YuNet (OpenCV)
download_file \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    "yunet_320_320.onnx" \
    "YuNet Face Detection"

# Face Swap - Inswapper (InsightFace)
download_file \
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx" \
    "inswapper_128.onnx" \
    "Inswapper 128"

# =============================================================================
# 5. PhotoMaker Model
# =============================================================================
echo ""
echo "[5/5] PhotoMaker Model"
echo "----------------------------------------"

# PhotoMaker v1
download_file \
    "https://huggingface.co/TencentARC/PhotoMaker/resolve/main/photomaker-v1.bin" \
    "photomaker-v1.bin" \
    "PhotoMaker v1"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "  Download Summary"
echo "========================================"
echo "Models directory: $MODEL_DIR"
echo ""
echo "Downloaded models:"
ls -lh "$MODEL_DIR"/*.pth "$MODEL_DIR"/*.bin "$MODEL_DIR"/*.onnx "$MODEL_DIR"/*.safetensors 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || true
echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo "Usage examples:"
echo "  Face Restoration:  --face-restore --face-restore-model $MODEL_DIR/GFPGANv1.4.pth"
echo "  IPAdapter:         --ipadapter --ipadapter-model $MODEL_DIR/ip-adapter_sd15.bin --ipadapter-clip-vision $MODEL_DIR/clip_vision_sd15.safetensors"
echo "  T2I-Adapter:       --t2i-adapter --t2i-adapter-model $MODEL_DIR/t2iadapter_sketch_sd15v2.pth"
echo "  Face Swap:         --face-swap --face-swap-source source.jpg --face-swap-detection-model $MODEL_DIR/yunet_320_320.onnx --face-swap-model $MODEL_DIR/inswapper_128.onnx"
echo "  PhotoMaker:        --photomaker --photomaker-model $MODEL_DIR/photomaker-v1.bin --photomaker-id-images id1.jpg,id2.jpg"
