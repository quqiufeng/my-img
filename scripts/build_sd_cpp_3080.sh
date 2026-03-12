#!/bin/bash

# stable-diffusion.cpp CUDA 编译脚本

echo "=========================================="
echo "stable-diffusion.cpp CUDA 编译脚本"
echo "=========================================="

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请先安装 CUDA Toolkit"
    exit 1
fi

echo "检测到 CUDA:"
nvcc --version | grep "release"

# 设置 CUDA 环境变量（兼容 WSL2）
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 克隆或更新 stable-diffusion.cpp
if [ ! -d "$HOME/stable-diffusion.cpp" ]; then
    echo ""
    echo "=== 克隆 stable-diffusion.cpp ==="
    git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git $HOME/stable-diffusion.cpp
fi

cd $HOME/stable-diffusion.cpp

# 清理并配置
echo ""
echo "=== 配置 CMake ==="
mkdir -p build && cd build
rm -rf *

# GPU 检测
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

echo "GPU: $GPU_NAME"
echo "Compute Capability: $GPU_CC"

# 根据 GPU 设置架构
case $GPU_CC in
    86) CUDA_ARCH=86 ;;
    89) CUDA_ARCH=89 ;;
    75) CUDA_ARCH=75 ;;
    80) CUDA_ARCH=80 ;;
    90) CUDA_ARCH=90 ;;
    12) CUDA_ARCH=120 ;;
    *) CUDA_ARCH=86 ;;
esac

echo "使用 CUDA_ARCH: $CUDA_ARCH"

# CMake 配置
cmake .. \
    -DSD_CUDA=ON \
    -DSD_FLASH_ATTN=ON \
    -DSD_FAST_SOFTMAX=ON \
    -DGGML_NATIVE=ON \
    -DGGML_LTO=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release

# 编译
echo ""
echo "=== 编译中 (使用 4 线程) ==="
make -j4

# 移动 bin 目录
cd $HOME/stable-diffusion.cpp
if [ -f "build/bin/sd-cli" ]; then
    echo ""
    echo "=== 整理文件 ==="
    mv build/bin bin
    # rm -rf build   # 保留 build 目录供 my-img 链接
    
    echo ""
    echo "=========================================="
    echo "✅ 编译成功！"
    echo "=========================================="
    echo "可执行文件: $HOME/stable-diffusion.cpp/bin/sd-cli"
    echo "API 服务:   $HOME/stable-diffusion.cpp/bin/sd-server"
else
    echo ""
    echo "❌ 编译失败"
    exit 1
fi
