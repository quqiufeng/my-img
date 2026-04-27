#!/bin/bash
# setup.sh - 环境设置脚本

set -e

echo "========================================"
echo "  my-img 环境设置"
echo "========================================"

# 检查 libtorch
if [ -d "/opt/libtorch" ] || [ -d "/usr/local/libtorch" ]; then
    echo "✓ libtorch 已安装"
else
    echo "✗ libtorch 未找到"
    echo ""
    echo "请下载 libtorch (CUDA 12.1 版本):"
    echo "  wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-latest.zip"
    echo "  unzip libtorch-cxx11-abi-shared-with-deps-latest.zip -d /opt/"
    echo ""
    echo "或者设置环境变量:"
    echo "  export LIBTORCH_ROOT=/path/to/libtorch"
    exit 1
fi

# 检查第三方库
cd "$(dirname "$0")"
if [ ! -d "third_party/json" ] || [ ! -d "third_party/ggml" ] || [ ! -d "third_party/stb" ]; then
    echo "正在 clone 第三方库..."
    mkdir -p third_party
    cd third_party
    
    [ ! -d "json" ] && git clone --depth 1 https://github.com/nlohmann/json.git
    [ ! -d "ggml" ] && git clone --depth 1 https://github.com/ggml-org/ggml.git
    [ ! -d "stb" ] && git clone --depth 1 https://github.com/nothings/stb.git
    
    cd ..
fi

echo "✓ 第三方库已就绪"
echo ""
echo "========================================"
echo "  环境设置完成"
echo "========================================"
echo ""
echo "编译命令:"
echo "  mkdir -p build && cd build"
echo "  cmake .."
echo "  make -j\$(nproc)"
echo ""
