#!/bin/bash
# =============================================================================
# my-img 构建脚本 - 默认启用 GPU (CUDA)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
NC="\033[0m"

# 配置
SD_DIR="/opt/stable-diffusion.cpp"
SD_BUILD_DIR="${SD_DIR}/build"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  my-img 构建脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ 检测到 GPU: ${GPU_INFO}${NC}"
    USE_CUDA=ON
else
    echo -e "${YELLOW}⚠ 未检测到 GPU，将使用 CPU 模式${NC}"
    USE_CUDA=OFF
fi

echo ""
echo -e "${CYAN}构建配置:${NC}"
echo "  Build Type: ${BUILD_TYPE}"
echo "  Jobs: ${JOBS}"
echo "  CUDA: ${USE_CUDA}"
echo ""

# =============================================================================
# 1. 编译 stable-diffusion.cpp
# =============================================================================
echo -e "${BLUE}[1/3] 编译 stable-diffusion.cpp...${NC}"

if [ ! -d "${SD_DIR}" ]; then
    echo -e "${RED}Error: stable-diffusion.cpp 未找到${NC}"
    echo "请确保 /opt/stable-diffusion.cpp 已安装"
    exit 1
fi

mkdir -p "${SD_BUILD_DIR}"
cd "${SD_BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DSD_CUDA=${USE_CUDA} \
    -DGGML_CUDA=${USE_CUDA}

make -j${JOBS} stable-diffusion

echo -e "${GREEN}✓ stable-diffusion.cpp 编译完成${NC}"
echo ""

# =============================================================================
# 2. 编译 my-img
# =============================================================================
echo -e "${BLUE}[2/3] 编译 my-img...${NC}"

cd "${SCRIPT_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

make -j${JOBS} myimg-cli

echo -e "${GREEN}✓ my-img 编译完成${NC}"
echo ""

# =============================================================================
# 3. 编译测试（可选）
# =============================================================================
echo -e "${BLUE}[3/3] 编译测试...${NC}"
make -j${JOBS} test_gguf_loader test_vae test_sdcpp_adapter

echo -e "${GREEN}✓ 测试编译完成${NC}"
echo ""

# =============================================================================
# 完成
# =============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  构建成功!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "可执行文件: ${BUILD_DIR}/myimg-cli"
echo ""
echo "使用示例:"
echo "  ./build/myimg-cli --help"
echo ""
echo "运行测试:"
echo "  cd build && LD_LIBRARY_PATH=.:third_party/ggml/src:\$LD_LIBRARY_PATH ./test_gguf_loader"
echo ""

# 验证二进制文件
if [ -f "${BUILD_DIR}/myimg-cli" ]; then
    FILE_SIZE=$(du -h "${BUILD_DIR}/myimg-cli" | cut -f1)
    echo -e "文件大小: ${CYAN}${FILE_SIZE}${NC}"
    
    # 检查是否链接了 CUDA
    if ldd "${BUILD_DIR}/myimg-cli" | grep -q "libcudart"; then
        echo -e "CUDA 支持: ${GREEN}已启用${NC}"
    else
        echo -e "CUDA 支持: ${YELLOW}未启用${NC}"
    fi
fi
