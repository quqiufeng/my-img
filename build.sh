#!/bin/bash
# =============================================================================
# my-img 构建脚本 - 默认启用 GPU (CUDA)
# =============================================================================
#
# 【编译踩坑记录 / 2025-05-29】
#
# 问题 1: 权限 denied
#   现象: /opt/stable-diffusion.cpp/build 为 root 所有，当前用户无法写入
#   修复: sudo chown -R $(whoami):$(whoami) /opt/stable-diffusion.cpp
#
# 问题 2: CUDA 非标准路径 (/data/cuda)
#   现象: libtorch CMake 找不到 CUDA，报 "Cannot find the CUDA libraries"
#   修复:
#     sudo ln -sf /data/cuda /usr/local/cuda
#     sudo ln -sf /usr/local/cuda /opt/libtorch/cuda
#     sudo mkdir -p /usr/local/cuda/include
#     sudo ln -sf /data/cuda/targets/x86_64-linux/include/* /usr/local/cuda/include/
#     sudo mkdir -p /usr/local/cuda/lib64
#     sudo ln -sf /data/cuda/targets/x86_64-linux/lib/* /usr/local/cuda/lib64/
#   注意: libcudart.so 不能循环链接，必须指向 libcudart.so.12.6.77
#
# 问题 3: nvToolsExt 缺失
#   现象: "Failed to find nvToolsExt"
#   修复: sudo ln -sf /opt/libtorch/lib/libnvToolsExt-847d78f2.so.1 /usr/local/cuda/lib64/libnvToolsExt.so
#
# 问题 4: 第三方库缺失 (third_party)
#   现象: "third_party/json" 等目录不存在，CMake 报错 add_subdirectory 失败
#   修复: bash setup.sh  # 自动 clone json/ggml/stb
#
# 问题 5: stable-diffusion.cpp API 不兼容
#   现象: "sd_ctx_params_t has no member named max_vram"
#   修复: 确保 sd.cpp 已升级到支持该字段的版本
#
# 问题 6: OpenCV 缺失
#   现象: "opencv2/imgproc.hpp: No such file or directory"
#   修复: sudo apt install libopencv-dev
#
# 问题 7: LTO 版本不匹配
#   现象: "bytecode stream generated with LTO version 12.0 instead of 13.1"
#   原因: stable-diffusion.cpp 用 gcc-12 编译，但 my-img 链接时用了 g++-13
#   修复: 统一通过环境变量 export CC/CXX，而不是仅在 cmake 中指定
#
# 环境要求:
#   - CUDA: /data/cuda (需符号链接到 /usr/local/cuda)
#   - libtorch: /opt/libtorch (CUDA 版本)
#   - GCC: /usr/bin/gcc-12, g++-12 (避免与 gcc-13 混用)
#   - OpenCV: libopencv-dev
#
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

# 编译器设置（统一使用 GCC-12，避免版本混用）
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

# CUDA 设置
export CUDA_HOME=/data/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  my-img 构建脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查编译器
echo -e "${CYAN}编译器:${NC}"
echo "  CC:  $CC ($($CC --version | head -1))"
echo "  CXX: $CXX ($($CXX --version | head -1))"
echo ""

# 检查 GPU 和 CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo -e "${GREEN}✓ 检测到 GPU: ${GPU_INFO}${NC}"
    echo "  Compute Capability: ${GPU_CC}"
    USE_CUDA=ON
    
    # 检测 CUDA 版本
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        CUDA_VERSION=$($CUDA_HOME/bin/nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo "  CUDA 版本: ${CUDA_VERSION}"
    fi
    
    # 根据 GPU 设置架构
    case $GPU_CC in
        86)  CUDA_ARCH=86 ;;
        89)  CUDA_ARCH=89 ;;
        75)  CUDA_ARCH=75 ;;
        80)  CUDA_ARCH=80 ;;
        90)  CUDA_ARCH=90 ;;
        120) CUDA_ARCH=120 ;;
        12)  CUDA_ARCH=120 ;;
        *)   CUDA_ARCH=86 ;;
    esac
    echo "  CUDA_ARCH: ${CUDA_ARCH}"
else
    echo -e "${YELLOW}⚠ 未检测到 GPU，将使用 CPU 模式${NC}"
    USE_CUDA=OFF
    CUDA_ARCH=""
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

# 确保子模块完整
if [ ! -f "${SD_DIR}/ggml/CMakeLists.txt" ]; then
    echo -e "${YELLOW}⚠ 子模块缺失，正在更新...${NC}"
    cd "${SD_DIR}"
    git submodule update --init --recursive
fi

# 清理并创建 build 目录
cd "${SD_DIR}"
rm -rf "${SD_BUILD_DIR}"
mkdir -p "${SD_BUILD_DIR}" && cd "${SD_BUILD_DIR}"

# CMake 配置
CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DSD_CUDA="${USE_CUDA}"
    -DGGML_CUDA="${USE_CUDA}"
)

if [ "$USE_CUDA" = "ON" ]; then
    CMAKE_FLAGS+=(
        -DSD_FLASH_ATTN=ON
        -DSD_FAST_SOFTMAX=ON
        -DGGML_NATIVE=OFF
        -DGGML_LTO=ON
        -DGGML_CUDA_FA_ALL_QUANTS=ON
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
        -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
    )
fi

cmake .. "${CMAKE_FLAGS[@]}"
make -j${JOBS} stable-diffusion

echo -e "${GREEN}✓ stable-diffusion.cpp 编译完成${NC}"
echo ""

# =============================================================================
# 2. 编译 my-img
# =============================================================================
echo -e "${BLUE}[2/3] 编译 my-img...${NC}"

cd "${SCRIPT_DIR}"
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

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
