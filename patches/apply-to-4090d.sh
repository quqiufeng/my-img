#!/bin/bash
# =============================================================================
# 4090d 部署脚本 - HiRes Fix 修复
# =============================================================================

set -e

echo "=========================================="
echo "HiRes Fix 修复部署到 4090d"
echo "=========================================="

# 1. 应用 stable-diffusion.cpp patch
echo "[1/3] 应用 stable-diffusion.cpp patch..."
cd ~/stable-diffusion.cpp
git checkout -- .  # 先清理
git apply ~/my-img/patches/hires-fix-stable-diffusion.patch
echo "✅ stable-diffusion.cpp patch 应用成功"

# 2. 应用 my-img patch
echo "[2/3] 应用 my-img patch..."
cd ~/my-img
git checkout -- .  # 先清理
git apply patches/hires-fix-myimg.patch
echo "✅ my-img patch 应用成功"

# 3. 编译 stable-diffusion.cpp (CUDA)
echo "[3/3] 编译 stable-diffusion.cpp..."
cd ~/stable-diffusion.cpp
rm -rf build
mkdir build && cd build
cmake .. -DSD_CUDA=ON
make -j$(nproc)
echo "✅ stable-diffusion.cpp 编译完成"

# 4. 编译 my-img
echo "[4/4] 编译 my-img..."
cd ~/my-img/build
rm -rf *
cmake .. -DSD_PATH=~/stable-diffusion.cpp
make -j$(nproc)
echo "✅ my-img 编译完成"

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "测试命令（2560x1440 大图）："
echo "  cd ~/my-shell/3080 && ./test_hires.sh"
echo ""
echo "或使用 sd-hires（推荐 --vae-tile-size 512）："
echo "  ~/my-img/build/sd-hires <参数> --vae-tile-size 512"
