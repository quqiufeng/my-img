#!/bin/bash
# =============================================================================
# apply_patches.sh - 应用 my-img 的修改到 stable-diffusion.cpp
# =============================================================================
#
# 用法：
#   ./apply_patches.sh
#
# 这个脚本使用 git patch 文件将 my-img 的修改应用到 stable-diffusion.cpp。
#
# 升级 stable-diffusion.cpp 后的工作流程：
#   1. cd stable-diffusion.cpp && git pull
#   2. cd ~/my-img && ./apply_patches.sh
#   3. 重新编译 stable-diffusion.cpp
#   4. 重新编译 my-img
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MY_IMG_DIR="$SCRIPT_DIR"
SD_DIR="$MY_IMG_DIR/stable-diffusion.cpp"
PATCHES_DIR="$MY_IMG_DIR/patches"

# Patch 文件
FULL_PATCH="$PATCHES_DIR/sd-engine-full.patch"
EXT_HEADER_PATCH="$PATCHES_DIR/sd-engine-ext-header.patch"

echo "========================================"
echo "Applying my-img patches to stable-diffusion.cpp"
echo "========================================"

# 检查目录
if [ ! -d "$SD_DIR" ]; then
    echo "Error: stable-diffusion.cpp not found at $SD_DIR"
    echo "Please create a symlink: ln -s /path/to/stable-diffusion.cpp $SD_DIR"
    exit 1
fi

if [ ! -d "$PATCHES_DIR" ]; then
    echo "Error: Patches directory not found at $PATCHES_DIR"
    exit 1
fi

cd "$SD_DIR"

# 检查 patch 文件是否存在
if [ ! -f "$FULL_PATCH" ]; then
    echo "Error: Main patch file not found: $FULL_PATCH"
    exit 1
fi

echo ""
echo "=== Applying patches via git apply ==="

# 1. 应用主 patch（stable-diffusion.cpp 修改）
echo ""
echo "[1/2] Applying sd-engine-full.patch..."
if git apply --check "$FULL_PATCH" 2>/dev/null; then
    git apply "$FULL_PATCH"
    echo "✅ Main patch applied successfully"
else
    echo "❌ Failed to apply main patch"
    echo ""
    echo "Possible reasons:"
    echo "  - stable-diffusion.cpp has been updated and the patch no longer applies cleanly"
    echo "  - The patch has already been applied"
    echo ""
    echo "To check the error details, run:"
    echo "  git apply --check $FULL_PATCH"
    exit 1
fi

# 2. 应用扩展头文件 patch（如果不存在则直接复制）
echo ""
echo "[2/2] Applying sd-engine-ext-header.patch..."
if [ -f "$EXT_HEADER_PATCH" ] && [ -s "$EXT_HEADER_PATCH" ]; then
    if git apply --check "$EXT_HEADER_PATCH" 2>/dev/null; then
        git apply "$EXT_HEADER_PATCH"
        echo "✅ Extension header patch applied successfully"
    else
        # 如果 patch 失败，直接复制文件
        echo "⚠️  Patch failed, copying file directly..."
        cp "$MY_IMG_DIR/stable-diffusion.cpp-patched/include/stable-diffusion-ext.h" \
           "$SD_DIR/include/stable-diffusion-ext.h"
        echo "✅ Extension header copied successfully"
    fi
else
    # 直接复制文件
    cp "$MY_IMG_DIR/stable-diffusion.cpp-patched/include/stable-diffusion-ext.h" \
       "$SD_DIR/include/stable-diffusion-ext.h"
    echo "✅ Extension header copied successfully"
fi

echo ""
echo "========================================"
echo "✅ All patches applied successfully!"
echo "========================================"
echo ""
echo "Modified files:"
echo "  - src/stable-diffusion.cpp (Deep HighRes Fix hooks + ComfyUI-style C API)"
echo "  - include/stable-diffusion-ext.h (Extension header)"
echo ""
echo "Next steps:"
echo "  1. cd $SD_DIR && mkdir -p build && cd build"
echo "  2. cmake .. -DSD_CUDA=ON -DSD_FLASH_ATTN=ON"
echo "  3. make -j\$(nproc)"
echo "  4. cd $MY_IMG_DIR/build && cmake .. && make -j\$(nproc)"
echo ""
echo "To verify the patches are applied:"
echo "  grep -n 'sd_latent_hook_t\|sd_sampler_run\|sd_apply_loras' $SD_DIR/src/stable-diffusion.cpp | head -5"
