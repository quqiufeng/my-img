#!/bin/bash
# =============================================================================
# apply_patches.sh - 应用 my-img 的修改到 stable-diffusion.cpp
# =============================================================================
#
# 用法：
#   ./apply_patches.sh
#
# 这个脚本会将 my-img 修改过的 stable-diffusion.cpp 文件
# 复制到 stable-diffusion.cpp 目录中。
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
PATCHED_DIR="$MY_IMG_DIR/stable-diffusion.cpp-patched"
PATCH_FILE="$MY_IMG_DIR/patches/deep_hires_hook.patch"

echo "========================================"
echo "Applying my-img patches to stable-diffusion.cpp"
echo "========================================"

# 检查目录
if [ ! -d "$SD_DIR" ]; then
    echo "Error: stable-diffusion.cpp not found at $SD_DIR"
    echo "Please create a symlink: ln -s /path/to/stable-diffusion.cpp $SD_DIR"
    exit 1
fi

if [ ! -d "$PATCHED_DIR" ]; then
    echo "Error: Patched files not found at $PATCHED_DIR"
    exit 1
fi

# 方法 1: 尝试使用 git apply 打 patch（推荐，可以检测冲突）
if [ -f "$PATCH_FILE" ]; then
    echo ""
    echo "=== Trying git apply patch ==="
    cd "$SD_DIR"
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
        git apply "$PATCH_FILE"
        echo "✅ Patch applied successfully via git apply"
        APPLY_METHOD="patch"
    else
        echo "⚠️  Git apply failed (possibly due to upstream changes), falling back to file copy"
        APPLY_METHOD=""
    fi
else
    echo "⚠️  Patch file not found, using file copy fallback"
    APPLY_METHOD=""
fi

# 方法 2: 文件覆盖（fallback）
if [ -z "$APPLY_METHOD" ]; then
    echo ""
    echo "=== Applying patches via file copy ==="
    
    # 备份原始文件（如果不存在）
    if [ ! -f "$PATCHED_DIR/src/stable-diffusion.cpp.bak" ]; then
        echo "Creating backup of original stable-diffusion.cpp..."
        cp "$SD_DIR/src/stable-diffusion.cpp" "$PATCHED_DIR/src/stable-diffusion.cpp.bak"
    fi
    
    # 1. 修改后的 src/stable-diffusion.cpp
    if [ -f "$PATCHED_DIR/src/stable-diffusion.cpp" ]; then
        echo "  - src/stable-diffusion.cpp"
        cp "$PATCHED_DIR/src/stable-diffusion.cpp" "$SD_DIR/src/stable-diffusion.cpp"
    fi
    
    # 2. 扩展头文件
    if [ -f "$PATCHED_DIR/include/stable-diffusion-ext.h" ]; then
        echo "  - include/stable-diffusion-ext.h"
        cp "$PATCHED_DIR/include/stable-diffusion-ext.h" "$SD_DIR/include/stable-diffusion-ext.h"
    fi
    
    echo "✅ Patches applied via file copy"
fi

echo ""
echo "Patches applied successfully!"
echo ""
echo "Next steps:"
echo "  1. cd $SD_DIR && mkdir -p build && cd build"
echo "  2. cmake .. -DSD_CUDA=ON -DSD_FLASH_ATTN=ON"
echo "  3. make -j4"
echo "  4. cd $MY_IMG_DIR/build && make -j2"
