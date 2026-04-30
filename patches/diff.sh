#!/bin/bash
# =============================================================================
# Patch Application Script for stable-diffusion.cpp
# =============================================================================
# 
# 用途: 在升级 stable-diffusion.cpp 后重新应用所有质量增强补丁
# 
# 修改的文件列表:
#   1. /opt/stable-diffusion.cpp/include/stable-diffusion.h
#   2. /opt/stable-diffusion.cpp/src/unet.hpp
#   3. /opt/stable-diffusion.cpp/src/diffusion_model.hpp
#   4. /opt/stable-diffusion.cpp/src/stable-diffusion.cpp
#
# 支持的功能:
#   - FreeU: 在 UNet 解码器的跳跃连接处注入频域加权
#     - backbone 特征 × b1/b2 (放大)
#     - skip 特征 × s1/s2 (缩小)
#     - 默认参数: b1=1.3, b2=1.4, s1=0.9, s2=0.2
#   
#   - SAG (Self-Attention Guidance): 自注意力引导
#     - 改善图像一致性和细节
#     - 默认参数: scale=1.0
#   
#   - Dynamic CFG: 动态 CFG 阈值
#     - 防止 CFG 过高导致的过饱和
#     - 自动调整 CFG 强度
#
# 使用方法:
#   cd /home/dministrator/my-img/patches
#   ./diff.sh apply    # 应用补丁
#   ./diff.sh revert   # 还原补丁
#   ./diff.sh status   # 查看当前状态
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDCPP_DIR="/opt/stable-diffusion.cpp"

PATCH_FILES=(
    "01_stable-diffusion.h.patch"
    "02_unet.hpp.patch"
    "03_diffusion_model.hpp.patch"
    "04_stable-diffusion.cpp.patch"
)

apply_patches() {
    echo "=== Applying patches to stable-diffusion.cpp ==="
    for patch in "${PATCH_FILES[@]}"; do
        echo "Applying: $patch"
        if [ -f "$SCRIPT_DIR/$patch" ]; then
            patch -p0 -d "$SDCPP_DIR" < "$SCRIPT_DIR/$patch" || echo "Warning: $patch may already be applied"
        else
            echo "Error: $patch not found"
            exit 1
        fi
    done
    echo "=== All patches applied successfully ==="
    echo ""
    echo "Next steps:"
    echo "  1. cd $SDCPP_DIR/build && make -j\$(nproc)"
    echo "  2. cd /home/dministrator/my-img/build && make -j\$(nproc)"
}

revert_patches() {
    echo "=== Reverting patches ==="
    for patch in "${PATCH_FILES[@]}"; do
        echo "Reverting: $patch"
        if [ -f "$SCRIPT_DIR/$patch" ]; then
            patch -p0 -R -d "$SDCPP_DIR" < "$SCRIPT_DIR/$patch" || echo "Warning: $patch may not be applied"
        fi
    done
    echo "=== Patches reverted ==="
}

check_status() {
    echo "=== Checking patch status ==="
    for patch in "${PATCH_FILES[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$patch" ]; then
            echo "  [MISSING   ] $patch"
            continue
        fi
        if patch -p0 --dry-run -d "$SDCPP_DIR" < "$SCRIPT_DIR/$patch" 2>/dev/null | grep -q "succeeded"; then
            echo "  [NEED APPLY] $patch"
        else
            echo "  [APPLIED   ] $patch"
        fi
    done
}

case "${1:-apply}" in
    apply)
        apply_patches
        ;;
    revert)
        revert_patches
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {apply|revert|status}"
        exit 1
        ;;
esac
