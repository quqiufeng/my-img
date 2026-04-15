#!/bin/bash
# =============================================================================
# apply-modular-patches.sh - Apply my-img modular patches to stable-diffusion.cpp
# =============================================================================
#
# These patches are logically modular but all modify src/stable-diffusion.cpp.
# They must be applied in order. We use git apply which handles line offsets
# automatically when patches are applied sequentially.
#
# Usage:
#   cd /path/to/stable-diffusion.cpp
#   bash /path/to/apply-modular-patches.sh
# =============================================================================

set -e

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Applying my-img modular patches"
echo "========================================"

patches=(
    "001-include-and-hooks.patch"
    "002-sampling-hooks.patch"
    "003-hook-setters.patch"
    "004-node-api.patch"
)

for p in "${patches[@]}"; do
    patch_file="$PATCH_DIR/$p"
    if [ ! -f "$patch_file" ]; then
        echo "❌ Patch not found: $patch_file"
        exit 1
    fi

    echo ""
    echo "Applying $p ..."
    
    if git apply --check "$patch_file" 2>/dev/null; then
        git apply "$patch_file"
        echo "  ✅ Applied with git apply"
    else
        echo "  ❌ Failed to apply $p"
        echo ""
        echo "This usually means stable-diffusion.cpp has changed significantly."
        echo "You may need to regenerate the patches from the patched source."
        exit 1
    fi
done

echo ""
echo "========================================"
echo "✅ All modular patches applied!"
echo "========================================"
