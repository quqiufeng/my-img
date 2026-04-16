// ============================================================================
// sd-engine/nodes/image_filter_nodes.cpp
// ============================================================================
// 图像滤镜/效果节点入口（薄封装，实际实现分散在子模块中）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

extern void init_image_blend_nodes();
extern void init_image_adjust_nodes();
extern void init_image_effect_nodes();

void init_image_filter_nodes() {
    init_image_blend_nodes();
    init_image_adjust_nodes();
    init_image_effect_nodes();
}

} // namespace sdengine
