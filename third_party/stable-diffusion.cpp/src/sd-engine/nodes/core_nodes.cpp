// ============================================================================
// sd-engine/nodes/core_nodes.cpp
// ============================================================================
// 核心节点初始化入口（确保节点文件被链接）
// ============================================================================

#include "nodes/node_utils.h"

namespace sdengine {

void init_core_nodes() {
    init_loader_nodes();
    init_image_nodes();
}

} // namespace sdengine
