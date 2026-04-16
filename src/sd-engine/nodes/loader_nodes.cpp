// ============================================================================
// sd-engine/nodes/loader_nodes.cpp
// ============================================================================
// 加载器类节点入口（薄封装，实际实现分散在子模块中）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

extern void init_model_loader_nodes();
extern void init_lora_loader_nodes();
extern void init_upscale_loader_nodes();
extern void init_ipadapter_loader_nodes();
extern void init_onnx_loader_nodes();

void init_loader_nodes() {
    init_model_loader_nodes();
    init_lora_loader_nodes();
    init_upscale_loader_nodes();
    init_ipadapter_loader_nodes();
    init_onnx_loader_nodes();
}

} // namespace sdengine
