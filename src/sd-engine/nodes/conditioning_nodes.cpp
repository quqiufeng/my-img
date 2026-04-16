// ============================================================================
// sd-engine/nodes/conditioning_nodes.cpp
// ============================================================================
// Conditioning Nodes 节点入口（薄封装，实际实现分散在子模块中）
// ============================================================================

namespace sdengine {

extern void init_clip_nodes();
extern void init_conditioning_combine_nodes();
extern void init_controlnet_ipadapter_nodes();

void init_conditioning_nodes() {
    init_clip_nodes();
    init_conditioning_combine_nodes();
    init_controlnet_ipadapter_nodes();
}

} // namespace sdengine
