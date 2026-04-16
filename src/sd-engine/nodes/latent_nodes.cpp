// ============================================================================
// sd-engine/nodes/latent_nodes.cpp
// ============================================================================
// Latent Nodes 节点入口（薄封装，实际实现分散在子模块中）
// ============================================================================

namespace sdengine {

extern void init_empty_latent_nodes();
extern void init_vae_nodes();
extern void init_sampler_nodes();
extern void init_deep_hires_nodes();

void init_latent_nodes() {
    init_empty_latent_nodes();
    init_vae_nodes();
    init_sampler_nodes();
    init_deep_hires_nodes();
}

} // namespace sdengine
