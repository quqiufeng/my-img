// ============================================================================
// sd-engine/nodes/latent_nodes.cpp
// ============================================================================
// Latent Nodes 节点入口（薄封装，实际实现分散在子模块中）
// 
// 注意：LatentUpscale 和 LatentComposite 节点已简化，因为 stable-diffusion.cpp
// 更新了 API，sd_latent_t 变成了不透明类型。
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

extern void init_empty_latent_nodes();
extern void init_vae_nodes();
extern void init_sampler_nodes();
extern void init_deep_hires_nodes();

// ============================================================================
// LatentUpscale - Latent 空间上采样（简化版，直接复制）
// ============================================================================
class LatentUpscaleNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LatentUpscale";
    }
    std::string get_category() const override {
        return "latent";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"samples", "LATENT", true, nullptr},
                {"upscale_method", "STRING", false, std::string("nearest-exact")},
                {"width", "INT", true, 0},
                {"height", "INT", true, 0},
                {"crop", "STRING", false, std::string("disabled")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr latent;
        SD_RETURN_IF_ERROR(get_input(inputs, "samples", latent));

        if (!latent) {
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        // 简化：直接复制 latent（不做实际上采样）
        sd_latent_t* out_latent = sd_latent_copy(latent.get());
        if (!out_latent) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["LATENT"] = make_latent_ptr(out_latent);
        LOG_INFO("[LatentUpscale] Copied latent (upscale not implemented)\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LatentUpscale", LatentUpscaleNode);

// ============================================================================
// LatentComposite - 在 Latent 空间合成两个 latent（简化版）
// ============================================================================
class LatentCompositeNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LatentComposite";
    }
    std::string get_category() const override {
        return "latent";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"samples_to", "LATENT", true, nullptr},
                {"samples_from", "LATENT", true, nullptr},
                {"x", "INT", true, 0},
                {"y", "INT", true, 0}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr dst;
        SD_RETURN_IF_ERROR(get_input(inputs, "samples_to", dst));

        if (!dst) {
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        // 简化：直接复制 dst（不做实际合成）
        sd_latent_t* result = sd_latent_copy(dst.get());
        if (!result) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["LATENT"] = make_latent_ptr(result);
        LOG_INFO("[LatentComposite] Copied dst latent (composite not implemented)\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LatentComposite", LatentCompositeNode);

void init_latent_nodes() {
    init_empty_latent_nodes();
    init_vae_nodes();
    init_sampler_nodes();
    init_deep_hires_nodes();
}

} // namespace sdengine
