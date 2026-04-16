// ============================================================================
// sd-engine/nodes/empty_latent_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// EmptyLatentImage - 创建空 Latent
// ============================================================================
class EmptyLatentImageNode : public Node {
  public:
    std::string get_class_type() const override {
        return "EmptyLatentImage";
    }
    std::string get_category() const override {
        return "latent";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"width", "INT", false, 512}, {"height", "INT", false, 512}, {"batch_size", "INT", false, 1}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int width;
        SD_RETURN_IF_ERROR(get_input(inputs, "width", width));
        int height;
        SD_RETURN_IF_ERROR(get_input(inputs, "height", height));

        sd_latent_t* latent = sd_create_empty_latent(nullptr, width, height);
        if (!latent) {
            LOG_ERROR("[ERROR] EmptyLatentImage: Failed to create latent\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["LATENT"] = make_latent_ptr(latent);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("EmptyLatentImage", EmptyLatentImageNode);

void init_empty_latent_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
