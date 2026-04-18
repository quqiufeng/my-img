// ============================================================================
// sd-engine/nodes/inpaint_nodes.cpp
// ============================================================================
// Inpaint 节点实现（LoadInpaintModel / ApplyInpaint）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// INPAINT_LoadInpaintModel - 加载 Inpaint 模型配置
// ============================================================================
class InpaintLoadInpaintModelNode : public Node {
  public:
    std::string get_class_type() const override {
        return "INPAINT_LoadInpaintModel";
    }
    std::string get_category() const override {
        return "inpaint";
    }

    bool is_placeholder() const override {
        return true;
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")}, {"model_type", "STRING", false, std::string("sd1.5")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"INPAINT_MODEL", "INPAINT_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string model_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "model_path", model_path));
        std::string model_type = get_input_opt<std::string>(inputs, "model_type", "sd1.5");

        if (model_path.empty()) {
            LOG_ERROR("[ERROR] INPAINT_LoadInpaintModel: model_path is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_ERROR("[ERROR] INPAINT_LoadInpaintModel: Real inpaint model loading is not yet implemented. "
                  "This node is a placeholder. Please use standard CheckpointLoaderSimple with an inpaint model.\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("INPAINT_LoadInpaintModel", InpaintLoadInpaintModelNode);

// ============================================================================
// INPAINT_ApplyInpaint - 应用 Inpaint 到 latent（传递 mask 给 KSampler）
// ============================================================================
class InpaintApplyInpaintNode : public Node {
  public:
    std::string get_class_type() const override {
        return "INPAINT_ApplyInpaint";
    }
    std::string get_category() const override {
        return "inpaint";
    }

    bool is_placeholder() const override {
        return true;
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"latent", "LATENT", true, nullptr},
                {"mask", "MASK", true, nullptr},
                {"inpaint_model", "INPAINT_MODEL", false, std::string("")},
                {"strength", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}, {"MASK", "MASK"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr latent;
        SD_RETURN_IF_ERROR(get_input(inputs, "latent", latent));
        ImagePtr mask;
        SD_RETURN_IF_ERROR(get_input(inputs, "mask", mask));
        float strength = get_input_opt<float>(inputs, "strength", 1.0f);
        (void)strength;

        if (!latent) {
            LOG_ERROR("[ERROR] INPAINT_ApplyInpaint: No latent data\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        if (!mask || !mask->data) {
            LOG_ERROR("[ERROR] INPAINT_ApplyInpaint: No mask data\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_ERROR("[ERROR] INPAINT_ApplyInpaint: Real inpaint masking is not yet implemented. "
                  "This node is a placeholder. Mask will not be applied correctly during sampling.\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("INPAINT_ApplyInpaint", InpaintApplyInpaintNode);

void init_inpaint_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
