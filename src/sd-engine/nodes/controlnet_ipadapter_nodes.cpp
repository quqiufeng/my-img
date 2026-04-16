// ============================================================================
// sd-engine/nodes/controlnet_ipadapter_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// ControlNetApply - 应用 ControlNet 条件
// ============================================================================
class ControlNetApplyNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ControlNetApply";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning", "CONDITIONING", true, nullptr},
                {"control_net", "CONTROL_NET", true, nullptr},
                {"image", "IMAGE", true, nullptr},
                {"strength", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning", cond));
        ImagePtr image;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", image));
        float strength = get_input_opt<float>(inputs, "strength", 1.0f);

        if (!cond) {
            LOG_ERROR("[ERROR] ControlNetApply: Missing conditioning\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[ControlNetApply] Applying ControlNet with strength=%.2f, image=%dx%d\n", strength,
                 image ? image->width : 0, image ? image->height : 0);

        outputs["CONDITIONING"] = cond;
        outputs["_control_image"] = image;
        outputs["_control_strength"] = strength;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ControlNetApply", ControlNetApplyNode);

// ============================================================================
// IPAdapterApply - 应用 IPAdapter 到 conditioning
// ============================================================================
class IPAdapterApplyNode : public Node {
  public:
    std::string get_class_type() const override {
        return "IPAdapterApply";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning", "CONDITIONING", true, nullptr},
                {"ipadapter", "IPADAPTER", true, nullptr},
                {"image", "IMAGE", true, nullptr},
                {"strength", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}, {"IPADAPTER", "IPADAPTER"}, {"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning", cond));
        IPAdapterInfo info;
        SD_RETURN_IF_ERROR(get_input(inputs, "ipadapter", info));
        ImagePtr image;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", image));
        float strength = get_input_opt<float>(inputs, "strength", 1.0f);

        if (!cond) {
            LOG_ERROR("[ERROR] IPAdapterApply: Missing conditioning\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        info.strength = strength;
        LOG_INFO("[IPAdapterApply] Applying IPAdapter strength=%.2f, image=%dx%d\n", strength, image ? image->width : 0,
                 image ? image->height : 0);

        outputs["CONDITIONING"] = cond;
        outputs["IPADAPTER"] = info;
        outputs["IMAGE"] = image;
        outputs["_ipadapter_info"] = info;
        outputs["_ipadapter_image"] = image;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("IPAdapterApply", IPAdapterApplyNode);

void init_controlnet_ipadapter_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
