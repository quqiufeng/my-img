// ============================================================================
// sd-engine/nodes/conditioning_nodes.cpp
// ============================================================================
// Conditioning / CLIP 相关节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// CLIPSetLastLayer - 设置 CLIP 跳过层
// ============================================================================
class CLIPSetLastLayerNode : public Node {
  public:
    std::string get_class_type() const override {
        return "CLIPSetLastLayer";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"clip", "CLIP", true, nullptr}, {"stop_at_clip_layer", "INT", false, -1}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CLIP", "CLIP"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "clip");
        int clip_skip = get_input_opt<int>(inputs, "stop_at_clip_layer", -1);

        if (!sd_ctx) {
            LOG_ERROR("[ERROR] CLIPSetLastLayer: Missing CLIP\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        CLIPWrapper wrapper;
        wrapper.sd_ctx = sd_ctx;
        const auto& clip_val = inputs.at("clip");
        if (clip_val.type() == typeid(SDContextPtr))
            wrapper.sd_ctx_ptr = std::any_cast<SDContextPtr>(clip_val);
        wrapper.clip_skip = clip_skip;

        outputs["CLIP"] = wrapper;
        LOG_INFO("[CLIPSetLastLayer] clip_skip set to %d\n", clip_skip);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CLIPSetLastLayer", CLIPSetLastLayerNode);

// ============================================================================
// CLIPVisionEncode - CLIP Vision 图像编码
// ============================================================================
class CLIPVisionEncodeNode : public Node {
  public:
    std::string get_class_type() const override {
        return "CLIPVisionEncode";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"clip", "CLIP", true, nullptr}, {"image", "IMAGE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "clip");
        ImagePtr image;
        if (sd_error_t err = get_input(inputs, "image", image); is_error(err)) {
            return err;
        }

        if (!sd_ctx || !image || !image->data) {
            LOG_ERROR("[ERROR] CLIPVisionEncode: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_clip_vision_output_t* output = sd_clip_vision_encode_image(sd_ctx, image.get(), true);
        if (!output) {
            LOG_ERROR("[ERROR] CLIPVisionEncode: Failed to encode image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CLIP_VISION_OUTPUT"] = make_clip_vision_output_ptr(output);
        LOG_INFO("[CLIPVisionEncode] Encoded image to CLIP Vision output (numel=%d)\n", output->numel);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CLIPVisionEncode", CLIPVisionEncodeNode);

// ============================================================================
// CLIPTextEncode - 真正的文本编码
// ============================================================================
class CLIPTextEncodeNode : public Node {
  public:
    std::string get_class_type() const override {
        return "CLIPTextEncode";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"text", "STRING", true, std::string("")},
                {"clip", "CLIP", true, nullptr},
                {"clip_skip", "INT", false, -1}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}, {"text", "STRING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string text;
        if (sd_error_t err = get_input(inputs, "text", text); is_error(err)) {
            return err;
        }
        int clip_skip = get_input_opt<int>(inputs, "clip_skip", -1);

        sd_ctx_t* sd_ctx = nullptr;
        const auto& clip_val = inputs.at("clip");
        if (clip_val.type() == typeid(CLIPWrapper)) {
            const auto& wrapper = std::any_cast<CLIPWrapper>(clip_val);
            sd_ctx = wrapper.sd_ctx;
            if (clip_skip == -1)
                clip_skip = wrapper.clip_skip;
        } else if (clip_val.type() == typeid(SDContextPtr)) {
            sd_ctx = std::any_cast<SDContextPtr>(clip_val).get();
        } else if (clip_val.type() == typeid(sd_ctx_t*)) {
            sd_ctx = std::any_cast<sd_ctx_t*>(clip_val);
        }

        if (!sd_ctx) {
            LOG_ERROR("[ERROR] CLIPTextEncode: Missing CLIP\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* cond = sd_encode_prompt(sd_ctx, text.c_str(), clip_skip);
        if (!cond) {
            LOG_ERROR("[ERROR] CLIPTextEncode: Failed to encode prompt\n");
            return sd_error_t::ERROR_ENCODING_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(cond);
        outputs["text"] = text;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CLIPTextEncode", CLIPTextEncodeNode);

// ============================================================================
// ConditioningCombine - 条件合并
// ============================================================================
class ConditioningCombineNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ConditioningCombine";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning_1", "CONDITIONING", true, nullptr}, {"conditioning_2", "CONDITIONING", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond1;
        if (sd_error_t err = get_input(inputs, "conditioning_1", cond1); is_error(err)) {
            return err;
        }
        ConditioningPtr cond2;
        if (sd_error_t err = get_input(inputs, "conditioning_2", cond2); is_error(err)) {
            return err;
        }

        if (!cond1 || !cond2) {
            LOG_ERROR("[ERROR] ConditioningCombine: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* combined = sd_conditioning_concat(cond1.get(), cond2.get());
        if (!combined) {
            LOG_ERROR("[ERROR] ConditioningCombine: Failed to combine conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(combined);
        LOG_INFO("[ConditioningCombine] Combined two conditionings\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningCombine", ConditioningCombineNode);

// ============================================================================
// ConditioningConcat - 条件拼接（与 Combine 行为相同，对齐 ComfyUI 命名）
// ============================================================================
class ConditioningConcatNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ConditioningConcat";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning_to", "CONDITIONING", true, nullptr},
                {"conditioning_from", "CONDITIONING", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond_to;
        if (sd_error_t err = get_input(inputs, "conditioning_to", cond_to); is_error(err)) {
            return err;
        }
        ConditioningPtr cond_from;
        if (sd_error_t err = get_input(inputs, "conditioning_from", cond_from); is_error(err)) {
            return err;
        }

        if (!cond_to || !cond_from) {
            LOG_ERROR("[ERROR] ConditioningConcat: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* concat = sd_conditioning_concat(cond_to.get(), cond_from.get());
        if (!concat) {
            LOG_ERROR("[ERROR] ConditioningConcat: Failed to concat conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(concat);
        LOG_INFO("[ConditioningConcat] Concatenated two conditionings\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningConcat", ConditioningConcatNode);

// ============================================================================
// ConditioningAverage - 条件加权平均
// ============================================================================
class ConditioningAverageNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ConditioningAverage";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning_to", "CONDITIONING", true, nullptr},
                {"conditioning_from", "CONDITIONING", true, nullptr},
                {"conditioning_to_strength", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond_to;
        if (sd_error_t err = get_input(inputs, "conditioning_to", cond_to); is_error(err)) {
            return err;
        }
        ConditioningPtr cond_from;
        if (sd_error_t err = get_input(inputs, "conditioning_from", cond_from); is_error(err)) {
            return err;
        }
        float strength = get_input_opt<float>(inputs, "conditioning_to_strength", 1.0f);

        if (!cond_to || !cond_from) {
            LOG_ERROR("[ERROR] ConditioningAverage: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* averaged = sd_conditioning_average(cond_to.get(), cond_from.get(), strength);
        if (!averaged) {
            LOG_ERROR("[ERROR] ConditioningAverage: Failed to average conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(averaged);
        LOG_INFO("[ConditioningAverage] Averaged conditionings (strength=%.2f)\n", strength);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningAverage", ConditioningAverageNode);

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
        if (sd_error_t err = get_input(inputs, "conditioning", cond); is_error(err)) {
            return err;
        }
        ImagePtr image;
        if (sd_error_t err = get_input(inputs, "image", image); is_error(err)) {
            return err;
        }
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
        if (sd_error_t err = get_input(inputs, "conditioning", cond); is_error(err)) {
            return err;
        }
        IPAdapterInfo info;
        if (sd_error_t err = get_input(inputs, "ipadapter", info); is_error(err)) {
            return err;
        }
        ImagePtr image;
        if (sd_error_t err = get_input(inputs, "image", image); is_error(err)) {
            return err;
        }
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

void init_conditioning_nodes() {
    // 空函数，仅确保本翻译单元被链接
}

} // namespace sdengine
