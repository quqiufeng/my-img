// ============================================================================
// sd-engine/nodes/clip_nodes.cpp
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
        auto it = inputs.find("clip");
        if (it != inputs.end() && it->second.type() == typeid(SDContextPtr))
            wrapper.sd_ctx_ptr = std::any_cast<SDContextPtr>(it->second);
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
        SD_RETURN_IF_ERROR(get_input(inputs, "image", image));

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
        SD_RETURN_IF_ERROR(get_input(inputs, "text", text));
        int clip_skip = get_input_opt<int>(inputs, "clip_skip", -1);

        sd_ctx_t* sd_ctx = nullptr;
        auto it = inputs.find("clip");
        if (it != inputs.end()) {
            const auto& clip_val = it->second;
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

void init_clip_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
