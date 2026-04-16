// ============================================================================
// sd-engine/nodes/upscale_loader_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// UpscaleModelLoader - 加载 ESRGAN 放大模型
// ============================================================================
class UpscaleModelLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "UpscaleModelLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_name", "STRING", true, std::string("")},
                {"use_gpu", "BOOLEAN", false, true},
                {"tile_size", "INT", false, 512}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"UPSCALE_MODEL", "UPSCALE_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string model_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "model_name", model_path));
        bool use_gpu = get_input_opt<bool>(inputs, "use_gpu", true);
        int tile_size = get_input_opt<int>(inputs, "tile_size", 512);

        if (model_path.empty()) {
            LOG_ERROR("[ERROR] UpscaleModelLoader: model_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[UpscaleModelLoader] Loading model: %s\n", model_path.c_str());

        upscaler_ctx_t* upscaler = new_upscaler_ctx(model_path.c_str(),
                                                    !use_gpu, // w_mode
                                                    false,    // no longer used
                                                    4,        // threads
                                                    tile_size);

        if (!upscaler) {
            LOG_ERROR("[ERROR] UpscaleModelLoader: Failed to load model\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        int scale = get_upscale_factor(upscaler);
        LOG_INFO("[UpscaleModelLoader] Model loaded, scale=%dx, tile_size=%d\n", scale, tile_size);

        outputs["UPSCALE_MODEL"] = make_upscaler_ptr(upscaler);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("UpscaleModelLoader", UpscaleModelLoaderNode);

void init_upscale_loader_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
