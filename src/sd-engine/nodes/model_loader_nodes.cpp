// ============================================================================
// sd-engine/nodes/model_loader_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// CheckpointLoaderSimple - 加载模型
// ============================================================================
class CheckpointLoaderSimpleNode : public Node {
  public:
    std::string get_class_type() const override {
        return "CheckpointLoaderSimple";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"ckpt_name", "STRING", true, std::string("")},
                {"vae_name", "STRING", false, std::string("")},
                {"clip_name", "STRING", false, std::string("")},
                {"control_net_path", "STRING", false, std::string("")},
                {"n_threads", "INT", false, 4},
                {"use_gpu", "BOOLEAN", false, true},
                {"flash_attn", "BOOLEAN", false, false}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"MODEL", "MODEL"}, {"CLIP", "CLIP"}, {"VAE", "VAE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string ckpt_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "ckpt_name", ckpt_path));
        std::string vae_path = get_input_opt<std::string>(inputs, "vae_name", "");
        std::string clip_path = get_input_opt<std::string>(inputs, "clip_name", "");
        std::string control_net_path = get_input_opt<std::string>(inputs, "control_net_path", "");
        int n_threads = get_input_opt<int>(inputs, "n_threads", 4);
        bool use_gpu = get_input_opt<bool>(inputs, "use_gpu", true);
        bool flash_attn = get_input_opt<bool>(inputs, "flash_attn", false);

        if (ckpt_path.empty()) {
            LOG_ERROR("[ERROR] CheckpointLoaderSimple: ckpt_name is required\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        LOG_INFO("[CheckpointLoaderSimple] Loading model: %s\n", ckpt_path.c_str());

        sd_ctx_params_t ctx_params;
        sd_ctx_params_init(&ctx_params);
        ctx_params.diffusion_model_path = ckpt_path.c_str();
        if (!vae_path.empty()) {
            ctx_params.vae_path = vae_path.c_str();
        }
        if (!clip_path.empty()) {
            ctx_params.llm_path = clip_path.c_str();
        }
        if (!control_net_path.empty()) {
            ctx_params.control_net_path = control_net_path.c_str();
            ctx_params.keep_control_net_on_cpu = !use_gpu;
            LOG_INFO("[CheckpointLoaderSimple] Loading ControlNet: %s\n", control_net_path.c_str());
        }
        ctx_params.n_threads = n_threads;
        ctx_params.offload_params_to_cpu = !use_gpu;
        ctx_params.keep_vae_on_cpu = !use_gpu;
        ctx_params.keep_clip_on_cpu = !use_gpu;
        ctx_params.flash_attn = use_gpu && flash_attn;
        ctx_params.diffusion_flash_attn = use_gpu && flash_attn;
        ctx_params.vae_decode_only = false;

        sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
        if (!sd_ctx) {
            LOG_ERROR("[ERROR] Failed to load checkpoint\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        LOG_INFO("[CheckpointLoaderSimple] Model loaded successfully\n");

        auto sd_ctx_ptr = make_sd_context_ptr(sd_ctx);
        outputs["MODEL"] = sd_ctx_ptr;
        outputs["CLIP"] = sd_ctx_ptr;
        outputs["VAE"] = sd_ctx_ptr;

        return sd_error_t::OK;
    }
};
REGISTER_NODE("CheckpointLoaderSimple", CheckpointLoaderSimpleNode);

// ============================================================================
// UnloadModel - 释放模型上下文
// ============================================================================
class UnloadModelNode : public Node {
  public:
    std::string get_class_type() const override {
        return "UnloadModel";
    }
    std::string get_category() const override {
        return "model_management";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model", "MODEL", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        auto it = inputs.find("model");
        if (it != inputs.end()) {
            const auto& model_val = it->second;
            if (model_val.type() == typeid(SDContextPtr)) {
                auto ctx_ptr = std::any_cast<SDContextPtr>(model_val);
                if (ctx_ptr) {
                    LOG_INFO("[UnloadModel] Releasing model context (ref_count=%ld)\n", ctx_ptr.use_count());
                }
            } else if (model_val.type() == typeid(sd_ctx_t*)) {
                sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(model_val);
                if (sd_ctx) {
                    LOG_INFO("[UnloadModel] Releasing raw model context\n");
                    free_sd_ctx(sd_ctx);
                }
            }
        }
        return sd_error_t::OK;
    }
};
REGISTER_NODE("UnloadModel", UnloadModelNode);

void init_model_loader_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
