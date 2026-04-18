// ============================================================================
// sd-engine/nodes/animatediff_nodes.cpp
// ============================================================================
// AnimateDiff 节点实现（Loader / Sampler）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// AnimateDiffLoader - 加载 AnimateDiff 运动模块配置
// ============================================================================
class AnimateDiffLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "AnimateDiffLoader";
    }
    std::string get_category() const override {
        return "animatediff";
    }

    bool is_placeholder() const override {
        return true;
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")},
                {"motion_module", "STRING", true, std::string("")},
                {"frame_number", "INT", false, 16}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"ANIMATEDIFF_MODEL", "ANIMATEDIFF_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string model_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "model_path", model_path));
        std::string motion_module = get_input_opt<std::string>(inputs, "motion_module", "");
        int frame_number = get_input_opt<int>(inputs, "frame_number", 16);

        if (model_path.empty()) {
            LOG_ERROR("[ERROR] AnimateDiffLoader: model_path is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_ERROR("[ERROR] AnimateDiffLoader: Real AnimateDiff motion module loading is not yet implemented. "
                  "This node is a placeholder. AnimateDiff video generation is not available.\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("AnimateDiffLoader", AnimateDiffLoaderNode);

// ============================================================================
// AnimateDiffSampler - AnimateDiff 采样器（生成多帧 latent）
// ============================================================================
class AnimateDiffSamplerNode : public Node {
  public:
    std::string get_class_type() const override {
        return "AnimateDiffSampler";
    }
    std::string get_category() const override {
        return "animatediff";
    }

    bool is_placeholder() const override {
        return true;
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model", "MODEL", true, nullptr},
                {"seed", "INT", false, 0},
                {"steps", "INT", false, 20},
                {"cfg", "FLOAT", false, 8.0f},
                {"sampler_name", "STRING", false, std::string("euler")},
                {"scheduler", "STRING", false, std::string("normal")},
                {"positive", "CONDITIONING", true, nullptr},
                {"negative", "CONDITIONING", true, nullptr},
                {"latent_image", "LATENT", true, nullptr},
                {"denoise", "FLOAT", false, 1.0f},
                {"frame_number", "INT", false, 16}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        int64_t seed = get_input_opt<int>(inputs, "seed", 0);
        int steps = get_input_opt<int>(inputs, "steps", 20);
        float cfg = get_input_opt<float>(inputs, "cfg", 8.0f);
        std::string sampler_name = get_input_opt<std::string>(inputs, "sampler_name", "euler");
        std::string scheduler_name = get_input_opt<std::string>(inputs, "scheduler", "normal");
        int frame_number = get_input_opt<int>(inputs, "frame_number", 16);
        (void)frame_number;

        if (!sd_ctx) {
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_node_sample_params_t sample_params = {};
        sample_params.seed = seed;
        sample_params.steps = steps;
        sample_params.cfg_scale = cfg;
        sample_params.sample_method = str_to_sample_method(sampler_name.c_str());
        sample_params.scheduler = str_to_scheduler(scheduler_name.c_str());
        sample_params.eta = 0.0f;
        sample_params.add_noise = true;

        if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        }
        if (sample_params.scheduler == SCHEDULER_COUNT) {
            sample_params.scheduler = DISCRETE_SCHEDULER;
        }

        LOG_ERROR("[ERROR] AnimateDiffSampler: Real AnimateDiff video generation is not yet implemented. "
                  "This node is a placeholder. Only standard single-frame sampling is available via KSampler.\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("AnimateDiffSampler", AnimateDiffSamplerNode);

void init_animatediff_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
