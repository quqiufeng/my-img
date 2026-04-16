// ============================================================================
// sd-engine/nodes/sampler_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// KSampler - 真正的采样器（调用分离式 API）
// ============================================================================
class KSamplerNode : public Node {
  public:
    std::string get_class_type() const override {
        return "KSampler";
    }
    std::string get_category() const override {
        return "sampling";
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
                {"lora_stack", "LORA_STACK", false, nullptr},
                {"control_image", "IMAGE", false, nullptr},
                {"control_strength", "FLOAT", false, 1.0f},
                {"mask", "MASK", false, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        int64_t seed = get_input_opt<int>(inputs, "seed", 0);
        int steps = get_input_opt<int>(inputs, "steps", 20);
        float cfg = get_input_opt<float>(inputs, "cfg", 8.0f);
        std::string sampler_name =
            get_input_opt<std::string>(inputs, "sampler_name", "euler");
        std::string scheduler_name =
            get_input_opt<std::string>(inputs, "scheduler", "normal");

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

        LOG_INFO("[KSampler] Running sampler: steps=%d, seed=%ld, cfg=%.2f\n", steps, (long)seed, cfg);

        sd_latent_t* result = nullptr;
        sd_error_t err = run_sampler_common(sd_ctx, inputs, sample_params, &result);
        if (is_error(err))
            return err;

        outputs["LATENT"] = make_latent_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("KSampler", KSamplerNode);

// ============================================================================
// KSamplerAdvanced - 高级采样器
// ============================================================================
class KSamplerAdvancedNode : public Node {
  public:
    std::string get_class_type() const override {
        return "KSamplerAdvanced";
    }
    std::string get_category() const override {
        return "sampling";
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
                {"start_at_step", "INT", false, 0},
                {"end_at_step", "INT", false, 10000},
                {"add_noise", "BOOLEAN", false, true},
                {"lora_stack", "LORA_STACK", false, nullptr},
                {"control_image", "IMAGE", false, nullptr},
                {"control_strength", "FLOAT", false, 1.0f},
                {"mask", "MASK", false, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        int64_t seed = get_input_opt<int>(inputs, "seed", 0);
        int steps = get_input_opt<int>(inputs, "steps", 20);
        float cfg = get_input_opt<float>(inputs, "cfg", 8.0f);
        std::string sampler_name =
            get_input_opt<std::string>(inputs, "sampler_name", "euler");
        std::string scheduler_name =
            get_input_opt<std::string>(inputs, "scheduler", "normal");
        int start_at_step = get_input_opt<int>(inputs, "start_at_step", 0);
        int end_at_step = get_input_opt<int>(inputs, "end_at_step", 10000);
        bool add_noise = get_input_opt<bool>(inputs, "add_noise", true);

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
        sample_params.start_at_step = start_at_step;
        sample_params.end_at_step = end_at_step;
        sample_params.add_noise = add_noise;

        if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        }
        if (sample_params.scheduler == SCHEDULER_COUNT) {
            sample_params.scheduler = DISCRETE_SCHEDULER;
        }

        LOG_INFO("[KSamplerAdvanced] Running sampler: steps=%d, seed=%ld, cfg=%.2f, start=%d, end=%d, add_noise=%s\n",
                 steps, (long)seed, cfg, start_at_step, end_at_step, add_noise ? "true" : "false");

        sd_latent_t* result = nullptr;
        sd_error_t err = run_sampler_common(sd_ctx, inputs, sample_params, &result);
        if (is_error(err))
            return err;

        outputs["LATENT"] = make_latent_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("KSamplerAdvanced", KSamplerAdvancedNode);

void init_sampler_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
