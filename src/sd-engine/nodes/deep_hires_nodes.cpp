// ============================================================================
// sd-engine/nodes/deep_hires_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// DeepHighResFix - 原生 Deep HighRes Fix 节点
// ============================================================================
class DeepHighResFixNode : public Node {
  public:
    std::string get_class_type() const override {
        return "DeepHighResFix";
    }
    std::string get_category() const override {
        return "sampling";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model", "MODEL", true, nullptr},
                {"positive", "CONDITIONING", true, nullptr},
                {"negative", "CONDITIONING", true, nullptr},
                {"positive_text", "STRING", false, std::string("")},
                {"negative_text", "STRING", false, std::string("")},
                {"init_image", "IMAGE", false, nullptr},
                {"seed", "INT", false, 0},
                {"steps", "INT", false, 30},
                {"cfg", "FLOAT", false, 7.0f},
                {"target_width", "INT", false, 1024},
                {"target_height", "INT", false, 1024},
                {"strength", "FLOAT", false, 1.0f},
                {"phase1_cfg", "FLOAT", false, 0.0f},
                {"phase2_cfg", "FLOAT", false, 0.0f},
                {"phase3_cfg", "FLOAT", false, 0.0f},
                {"vae_tiling", "BOOLEAN", false, false}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        ConditioningPtr positive;
        SD_RETURN_IF_ERROR(get_input(inputs, "positive", positive));
        ConditioningPtr negative =
            get_input_opt<ConditioningPtr>(inputs, "negative", nullptr);

        int64_t seed = get_input_opt<int>(inputs, "seed", 0);
        int total_steps = get_input_opt<int>(inputs, "steps", 30);
        float cfg = get_input_opt<float>(inputs, "cfg", 7.0f);
        int target_width = get_input_opt<int>(inputs, "target_width", 1024);
        int target_height = get_input_opt<int>(inputs, "target_height", 1024);
        float strength = get_input_opt<float>(inputs, "strength", 1.0f);
        float phase1_cfg = get_input_opt<float>(inputs, "phase1_cfg", 0.0f);
        float phase2_cfg = get_input_opt<float>(inputs, "phase2_cfg", 0.0f);
        float phase3_cfg = get_input_opt<float>(inputs, "phase3_cfg", 0.0f);
        bool vae_tiling = get_input_opt<bool>(inputs, "vae_tiling", false);

        if (!sd_ctx || !positive) {
            LOG_ERROR("[ERROR] DeepHighResFix: Missing required inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int target_w = (target_width + 63) & ~63;
        int target_h = (target_height + 63) & ~63;

        int phase1_steps = std::max(6, total_steps / 4);
        int phase3_steps = std::max(8, total_steps * 3 / 4);
        int phase2_steps = std::max(4, total_steps - phase1_steps - phase3_steps);

        int phase1_w = std::min(512, target_w / 2);
        int phase1_h = std::min(512, target_h / 2);
        phase1_w = (phase1_w + 63) & ~63;
        phase1_h = (phase1_h + 63) & ~63;

        int phase2_w = target_w * 3 / 4;
        int phase2_h = target_h * 3 / 4;
        phase2_w = (phase2_w + 63) & ~63;
        phase2_h = (phase2_h + 63) & ~63;

        LOG_INFO("[DeepHighResFix] Target: %dx%d, Phases: %dx%d(%d) -> %dx%d(%d) -> %dx%d(%d)\n", target_w, target_h,
                 phase1_w, phase1_h, phase1_steps, phase2_w, phase2_h, phase2_steps, target_w, target_h, phase3_steps);

        DeepHiresNodeState state = {};
        state.phase1_steps = phase1_steps;
        state.phase2_steps = phase2_steps;
        state.phase1_w = phase1_w;
        state.phase1_h = phase1_h;
        state.phase2_w = phase2_w;
        state.phase2_h = phase2_h;
        state.target_w = target_w;
        state.target_h = target_h;
        state.phase1_cfg_scale = phase1_cfg > 0 ? phase1_cfg : cfg;
        state.phase2_cfg_scale = phase2_cfg > 0 ? phase2_cfg : cfg;
        state.phase3_cfg_scale = phase3_cfg > 0 ? phase3_cfg : cfg;

        sd_set_latent_hook(deep_hires_node_latent_hook, &state);
        sd_set_guidance_hook(deep_hires_node_guidance_hook, &state);

        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);

        std::string positive_text =
            get_input_opt<std::string>(inputs, "positive_text", "");
        std::string negative_text =
            get_input_opt<std::string>(inputs, "negative_text", "");
        gen_params.prompt = positive_text.c_str();
        gen_params.negative_prompt = negative_text.c_str();
        gen_params.width = phase1_w;
        gen_params.height = phase1_h;
        gen_params.strength = strength;
        gen_params.seed = seed;
        gen_params.sample_params.sample_steps = total_steps;
        gen_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        gen_params.sample_params.scheduler = KARRAS_SCHEDULER;
        gen_params.sample_params.guidance.txt_cfg = cfg;

        if (inputs.count("init_image")) {
            ImagePtr init_img;
            SD_RETURN_IF_ERROR(get_input(inputs, "init_image", init_img));
            if (init_img && init_img->data) {
                gen_params.init_image = *init_img;
            }
        }

        if (vae_tiling) {
            gen_params.vae_tiling_params.enabled = true;
            gen_params.vae_tiling_params.tile_size_x = 512;
            gen_params.vae_tiling_params.tile_size_y = 512;
            gen_params.vae_tiling_params.target_overlap = 64;
        }

        sd_image_t* result = generate_image(sd_ctx, &gen_params);

        sd_clear_latent_hook();
        sd_clear_guidance_hook();

        if (!result || !result->data) {
            LOG_ERROR("[ERROR] DeepHighResFix: Generation failed\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        LOG_INFO("[DeepHighResFix] Generation completed: %dx%d\n", result->width, result->height);
        outputs["IMAGE"] = make_image_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("DeepHighResFix", DeepHighResFixNode);

void init_deep_hires_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
