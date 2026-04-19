// ============================================================================
// sd-engine/nodes/deep_hires_nodes.cpp
// ============================================================================
// Deep HighRes Fix 节点（简化版）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// DeepHighResFix - 原生 Deep HighRes Fix 节点（简化版）
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

        if (!sd_ctx) {
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int seed = get_input_opt<int>(inputs, "seed", 0);
        int steps = get_input_opt<int>(inputs, "steps", 30);
        float cfg = get_input_opt<float>(inputs, "cfg", 7.0f);
        int target_width = get_input_opt<int>(inputs, "target_width", 1024);
        int target_height = get_input_opt<int>(inputs, "target_height", 1024);
        float strength = get_input_opt<float>(inputs, "strength", 1.0f);
        bool vae_tiling = get_input_opt<bool>(inputs, "vae_tiling", false);

        std::string positive_text =
            get_input_opt<std::string>(inputs, "positive_text", "");
        std::string negative_text =
            get_input_opt<std::string>(inputs, "negative_text", "");

        // 使用 generate_image 直接生成（不使用 hook）
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        gen_params.prompt = positive_text.c_str();
        gen_params.negative_prompt = negative_text.c_str();
        gen_params.width = target_width;
        gen_params.height = target_height;
        gen_params.strength = strength;
        gen_params.seed = seed;
        gen_params.sample_params.sample_steps = steps;
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

        if (!result || !result->data) {
            LOG_ERROR("[DeepHighResFix] generate_image failed\n");
            return sd_error_t::ERROR_SAMPLING_FAILED;
        }

        outputs["IMAGE"] = make_image_ptr(result);
        LOG_INFO("[DeepHighResFix] Generated %dx%d image\n", result->width, result->height);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("DeepHighResFix", DeepHighResFixNode);

void init_deep_hires_nodes() {
    // 节点已通过 REGISTER_NODE 宏自动注册
}

} // namespace sdengine
