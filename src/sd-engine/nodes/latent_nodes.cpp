// ============================================================================
// sd-engine/nodes/latent_nodes.cpp
// ============================================================================
// Latent / VAE / Sampler 相关节点实现
// ============================================================================

#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// EmptyLatentImage - 创建空 Latent
// ============================================================================
class EmptyLatentImageNode : public Node {
public:
    std::string get_class_type() const override { return "EmptyLatentImage"; }
    std::string get_category() const override { return "latent"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"width", "INT", false, 512},
            {"height", "INT", false, 512},
            {"batch_size", "INT", false, 1}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int width = std::any_cast<int>(inputs.at("width"));
        int height = std::any_cast<int>(inputs.at("height"));

        sd_latent_t* latent = sd_create_empty_latent(nullptr, width, height);
        if (!latent) {
            fprintf(stderr, "[ERROR] EmptyLatentImage: Failed to create latent\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["LATENT"] = make_latent_ptr(latent);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("EmptyLatentImage", EmptyLatentImageNode);

// ============================================================================
// VAEEncode - 真正的 VAE 编码
// ============================================================================
class VAEEncodeNode : public Node {
public:
    std::string get_class_type() const override { return "VAEEncode"; }
    std::string get_category() const override { return "latent"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"pixels", "IMAGE", true, nullptr},
            {"vae", "VAE", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_image_t image = std::any_cast<sd_image_t>(inputs.at("pixels"));
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "vae");

        if (!image.data) {
            fprintf(stderr, "[ERROR] VAEEncode: No image data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_latent_t* latent = sd_encode_image(sd_ctx, &image);
        if (!latent) {
            fprintf(stderr, "[ERROR] VAEEncode: Failed to encode image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("[VAEEncode] Image encoded to latent\n");
        outputs["LATENT"] = make_latent_ptr(latent);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("VAEEncode", VAEEncodeNode);

// ============================================================================
// VAEDecode - 真正的 VAE 解码
// ============================================================================
class VAEDecodeNode : public Node {
public:
    std::string get_class_type() const override { return "VAEDecode"; }
    std::string get_category() const override { return "latent"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"samples", "LATENT", true, nullptr},
            {"vae", "VAE", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr latent = std::any_cast<LatentPtr>(inputs.at("samples"));
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "vae");

        if (!latent) {
            fprintf(stderr, "[ERROR] VAEDecode: No latent data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_image_t* image = sd_decode_latent(sd_ctx, latent.get());
        if (!image) {
            fprintf(stderr, "[ERROR] VAEDecode: Failed to decode latent\n");
            return sd_error_t::ERROR_DECODING_FAILED;
        }

        printf("[VAEDecode] Latent decoded: %dx%d\n", image->width, image->height);
        outputs["IMAGE"] = make_image_ptr(image);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("VAEDecode", VAEDecodeNode);

// ============================================================================
// KSampler - 真正的采样器（调用分离式 API）
// ============================================================================
class KSamplerNode : public Node {
public:
    std::string get_class_type() const override { return "KSampler"; }
    std::string get_category() const override { return "sampling"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
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
            {"mask", "MASK", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        int64_t seed = inputs.count("seed") ? std::any_cast<int>(inputs.at("seed")) : 0;
        int steps = inputs.count("steps") ? std::any_cast<int>(inputs.at("steps")) : 20;
        float cfg = inputs.count("cfg") ? std::any_cast<float>(inputs.at("cfg")) : 8.0f;
        std::string sampler_name = inputs.count("sampler_name") ?
            std::any_cast<std::string>(inputs.at("sampler_name")) : "euler";
        std::string scheduler_name = inputs.count("scheduler") ?
            std::any_cast<std::string>(inputs.at("scheduler")) : "normal";

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

        printf("[KSampler] Running sampler: steps=%d, seed=%ld, cfg=%.2f\n",
               steps, (long)seed, cfg);

        sd_latent_t* result = nullptr;
        sd_error_t err = run_sampler_common(sd_ctx, inputs, sample_params, &result);
        if (is_error(err)) return err;

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
    std::string get_class_type() const override { return "KSamplerAdvanced"; }
    std::string get_category() const override { return "sampling"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
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
            {"mask", "MASK", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        int64_t seed = inputs.count("seed") ? std::any_cast<int>(inputs.at("seed")) : 0;
        int steps = inputs.count("steps") ? std::any_cast<int>(inputs.at("steps")) : 20;
        float cfg = inputs.count("cfg") ? std::any_cast<float>(inputs.at("cfg")) : 8.0f;
        std::string sampler_name = inputs.count("sampler_name") ?
            std::any_cast<std::string>(inputs.at("sampler_name")) : "euler";
        std::string scheduler_name = inputs.count("scheduler") ?
            std::any_cast<std::string>(inputs.at("scheduler")) : "normal";
        int start_at_step = inputs.count("start_at_step") ? std::any_cast<int>(inputs.at("start_at_step")) : 0;
        int end_at_step = inputs.count("end_at_step") ? std::any_cast<int>(inputs.at("end_at_step")) : 10000;
        bool add_noise = inputs.count("add_noise") ? std::any_cast<bool>(inputs.at("add_noise")) : true;

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

        printf("[KSamplerAdvanced] Running sampler: steps=%d, seed=%ld, cfg=%.2f, start=%d, end=%d, add_noise=%s\n",
               steps, (long)seed, cfg, start_at_step, end_at_step, add_noise ? "true" : "false");

        sd_latent_t* result = nullptr;
        sd_error_t err = run_sampler_common(sd_ctx, inputs, sample_params, &result);
        if (is_error(err)) return err;

        outputs["LATENT"] = make_latent_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("KSamplerAdvanced", KSamplerAdvancedNode);

// ============================================================================
// DeepHighResFix - 原生 Deep HighRes Fix 节点
// ============================================================================
class DeepHighResFixNode : public Node {
public:
    std::string get_class_type() const override { return "DeepHighResFix"; }
    std::string get_category() const override { return "sampling"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
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
            {"vae_tiling", "BOOLEAN", false, false}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        ConditioningPtr positive = std::any_cast<ConditioningPtr>(inputs.at("positive"));
        ConditioningPtr negative = inputs.count("negative") ?
            std::any_cast<ConditioningPtr>(inputs.at("negative")) : nullptr;

        int64_t seed = inputs.count("seed") ? std::any_cast<int>(inputs.at("seed")) : 0;
        int total_steps = inputs.count("steps") ? std::any_cast<int>(inputs.at("steps")) : 30;
        float cfg = inputs.count("cfg") ? std::any_cast<float>(inputs.at("cfg")) : 7.0f;
        int target_width = inputs.count("target_width") ? std::any_cast<int>(inputs.at("target_width")) : 1024;
        int target_height = inputs.count("target_height") ? std::any_cast<int>(inputs.at("target_height")) : 1024;
        float strength = inputs.count("strength") ? std::any_cast<float>(inputs.at("strength")) : 1.0f;
        float phase1_cfg = inputs.count("phase1_cfg") ? std::any_cast<float>(inputs.at("phase1_cfg")) : 0.0f;
        float phase2_cfg = inputs.count("phase2_cfg") ? std::any_cast<float>(inputs.at("phase2_cfg")) : 0.0f;
        float phase3_cfg = inputs.count("phase3_cfg") ? std::any_cast<float>(inputs.at("phase3_cfg")) : 0.0f;
        bool vae_tiling = inputs.count("vae_tiling") ? std::any_cast<bool>(inputs.at("vae_tiling")) : false;

        if (!sd_ctx || !positive) {
            fprintf(stderr, "[ERROR] DeepHighResFix: Missing required inputs\n");
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

        printf("[DeepHighResFix] Target: %dx%d, Phases: %dx%d(%d) -> %dx%d(%d) -> %dx%d(%d)\n",
               target_w, target_h,
               phase1_w, phase1_h, phase1_steps,
               phase2_w, phase2_h, phase2_steps,
               target_w, target_h, phase3_steps);

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

        std::string positive_text = inputs.count("positive_text") ?
            std::any_cast<std::string>(inputs.at("positive_text")) : "";
        std::string negative_text = inputs.count("negative_text") ?
            std::any_cast<std::string>(inputs.at("negative_text")) : "";
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
            ImagePtr init_img = std::any_cast<ImagePtr>(inputs.at("init_image"));
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
            fprintf(stderr, "[ERROR] DeepHighResFix: Generation failed\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("[DeepHighResFix] Generation completed: %dx%d\n", result->width, result->height);
        outputs["IMAGE"] = make_image_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("DeepHighResFix", DeepHighResFixNode);

void init_latent_nodes() {
    // 空函数，仅确保本翻译单元被链接
}

} // namespace sdengine
