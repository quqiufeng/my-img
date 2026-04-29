// ============================================================================
// sd-engine/nodes/generate_nodes.cpp
// ============================================================================
// 图像生成节点（Txt2Img / Img2Img / HiResFix）
// 粗粒度节点：直接调用 SDAdapter::generate()
// ============================================================================

#include "adapter/sd_adapter.h"
#include "core/log.h"
#include "core/memory_monitor.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// Txt2Img - 文生图
// ============================================================================
class Txt2ImgNode : public Node {
  public:
    std::string get_class_type() const override { return "Txt2Img"; }
    std::string get_category() const override { return "generate"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},           // SDContextPtr
            {"prompt", "STRING", true, std::string("")},
            {"negative_prompt", "STRING", false, std::string("")},
            {"width", "INT", false, 512},
            {"height", "INT", false, 512},
            {"steps", "INT", false, 20},
            {"cfg_scale", "FLOAT", false, 7.5f},
            {"sample_method", "STRING", false, std::string("euler")},
            {"scheduler", "STRING", false, std::string("discrete")},
            {"seed", "INT", false, 42},
            {"lora_stack", "LORA_STACK", false, nullptr},
            {"control_image", "IMAGE", false, nullptr},
            {"control_strength", "FLOAT", false, 1.0f},
            {"ref_images", "IMAGE_LIST", false, nullptr},
            {"ref_strength", "FLOAT", false, 1.0f},
            {"vae_tiling", "BOOLEAN", false, false},
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        // 获取模型上下文
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        SD_RETURN_IF_NULL(sd_ctx, sd_error_t::ERROR_MODEL_LOADING);

        // 收集参数
        GenerateParams params;
        SD_RETURN_IF_ERROR(get_required_string(inputs, "prompt", params.prompt));
        params.negative_prompt = get_input_opt<std::string>(inputs, "negative_prompt", "");
        params.width = get_input_opt<int>(inputs, "width", 512);
        params.height = get_input_opt<int>(inputs, "height", 512);
        params.sample_steps = get_input_opt<int>(inputs, "steps", 20);
        params.cfg_scale = get_input_opt<float>(inputs, "cfg_scale", 7.5f);
        params.seed = get_input_opt<int64_t>(inputs, "seed", 42);

        // 验证输入参数
        SD_RETURN_IF_ERROR(validate_generation_params(params.width, params.height, params.sample_steps, params.cfg_scale));

        // 解析采样方法
        std::string method_str = get_input_opt<std::string>(inputs, "sample_method", "euler");
        params.sample_method = str_to_sample_method(method_str.c_str());

        // 解析调度器
        std::string scheduler_str = get_input_opt<std::string>(inputs, "scheduler", "discrete");
        params.scheduler = str_to_scheduler(scheduler_str.c_str());

        // 显存检测与自动降分辨率
        bool auto_optimize = get_input_opt<bool>(inputs, "auto_optimize", true);
        if (auto_optimize) {
            auto mem_info = detect_gpu_memory();
            if (mem_info.available) {
                auto [new_w, new_h] = auto_adjust_resolution(params.width, params.height, 
                                                              mem_info.free_bytes, true);
                if (new_w != params.width || new_h != params.height) {
                    params.width = new_w;
                    params.height = new_h;
                }
            }
        }

        LOG_INFO("[Txt2Img] Generating: %dx%d, steps=%d, cfg=%.1f, seed=%ld\n",
                 params.width, params.height, params.sample_steps, params.cfg_scale, params.seed);

        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        
        gen_params.prompt = params.prompt.c_str();
        gen_params.negative_prompt = params.negative_prompt.c_str();
        gen_params.width = params.width;
        gen_params.height = params.height;
        gen_params.sample_params.sample_method = params.sample_method;
        gen_params.sample_params.sample_steps = params.sample_steps;
        gen_params.sample_params.scheduler = params.scheduler;
        gen_params.seed = params.seed;
        
        // 填充 LoRA、ControlNet、IPAdapter、VAE Tiling（消除重复代码）
        SD_RETURN_IF_ERROR(fill_gen_params_from_inputs(inputs, gen_params, "[Txt2Img]"));

        sd_image_t* result = generate_image(sd_ctx, &gen_params);
        if (!result) {
            LOG_ERROR("[Txt2Img] Generation failed\n");
            return sd_error_t::ERROR_SAMPLING_FAILED;
        }

        outputs["IMAGE"] = make_image_ptr(result);
        LOG_INFO("[Txt2Img] Generation completed: %dx%d\n", result->width, result->height);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("Txt2Img", Txt2ImgNode);

// ============================================================================
// Img2Img - 图生图
// ============================================================================
class Img2ImgNode : public Node {
  public:
    std::string get_class_type() const override { return "Img2Img"; }
    std::string get_category() const override { return "generate"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
            {"image", "IMAGE", true, nullptr},           // 输入图像
            {"prompt", "STRING", true, std::string("")},
            {"negative_prompt", "STRING", false, std::string("")},
            {"strength", "FLOAT", false, 0.75f},         // denoise strength
            {"width", "INT", false, 512},
            {"height", "INT", false, 512},
            {"steps", "INT", false, 20},
            {"cfg_scale", "FLOAT", false, 7.5f},
            {"seed", "INT", false, 42},
            {"lora_stack", "LORA_STACK", false, nullptr},
            {"control_image", "IMAGE", false, nullptr},
            {"control_strength", "FLOAT", false, 1.0f},
            {"ref_images", "IMAGE_LIST", false, nullptr},
            {"ref_strength", "FLOAT", false, 1.0f},
            {"vae_tiling", "BOOLEAN", false, false},
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        SD_RETURN_IF_NULL(sd_ctx, sd_error_t::ERROR_MODEL_LOADING);

        ImagePtr init_img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", init_img));
        SD_RETURN_IF_NULL(init_img.get(), sd_error_t::ERROR_INVALID_INPUT);

        GenerateParams params;
        SD_RETURN_IF_ERROR(get_required_string(inputs, "prompt", params.prompt));
        params.negative_prompt = get_input_opt<std::string>(inputs, "negative_prompt", "");
        params.strength = get_input_opt<float>(inputs, "strength", 0.75f);
        params.width = get_input_opt<int>(inputs, "width", init_img->width);
        params.height = get_input_opt<int>(inputs, "height", init_img->height);
        params.sample_steps = get_input_opt<int>(inputs, "steps", 20);
        params.cfg_scale = get_input_opt<float>(inputs, "cfg_scale", 7.5f);
        params.seed = get_input_opt<int64_t>(inputs, "seed", 42);

        // 验证输入参数
        SD_RETURN_IF_ERROR(validate_generation_params(params.width, params.height, params.sample_steps, params.cfg_scale, params.strength));

        // 显存检测与自动降分辨率
        bool auto_optimize = get_input_opt<bool>(inputs, "auto_optimize", true);
        if (auto_optimize) {
            auto mem_info = detect_gpu_memory();
            if (mem_info.available) {
                auto [new_w, new_h] = auto_adjust_resolution(params.width, params.height,
                                                              mem_info.free_bytes, true);
                if (new_w != params.width || new_h != params.height) {
                    params.width = new_w;
                    params.height = new_h;
                }
            }
        }

        LOG_INFO("[Img2Img] Generating from image: %dx%d, strength=%.2f\n",
                 init_img->width, init_img->height, params.strength);

        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        
        gen_params.prompt = params.prompt.c_str();
        gen_params.negative_prompt = params.negative_prompt.c_str();
        gen_params.width = params.width;
        gen_params.height = params.height;
        gen_params.sample_params.sample_steps = params.sample_steps;
        gen_params.strength = params.strength;
        gen_params.seed = params.seed;
        gen_params.init_image = *init_img;
        
        // 填充 LoRA、ControlNet、IPAdapter、VAE Tiling（消除重复代码）
        SD_RETURN_IF_ERROR(fill_gen_params_from_inputs(inputs, gen_params, "[Img2Img]"));

        sd_image_t* result = generate_image(sd_ctx, &gen_params);
        if (!result) {
            LOG_ERROR("[Img2Img] Generation failed\n");
            return sd_error_t::ERROR_SAMPLING_FAILED;
        }

        outputs["IMAGE"] = make_image_ptr(result);
        LOG_INFO("[Img2Img] Generation completed\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("Img2Img", Img2ImgNode);

// ============================================================================
// HiResFix - 高清修复
// ============================================================================
class HiResFixNode : public Node {
  public:
    std::string get_class_type() const override { return "HiResFix"; }
    std::string get_category() const override { return "generate"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
            {"prompt", "STRING", true, std::string("")},
            {"negative_prompt", "STRING", false, std::string("")},
            {"width", "INT", true, 0},                   // 目标宽度
            {"height", "INT", true, 0},                  // 目标高度
            {"steps", "INT", false, 20},
            {"hires_steps", "INT", false, 20},
            {"cfg_scale", "FLOAT", false, 7.5f},
            {"hires_strength", "FLOAT", false, 0.5f},
            {"seed", "INT", false, 42},
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "model");
        SD_RETURN_IF_NULL(sd_ctx, sd_error_t::ERROR_MODEL_LOADING);

        GenerateParams params;
        SD_RETURN_IF_ERROR(get_required_string(inputs, "prompt", params.prompt));
        params.negative_prompt = get_input_opt<std::string>(inputs, "negative_prompt", "");
        params.width = get_input_opt<int>(inputs, "width", 1024);
        params.height = get_input_opt<int>(inputs, "height", 1024);
        params.sample_steps = get_input_opt<int>(inputs, "steps", 20);
        params.hires_steps = get_input_opt<int>(inputs, "hires_steps", 20);
        params.cfg_scale = get_input_opt<float>(inputs, "cfg_scale", 7.5f);
        params.hires_strength = get_input_opt<float>(inputs, "hires_strength", 0.5f);
        params.seed = get_input_opt<int64_t>(inputs, "seed", 42);
        params.enable_hires = true;
        params.hires_width = params.width;
        params.hires_height = params.height;

        // 验证输入参数
        SD_RETURN_IF_ERROR(validate_generation_params(params.width, params.height, params.sample_steps, params.cfg_scale));

        LOG_INFO("[HiResFix] Generating: %dx%d -> %dx%d, strength=%.2f\n",
                 params.width / 2, params.height / 2, params.width, params.height, params.hires_strength);

        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        gen_params.prompt = params.prompt.c_str();
        gen_params.negative_prompt = params.negative_prompt.c_str();
        gen_params.width = params.width / 2;  // 先生成一半分辨率
        gen_params.height = params.height / 2;
        gen_params.sample_params.sample_steps = params.sample_steps;
        gen_params.seed = params.seed;
        gen_params.enable_hires = true;
        gen_params.hires_width = params.width;
        gen_params.hires_height = params.height;
        gen_params.hires_strength = params.hires_strength;
        gen_params.hires_sample_steps = params.hires_steps;

        sd_image_t* result = generate_image(sd_ctx, &gen_params);
        if (!result) {
            LOG_ERROR("[HiResFix] Generation failed\n");
            return sd_error_t::ERROR_SAMPLING_FAILED;
        }

        outputs["IMAGE"] = make_image_ptr(result);
        LOG_INFO("[HiResFix] Generation completed: %dx%d\n", result->width, result->height);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("HiResFix", HiResFixNode);

// ============================================================================
// Upscale - ESRGAN 放大
// ============================================================================
class UpscaleNode : public Node {
  public:
    std::string get_class_type() const override { return "Upscale"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"upscale_model", "STRING", true, std::string("")},
            {"scale", "INT", false, 2},
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr input_img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", input_img));
        SD_RETURN_IF_NULL(input_img.get(), sd_error_t::ERROR_INVALID_INPUT);

        std::string model_path;
        SD_RETURN_IF_ERROR(get_required_string(inputs, "upscale_model", model_path));
        if (model_path.empty()) {
            LOG_ERROR("[Upscale] upscale_model is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        uint32_t scale = get_input_opt<int>(inputs, "scale", 2);

        LOG_INFO("[Upscale] Upscaling %dx%d by %dx using %s\n",
                 input_img->width, input_img->height, scale, model_path.c_str());

        // 加载 upscaler
        upscaler_ctx_t* upscaler = new_upscaler_ctx(
            model_path.c_str(), false, false, 4, 256
        );
        if (!upscaler) {
            LOG_ERROR("[Upscale] Failed to load upscaler: %s\n", model_path.c_str());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        sd_image_t result = ::upscale(upscaler, *input_img, scale);
        free_upscaler_ctx(upscaler);

        if (!result.data) {
            LOG_ERROR("[Upscale] Upscale failed\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_image_t* result_ptr = new sd_image_t();
        *result_ptr = result;
        outputs["IMAGE"] = make_image_ptr(result_ptr);
        
        LOG_INFO("[Upscale] Completed: %dx%d -> %dx%d\n",
                 input_img->width, input_img->height, result.width, result.height);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("Upscale", UpscaleNode);

// ============================================================================
// ImageUpscaleWithModel - 使用已加载的 ESRGAN 模型放大
// 与 UpscaleModelLoader 配合使用，避免每次重新加载模型
// ============================================================================
class ImageUpscaleWithModelNode : public Node {
  public:
    std::string get_class_type() const override { return "ImageUpscaleWithModel"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"upscale_model", "UPSCALE_MODEL", true, nullptr},  // UpscalerPtr
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr input_img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", input_img));
        SD_RETURN_IF_NULL(input_img.get(), sd_error_t::ERROR_INVALID_INPUT);

        UpscalerPtr upscaler;
        auto it = inputs.find("upscale_model");
        if (it == inputs.end()) {
            LOG_ERROR("[ImageUpscaleWithModel] Missing upscale_model input\n");
            return sd_error_t::ERROR_MISSING_INPUT;
        }
        const auto& val = it->second;
        if (val.type() == typeid(UpscalerPtr)) {
            auto opt = any_cast_safe<UpscalerPtr>(val);
            if (!opt) {
                LOG_ERROR("[ImageUpscaleWithModel] Failed to cast upscale_model\n");
                return sd_error_t::ERROR_INVALID_INPUT;
            }
            upscaler = *opt;
        } else {
            LOG_ERROR("[ImageUpscaleWithModel] Invalid upscale_model type\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        SD_RETURN_IF_NULL(upscaler.get(), sd_error_t::ERROR_MODEL_LOADING);

        uint32_t scale = get_upscale_factor(upscaler.get());

        LOG_INFO("[ImageUpscaleWithModel] Upscaling %dx%d by %dx\n",
                 input_img->width, input_img->height, scale);

        sd_image_t result = ::upscale(upscaler.get(), *input_img, scale);

        if (!result.data) {
            LOG_ERROR("[ImageUpscaleWithModel] Upscale failed\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_image_t* result_ptr = new sd_image_t();
        *result_ptr = result;
        outputs["IMAGE"] = make_image_ptr(result_ptr);
        
        LOG_INFO("[ImageUpscaleWithModel] Completed: %dx%d -> %dx%d\n",
                 input_img->width, input_img->height, result.width, result.height);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageUpscaleWithModel", ImageUpscaleWithModelNode);

// ============================================================================
// IPAdapterApply - IPAdapter 应用
// ============================================================================
class IPAdapterApplyNode : public Node {
  public:
    std::string get_class_type() const override { return "IPAdapterApply"; }
    std::string get_category() const override { return "ipadapter"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"weight", "FLOAT", false, 1.0f},
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE_LIST", "IMAGE_LIST"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", img));
        SD_RETURN_IF_NULL(img.get(), sd_error_t::ERROR_INVALID_INPUT);

        float weight = get_input_opt<float>(inputs, "weight", 1.0f);
        (void)weight; // upstream handles weight internally

        std::vector<ImagePtr> image_list = {img};
        outputs["IMAGE_LIST"] = image_list;

        LOG_INFO("[IPAdapterApply] Prepared 1 reference image\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("IPAdapterApply", IPAdapterApplyNode);

} // namespace sdengine
