// ============================================================================
// sd-engine/adapter/sd_adapter.cpp
// ============================================================================

#include "adapter/sd_adapter.h"
#include "core/log.h"

namespace sdengine {

bool SDAdapter::init(const SDAdapterConfig& config) {
    if (sd_ctx_) {
        LOG_WARN("[SDAdapter] Already initialized, releasing previous context\n");
        release();
    }
    
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.diffusion_model_path = config.diffusion_model_path.c_str();
    
    if (!config.vae_path.empty()) {
        ctx_params.vae_path = config.vae_path.c_str();
    }
    if (!config.llm_path.empty()) {
        ctx_params.llm_path = config.llm_path.c_str();
    }
    
    ctx_params.n_threads = config.n_threads;
    ctx_params.offload_params_to_cpu = !config.use_gpu;
    ctx_params.keep_vae_on_cpu = !config.use_gpu;
    ctx_params.keep_clip_on_cpu = !config.use_gpu;
    ctx_params.flash_attn = config.use_gpu && config.flash_attn;
    ctx_params.diffusion_flash_attn = config.use_gpu && config.flash_attn;
    
    LOG_INFO("[SDAdapter] Loading model: %s\n", config.diffusion_model_path.c_str());
    
    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) {
        LOG_ERROR("[SDAdapter] Failed to load model\n");
        return false;
    }
    
    sd_ctx_ = make_sd_context_ptr(ctx);
    LOG_INFO("[SDAdapter] Model loaded successfully\n");
    return true;
}

void SDAdapter::release() {
    upscaler_ctx_.reset();
    sd_ctx_.reset();
    LOG_INFO("[SDAdapter] Released\n");
}

ImagePtr SDAdapter::generate(const GenerateParams& params) {
    if (!sd_ctx_) {
        LOG_ERROR("[SDAdapter] Not initialized\n");
        return nullptr;
    }
    
    // 构建生成参数
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    
    // 准备 LoRA 数据
    std::vector<sd_lora_t> lora_array;
    std::vector<std::string> lora_paths;
    if (!params.loras.empty()) {
        lora_array.reserve(params.loras.size());
        lora_paths.reserve(params.loras.size());
        for (const auto& lora : params.loras) {
            lora_paths.push_back(lora.path);
            lora_array.push_back({
                false,  // is_high_noise (default)
                lora.strength,
                lora_paths.back().c_str()
            });
        }
        gen_params.loras = lora_array.data();
        gen_params.lora_count = (uint32_t)lora_array.size();
        LOG_INFO("[SDAdapter] Applying %zu LoRA(s)\n", params.loras.size());
    }
    
    gen_params.prompt = params.prompt.c_str();
    gen_params.negative_prompt = params.negative_prompt.c_str();
    gen_params.width = params.width;
    gen_params.height = params.height;
    gen_params.sample_params.sample_method = params.sample_method;
    gen_params.sample_params.sample_steps = params.sample_steps;
    gen_params.sample_params.scheduler = params.scheduler;
    gen_params.strength = params.strength;
    gen_params.seed = params.seed;
    
    // img2img
    if (params.init_image && params.init_image->data) {
        gen_params.init_image = *params.init_image;
    }
    
    // inpaint mask
    if (params.mask_image && params.mask_image->data) {
        gen_params.mask_image = *params.mask_image;
    }
    
    // HiRes Fix
    if (params.enable_hires) {
        gen_params.enable_hires = true;
        gen_params.hires_width = params.hires_width;
        gen_params.hires_height = params.hires_height;
        gen_params.hires_strength = params.hires_strength;
        gen_params.hires_sample_steps = params.hires_steps;
    }
    
    LOG_INFO("[SDAdapter] Generating: %dx%d, steps=%d, cfg=%.1f, seed=%ld\n",
             params.width, params.height, params.sample_steps, params.cfg_scale, params.seed);
    
    sd_image_t* result = generate_image(sd_ctx_.get(), &gen_params);
    
    if (!result) {
        LOG_ERROR("[SDAdapter] Generation failed\n");
        return nullptr;
    }
    
    LOG_INFO("[SDAdapter] Generation completed\n");
    return make_image_ptr(result);
}

bool SDAdapter::init_upscaler(const UpscaleParams& params) {
    if (upscaler_ctx_) {
        upscaler_ctx_.reset();
    }
    
    LOG_INFO("[SDAdapter] Loading upscaler: %s\n", params.model_path.c_str());
    
    upscaler_ctx_t* ctx = new_upscaler_ctx(
        params.model_path.c_str(),
        false,  // offload_params_to_cpu
        false,  // direct
        4,      // n_threads
        256     // tile_size
    );
    
    if (!ctx) {
        LOG_ERROR("[SDAdapter] Failed to load upscaler\n");
        return false;
    }
    
    upscaler_ctx_ = make_upscaler_ptr(ctx);
    LOG_INFO("[SDAdapter] Upscaler loaded successfully\n");
    return true;
}

ImagePtr SDAdapter::upscale(const sd_image_t* input_image) {
    if (!upscaler_ctx_) {
        LOG_ERROR("[SDAdapter] Upscaler not initialized\n");
        return nullptr;
    }
    if (!input_image || !input_image->data) {
        LOG_ERROR("[SDAdapter] Invalid input image\n");
        return nullptr;
    }
    
    LOG_INFO("[SDAdapter] Upscaling: %dx%d\n", input_image->width, input_image->height);
    
    sd_image_t result = ::upscale(upscaler_ctx_.get(), *input_image, 2);
    
    if (!result.data) {
        LOG_ERROR("[SDAdapter] Upscale failed\n");
        return nullptr;
    }
    
    // upscale 返回的是值，需要拷贝到堆上
    sd_image_t* result_ptr = new sd_image_t();
    *result_ptr = result;
    
    LOG_INFO("[SDAdapter] Upscale completed: %dx%d\n", result.width, result.height);
    return make_image_ptr(result_ptr);
}

void SDAdapter::apply_loras(const std::vector<LoRAInfo>& loras) {
    if (loras.empty() || !sd_ctx_) {
        return;
    }
    
    // upstream 的 sd_img_gen_params_t 支持 loras 字段
    // 但我们已经在 generate() 中通过 gen_params.loras 传递了
    // 这里不需要额外操作
    LOG_INFO("[SDAdapter] Applying %zu LoRA(s)\n", loras.size());
}

void SDAdapter::clear_loras() {
    // 如果 upstream 有清除 LoRA 的 API，在这里调用
    // 目前 generate_image 会在每次调用后自动处理
}

} // namespace sdengine
