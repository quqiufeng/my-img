#include "adapters/sdcpp_adapter.h"
#include <stable-diffusion.h>
#include <iostream>
#include <cstring>
#include <chrono>

#include <stb_image_write.h>

// 日志回调函数
static void sd_log_callback(enum sd_log_level_t level, const char* text, void* data) {
    (void)data;
    const char* level_str = "";
    switch (level) {
        case SD_LOG_DEBUG: level_str = "DEBUG"; break;
        case SD_LOG_INFO:  level_str = "INFO";  break;
        case SD_LOG_WARN:  level_str = "WARN";  break;
        case SD_LOG_ERROR: level_str = "ERROR"; break;
    }
    std::cerr << "[SD " << level_str << "] " << text;
}

namespace myimg {

// ============================================================
// 辅助函数：枚举转换
// ============================================================
static sample_method_t convert_sample_method(SampleMethod method) {
    switch (method) {
        case SampleMethod::Euler: return EULER_SAMPLE_METHOD;
        case SampleMethod::EulerAncestral: return EULER_A_SAMPLE_METHOD;
        case SampleMethod::Heun: return HEUN_SAMPLE_METHOD;
        case SampleMethod::DPM2: return DPM2_SAMPLE_METHOD;
        case SampleMethod::DPMPP2S_A: return DPMPP2S_A_SAMPLE_METHOD;
        case SampleMethod::DPMPP2M: return DPMPP2M_SAMPLE_METHOD;
        case SampleMethod::DPMPP2Mv2: return DPMPP2Mv2_SAMPLE_METHOD;
        case SampleMethod::IPNDM: return IPNDM_SAMPLE_METHOD;
        case SampleMethod::IPNDM_V: return IPNDM_V_SAMPLE_METHOD;
        case SampleMethod::LCM: return LCM_SAMPLE_METHOD;
        case SampleMethod::DDIM_Trailing: return DDIM_TRAILING_SAMPLE_METHOD;
        case SampleMethod::TCD: return TCD_SAMPLE_METHOD;
        case SampleMethod::RES_Multistep: return RES_MULTISTEP_SAMPLE_METHOD;
        case SampleMethod::RES_2S: return RES_2S_SAMPLE_METHOD;
        case SampleMethod::ER_SDE: return ER_SDE_SAMPLE_METHOD;
        default: return EULER_SAMPLE_METHOD;
    }
}

static scheduler_t convert_scheduler(Scheduler scheduler) {
    switch (scheduler) {
        case Scheduler::Discrete: return DISCRETE_SCHEDULER;
        case Scheduler::Karras: return KARRAS_SCHEDULER;
        case Scheduler::Exponential: return EXPONENTIAL_SCHEDULER;
        case Scheduler::AYS: return AYS_SCHEDULER;
        case Scheduler::GITS: return GITS_SCHEDULER;
        case Scheduler::SGM_Uniform: return SGM_UNIFORM_SCHEDULER;
        case Scheduler::Simple: return SIMPLE_SCHEDULER;
        case Scheduler::Smoothstep: return SMOOTHSTEP_SCHEDULER;
        case Scheduler::KL_Optimal: return KL_OPTIMAL_SCHEDULER;
        case Scheduler::LCM: return LCM_SCHEDULER;
        case Scheduler::Bong_Tangent: return BONG_TANGENT_SCHEDULER;
        default: return SIMPLE_SCHEDULER;
    }
}

static sd_type_t convert_wtype(const std::string& wtype) {
    if (wtype == "f32" || wtype == "F32") return SD_TYPE_F32;
    if (wtype == "f16" || wtype == "F16") return SD_TYPE_F16;
    if (wtype == "q4_0" || wtype == "Q4_0") return SD_TYPE_Q4_0;
    if (wtype == "q4_1" || wtype == "Q4_1") return SD_TYPE_Q4_1;
    if (wtype == "q5_0" || wtype == "Q5_0") return SD_TYPE_Q5_0;
    if (wtype == "q5_1" || wtype == "Q5_1") return SD_TYPE_Q5_1;
    if (wtype == "q8_0" || wtype == "Q8_0") return SD_TYPE_Q8_0;
    if (wtype == "q2_k" || wtype == "Q2_K") return SD_TYPE_Q2_K;
    if (wtype == "q3_k" || wtype == "Q3_K") return SD_TYPE_Q3_K;
    if (wtype == "q4_k" || wtype == "Q4_K") return SD_TYPE_Q4_K;
    if (wtype == "q5_k" || wtype == "Q5_K") return SD_TYPE_Q5_K;
    if (wtype == "q6_k" || wtype == "Q6_K") return SD_TYPE_Q6_K;
    if (wtype == "q8_k" || wtype == "Q8_K") return SD_TYPE_Q8_K;
    return SD_TYPE_COUNT;  // 使用模型默认类型
}

// ============================================================
// SDCPPAdapter 实现
// ============================================================
SDCPPAdapter::SDCPPAdapter() = default;

SDCPPAdapter::~SDCPPAdapter() {
    if (ctx_) {
        free_sd_ctx(ctx_);
        ctx_ = nullptr;
    }
}

SDCPPAdapter::SDCPPAdapter(SDCPPAdapter&& other) noexcept
    : ctx_(other.ctx_), progress_callback_(std::move(other.progress_callback_)) {
    other.ctx_ = nullptr;
}

SDCPPAdapter& SDCPPAdapter::operator=(SDCPPAdapter&& other) noexcept {
    if (this != &other) {
        if (ctx_) {
            free_sd_ctx(ctx_);
        }
        ctx_ = other.ctx_;
        progress_callback_ = std::move(other.progress_callback_);
        other.ctx_ = nullptr;
    }
    return *this;
}

bool SDCPPAdapter::initialize(const GenerationParams& params) {
    if (ctx_) {
        free_sd_ctx(ctx_);
        ctx_ = nullptr;
    }
    
    return load_model(params);
}

bool SDCPPAdapter::load_model(const GenerationParams& params) {
    // 设置日志回调
    static bool log_initialized = false;
    if (!log_initialized) {
        sd_set_log_callback(sd_log_callback, nullptr);
        log_initialized = true;
    }
    
    sd_ctx_params_t sd_params;
    sd_ctx_params_init(&sd_params);
    
    // 设置模型路径
    // 注意：Z-Image 模型需要使用 diffusion_model_path 而不是 model_path
    if (!params.diffusion_model_path.empty()) {
        sd_params.diffusion_model_path = params.diffusion_model_path.c_str();
    } else if (!params.model_path.empty()) {
        // 如果只有 model_path，尝试判断是否为 diffusion-only 模型
        sd_params.model_path = params.model_path.c_str();
    }
    
    sd_params.vae_path = params.vae_path.empty() ? nullptr : params.vae_path.c_str();
    sd_params.clip_l_path = params.clip_l_path.empty() ? nullptr : params.clip_l_path.c_str();
    sd_params.clip_g_path = params.clip_g_path.empty() ? nullptr : params.clip_g_path.c_str();
    sd_params.t5xxl_path = params.t5xxl_path.empty() ? nullptr : params.t5xxl_path.c_str();
    sd_params.llm_path = params.llm_path.empty() ? nullptr : params.llm_path.c_str();
    sd_params.control_net_path = params.control_net_path.empty() ? nullptr : params.control_net_path.c_str();
    
    // 系统设置
    sd_params.n_threads = params.n_threads > 0 ? params.n_threads : sd_get_num_physical_cores();
    sd_params.offload_params_to_cpu = params.offload_params_to_cpu;
    sd_params.enable_mmap = params.enable_mmap;
    sd_params.flash_attn = params.flash_attn;
    
    // 权重类型
    if (!params.wtype.empty() && params.wtype != "default") {
        sd_params.wtype = convert_wtype(params.wtype);
    }
    
    // VAE 设置
    sd_params.vae_decode_only = false;
    sd_params.keep_vae_on_cpu = false;
    
    std::cout << "[SDCPPAdapter] Loading model from: " << params.model_path << std::endl;
    std::cout << "[SDCPPAdapter] Using " << sd_params.n_threads << " threads" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    ctx_ = new_sd_ctx(&sd_params);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    if (!ctx_) {
        std::cerr << "[SDCPPAdapter] Failed to load model after " << duration << " seconds" << std::endl;
        return false;
    }
    
    std::cout << "[SDCPPAdapter] Model loaded successfully in " << duration << " seconds" << std::endl;
    
    // 设置进度回调
    if (progress_callback_) {
        sd_set_progress_callback(progress_callback_wrapper, this);
    }
    
    return true;
}

std::vector<Image> SDCPPAdapter::generate(const GenerationParams& params) {
    std::vector<Image> images;
    
    if (!ctx_) {
        std::cerr << "[SDCPPAdapter] Model not initialized!" << std::endl;
        return images;
    }
    
    // 准备生成参数
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    
    gen_params.prompt = params.prompt.c_str();
    gen_params.negative_prompt = params.negative_prompt.c_str();
    gen_params.clip_skip = params.clip_skip;
    gen_params.width = params.width;
    gen_params.height = params.height;
    gen_params.seed = params.seed;
    gen_params.batch_count = params.batch_count;
    gen_params.strength = params.strength;
    
    // 采样参数
    gen_params.sample_params.sample_method = convert_sample_method(params.sample_method);
    gen_params.sample_params.scheduler = convert_scheduler(params.scheduler);
    gen_params.sample_params.sample_steps = params.sample_steps;
    gen_params.sample_params.eta = params.eta;
    gen_params.sample_params.flow_shift = params.flow_shift;
    
    // img2img
    if (!params.init_image.empty() && params.strength < 1.0f) {
        gen_params.init_image = image_to_sd_image(params.init_image);
    }
    
    // Inpainting
    if (!params.mask_image.empty()) {
        gen_params.mask_image = image_to_sd_image(params.mask_image);
    }
    
    // ControlNet
    if (!params.control_image.empty()) {
        gen_params.control_image = image_to_sd_image(params.control_image);
        gen_params.control_strength = params.control_strength;
    }
    
    // LoRA
    std::vector<sd_lora_t> sd_loras;
    if (!params.loras.empty()) {
        sd_loras.reserve(params.loras.size());
        for (const auto& lora : params.loras) {
            sd_lora_t sd_lora;
            sd_lora.path = lora.path.c_str();
            sd_lora.multiplier = lora.multiplier;
            sd_lora.is_high_noise = lora.is_high_noise;
            sd_loras.push_back(sd_lora);
        }
        gen_params.loras = sd_loras.data();
        gen_params.lora_count = static_cast<uint32_t>(sd_loras.size());
    }
    
    // HiRes Fix
    gen_params.enable_hires = params.enable_hires;
    if (params.enable_hires) {
        gen_params.hires_width = params.hires_width;
        gen_params.hires_height = params.hires_height;
        gen_params.hires_strength = params.hires_strength;
        gen_params.hires_sample_steps = params.hires_sample_steps;
    }
    
    // VAE Tiling
    gen_params.vae_tiling_params.enabled = params.vae_tiling;
    if (params.vae_tiling) {
        gen_params.vae_tiling_params.tile_size_x = params.vae_tile_size_x;
        gen_params.vae_tiling_params.tile_size_y = params.vae_tile_size_y;
        gen_params.vae_tiling_params.target_overlap = params.vae_tile_overlap;
        std::cout << "  VAE Tiling: " << params.vae_tile_size_x << "x" << params.vae_tile_size_y 
                  << " (overlap: " << params.vae_tile_overlap << ")" << std::endl;
    }
    
    std::cout << "[SDCPPAdapter] Generating image..." << std::endl;
    std::cout << "  Prompt: " << params.prompt << std::endl;
    std::cout << "  Size: " << params.width << "x" << params.height << std::endl;
    std::cout << "  Steps: " << params.sample_steps << ", CFG: " << params.cfg_scale << std::endl;
    std::cout << "  Seed: " << params.seed << std::endl;
    
    if (params.enable_hires) {
        std::cout << "  HiRes: " << params.hires_width << "x" << params.hires_height 
                  << " (strength: " << params.hires_strength << ")" << std::endl;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    sd_image_t* sd_images = generate_image(ctx_, &gen_params);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "[SDCPPAdapter] Generation completed in " << duration << " seconds" << std::endl;
    
    if (sd_images) {
        for (int i = 0; i < params.batch_count; i++) {
            images.push_back(sd_image_to_image(sd_images[i]));
        }
        free(sd_images);
    }
    
    return images;
}

Image SDCPPAdapter::generate_single(const GenerationParams& params) {
    GenerationParams p = params;
    p.batch_count = 1;
    auto images = generate(p);
    if (!images.empty()) {
        return images[0];
    }
    return Image();
}

std::vector<float> SDCPPAdapter::encode_prompt(const std::string& prompt, int clip_skip) {
    // TODO: 实现文本编码，返回 conditioning
    // 这需要更底层的 API，可能需要修改 sd.cpp 暴露更多接口
    std::cerr << "[SDCPPAdapter] encode_prompt not yet implemented" << std::endl;
    return std::vector<float>();
}

void SDCPPAdapter::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
    if (ctx_) {
        sd_set_progress_callback(progress_callback_wrapper, this);
    }
}

void SDCPPAdapter::progress_callback_wrapper(int step, int steps, float time, void* data) {
    auto* adapter = static_cast<SDCPPAdapter*>(data);
    if (adapter && adapter->progress_callback_) {
        adapter->progress_callback_(step, steps, time);
    }
}

// ============================================================
// 静态工具函数
// ============================================================
Image SDCPPAdapter::sd_image_to_image(const sd_image_t& sd_img) {
    Image img;
    img.width = sd_img.width;
    img.height = sd_img.height;
    img.channels = sd_img.channel;
    
    size_t data_size = img.width * img.height * img.channels;
    img.data.resize(data_size);
    std::memcpy(img.data.data(), sd_img.data, data_size);
    
    return img;
}

sd_image_t SDCPPAdapter::image_to_sd_image(const Image& img) {
    sd_image_t sd_img;
    sd_img.width = img.width;
    sd_img.height = img.height;
    sd_img.channel = img.channels;
    
    size_t data_size = img.width * img.height * img.channels;
    sd_img.data = (uint8_t*)malloc(data_size);
    std::memcpy(sd_img.data, img.data.data(), data_size);
    
    return sd_img;
}

std::vector<std::string> SDCPPAdapter::get_available_sample_methods() {
    std::vector<std::string> methods;
    for (int i = 0; i < SAMPLE_METHOD_COUNT; i++) {
        methods.push_back(sd_sample_method_name(static_cast<sample_method_t>(i)));
    }
    return methods;
}

std::vector<std::string> SDCPPAdapter::get_available_schedulers() {
    std::vector<std::string> schedulers;
    for (int i = 0; i < SCHEDULER_COUNT; i++) {
        schedulers.push_back(sd_scheduler_name(static_cast<scheduler_t>(i)));
    }
    return schedulers;
}

std::string SDCPPAdapter::get_version() {
    return sd_version();
}

std::string SDCPPAdapter::get_commit() {
    return sd_commit();
}

// ============================================================
// Image 保存
// ============================================================
bool Image::save_to_file(const std::string& path) const {
    if (empty()) {
        std::cerr << "[Image] Cannot save empty image" << std::endl;
        return false;
    }
    int success = stbi_write_png(path.c_str(), width, height, channels, data.data(), width * channels);
    if (success) {
        std::cout << "[Image] Saved to " << path << " (" << width << "x" << height << ")" << std::endl;
        return true;
    } else {
        std::cerr << "[Image] Failed to save to " << path << std::endl;
        return false;
    }
}

// ============================================================
// ESRGAN 放大
// ============================================================
Image SDCPPAdapter::upscale_with_esrgan(const Image& image, const std::string& model_path, int repeats, int tile_size) {
    if (image.empty()) {
        std::cerr << "[SDCPPAdapter] Cannot upscale empty image" << std::endl;
        return Image();
    }
    
    upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(model_path.c_str(), false, false, -1, tile_size);
    if (!upscaler_ctx) {
        std::cerr << "[SDCPPAdapter] Failed to load upscaler model: " << model_path << std::endl;
        return Image();
    }
    
    sd_image_t sd_img = image_to_sd_image(image);
    Image result = image;
    
    for (int i = 0; i < repeats; ++i) {
        sd_image_t upscaled = upscale(upscaler_ctx, sd_img, 4);  // ESRGAN 默认 4x
        if (upscaled.data == nullptr) {
            std::cerr << "[SDCPPAdapter] Upscale failed at iteration " << (i + 1) << std::endl;
            free(sd_img.data);
            break;
        }
        free(sd_img.data);
        sd_img = upscaled;
        result = sd_image_to_image(sd_img);
    }
    
    free(sd_img.data);
    free_upscaler_ctx(upscaler_ctx);
    
    return result;
}

} // namespace myimg
