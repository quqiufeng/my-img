#include "adapters/sdcpp_adapter.h"
#include <stable-diffusion.h>
#include <stable-diffusion-ext.h>

#ifdef MYIMG_ENABLE_LIBTORCH
#include <torch/torch.h>
#endif

#include <iostream>
#include <cstring>
#include <chrono>
#include <cctype>
#include <filesystem>

#include <stb_image_write.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <webp/encode.h>
#include <webp/mux.h>

namespace fs = std::filesystem;

// JPEG quality (0-100, default 95)
static int g_jpeg_quality = 95;

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
// JPEG 质量设置
// ============================================================
void set_jpeg_quality(int quality) {
    g_jpeg_quality = std::max(1, std::min(100, quality));
}

int get_jpeg_quality() {
    return g_jpeg_quality;
}

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

static sd_hires_upscaler_t convert_hires_upscaler(HiresUpscaler upscaler) {
    switch (upscaler) {
        case HiresUpscaler::Latent:                    return SD_HIRES_UPSCALER_LATENT;
        case HiresUpscaler::LatentNearest:             return SD_HIRES_UPSCALER_LATENT_NEAREST;
        case HiresUpscaler::LatentNearestExact:        return SD_HIRES_UPSCALER_LATENT_NEAREST_EXACT;
        case HiresUpscaler::LatentAntialiased:         return SD_HIRES_UPSCALER_LATENT_ANTIALIASED;
        case HiresUpscaler::LatentBicubic:             return SD_HIRES_UPSCALER_LATENT_BICUBIC;
        case HiresUpscaler::LatentBicubicAntialiased:  return SD_HIRES_UPSCALER_LATENT_BICUBIC_ANTIALIASED;
        case HiresUpscaler::Lanczos:                   return SD_HIRES_UPSCALER_LANCZOS;
        case HiresUpscaler::Nearest:                   return SD_HIRES_UPSCALER_NEAREST;
        case HiresUpscaler::Model:                     return SD_HIRES_UPSCALER_MODEL;
        default:                                       return SD_HIRES_UPSCALER_LATENT;
    }
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
    sd_params.diffusion_flash_attn = params.flash_attn;
    
    // 权重类型
    if (!params.wtype.empty() && params.wtype != "default") {
        sd_params.wtype = convert_wtype(params.wtype);
    }
    
    // VAE 设置
    sd_params.vae_decode_only = false;
    sd_params.keep_vae_on_cpu = false;
    
    // Embeddings (Textual Inversion)
    std::vector<sd_embedding_t> embeddings;
    std::vector<std::string> embedding_names;
    std::vector<std::string> embedding_paths;
    if (!params.embedding_dir.empty() && fs::exists(params.embedding_dir)) {
        std::cout << "[SDCPPAdapter] Scanning embeddings from: " << params.embedding_dir << std::endl;
        for (const auto& entry : fs::directory_iterator(params.embedding_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            for (auto& c : ext) c = std::tolower(c);
            if (ext == ".pt" || ext == ".safetensors" || ext == ".bin") {
                std::string name = entry.path().stem().string();
                std::string path = entry.path().string();
                embedding_names.push_back(name);
                embedding_paths.push_back(path);
                std::cout << "  - Embedding: " << name << " (" << path << ")" << std::endl;
            }
        }
        if (!embedding_names.empty()) {
            embeddings.reserve(embedding_names.size());
            for (size_t i = 0; i < embedding_names.size(); ++i) {
                embeddings.push_back({embedding_names[i].c_str(), embedding_paths[i].c_str()});
            }
            sd_params.embeddings = embeddings.data();
            sd_params.embedding_count = embeddings.size();
        }
    }
    
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
    if (params.flow_shift != 0.0f) {
        gen_params.sample_params.flow_shift = params.flow_shift;
    }
    
    // Guidance (CFG scale)
    gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;
    
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
    gen_params.hires.enabled = params.enable_hires;
    if (params.enable_hires) {
        gen_params.hires.upscaler = convert_hires_upscaler(params.hires_upscaler);
        gen_params.hires.target_width = params.hires_width;
        gen_params.hires.target_height = params.hires_height;
        gen_params.hires.scale = params.hires_scale;
        gen_params.hires.denoising_strength = params.hires_strength;
        gen_params.hires.steps = params.hires_sample_steps;
        gen_params.hires.upscale_tile_size = params.hires_tile_size;
        if (!params.hires_model_path.empty() && params.hires_upscaler == HiresUpscaler::Model) {
            gen_params.hires.model_path = params.hires_model_path.c_str();
        }
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
                  << " (upscaler: " << sd_hires_upscaler_name(convert_hires_upscaler(params.hires_upscaler))
                  << ", scale: " << params.hires_scale
                  << ", strength: " << params.hires_strength
                  << ", steps: " << params.hires_sample_steps << ")" << std::endl;
        if (params.hires_upscaler == HiresUpscaler::Model && !params.hires_model_path.empty()) {
            std::cout << "  HiRes Model: " << params.hires_model_path << std::endl;
        }
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

Image SDCPPAdapter::generate_hires_libtorch(const GenerationParams& params) {
#ifdef MYIMG_ENABLE_LIBTORCH
    if (!ctx_) {
        std::cerr << "[SDCPPAdapter] Model not initialized!" << std::endl;
        return Image();
    }

    std::cout << "[SDCPPAdapter] libTorch HiRes Fix: generating base latent..." << std::endl;

    // 1. 构造基础生成参数（禁用 hires，使用基础分辨率）
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    gen_params.prompt = params.prompt.c_str();
    gen_params.negative_prompt = params.negative_prompt.c_str();
    gen_params.clip_skip = params.clip_skip;
    gen_params.width = params.width;
    gen_params.height = params.height;
    gen_params.seed = params.seed;
    gen_params.batch_count = 1;
    gen_params.strength = params.strength;

    // 采样参数
    gen_params.sample_params.sample_method = convert_sample_method(params.sample_method);
    gen_params.sample_params.scheduler = convert_scheduler(params.scheduler);
    gen_params.sample_params.sample_steps = params.sample_steps;
    gen_params.sample_params.eta = params.eta;
    if (params.flow_shift != 0.0f) {
        gen_params.sample_params.flow_shift = params.flow_shift;
    }
    gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;

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

    // 禁用内置 hires
    gen_params.hires.enabled = false;

    // VAE Tiling
    gen_params.vae_tiling_params.enabled = params.vae_tiling;
    if (params.vae_tiling) {
        gen_params.vae_tiling_params.tile_size_x = params.vae_tile_size_x;
        gen_params.vae_tiling_params.tile_size_y = params.vae_tile_size_y;
        gen_params.vae_tiling_params.target_overlap = params.vae_tile_overlap;
    }

    // 2. 生成基础 latent
    sd_tensor_t* base_latent = sd_ext_generate_latent(ctx_, &gen_params);
    if (!base_latent) {
        std::cerr << "[SDCPPAdapter] Failed to generate base latent" << std::endl;
        return Image();
    }

    // 获取 tensor 信息
    int ndim = sd_ext_tensor_ndim(base_latent);
    if (ndim != 3) {
        std::cerr << "[SDCPPAdapter] Unexpected latent ndim: " << ndim << std::endl;
        sd_ext_tensor_free(base_latent);
        return Image();
    }

    int64_t h = sd_ext_tensor_shape(base_latent, 0);
    int64_t w = sd_ext_tensor_shape(base_latent, 1);
    int64_t c = sd_ext_tensor_shape(base_latent, 2);

    std::cout << "[SDCPPAdapter] Base latent shape: [" << h << ", " << w << ", " << c << "]" << std::endl;
    std::cout << "[SDCPPAdapter] Upscaling to target: " << params.hires_width << "x" << params.hires_height << std::endl;

    // 3. 转换为 libTorch tensor
    void* data_ptr = sd_ext_tensor_data_ptr(base_latent);
    if (!data_ptr) {
        std::cerr << "[SDCPPAdapter] Failed to get latent data pointer" << std::endl;
        sd_ext_tensor_free(base_latent);
        return Image();
    }

    // sd.cpp latent 格式: [H, W, C] (HWC)
    // 转换为 torch: [C, H, W] 再添加 batch -> [1, C, H, W]
    auto cpu_tensor = torch::from_blob(data_ptr, {h, w, c}, torch::kFloat32);
    auto torch_latent = cpu_tensor.permute({2, 0, 1}).unsqueeze(0).to(torch::kCUDA);

    // 计算目标 latent 尺寸（VAE 缩放因子通常为 8）
    int64_t target_h = params.hires_height / 8;
    int64_t target_w = params.hires_width / 8;

    // 4. libTorch 上采样
    torch::nn::functional::InterpolateFuncOptions options;
    options.size(std::vector<int64_t>{target_h, target_w});

    switch (params.hires_upscaler) {
        case HiresUpscaler::LatentNearest:
        case HiresUpscaler::LatentNearestExact:
        case HiresUpscaler::Nearest:
            options.mode(torch::kNearest);
            break;
        case HiresUpscaler::LatentBicubic:
        case HiresUpscaler::LatentBicubicAntialiased:
            options.mode(torch::kBicubic);
            break;
        case HiresUpscaler::Lanczos:
        case HiresUpscaler::Latent:
        case HiresUpscaler::LatentAntialiased:
        case HiresUpscaler::Model:
        default:
            options.mode(torch::kBilinear);
            options.align_corners(false);
            break;
    }

    auto upscaled = torch::nn::functional::interpolate(torch_latent, options);

    // 转回 [H, W, C] 格式并移回 CPU
    auto upscaled_hwc = upscaled.squeeze(0).permute({1, 2, 0}).contiguous().to(torch::kCPU);

    // 5. 转换回 sd tensor
    std::vector<int64_t> target_shape = {target_h, target_w, c};
    sd_tensor_t* upscaled_sd = sd_ext_tensor_from_data(
        upscaled_hwc.data_ptr<float>(),
        target_shape.data(),
        3,
        0);  // f32

    // 释放基础 latent
    sd_ext_tensor_free(base_latent);

    if (!upscaled_sd) {
        std::cerr << "[SDCPPAdapter] Failed to create upscaled sd tensor" << std::endl;
        return Image();
    }

    std::cout << "[SDCPPAdapter] Upscaled latent shape: [" << target_h << ", " << target_w << ", " << c << "]" << std::endl;
    std::cout << "[SDCPPAdapter] Sampling with strength=" << params.hires_strength << ", steps=" << params.hires_sample_steps << std::endl;

    // 6. 构造采样参数并继续采样
    sd_sample_params_t sample_params;
    sd_sample_params_init(&sample_params);
    sample_params.sample_method = convert_sample_method(params.sample_method);
    sample_params.scheduler = convert_scheduler(params.scheduler);
    sample_params.sample_steps = params.hires_sample_steps;
    sample_params.eta = params.eta;
    sample_params.guidance.txt_cfg = params.cfg_scale;
    if (params.flow_shift != 0.0f) {
        sample_params.flow_shift = params.flow_shift;
    }

    sd_tensor_t* refined_latent = sd_ext_sample_latent(
        ctx_,
        upscaled_sd,
        nullptr,  // noise: 让 sd.cpp 内部生成并混合
        params.prompt.c_str(),
        params.negative_prompt.c_str(),
        &sample_params,
        params.hires_width,
        params.hires_height,
        params.hires_strength);

    sd_ext_tensor_free(upscaled_sd);

    if (!refined_latent) {
        std::cerr << "[SDCPPAdapter] HiRes sampling failed" << std::endl;
        return Image();
    }

    // 7. VAE 解码
    sd_image_t sd_img = sd_ext_vae_decode(ctx_, refined_latent);
    sd_ext_tensor_free(refined_latent);

    if (!sd_img.data) {
        std::cerr << "[SDCPPAdapter] VAE decode failed" << std::endl;
        return Image();
    }

    // 8. 转换为 Image 并返回
    Image result = sd_image_to_image(sd_img);
    free(sd_img.data);

    std::cout << "[SDCPPAdapter] libTorch HiRes Fix completed: " << result.width << "x" << result.height << std::endl;
    return result;
#else
    (void)params;
    std::cerr << "[SDCPPAdapter] libTorch HiRes Fix not available (compiled without MYIMG_ENABLE_LIBTORCH)" << std::endl;
    std::cerr << "              Falling back to standard generate_single()" << std::endl;
    return generate_single(params);
#endif
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
    int success = 0;
    std::string ext = "";
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        for (auto& c : ext) c = std::tolower(c);
    }
    if (ext == "bmp") {
        success = stbi_write_bmp(path.c_str(), width, height, channels, data.data());
    } else if (ext == "tga") {
        success = stbi_write_tga(path.c_str(), width, height, channels, data.data());
    } else if (ext == "webp") {
        // Use libwebp for WebP output
        WebPConfig config;
        WebPPicture picture;
        if (!WebPConfigInit(&config) || !WebPPictureInit(&picture)) {
            std::cerr << "[Image] Failed to init WebP" << std::endl;
            return false;
        }
        config.quality = g_jpeg_quality; // Reuse quality setting (0-100)
        picture.width = width;
        picture.height = height;
        picture.use_argb = 1;
        if (!WebPPictureAlloc(&picture)) {
            std::cerr << "[Image] Failed to alloc WebP picture" << std::endl;
            return false;
        }
        
        // Import RGB/RGBA data
        if (channels == 3) {
            WebPPictureImportRGB(&picture, data.data(), width * channels);
        } else if (channels == 4) {
            WebPPictureImportRGBA(&picture, data.data(), width * channels);
        } else {
            // Convert grayscale to RGB
            std::vector<uint8_t> rgb_data(width * height * 3);
            for (int i = 0; i < width * height; ++i) {
                rgb_data[i * 3] = data[i];
                rgb_data[i * 3 + 1] = data[i];
                rgb_data[i * 3 + 2] = data[i];
            }
            WebPPictureImportRGB(&picture, rgb_data.data(), width * 3);
        }
        
        WebPMemoryWriter writer;
        WebPMemoryWriterInit(&writer);
        picture.writer = WebPMemoryWrite;
        picture.custom_ptr = &writer;
        
        if (!WebPEncode(&config, &picture)) {
            std::cerr << "[Image] Failed to encode WebP" << std::endl;
            WebPMemoryWriterClear(&writer);
            WebPPictureFree(&picture);
            return false;
        }
        
        FILE* fp = fopen(path.c_str(), "wb");
        if (fp) {
            fwrite(writer.mem, 1, writer.size, fp);
            fclose(fp);
            success = 1;
        }
        
        WebPMemoryWriterClear(&writer);
        WebPPictureFree(&picture);
    } else if (ext == "jpg" || ext == "jpeg") {
        // Use libjpeg for JPEG output
        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;
        FILE* outfile = fopen(path.c_str(), "wb");
        if (!outfile) {
            std::cerr << "[Image] Failed to open file for writing: " << path << std::endl;
            return false;
        }
        
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, outfile);
        
        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = channels;
        cinfo.in_color_space = (channels == 3) ? JCS_RGB : JCS_GRAYSCALE;
        
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, g_jpeg_quality, TRUE);
        jpeg_start_compress(&cinfo, TRUE);
        
        int row_stride = width * channels;
        const uint8_t* image_data = data.data();
        while (cinfo.next_scanline < cinfo.image_height) {
            JSAMPROW row_pointer = (JSAMPROW)&image_data[cinfo.next_scanline * row_stride];
            jpeg_write_scanlines(&cinfo, &row_pointer, 1);
        }
        
        jpeg_finish_compress(&cinfo);
        fclose(outfile);
        jpeg_destroy_compress(&cinfo);
        success = 1;
    } else {
        success = stbi_write_png(path.c_str(), width, height, channels, data.data(), width * channels);
    }
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
