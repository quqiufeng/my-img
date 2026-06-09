#include "adapters/sdcpp_adapter.h"
#include "utils/log.h"
#include "utils/prompt_schedule.h"
#include "utils/regional_prompting.h"
#include "utils/face_restoration.h"
#include "utils/face_swap.h"
#include "utils/latent_ops.h"
#include "utils/ipadapter.h"
#include "utils/image_postproc.h"
#include <stable-diffusion.h>

#include <iostream>
#include <cstring>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <numeric>
#include <algorithm>

#include <stb_image_write.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <webp/encode.h>
#include <webp/mux.h>

namespace fs = std::filesystem;

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
    std::cerr << "[SD " << level_str << "] " << text; // sd.cpp callback uses cerr
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
        case SampleMethod::EulerCfgPP: return EULER_CFG_PP_SAMPLE_METHOD;
        case SampleMethod::EulerACfgPP: return EULER_A_CFG_PP_SAMPLE_METHOD;
        case SampleMethod::EulerGE: return EULER_GE_SAMPLE_METHOD;
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
        case Scheduler::LTX2: return LTX2_SCHEDULER;
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
    if (wtype == "q1_0" || wtype == "Q1_0") return SD_TYPE_Q1_0;
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
    
    sd_ctx_params_t sd_params = {};
    sd_ctx_params_init(&sd_params);
    
    // 设置模型路径
    // 完整 checkpoint（如 SDXL Base safetensors）包含 UNet+VAE+CLIP，需要用 model_path 加载全部权重
    // diffusion-only 模型（如 Z-Image GGUF）只含 UNet，需要用 diffusion_model_path
    bool is_full_checkpoint = false;
    std::string model_file = !params.diffusion_model_path.empty() ? params.diffusion_model_path : params.model_path;
    if (!model_file.empty()) {
        // 快速检测 safetensors 文件是否包含 VAE/CLIP 权重
        std::ifstream fs(model_file, std::ios::binary);
        if (fs.good()) {
            uint64_t header_len = 0;
            fs.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
            if (header_len > 0 && header_len < 100 * 1024 * 1024) {  // header < 100MB
                std::string header(header_len, '\0');
                fs.read(header.data(), header_len);
                is_full_checkpoint = (header.find("first_stage_model") != std::string::npos) ||
                                     (header.find("conditioner.embedders") != std::string::npos) ||
                                     (header.find("cond_stage_model") != std::string::npos);
            }
        }
    }
    
    if (!params.diffusion_model_path.empty()) {
        if (is_full_checkpoint) {
            sd_params.model_path = params.diffusion_model_path.c_str();
            std::cout << "[SDCPPAdapter] Detected full checkpoint, using model_path (UNet+VAE+CLIP)" << std::endl;
        } else {
            sd_params.diffusion_model_path = params.diffusion_model_path.c_str();
        }
    } else if (!params.model_path.empty()) {
        sd_params.model_path = params.model_path.c_str();
    }
    
    sd_params.vae_path = params.vae_path.empty() ? nullptr : params.vae_path.c_str();
    sd_params.clip_l_path = params.clip_l_path.empty() ? nullptr : params.clip_l_path.c_str();
    sd_params.clip_g_path = params.clip_g_path.empty() ? nullptr : params.clip_g_path.c_str();
    sd_params.t5xxl_path = params.t5xxl_path.empty() ? nullptr : params.t5xxl_path.c_str();
    sd_params.llm_path = params.llm_path.empty() ? nullptr : params.llm_path.c_str();
    sd_params.control_net_path = params.control_net_path.empty() ? nullptr : params.control_net_path.c_str();
    sd_params.audio_vae_path = params.audio_vae_path.empty() ? nullptr : params.audio_vae_path.c_str();
    sd_params.embeddings_connectors_path = params.embeddings_connectors_path.empty() ? nullptr : params.embeddings_connectors_path.c_str();

    // IPAdapter UNet cross-attention injection (must be set at load time to create extra tensors)
    sd_params.ipadapter_unet_mode = !params.ipadapter_unet_weights_path.empty();
    sd_params.ipadapter_unet_weights_path = params.ipadapter_unet_weights_path.empty() ? nullptr
                                                                                       : params.ipadapter_unet_weights_path.c_str();

    // 系统设置
    sd_params.n_threads = params.n_threads > 0 ? params.n_threads : sd_get_num_physical_cores();
    sd_params.offload_params_to_cpu = params.offload_params_to_cpu;
    sd_params.enable_mmap = params.enable_mmap;
    sd_params.flash_attn = params.flash_attn;
    sd_params.diffusion_flash_attn = params.flash_attn;

    // 显存管理: 上游支持 max_vram (GiB, 0=禁用, -1=自动)
    sd_params.max_vram = params.max_vram;

    // 根据分辨率自动启用 CPU offloading
    // 20GB VRAM 的 RTX 3080 可轻松容纳 VAE(160MB) 和 LLM(1.5GB)，
    // 因此仅在极高分辨率（>4MP）时才 offload
    int64_t pixel_count = static_cast<int64_t>(params.width) * params.height;
    bool auto_vae_cpu = false;
    bool auto_clip_cpu = false;
    if (pixel_count > 2048 * 2048) {
        auto_vae_cpu = true;
    }
    if (pixel_count > 2560 * 2560) {
        auto_clip_cpu = true;
    }
    if (params.max_vram > 0 && params.max_vram < 10.0f) {
        auto_vae_cpu = true;
        auto_clip_cpu = true;
        sd_params.offload_params_to_cpu = true;
        LOG_INFO("User VRAM limit %.1fGB, enabling aggressive CPU offloading", params.max_vram);
    }

    // 权重类型
    if (!params.wtype.empty() && params.wtype != "default") {
        sd_params.wtype = convert_wtype(params.wtype);
    }

    // VAE 格式
    if (params.vae_format == "flux") {
        sd_params.vae_format = SD_VAE_FORMAT_FLUX;
    } else if (params.vae_format == "sd3") {
        sd_params.vae_format = SD_VAE_FORMAT_SD3;
    } else if (params.vae_format == "flux2") {
        sd_params.vae_format = SD_VAE_FORMAT_FLUX2;
    } else {
        sd_params.vae_format = SD_VAE_FORMAT_AUTO;
    }

    // 后端选择
    sd_params.backend = params.backend.empty() ? nullptr : params.backend.c_str();
    sd_params.params_backend = params.params_backend.empty() ? nullptr : params.params_backend.c_str();

    // VAE 设置
    // txt2img 只需要 decoder；img2img / hires 需要 encoder
    sd_params.vae_decode_only = params.init_image.empty();
    sd_params.keep_vae_on_cpu = auto_vae_cpu;
    sd_params.keep_clip_on_cpu = auto_clip_cpu;
    
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
        LOG_ERROR("Failed to load model after %ld seconds", duration);
        return false;
    }
    
    std::cout << "[SDCPPAdapter] Model loaded successfully in " << duration << " seconds" << std::endl;
    
    // 设置进度回调
    if (progress_callback_) {
        sd_set_progress_callback(progress_callback_wrapper, this);
    }
    
    // 设置预览回调
    if (preview_callback_) {
        preview_t preview_mode = PREVIEW_NONE;
        if (preview_mode_ == "proj") preview_mode = PREVIEW_PROJ;
        else if (preview_mode_ == "tae") preview_mode = PREVIEW_TAE;
        else if (preview_mode_ == "vae") preview_mode = PREVIEW_VAE;
        
        sd_set_preview_callback(preview_callback_wrapper, preview_mode, preview_interval_, true, false, this);
    }
    
    return true;
}

std::vector<Image> SDCPPAdapter::generate(const GenerationParams& params) {
    std::vector<Image> images;
    
    if (!ctx_) {
        LOG_ERROR("Model not initialized!");
        return images;
    }
    
    // 分发到高级生成模式
    if (!params.prompt_schedule.empty() && params.batch_count == 1) {
        LOG_INFO("Using Prompt Schedule generation mode");
        Image img = generate_with_schedule(params);
        if (!img.empty()) {
            images.push_back(img);
        }
        return images;
    }
    
    if (!params.regional_prompts.empty() && params.batch_count == 1) {
        LOG_INFO("Using Regional Prompting generation mode");
        Image img = generate_with_regional_prompts(params);
        if (!img.empty()) {
            images.push_back(img);
        }
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
    
    // img2img (使用 RAII 管理临时 sd_image_t 内存)
    SDImageGuard init_image_guard;
    if (!params.init_image.empty() && params.strength < 1.0f) {
        init_image_guard = image_to_sd_image(params.init_image);
        gen_params.init_image = *init_image_guard.get();
    }
    
    // Inpainting
    SDImageGuard mask_image_guard;
    if (!params.mask_image.empty()) {
        mask_image_guard = image_to_sd_image(params.mask_image);
        gen_params.mask_image = *mask_image_guard.get();
    }
    
    // ControlNet
    SDImageGuard control_image_guard;
    if (!params.control_image.empty()) {
        control_image_guard = image_to_sd_image(params.control_image);
        gen_params.control_image = *control_image_guard.get();
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
    
    // 采样器额外参数
    gen_params.sample_params.extra_sample_args = params.extra_sample_args.empty() ? nullptr : params.extra_sample_args.c_str();

    // Prompt Schedule: 解析并使用第一个 prompt
    PromptSchedule schedule;
    std::string effective_prompt = params.prompt;
    std::string effective_negative = params.negative_prompt;
    if (!params.prompt_schedule.empty()) {
        if (schedule.parse(params.prompt_schedule)) {
            auto first_entry = schedule.get_entry(0, params.sample_steps);
            if (first_entry.has_value()) {
                effective_prompt = first_entry->prompt;
                if (!first_entry->negative_prompt.empty()) {
                    effective_negative = first_entry->negative_prompt;
                }
                if (first_entry->cfg_scale > 0) {
                    gen_params.sample_params.guidance.txt_cfg = first_entry->cfg_scale;
                }
                std::cout << "  Prompt Schedule: using step 0 prompt ("
                          << schedule.entries().size() << " total entries)" << std::endl;
            }
        } else {
            LOG_WARN("Failed to parse prompt schedule: %s", params.prompt_schedule.c_str());
        }
    }
    gen_params.prompt = effective_prompt.c_str();
    gen_params.negative_prompt = effective_negative.c_str();

    // FreeU
    gen_params.freeu.enabled = params.freeu_enabled;
    if (params.freeu_enabled) {
        gen_params.freeu.b1 = params.freeu_b1;
        gen_params.freeu.b2 = params.freeu_b2;
        gen_params.freeu.s1 = params.freeu_s1;
        gen_params.freeu.s2 = params.freeu_s2;
        std::cout << "  FreeU: enabled (b1=" << params.freeu_b1 << ", b2=" << params.freeu_b2
                  << ", s1=" << params.freeu_s1 << ", s2=" << params.freeu_s2 << ")" << std::endl;
    }

    // SAG
    gen_params.sag.enabled = params.sag_enabled;
    if (params.sag_enabled) {
        gen_params.sag.scale = params.sag_scale;
        std::cout << "  SAG: enabled (scale=" << params.sag_scale << ")" << std::endl;
    }

    // VAE Tiling
    gen_params.vae_tiling_params.enabled = params.vae_tiling;
    gen_params.vae_tiling_params.temporal_tiling = params.vae_temporal_tiling;
    gen_params.vae_tiling_params.extra_tiling_args = params.extra_tiling_args.empty() ? nullptr : params.extra_tiling_args.c_str();
    if (params.vae_tiling) {
        gen_params.vae_tiling_params.tile_size_x = params.vae_tile_size_x;
        gen_params.vae_tiling_params.tile_size_y = params.vae_tile_size_y;
        gen_params.vae_tiling_params.target_overlap = params.vae_tile_overlap;
        std::cout << "  VAE Tiling: " << params.vae_tile_size_x << "x" << params.vae_tile_size_y
                  << " (overlap: " << params.vae_tile_overlap << ")" << std::endl;
        if (params.vae_temporal_tiling) {
            std::cout << "  VAE Temporal Tiling: enabled" << std::endl;
        }
    }

    // Advanced Features Logging
    if (!params.prompt_schedule.empty()) {
        std::cout << "  Prompt Schedule: " << params.prompt_schedule << std::endl;
    }
    if (!params.regional_prompts.empty()) {
        std::cout << "  Regional Prompts: " << params.regional_prompts << std::endl;
    }
    if (params.face_restoration) {
        std::cout << "  Face Restoration: enabled (model: " << params.face_restore_model << ")" << std::endl;
    }
    if (params.ipadapter) {
        std::cout << "  IPAdapter: enabled (weight: " << params.ipadapter_weight << ")" << std::endl;
        std::cout << "  IPAdapter: loading models..." << std::endl;
        try {
            IPAdapterConfig ipa_config;
            ipa_config.model_path = params.ipadapter_model;
            ipa_config.clip_vision_path = params.ipadapter_clip_vision;
            ipa_config.image_path = params.ipadapter_image;
            ipa_config.weight = params.ipadapter_weight;

            ipadapter_ = std::make_unique<IPAdapter>(ipa_config);

            if (ipadapter_ && ipadapter_->is_loaded()) {
                auto& tokens = ipadapter_->get_image_tokens();
                std::cout << "  IPAdapter: image tokens computed, size="
                          << tokens.size() << " floats" << std::endl;
                if (!tokens.empty()) {
                    float min_val = *std::min_element(tokens.begin(), tokens.end());
                    float max_val = *std::max_element(tokens.begin(), tokens.end());
                    float sum = std::accumulate(tokens.begin(), tokens.end(), 0.0f);
                    float mean_val = sum / tokens.size();
                    std::cout << "  IPAdapter: token stats: min=" << min_val
                              << " max=" << max_val << " mean=" << mean_val << std::endl;
                }
            } else {
                std::cout << "  IPAdapter: model loading failed ("
                          << (params.ipadapter_model.empty() ? "no model" : params.ipadapter_model)
                          << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            LOG_WARN("IPAdapter initialization failed: %s", e.what());
        }
    }
    // Pass IPAdapter tokens to sd_img_gen_params_t (for UNet cross-attention injection)
    gen_params.ipadapter_unet_mode = !params.ipadapter_unet_weights_path.empty();
    gen_params.ipadapter_unet_weights_path = params.ipadapter_unet_weights_path.empty() ? nullptr
                                                                                       : params.ipadapter_unet_weights_path.c_str();
    if (ipadapter_ && ipadapter_->is_loaded() && !ipadapter_->get_image_tokens().empty()) {
        const auto& tokens = ipadapter_->get_image_tokens();
        int n_tokens = ipadapter_->get_num_tokens();
        gen_params.ipadapter_tokens     = tokens.data();
        gen_params.ipadapter_num_tokens = n_tokens;
        std::cout << "  IPAdapter: passing " << tokens.size()
                  << " floats (" << n_tokens << " tokens x 2048-dim) to sd.cpp UNet"
                  << std::endl;
        gen_params.ipadapter_weight    = params.ipadapter_weight;
    } else {
        gen_params.ipadapter_tokens    = nullptr;
        gen_params.ipadapter_num_tokens = 0;
        gen_params.ipadapter_weight    = 0.0f;
    }

    if (params.t2i_adapter) {
        std::cout << "  T2I-Adapter: enabled (strength: " << params.t2i_adapter_strength << ")" << std::endl;
    }
    if (params.face_swap) {
        std::cout << "  Face Swap: enabled" << std::endl;
    }
    if (params.photo_maker) {
        std::cout << "  PhotoMaker: enabled (id images: " << params.photo_maker_id_images.size() << ")" << std::endl;
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
    SDImageArrayGuard sd_images(generate_image(ctx_, &gen_params));
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "[SDCPPAdapter] Generation completed in " << duration << " seconds" << std::endl;
    
    if (!sd_images.empty()) {
        for (int i = 0; i < params.batch_count; i++) {
            Image img = sd_image_to_image(sd_images[i]);
            
            // Face Restoration (post-processing)
            if (params.face_restoration && !params.face_restore_model.empty()) {
                try {
                    FaceRestorationConfig fr_config;
                    fr_config.model_path = params.face_restore_model;
                    fr_config.fidelity = params.face_restore_fidelity;
                    FaceRestoration fr(fr_config);
                    if (fr.is_loaded()) {
                        ImageData img_data;
                        img_data.width = img.width;
                        img_data.height = img.height;
                        img_data.channels = img.channels;
                        img_data.data = img.data;
                        auto restored = fr.restore_faces(img_data);
                        img.data = restored.data;
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Face restoration failed: %s", e.what());
                }
            }
            
            // Face Swap (post-processing)
            if (params.face_swap && !params.face_swap_source.empty()) {
                try {
                    FaceSwapConfig fs_config;
                    fs_config.source_image = params.face_swap_source;
                    fs_config.detection_model = params.face_swap_detection_model;
                    fs_config.swap_model = params.face_swap_model;
                    FaceSwap fs(fs_config);
                    if (fs.is_loaded()) {
                        ImageData target_data;
                        target_data.width = img.width;
                        target_data.height = img.height;
                        target_data.channels = img.channels;
                        target_data.data = img.data;
                        
                        auto source_data = load_image_from_file(params.face_swap_source);
                        if (!source_data.empty()) {
                            auto swapped = fs.swap_faces(source_data, target_data);
                            img.data = swapped.data;
                        }
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Face swap failed: %s", e.what());
                }
            }
            
            images.push_back(img);
        }
    }
    
    // Regional Prompting: post-process by generating masked regions
    if (!params.regional_prompts.empty() && !images.empty()) {
        try {
            RegionalPromptManager regional_mgr;
            if (regional_mgr.parse(params.regional_prompts)) {
                LOG_INFO("Applying regional prompting to %zu images...", images.size());
                // TODO: Generate per-region variations and composite
                // For now, log that this feature needs further integration
            }
        } catch (const std::exception& e) {
            LOG_WARN("Regional prompting failed: %s", e.what());
        }
    }
    
    // 图像后处理 (clarity / sharpen / smart-sharpen / edge-sharpen)
    {
        PostProcessParams pp;
        pp.clarity = params.postproc_clarity;
        pp.sharpen_amount = params.postproc_sharpen_amount;
        pp.sharpen_radius = params.postproc_sharpen_radius;
        pp.sharpen_threshold = params.postproc_sharpen_threshold;
        pp.smart_sharpen_strength = params.postproc_smart_sharpen_strength;
        pp.smart_sharpen_radius = params.postproc_smart_sharpen_radius;
        pp.edge_sharpen_amount = params.postproc_edge_sharpen_amount;
        pp.edge_sharpen_radius = params.postproc_edge_sharpen_radius;
        pp.edge_sharpen_threshold = params.postproc_edge_sharpen_threshold;

        bool has_work = (pp.clarity > 0.0f ||
                         pp.sharpen_amount > 0.0f ||
                         pp.smart_sharpen_strength > 0.0f ||
                         pp.edge_sharpen_amount > 0.0f);
        if (has_work) {
            LOG_INFO("Applying image post-processing to %zu images...", images.size());
            for (auto& img : images) {
                apply_image_postprocessing(img, pp);
            }
        }
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

// ============================================================
// Prompt Schedule 多阶段生成
// ============================================================
Image SDCPPAdapter::generate_with_schedule(const GenerationParams& params) {
    if (!ctx_) {
        LOG_ERROR("Model not initialized!");
        return Image();
    }
    
    PromptSchedule schedule;
    if (!schedule.parse(params.prompt_schedule)) {
        LOG_WARN("Failed to parse prompt schedule, falling back to normal generation");
        return generate_single(params);
    }
    
    const auto& entries = schedule.entries();
    if (entries.empty()) {
        return generate_single(params);
    }
    
    LOG_INFO("========================================");
    LOG_INFO("  Prompt Schedule Generation");
    LOG_INFO("  Total stages: %zu", entries.size());
    LOG_INFO("========================================");
    
    Image current_image;
    
    for (size_t stage = 0; stage < entries.size(); stage++) {
        const auto& entry = entries[stage];
        LOG_INFO("Stage %zu/%zu: steps %d-%d, prompt: %s",
                 stage + 1, entries.size(), entry.start_step, entry.end_step,
                 entry.prompt.c_str());
        
        GenerationParams stage_params = params;
        stage_params.prompt = entry.prompt;
        stage_params.prompt_schedule.clear(); // 防止递归！
        if (!entry.negative_prompt.empty()) {
            stage_params.negative_prompt = entry.negative_prompt;
        }
        if (entry.cfg_scale > 0) {
            stage_params.cfg_scale = entry.cfg_scale;
        }
        
        // 计算阶段步数
        int stage_steps = entry.end_step - entry.start_step;
        if (stage_steps <= 0) {
            stage_steps = params.sample_steps / entries.size();
        }
        stage_params.sample_steps = std::min(stage_steps, params.sample_steps);
        
        // 如果不是第一阶段，使用上一阶段的输出作为 init_image
        if (stage > 0 && !current_image.empty()) {
            stage_params.init_image = current_image;
            // 渐进式降低 strength：早期阶段变化大，后期阶段变化小
            float progress = static_cast<float>(stage) / entries.size();
            stage_params.strength = 0.3f + 0.4f * (1.0f - progress);
            stage_params.sample_steps = std::max(stage_params.sample_steps, 15);
            LOG_INFO("  Using previous output as init (strength: %.2f)", stage_params.strength);
        }
        
        stage_params.batch_count = 1;
        auto images = generate(stage_params);
        if (!images.empty()) {
            current_image = images[0];
        } else {
            LOG_ERROR("Stage %zu failed, aborting schedule generation", stage + 1);
            break;
        }
    }
    
    LOG_INFO("Prompt schedule generation completed");
    return current_image;
}

// ============================================================
// Regional Prompting 分区生成
// ============================================================
Image SDCPPAdapter::generate_with_regional_prompts(const GenerationParams& params) {
    if (!ctx_) {
        LOG_ERROR("Model not initialized!");
        return Image();
    }
    
    RegionalPromptManager regional_mgr;
    if (!regional_mgr.parse(params.regional_prompts)) {
        LOG_WARN("Failed to parse regional prompts, falling back to normal generation");
        return generate_single(params);
    }
    
    const auto& regions = regional_mgr.regions();
    if (regions.empty()) {
        return generate_single(params);
    }
    
    LOG_INFO("========================================");
    LOG_INFO("  Regional Prompting Generation");
    LOG_INFO("  Total regions: %zu", regions.size());
    LOG_INFO("========================================");
    
    // 阶段 1: 生成基础图（使用所有区域 prompt 的组合）
    std::string combined_prompt = params.prompt;
    for (const auto& region : regions) {
        if (!region.prompt.empty()) {
            combined_prompt += ", " + region.prompt;
        }
    }
    
    GenerationParams base_params = params;
    base_params.prompt = combined_prompt;
    base_params.regional_prompts.clear(); // 防止递归！
    base_params.batch_count = 1;
    LOG_INFO("Generating base image with combined prompt...");
    auto base_images = generate(base_params);
    if (base_images.empty()) {
        LOG_ERROR("Base image generation failed");
        return Image();
    }
    Image result = base_images[0];
    
    // 阶段 2: 对每个区域生成变体
    for (size_t i = 0; i < regions.size(); i++) {
        const auto& region = regions[i];
        if (region.prompt.empty()) continue;
        
        LOG_INFO("Generating region %zu/%zu: %s (%.0f%% of image)",
                 i + 1, regions.size(), region.prompt.c_str(), region.weight * 100);
        
        GenerationParams region_params = params;
        region_params.prompt = region.prompt;
        if (!region.negative_prompt.empty()) {
            region_params.negative_prompt = region.negative_prompt;
        }
        
        // 使用基础图作为 init_image
        region_params.init_image = result;
        region_params.strength = region.weight;
        region_params.batch_count = 1;
        region_params.sample_steps = std::max(params.sample_steps / 2, 15);
        
        auto region_images = generate(region_params);
        if (region_images.empty()) {
            LOG_WARN("Region %zu generation failed, skipping", i + 1);
            continue;
        }
        
        // 使用 Latent Composite 合成
        try {
            auto base_tensor = image_data_to_tensor({result.width, result.height, result.channels, result.data});
            auto region_tensor = image_data_to_tensor({region_images[0].width, region_images[0].height, region_images[0].channels, region_images[0].data});
            
            // 生成区域 mask
            int latent_w = result.width / 8;
            int latent_h = result.height / 8;
            auto masks = regional_mgr.generate_masks(latent_w, latent_h);
            
            if (i < masks.size()) {
                // 将 mask 上采样到图像尺寸
                auto mask = masks[i];
                mask = torch::nn::functional::interpolate(
                    mask.unsqueeze(0),
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{result.height, result.width})
                        .mode(torch::kBilinear)
                        .align_corners(false)
                ).squeeze(0);
                
                // 扩展到 3 通道
                mask = mask.expand({3, result.height, result.width});
                
                // 混合: result = base * (1 - mask) + region * mask
                auto blended = base_tensor * (1.0f - mask) + region_tensor * mask;
                
                auto blended_data = tensor_to_image_data(blended);
                result.data = blended_data.data;
                LOG_INFO("Region %zu composited successfully", i + 1);
            }
        } catch (const std::exception& e) {
            LOG_WARN("Region %zu compositing failed: %s", i + 1, e.what());
        }
    }
    
    LOG_INFO("Regional prompting generation completed");
    return result;
}

std::vector<float> SDCPPAdapter::encode_prompt(const std::string& /*prompt*/, int /*clip_skip*/) {
    // TODO: 实现文本编码，返回 conditioning
    // 这需要更底层的 API，可能需要修改 sd.cpp 暴露更多接口
    LOG_WARN("encode_prompt not yet implemented");
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

void SDCPPAdapter::set_preview_callback(PreviewCallback callback, int interval, const std::string& mode) {
    preview_callback_ = callback;
    preview_interval_ = interval;
    preview_mode_ = mode;
    
    if (ctx_) {
        preview_t preview_mode = PREVIEW_NONE;
        if (mode == "proj") preview_mode = PREVIEW_PROJ;
        else if (mode == "tae") preview_mode = PREVIEW_TAE;
        else if (mode == "vae") preview_mode = PREVIEW_VAE;
        
        sd_set_preview_callback(preview_callback_wrapper, preview_mode, interval, true, false, this);
    }
}

void SDCPPAdapter::preview_callback_wrapper(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
    auto* adapter = static_cast<SDCPPAdapter*>(data);
    if (adapter && adapter->preview_callback_ && frames && frame_count > 0) {
        Image img = sd_image_to_image(frames[0]);
        adapter->preview_callback_(step, img, is_noisy);
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

SDImageGuard SDCPPAdapter::image_to_sd_image(const Image& img) {
    sd_image_t sd_img;
    sd_img.width = img.width;
    sd_img.height = img.height;
    sd_img.channel = img.channels;
    
    size_t data_size = img.width * img.height * img.channels;
    sd_img.data = static_cast<uint8_t*>(std::malloc(data_size));
    if (!sd_img.data) {
        LOG_FATAL("Failed to allocate %zu bytes for sd_image_t", data_size);
        return SDImageGuard();  // Return empty guard
    }
    std::memcpy(sd_img.data, img.data.data(), data_size);
    
    return SDImageGuard(sd_img);
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
// Image 保存辅助函数
// ============================================================
namespace {

bool save_webp_internal(const std::string& path, int width, int height, int channels,
                        const std::vector<uint8_t>& data, int quality) {
    WebPConfig config;
    WebPPicture picture;
    if (!WebPConfigInit(&config) || !WebPPictureInit(&picture)) {
        LOG_ERROR("Failed to init WebP");
        return false;
    }
    config.quality = quality;
    picture.width = width;
    picture.height = height;
    picture.use_argb = 1;
    if (!WebPPictureAlloc(&picture)) {
        LOG_ERROR("Failed to alloc WebP picture");
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
        LOG_ERROR("Failed to encode WebP");
        WebPMemoryWriterClear(&writer);
        WebPPictureFree(&picture);
        return false;
    }

    FILE* fp = fopen(path.c_str(), "wb");
    bool success = false;
    if (fp) {
        fwrite(writer.mem, 1, writer.size, fp);
        fclose(fp);
        success = true;
    }

    WebPMemoryWriterClear(&writer);
    WebPPictureFree(&picture);
    return success;
}

bool save_jpeg_internal(const std::string& path, int width, int height, int channels,
                        const std::vector<uint8_t>& data, int quality) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE* outfile = fopen(path.c_str(), "wb");
    if (!outfile) {
        LOG_ERROR("Failed to open file for writing: %s", path.c_str());
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
    jpeg_set_quality(&cinfo, quality, TRUE);
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
    return true;
}

} // anonymous namespace

// ============================================================
// Image 保存
// ============================================================
bool Image::save_to_file(const std::string& path) const {
    if (empty()) {
        LOG_ERROR("Cannot save empty image");
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
        success = save_webp_internal(path, width, height, channels, data, jpeg_quality);
    } else if (ext == "jpg" || ext == "jpeg") {
        success = save_jpeg_internal(path, width, height, channels, data, jpeg_quality);
    } else {
        success = stbi_write_png(path.c_str(), width, height, channels, data.data(), width * channels);
    }
    if (success) {
        std::cout << "[Image] Saved to " << path << " (" << width << "x" << height << ")" << std::endl;
        return true;
    } else {
        LOG_ERROR("Failed to save to %s", path.c_str());
        return false;
    }
}

// ============================================================
// ESRGAN 放大
// ============================================================
Image SDCPPAdapter::upscale_with_esrgan(const Image& image, const std::string& model_path, int repeats, int tile_size) {
    if (image.empty()) {
        LOG_ERROR("Cannot upscale empty image");
        return Image();
    }
    
    upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(model_path.c_str(), false, false, -1, tile_size, nullptr, nullptr);
    if (!upscaler_ctx) {
        LOG_ERROR("Failed to load upscaler model: %s", model_path.c_str());
        return Image();
    }
    
    SDImageGuard sd_img_guard = image_to_sd_image(image);
    Image result = image;
    
    for (int i = 0; i < repeats; ++i) {
        sd_image_t upscaled = upscale(upscaler_ctx, *sd_img_guard.get(), 4);  // ESRGAN 默认 4x
        if (upscaled.data == nullptr) {
            LOG_ERROR("Upscale failed at iteration %d", i + 1);
            break;
        }
        sd_img_guard = SDImageGuard(upscaled);
        result = sd_image_to_image(*sd_img_guard.get());
    }
    
    free_upscaler_ctx(upscaler_ctx);
    
    return result;
}

} // namespace myimg
