// ============================================================================
// stable-diffusion-ext.h
// ============================================================================
// 
// my-img 项目扩展头文件。
// 
// 这个文件应该被复制到 stable-diffusion.cpp/include/ 目录下，
// 作为 stable-diffusion.h 的补充。
// 
// 它声明了 Deep HighRes Fix 所需的 C++ hook 接口，
// 以及真正的 ComfyUI 风格节点级 API（分离的 encode/sample/decode）。
// 
// 注意：这个头文件是 C++ only 的，不能用于纯 C 编译单元。
// ============================================================================

#pragma once

#include "stable-diffusion.h"
#include <vector>

// 前向声明 sd::Tensor
namespace sd {
template <typename T>
class Tensor;
}

// ============================================================================
// 原有的 Deep HighRes Fix Hook API
// ============================================================================

// Latent hook 类型定义
typedef sd::Tensor<float> (*sd_latent_hook_t)(
    sd::Tensor<float>& latent,
    int step,
    int total_steps,
    void* user_data);

// Guidance hook 类型定义
typedef void (*sd_guidance_hook_t)(
    float* txt_cfg,
    float* img_cfg,
    float* distilled_guidance,
    int step,
    int total_steps,
    void* user_data);

SD_API void sd_set_latent_hook(sd_latent_hook_t hook, void* user_data);
SD_API void sd_clear_latent_hook();
SD_API void sd_set_guidance_hook(sd_guidance_hook_t hook, void* user_data);
SD_API void sd_clear_guidance_hook();

// Deep HighRes Fix 参数结构
typedef struct {
    const char* prompt;
    const char* negative_prompt;
    float cfg_scale;
    int sample_method;
    int scheduler;
    int64_t seed;
    int total_steps;
    int target_width;
    int target_height;
    bool vae_tiling;
    sd_image_t init_image;
    float strength;
    float phase1_cfg_scale;
    float phase2_cfg_scale;
    float phase3_cfg_scale;
} sd_deep_hires_params_t;

SD_API void sd_deep_hires_params_init(sd_deep_hires_params_t* params);
SD_API sd_image_t* generate_image_deep_hires(
    sd_ctx_t* sd_ctx,
    const sd_deep_hires_params_t* params
);

// ============================================================================
// 新的 ComfyUI 风格节点级 API
// ============================================================================

// Opaque handle types
typedef struct sd_latent_t sd_latent_t;
typedef struct sd_conditioning_t sd_conditioning_t;

// ---------------------------------------------------------------------------
// CLIP Text Encoding
// ---------------------------------------------------------------------------

// 编码文本为 conditioning
// 返回的 sd_conditioning_t* 需要用 sd_free_conditioning() 释放
SD_API sd_conditioning_t* sd_encode_prompt(
    sd_ctx_t* sd_ctx,
    const char* prompt,
    int clip_skip
);

// 释放 conditioning
SD_API void sd_free_conditioning(sd_conditioning_t* cond);

// ---------------------------------------------------------------------------
// VAE Encode/Decode
// ---------------------------------------------------------------------------

// 将图像编码为 latent
// 返回的 sd_latent_t* 需要用 sd_free_latent() 释放
SD_API sd_latent_t* sd_encode_image(
    sd_ctx_t* sd_ctx,
    const sd_image_t* image
);

// 将 latent 解码为图像
// 返回的 sd_image_t* 需要用 sd_free_image() 释放（注意：不是 stbi_image_free）
SD_API sd_image_t* sd_decode_latent(
    sd_ctx_t* sd_ctx,
    const sd_latent_t* latent
);

// 创建空的 latent（用于 txt2img）
// 返回的 sd_latent_t* 需要用 sd_free_latent() 释放
SD_API sd_latent_t* sd_create_empty_latent(
    sd_ctx_t* sd_ctx,
    int width,
    int height
);

// 释放 latent
SD_API void sd_free_latent(sd_latent_t* latent);

// 释放 sd_image_t（sd_decode_latent 返回的）
SD_API void sd_free_image(sd_image_t* image);

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

// 节点级采样参数结构（注意：不要与 sd_sample_params_t 混淆）
typedef struct {
    int64_t seed;
    int steps;
    float cfg_scale;
    enum sample_method_t sample_method;
    enum scheduler_t scheduler;
    float eta;  // 用于某些采样器
} sd_node_sample_params_t;

// 执行采样
// init_latent: 初始 latent（可以是空 latent 或编码后的图像）
// positive: 正向 conditioning
// negative: 负向 conditioning
// strength: denoise 强度 (0.0-1.0)，1.0 = 完全重绘，0.0 = 不改变
// 返回的 sd_latent_t* 需要用 sd_free_latent() 释放
SD_API sd_latent_t* sd_sampler_run(
    sd_ctx_t* sd_ctx,
    const sd_latent_t* init_latent,
    const sd_conditioning_t* positive,
    const sd_conditioning_t* negative,
    const sd_node_sample_params_t* params,
    float strength
);

// ============================================================================
// 辅助函数
// ============================================================================

// 获取 latent 的尺寸信息
SD_API void sd_latent_get_shape(
    const sd_latent_t* latent,
    int* width,
    int* height,
    int* channels
);

// 复制 latent（用于缓存）
SD_API sd_latent_t* sd_latent_copy(const sd_latent_t* latent);

// 复制 conditioning（用于缓存）
SD_API sd_conditioning_t* sd_conditioning_copy(const sd_conditioning_t* cond);

// ============================================================================
// LoRA 支持
// ============================================================================

// 应用 LoRA 到模型上下文（在采样前调用）
// loras: LoRA 配置数组
// lora_count: 数组长度
SD_API void sd_apply_loras(
    sd_ctx_t* sd_ctx,
    const sd_lora_t* loras,
    uint32_t lora_count
);

// 清除已应用的所有 LoRA
SD_API void sd_clear_loras(sd_ctx_t* sd_ctx);
