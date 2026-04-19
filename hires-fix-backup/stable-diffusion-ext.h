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

// 包含 Tensor 定义
#include "tensor.hpp"

// 暴露 sd_latent_t 定义（sd-engine 需要访问 tensor 成员）
struct sd_latent_t {
    sd::Tensor<float> tensor;
};

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
    sd_image_t control_image;  // ControlNet 输入图像（可选，data 为空表示不使用）
    float control_strength;    // ControlNet 强度
    sd_image_t mask_image;     // Inpaint mask（可选，data 为空表示不使用）
    int start_at_step;         // 高级采样：开始步骤（默认 0）
    int end_at_step;           // 高级采样：结束步骤（默认 0 = steps）
    bool add_noise;            // 高级采样：是否添加噪声（默认 true）
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

// 拼接两个 conditioning（在 token 维度 dim=1 拼接 c_crossattn）
SD_API sd_conditioning_t* sd_conditioning_concat(const sd_conditioning_t* cond1,
                                                  const sd_conditioning_t* cond2);

// 对两个 conditioning 做加权平均（conditioning_to_strength: 0.0=全cond1, 1.0=全cond2）
SD_API sd_conditioning_t* sd_conditioning_average(const sd_conditioning_t* cond1,
                                                   const sd_conditioning_t* cond2,
                                                   float conditioning_to_strength);

// ============================================================================
// IPAdapter 支持
// ============================================================================

// 加载 IPAdapter 模型
// ipadapter_path: IPAdapter safetensors 文件路径
// cross_attention_dim: UNet 的 cross_attention_dim（SD1.5=768, SDXL=2048）
// num_tokens: 图像 token 数量（通常为 4）
// clip_embeddings_dim: CLIP Vision 输出维度（ViT-H=1024, ViT-bigG=1280）
SD_API bool sd_load_ipadapter(
    sd_ctx_t* sd_ctx,
    const char* ipadapter_path,
    int cross_attention_dim,
    int num_tokens,
    int clip_embeddings_dim
);

// 设置 IPAdapter 的参考图像（CLIP Vision 编码后的输出）
// image: 参考图像
// strength: IPAdapter 强度（0.0 - 1.0+）
SD_API void sd_set_ipadapter_image(
    sd_ctx_t* sd_ctx,
    const sd_image_t* image,
    float strength
);

// 清除已加载的 IPAdapter
SD_API void sd_clear_ipadapter(sd_ctx_t* sd_ctx);

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

// ============================================================================
// CLIP Vision 支持
// ============================================================================

typedef struct {
    float* data;
    int numel;
} sd_clip_vision_output_t;

// 使用 CLIP Vision 编码图像
// image: 输入图像
// return_pooled: true 返回 pooled 特征（一维向量），false 返回完整特征图
// 返回的 sd_clip_vision_output_t* 需要用 sd_free_clip_vision_output() 释放
SD_API sd_clip_vision_output_t* sd_clip_vision_encode_image(
    sd_ctx_t* sd_ctx,
    const sd_image_t* image,
    bool return_pooled
);

// 释放 CLIP Vision 输出
SD_API void sd_free_clip_vision_output(sd_clip_vision_output_t* output);
