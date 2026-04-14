// ============================================================================
// stable-diffusion-ext.h
// ============================================================================
// 
// my-img 项目扩展头文件。
// 
// 这个文件应该被复制到 stable-diffusion.cpp/include/ 目录下，
// 作为 stable-diffusion.h 的补充。
// 
// 它声明了 Deep HighRes Fix 所需的 C++ hook 接口。
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

// Latent hook 类型定义
// 在采样过程中被调用，允许修改/替换当前 latent
typedef sd::Tensor<float> (*sd_latent_hook_t)(
    sd::Tensor<float>& latent,
    int step,
    int total_steps,
    void* user_data);

// 设置 latent hook
// 调用后，stable-diffusion.cpp 的 sample() 函数会在每个采样步骤前调用此 hook
SD_API void sd_set_latent_hook(sd_latent_hook_t hook, void* user_data);

// 清除 latent hook
SD_API void sd_clear_latent_hook();

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
} sd_deep_hires_params_t;

// 初始化默认参数
SD_API void sd_deep_hires_params_init(sd_deep_hires_params_t* params);

// 原生 Deep HighRes Fix 生成 API
// 在 my-img 项目中实现
SD_API sd_image_t* generate_image_deep_hires(
    sd_ctx_t* sd_ctx,
    const sd_deep_hires_params_t* params
);
