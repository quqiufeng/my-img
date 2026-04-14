// =============================================================================
// sd-core/deep_hires.cpp
// =============================================================================
// 
// 原生 Deep HighRes Fix 实现
// 
// 实现方式：
// 1. 利用 stable-diffusion.cpp 中添加的 latent hook
// 2. 在采样过程中，在特定 step 将 latent interpolate 到更高分辨率
// 3. 继续在高分辨率下完成剩余采样步骤
// =============================================================================

#include "deep_hires.h"

// 需要包含 stable-diffusion.cpp 的内部头文件
#include "stable-diffusion.h"
#include "stable-diffusion-ext.h"
#include "tensor.hpp"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

struct DeepHiresState {
    int phase1_steps;
    int phase2_steps;
    int phase1_w;
    int phase1_h;
    int phase2_w;
    int phase2_h;
    int target_w;
    int target_h;
    float phase1_cfg_scale;
    float phase2_cfg_scale;
    float phase3_cfg_scale;
    bool phase1_done;
    bool phase2_done;
};

// 使用 nearest neighbor 插值放大 latent
// 双线性插值放大 latent
static sd::Tensor<float> upscale_latent_bilinear(
    const sd::Tensor<float>& latent,
    int target_w,
    int target_h,
    int channels) {
    
    int current_w = (int)latent.shape()[0];
    int current_h = (int)latent.shape()[1];
    
    if (current_w == target_w && current_h == target_h) {
        return latent;
    }
    
    sd::Tensor<float> result({target_w, target_h, channels, 1});
    
    float scale_x = (float)current_w / target_w;
    float scale_y = (float)current_h / target_h;
    
    for (int y = 0; y < target_h; y++) {
        float fy = y * scale_y;
        int y0 = (int)fy;
        int y1 = std::min(y0 + 1, current_h - 1);
        float dy = fy - y0;
        
        for (int x = 0; x < target_w; x++) {
            float fx = x * scale_x;
            int x0 = (int)fx;
            int x1 = std::min(x0 + 1, current_w - 1);
            float dx = fx - x0;
            
            for (int c = 0; c < channels; c++) {
                float v00 = latent.data()[((y0 * current_w + x0) * channels + c)];
                float v01 = latent.data()[((y0 * current_w + x1) * channels + c)];
                float v10 = latent.data()[((y1 * current_w + x0) * channels + c)];
                float v11 = latent.data()[((y1 * current_w + x1) * channels + c)];
                
                float v0 = v00 * (1.0f - dx) + v01 * dx;
                float v1 = v10 * (1.0f - dx) + v11 * dx;
                float v = v0 * (1.0f - dy) + v1 * dy;
                
                result.data()[((y * target_w + x) * channels + c)] = v;
            }
        }
    }
    
    return result;
}

// Latent hook 回调
static sd::Tensor<float> deep_hires_latent_hook(
    sd::Tensor<float>& latent,
    int step,
    int total_steps,
    void* user_data) {
    
    DeepHiresState* state = (DeepHiresState*)user_data;
    if (!state) return latent;
    
    // 自动从 latent shape 获取 channel 数
    int latent_channel = (int)latent.shape()[2];
    
    // Phase 1 -> Phase 2 过渡
    if (!state->phase1_done && step > state->phase1_steps) {
        state->phase1_done = true;
        printf("[DeepHires Hook] Step %d/%d: Upsampling latent from %dx%dx%d to %dx%dx%d\n",
               step, total_steps,
               (int)latent.shape()[0], (int)latent.shape()[1], latent_channel,
               state->phase2_w, state->phase2_h, latent_channel);
        return upscale_latent_bilinear(latent, state->phase2_w, state->phase2_h, latent_channel);
    }
    
    // Phase 2 -> Phase 3 过渡
    if (!state->phase2_done && step > (state->phase1_steps + state->phase2_steps)) {
        state->phase2_done = true;
        printf("[DeepHires Hook] Step %d/%d: Upsampling latent from %dx%dx%d to %dx%dx%d\n",
               step, total_steps,
               (int)latent.shape()[0], (int)latent.shape()[1], latent_channel,
               state->target_w, state->target_h, latent_channel);
        return upscale_latent_bilinear(latent, state->target_w, state->target_h, latent_channel);
    }
    
    return latent;
}

// Guidance hook 回调：动态调整 cfg_scale
static void deep_hires_guidance_hook(
    float* txt_cfg,
    float* img_cfg,
    float* distilled_guidance,
    int step,
    int total_steps,
    void* user_data) {
    
    DeepHiresState* state = (DeepHiresState*)user_data;
    if (!state) return;
    
    (void)img_cfg;
    (void)distilled_guidance;
    
    // Phase 1: 使用 phase1_cfg_scale
    if (step <= state->phase1_steps) {
        if (state->phase1_cfg_scale > 0) {
            *txt_cfg = state->phase1_cfg_scale;
        }
    }
    // Phase 2: 使用 phase2_cfg_scale
    else if (step <= state->phase1_steps + state->phase2_steps) {
        if (state->phase2_cfg_scale > 0) {
            *txt_cfg = state->phase2_cfg_scale;
        }
    }
    // Phase 3: 使用 phase3_cfg_scale
    else {
        if (state->phase3_cfg_scale > 0) {
            *txt_cfg = state->phase3_cfg_scale;
        }
    }
}

void sd_deep_hires_params_init(sd_deep_hires_params_t* params) {
    if (!params) return;
    memset(params, 0, sizeof(sd_deep_hires_params_t));
    params->prompt = "";
    params->negative_prompt = "";
    params->cfg_scale = 7.0f;
    params->sample_method = EULER_A_SAMPLE_METHOD;
    params->scheduler = KARRAS_SCHEDULER;
    params->seed = 42;
    params->total_steps = 30;
    params->target_width = 1024;
    params->target_height = 1024;
    params->strength = 1.0f;
}

sd_image_t* generate_image_deep_hires(
    sd_ctx_t* sd_ctx,
    const sd_deep_hires_params_t* params) {
    
    if (!sd_ctx || !params) {
        fprintf(stderr, "[ERROR] Invalid parameters for generate_image_deep_hires\n");
        return nullptr;
    }
    
    // 计算各阶段参数
    int target_w = (params->target_width + 63) & ~63;
    int target_h = (params->target_height + 63) & ~63;
    int total_steps = params->total_steps;
    
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
    
    // 获取 latent channel（通过 VAE scale factor 推断）
    // 对于 SD1.5 是 4，SDXL/Flux 可能不同
    // 由于无法直接访问 get_latent_channel()，我们假设 4
    int latent_channel = 4;
    
    printf("[DeepHires] Target: %dx%d, Phases: %dx%d(%d) -> %dx%d(%d) -> %dx%d(%d)\n",
           target_w, target_h,
           phase1_w, phase1_h, phase1_steps,
           phase2_w, phase2_h, phase2_steps,
           target_w, target_h, phase3_steps);
    
    // 准备 hook 状态
    DeepHiresState state = {};
    state.phase1_steps = phase1_steps;
    state.phase2_steps = phase2_steps;
    state.phase1_w = phase1_w;
    state.phase1_h = phase1_h;
    state.phase2_w = phase2_w;
    state.phase2_h = phase2_h;
    state.target_w = target_w;
    state.target_h = target_h;
    state.phase1_cfg_scale = params->phase1_cfg_scale > 0 ? params->phase1_cfg_scale : params->cfg_scale;
    state.phase2_cfg_scale = params->phase2_cfg_scale > 0 ? params->phase2_cfg_scale : params->cfg_scale;
    state.phase3_cfg_scale = params->phase3_cfg_scale > 0 ? params->phase3_cfg_scale : params->cfg_scale;
    
    // 注册 hook
    sd_set_latent_hook(deep_hires_latent_hook, &state);
    sd_set_guidance_hook(deep_hires_guidance_hook, &state);
    
    // 构建生成参数
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt = params->prompt;
    gen_params.negative_prompt = params->negative_prompt;
    gen_params.width = phase1_w;  // 从低分辨率开始
    gen_params.height = phase1_h;
    gen_params.strength = params->init_image.data ? params->strength : 1.0f;
    if (params->init_image.data) {
        gen_params.init_image = params->init_image;
    }
    gen_params.seed = params->seed;
    gen_params.sample_params.sample_steps = total_steps;
    gen_params.sample_params.sample_method = (sample_method_t)params->sample_method;
    gen_params.sample_params.scheduler = (scheduler_t)params->scheduler;
    gen_params.sample_params.guidance.txt_cfg = params->cfg_scale;
    
    if (params->vae_tiling) {
        gen_params.vae_tiling_params.enabled = true;
        gen_params.vae_tiling_params.tile_size_x = 512;
        gen_params.vae_tiling_params.tile_size_y = 512;
        gen_params.vae_tiling_params.target_overlap = 64;
    }
    
    // 调用生成
    sd_image_t* result = generate_image(sd_ctx, &gen_params);
    
    // 清除 hook
    sd_clear_latent_hook();
    sd_clear_guidance_hook();
    
    return result;
}
