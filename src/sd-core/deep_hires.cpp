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
    int latent_channel;
    bool phase1_done;
    bool phase2_done;
};

// 使用 nearest neighbor 插值放大 latent
// TODO: 实现更高质量的插值（如双线性）
static sd::Tensor<float> upscale_latent_nearest(
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
    
    for (int y = 0; y < target_h; y++) {
        int src_y = y * current_h / target_h;
        for (int x = 0; x < target_w; x++) {
            int src_x = x * current_w / target_w;
            for (int c = 0; c < channels; c++) {
                result.data()[((y * target_w + x) * channels + c)] =
                    latent.data()[((src_y * current_w + src_x) * channels + c)];
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
    
    // Phase 1 -> Phase 2 过渡
    if (!state->phase1_done && step > state->phase1_steps) {
        state->phase1_done = true;
        printf("[DeepHires Hook] Step %d/%d: Upsampling latent from %dx%d to %dx%d\n",
               step, total_steps,
               (int)latent.shape()[0], (int)latent.shape()[1],
               state->phase2_w, state->phase2_h);
        return upscale_latent_nearest(latent, state->phase2_w, state->phase2_h, state->latent_channel);
    }
    
    // Phase 2 -> Phase 3 过渡
    if (!state->phase2_done && step > (state->phase1_steps + state->phase2_steps)) {
        state->phase2_done = true;
        printf("[DeepHires Hook] Step %d/%d: Upsampling latent from %dx%d to %dx%d\n",
               step, total_steps,
               (int)latent.shape()[0], (int)latent.shape()[1],
               state->target_w, state->target_h);
        return upscale_latent_nearest(latent, state->target_w, state->target_h, state->latent_channel);
    }
    
    return latent;
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
    state.latent_channel = latent_channel;
    
    // 注册 hook
    sd_set_latent_hook(deep_hires_latent_hook, &state);
    
    // 构建生成参数
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    gen_params.prompt = params->prompt;
    gen_params.negative_prompt = params->negative_prompt;
    gen_params.width = phase1_w;  // 从低分辨率开始
    gen_params.height = phase1_h;
    gen_params.strength = 1.0f;   // txt2img
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
    
    return result;
}
