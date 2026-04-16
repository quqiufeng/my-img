// ============================================================================
// sd-engine/nodes/node_utils.cpp
// ============================================================================
// 核心节点公共辅助函数实现
// ============================================================================

#include "nodes/node_utils.h"
#include "core/log.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"

namespace sdengine {

// ============================================================================
// DeepHighResFix 辅助函数
// ============================================================================

sd::Tensor<float> upscale_latent_bilinear_node(const sd::Tensor<float>& latent, int target_w, int target_h,
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

sd::Tensor<float> deep_hires_node_latent_hook(sd::Tensor<float>& latent, int step, int total_steps, void* user_data) {
    DeepHiresNodeState* state = (DeepHiresNodeState*)user_data;
    if (!state)
        return latent;

    int latent_channel = (int)latent.shape()[2];

    // Phase 1 -> Phase 2 过渡
    if (!state->phase1_done && step > state->phase1_steps) {
        state->phase1_done = true;
        LOG_INFO("[DeepHires Hook] Step %d/%d: Upsampling %dx%d -> %dx%d\n", step, total_steps, (int)latent.shape()[0],
                 (int)latent.shape()[1], state->phase2_w, state->phase2_h);
        return upscale_latent_bilinear_node(latent, state->phase2_w, state->phase2_h, latent_channel);
    }

    // Phase 2 -> Phase 3 过渡
    if (!state->phase2_done && step > (state->phase1_steps + state->phase2_steps)) {
        state->phase2_done = true;
        LOG_INFO("[DeepHires Hook] Step %d/%d: Upsampling %dx%d -> %dx%d\n", step, total_steps, (int)latent.shape()[0],
                 (int)latent.shape()[1], state->target_w, state->target_h);
        return upscale_latent_bilinear_node(latent, state->target_w, state->target_h, latent_channel);
    }

    return latent;
}

void deep_hires_node_guidance_hook(float* txt_cfg, float* img_cfg, float* distilled_guidance, int step, int total_steps,
                                   void* user_data) {
    DeepHiresNodeState* state = (DeepHiresNodeState*)user_data;
    if (!state)
        return;

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

// ============================================================================
// KSampler 通用执行逻辑
// ============================================================================

sd_error_t run_sampler_common(sd_ctx_t* sd_ctx, const NodeInputs& inputs, sd_node_sample_params_t& sample_params,
                              sd_latent_t** out_result) {
    // 处理 LoRA Stack
    std::vector<sd_lora_t> loras;
    if (inputs.count("lora_stack")) {
        auto lora_stack = std::any_cast<std::vector<LoRAInfo>>(inputs.at("lora_stack"));
        for (const auto& info : lora_stack) {
            sd_lora_t lora;
            lora.path = info.path.c_str();
            lora.multiplier = info.strength;
            lora.is_high_noise = false;
            loras.push_back(lora);
            LOG_INFO("[KSampler] Applying LoRA: %s (strength=%.2f)\n", info.path.c_str(), info.strength);
        }
    }

    if (!loras.empty()) {
        sd_apply_loras(sd_ctx, loras.data(), static_cast<uint32_t>(loras.size()));
    } else {
        sd_clear_loras(sd_ctx);
    }

    // 处理 ControlNet 输入
    if (inputs.count("_control_image")) {
        ImagePtr ctrl_img = std::any_cast<ImagePtr>(inputs.at("_control_image"));
        if (ctrl_img && ctrl_img->data) {
            sample_params.control_image = *ctrl_img;
            sample_params.control_strength =
                inputs.count("_control_strength") ? std::any_cast<float>(inputs.at("_control_strength")) : 1.0f;
            LOG_INFO("[KSampler] Using ControlNet (from Apply): strength=%.2f, image=%dx%d\n",
                     sample_params.control_strength, ctrl_img->width, ctrl_img->height);
        }
    } else if (inputs.count("control_image")) {
        ImagePtr ctrl_img = std::any_cast<ImagePtr>(inputs.at("control_image"));
        if (ctrl_img && ctrl_img->data) {
            sample_params.control_image = *ctrl_img;
            sample_params.control_strength =
                inputs.count("control_strength") ? std::any_cast<float>(inputs.at("control_strength")) : 1.0f;
            LOG_INFO("[KSampler] Using ControlNet: strength=%.2f, image=%dx%d\n", sample_params.control_strength,
                     ctrl_img->width, ctrl_img->height);
        }
    }

    // 处理 Inpaint mask 输入
    if (inputs.count("mask")) {
        ImagePtr mask = std::any_cast<ImagePtr>(inputs.at("mask"));
        if (mask && mask->data) {
            sample_params.mask_image = *mask;
            LOG_INFO("[KSampler] Using Inpaint mask: %dx%d\n", mask->width, mask->height);
        }
    }

    // 处理 IPAdapter 输入
    if (inputs.count("_ipadapter_info")) {
        IPAdapterInfo info = std::any_cast<IPAdapterInfo>(inputs.at("_ipadapter_info"));
        ImagePtr ip_image = std::any_cast<ImagePtr>(inputs.at("_ipadapter_image"));
        if (!info.path.empty() && ip_image && ip_image->data) {
            LOG_INFO("[KSampler] Loading IPAdapter: %s\n", info.path.c_str());
            bool loaded = sd_load_ipadapter(sd_ctx, info.path.c_str(), info.cross_attention_dim, info.num_tokens,
                                            info.clip_embeddings_dim);
            if (loaded) {
                sd_set_ipadapter_image(sd_ctx, ip_image.get(), info.strength);
                LOG_INFO("[KSampler] IPAdapter applied: strength=%.2f, image=%dx%d\n", info.strength, ip_image->width,
                         ip_image->height);
            } else {
                LOG_ERROR("[ERROR] KSampler: Failed to load IPAdapter %s\n", info.path.c_str());
                sd_clear_loras(sd_ctx);
                return sd_error_t::ERROR_MODEL_LOADING;
            }
        }
    }

    ConditioningPtr positive = std::any_cast<ConditioningPtr>(inputs.at("positive"));
    ConditioningPtr negative =
        inputs.count("negative") ? std::any_cast<ConditioningPtr>(inputs.at("negative")) : nullptr;
    LatentPtr init_latent = std::any_cast<LatentPtr>(inputs.at("latent_image"));
    float denoise = inputs.count("denoise") ? std::any_cast<float>(inputs.at("denoise")) : 1.0f;

    *out_result = sd_sampler_run(sd_ctx, init_latent.get(), positive.get(), negative.get(), &sample_params, denoise);

    sd_clear_loras(sd_ctx);
    sd_clear_ipadapter(sd_ctx);

    if (!*out_result) {
        LOG_ERROR("[ERROR] KSampler: Sampling failed\n");
        return sd_error_t::ERROR_SAMPLING_FAILED;
    }

    LOG_INFO("[KSampler] Sampling completed\n");
    return sd_error_t::OK;
}

} // namespace sdengine
