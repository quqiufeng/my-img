/**
 * stable-diffusion.cpp Extension API Implementation
 * 
 * 此文件提供原子操作 API，允许外部代码接管 HiRes Fix 等复杂流程。
 * 
 * 修改记录：
 *   - 依赖 stable-diffusion.cpp 中的非 static 辅助函数：
 *     prepare_image_generation_latents
 *     prepare_image_generation_embeds
 *     decode_image_outputs
 *     resolve_seed
 *     resolve_sample_method
 *     resolve_scheduler
 *     resolve_eta
 *     upscale_hires_latent
 */

// stable-diffusion-ext.cpp
// 注意：此文件被 stable-diffusion.cpp #include，不作为独立编译单元编译

// 包含 sd.cpp 内部头文件
#include "model.h"
#include "tensor.hpp"
#include "vae.hpp"
#include "denoiser.hpp"
#include "diffusion_model.hpp"
#include "conditioner.hpp"
#include "upscaler.h"
#include "util.h"

#include <cstring>
#include <cmath>

// ========== 前向声明：sd.cpp 中的辅助函数 ==========

extern std::optional<ImageGenerationLatents> prepare_image_generation_latents(
    sd_ctx_t* sd_ctx,
    const sd_img_gen_params_t* sd_img_gen_params,
    GenerationRequest* request,
    SamplePlan* plan);

extern std::optional<ImageGenerationEmbeds> prepare_image_generation_embeds(
    sd_ctx_t* sd_ctx,
    const sd_img_gen_params_t* sd_img_gen_params,
    GenerationRequest* request,
    SamplePlan* plan,
    ImageGenerationLatents* latents);

extern sd_image_t* decode_image_outputs(
    sd_ctx_t* sd_ctx,
    const GenerationRequest& request,
    const std::vector<sd::Tensor<float>>& final_latents);

extern int64_t resolve_seed(int64_t seed);
extern enum sample_method_t resolve_sample_method(sd_ctx_t* sd_ctx, enum sample_method_t sample_method);
extern scheduler_t resolve_scheduler(sd_ctx_t* sd_ctx, scheduler_t scheduler, enum sample_method_t sample_method);
extern float resolve_eta(sd_ctx_t* sd_ctx, float eta, enum sample_method_t sample_method);
extern sd::Tensor<float> upscale_hires_latent(sd_ctx_t* sd_ctx,
                                               const sd::Tensor<float>& latent,
                                               const GenerationRequest& request,
                                               UpscalerGGML* upscaler);

// ========== 内部辅助 ==========

struct sd_tensor_t {
    sd::Tensor<float> tensor;
};

static StableDiffusionGGML* get_sd(sd_ctx_t* ctx) {
    if (!ctx) return nullptr;
    return ctx->sd;
}

// ========== C API ==========

extern "C" {

// ========== Tensor API ==========

int sd_ext_tensor_ndim(sd_tensor_t* tensor) {
    if (!tensor) return 0;
    return (int)tensor->tensor.shape().size();
}

int64_t sd_ext_tensor_shape(sd_tensor_t* tensor, int dim) {
    if (!tensor || dim < 0 || dim >= (int)tensor->tensor.shape().size()) return 0;
    return tensor->tensor.shape()[dim];
}

int64_t sd_ext_tensor_nelements(sd_tensor_t* tensor) {
    if (!tensor) return 0;
    return tensor->tensor.numel();
}

int sd_ext_tensor_dtype(sd_tensor_t* tensor) {
    if (!tensor) return -1;
    // sd::Tensor<float> 目前只支持 float32
    return 0;  // 0 = f32
}

void* sd_ext_tensor_data_ptr(sd_tensor_t* tensor) {
    if (!tensor) return nullptr;
    return tensor->tensor.data();
}

void sd_ext_tensor_free(sd_tensor_t* tensor) {
    delete tensor;
}

sd_tensor_t* sd_ext_tensor_from_data(const void* data,
                                      const int64_t* shape,
                                      int ndim,
                                      int dtype) {
    if (!data || !shape || ndim <= 0) return nullptr;
    
    std::vector<int64_t> tensor_shape(shape, shape + ndim);
    int64_t numel = 1;
    for (auto s : tensor_shape) numel *= s;
    
    sd_tensor_t* result = new sd_tensor_t();
    result->tensor = sd::Tensor<float>(tensor_shape);
    
    size_t bytes = numel * sizeof(float);
    std::memcpy(result->tensor.data(), data, bytes);
    
    return result;
}

// ========== 原子操作 API ==========

sd_tensor_t* sd_ext_generate_latent(sd_ctx_t* ctx,
                                     const sd_img_gen_params_t* params) {
    auto* sd = get_sd(ctx);
    if (!sd || !params) return nullptr;
    
    try {
        int64_t t0 = ggml_time_ms();
        
        // 1. 构造请求（禁用 hires，因为我们只生成基础 latent）
        sd_img_gen_params_t params_copy = *params;
        params_copy.hires.enabled = false;  // 禁用内置 hires
        
        GenerationRequest request(ctx, &params_copy);
        LOG_INFO("sd_ext_generate_latent %dx%d", request.width, request.height);
        
        // 2. 设置随机种子和参数
        sd->rng->manual_seed(request.seed);
        if (sd->sampler_rng) {
            sd->sampler_rng->manual_seed(request.seed);
        }
        sd->set_flow_shift(params_copy.sample_params.flow_shift);
        sd->apply_loras(params_copy.loras, params_copy.lora_count);
        
        // 3. 配置 VAE 轴（RAII）
        ImageVaeAxesGuard axes_guard(ctx, &params_copy, request);
        
        // 4. 构造采样计划
        SamplePlan plan(ctx, &params_copy, request);
        
        // 5. 准备 latents
        auto latents_opt = prepare_image_generation_latents(ctx, &params_copy, &request, &plan);
        if (!latents_opt.has_value()) {
            LOG_ERROR("prepare_image_generation_latents failed");
            return nullptr;
        }
        ImageGenerationLatents latents = std::move(*latents_opt);
        
        // 6. 准备文本条件
        auto embeds_opt = prepare_image_generation_embeds(ctx, &params_copy, &request, &plan, &latents);
        if (!embeds_opt.has_value()) {
            LOG_ERROR("prepare_image_generation_embeds failed");
            return nullptr;
        }
        ImageGenerationEmbeds embeds = std::move(*embeds_opt);
        
        // 7. 采样（只处理 batch_count=1 的情况，简化实现）
        std::vector<sd::Tensor<float>> final_latents;
        int64_t denoise_start = ggml_time_ms();
        
        for (int b = 0; b < request.batch_count; b++) {
            int64_t cur_seed = request.seed + b;
            sd->rng->manual_seed(cur_seed);
            if (sd->sampler_rng) {
                sd->sampler_rng->manual_seed(cur_seed);
            }
            
            sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd->rng);
            
            sd::Tensor<float> x_0 = sd->sample(sd->diffusion_model,
                                               true,
                                               latents.init_latent,
                                               std::move(noise),
                                               embeds.cond,
                                               embeds.uncond,
                                               embeds.img_cond,
                                               embeds.id_cond,
                                               latents.control_image,
                                               request.control_strength,
                                               request.guidance,
                                               plan.eta,
                                               request.shifted_timestep,
                                               plan.sample_method,
                                               sd->is_flow_denoiser(),
                                               plan.sigmas,
                                               plan.start_merge_step,
                                               latents.ref_latents,
                                               request.increase_ref_index,
                                               latents.denoise_mask,
                                               sd::Tensor<float>(),
                                               1.f,
                                               request.cache_params);
            
            if (!x_0.empty()) {
                final_latents.push_back(std::move(x_0));
            } else {
                LOG_ERROR("sampling failed for batch %d/%d", b + 1, request.batch_count);
                if (sd->free_params_immediately) {
                    sd->diffusion_model->free_params_buffer();
                }
                return nullptr;
            }
        }
        
        if (sd->free_params_immediately) {
            sd->diffusion_model->free_params_buffer();
        }
        
        int64_t denoise_end = ggml_time_ms();
        LOG_INFO("sd_ext_generate_latent completed, generating %zu latent(s), taking %.2fs",
                 final_latents.size(), (denoise_end - denoise_start) * 1.0f / 1000);
        
        // 8. 返回第一个 latent（简化：暂不支持 batch_count > 1）
        if (final_latents.empty()) {
            return nullptr;
        }
        
        sd_tensor_t* result = new sd_tensor_t();
        result->tensor = std::move(final_latents[0]);
        
        int64_t t1 = ggml_time_ms();
        LOG_INFO("sd_ext_generate_latent total time: %.2fs", (t1 - t0) * 1.0f / 1000);
        
        return result;
        
    } catch (const std::exception& e) {
        LOG_ERROR("sd_ext_generate_latent failed: %s", e.what());
        return nullptr;
    }
}

sd_tensor_t* sd_ext_sample_latent(sd_ctx_t* ctx,
                                   sd_tensor_t* init_latent,
                                   sd_tensor_t* noise,
                                   const char* prompt,
                                   const char* negative_prompt,
                                   sd_sample_params_t* sample_params,
                                   int width,
                                   int height,
                                   float strength) {
    auto* sd = get_sd(ctx);
    if (!sd || !init_latent || !sample_params || !prompt) return nullptr;
    
    try {
        int64_t t0 = ggml_time_ms();
        LOG_INFO("sd_ext_sample_latent %dx%d, strength=%.2f", width, height, strength);
        
        // 1. 解析采样参数
        enum sample_method_t method = resolve_sample_method(ctx, sample_params->sample_method);
        scheduler_t scheduler = resolve_scheduler(ctx, sample_params->scheduler, method);
        float eta = resolve_eta(ctx, sample_params->eta, method);
        int steps = sample_params->sample_steps;
        
        // 2. 计算 sigma 调度（支持 strength 截断，复现 sd.cpp 内置 HiRes Fix 逻辑）
        std::vector<float> sigmas;
        if (strength < 1.0f && strength > 0.0f) {
            // sd-webui behavior: scale up total steps so trimming by denoising_strength
            // yields exactly 'steps' effective steps
            int total_steps = static_cast<int>(steps / strength);
            std::vector<float> hires_sigmas = sd->denoiser->get_sigmas(
                total_steps,
                sd->get_image_seq_len(height, width),
                scheduler,
                sd->version);
            
            size_t t_enc = static_cast<size_t>(total_steps * strength);
            if (t_enc >= static_cast<size_t>(total_steps)) {
                t_enc = static_cast<size_t>(total_steps) - 1;
            }
            sigmas.assign(hires_sigmas.begin() + total_steps - static_cast<int>(t_enc) - 1,
                          hires_sigmas.end());
            LOG_INFO("sd_ext_sample_latent: strength=%.2f, total_steps=%d, t_enc=%zu, sigma_sched_size=%zu",
                     strength, total_steps, t_enc, sigmas.size());
        } else {
            sigmas = sd->denoiser->get_sigmas(
                steps,
                sd->get_image_seq_len(height, width),
                scheduler,
                sd->version);
        }
        
        // 3. 准备噪声
        sd::Tensor<float> sample_noise;
        if (noise) {
            sample_noise = noise->tensor;
        } else {
            sample_noise = sd::randn_like<float>(init_latent->tensor, sd->rng);
        }
        
        // 4. 文本编码
        ConditionerParams condition_params;
        condition_params.text = prompt;
        condition_params.clip_skip = -1;
        condition_params.width = width;
        condition_params.height = height;
        condition_params.adm_in_channels = static_cast<int>(sd->diffusion_model->get_adm_in_channels());
        
        auto id_cond = sd->get_pmid_conditon({}, condition_params);
        auto cond = sd->cond_stage_model->get_learned_condition(sd->n_threads, condition_params);
        
        SDCondition uncond;
        sd_guidance_params_t guidance = sample_params->guidance;
        if (guidance.txt_cfg != 1.f) {
            condition_params.text = negative_prompt ? negative_prompt : "";
            uncond = sd->cond_stage_model->get_learned_condition(sd->n_threads, condition_params);
        }
        
        if (sd->free_params_immediately) {
            sd->cond_stage_model->free_params_buffer();
        }
        
        // 5. 采样
        sd::Tensor<float> x_0 = sd->sample(sd->diffusion_model,
                                           true,
                                           init_latent->tensor,
                                           std::move(sample_noise),
                                           cond,
                                           uncond,
                                           SDCondition(),  // img_cond
                                           id_cond,
                                           sd::Tensor<float>(),  // control_image
                                           0.f,  // control_strength
                                           guidance,
                                           eta,
                                           sample_params->shifted_timestep,
                                           method,
                                           sd->is_flow_denoiser(),
                                           sigmas,
                                           -1,  // start_merge_step
                                           std::vector<sd::Tensor<float>>(),  // ref_latents
                                           false,
                                           sd::Tensor<float>(),  // denoise_mask
                                           sd::Tensor<float>(),  // vace_context
                                           1.f,
                                           nullptr);  // cache_params
        
        if (x_0.empty()) {
            LOG_ERROR("sd_ext_sample_latent: sampling failed");
            if (sd->free_params_immediately) {
                sd->diffusion_model->free_params_buffer();
            }
            return nullptr;
        }
        
        if (sd->free_params_immediately) {
            sd->diffusion_model->free_params_buffer();
        }
        
        int64_t t1 = ggml_time_ms();
        LOG_INFO("sd_ext_sample_latent completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        
        sd_tensor_t* result = new sd_tensor_t();
        result->tensor = std::move(x_0);
        return result;
        
    } catch (const std::exception& e) {
        LOG_ERROR("sd_ext_sample_latent failed: %s", e.what());
        return nullptr;
    }
}

sd_tensor_t* sd_ext_vae_encode(sd_ctx_t* ctx, sd_image_t image) {
    auto* sd = get_sd(ctx);
    if (!sd) return nullptr;
    
    try {
        // 图像 → tensor
        sd::Tensor<float> img_tensor = sd_image_to_tensor(image);
        if (img_tensor.empty()) {
            LOG_ERROR("sd_ext_vae_encode: failed to convert image to tensor");
            return nullptr;
        }
        
        // VAE 编码
        sd::Tensor<float> latent = sd->encode_first_stage(img_tensor);
        if (latent.empty()) {
            LOG_ERROR("sd_ext_vae_encode: encode_first_stage failed");
            return nullptr;
        }
        
        sd_tensor_t* result = new sd_tensor_t();
        result->tensor = std::move(latent);
        return result;
        
    } catch (const std::exception& e) {
        LOG_ERROR("sd_ext_vae_encode failed: %s", e.what());
        return nullptr;
    }
}

sd_image_t sd_ext_vae_decode(sd_ctx_t* ctx, sd_tensor_t* latent) {
    sd_image_t result = {0, 0, 0, nullptr};
    
    auto* sd = get_sd(ctx);
    if (!sd || !latent) return result;
    
    try {
        // VAE 解码
        sd::Tensor<float> decoded = sd->decode_first_stage(latent->tensor);
        if (decoded.empty()) {
            LOG_ERROR("sd_ext_vae_decode: decode_first_stage failed");
            return result;
        }
        
        // tensor → 图像
        result = tensor_to_sd_image(decoded);
        return result;
        
    } catch (const std::exception& e) {
        LOG_ERROR("sd_ext_vae_decode failed: %s", e.what());
        return result;
    }
}

sd_tensor_t* sd_ext_create_noise(sd_ctx_t* ctx,
                                  int width,
                                  int height,
                                  int64_t seed) {
    auto* sd = get_sd(ctx);
    if (!sd) return nullptr;
    
    try {
        sd->rng->manual_seed(seed);
        
        int vae_scale = sd->get_vae_scale_factor();
        int latent_c = sd->get_latent_channel();
        
        std::vector<int64_t> latent_shape = {
            width / vae_scale,
            height / vae_scale,
            latent_c
        };
        
        sd::Tensor<float> noise = sd::randn<float>(latent_shape, sd->rng);
        
        sd_tensor_t* result = new sd_tensor_t();
        result->tensor = std::move(noise);
        return result;
        
    } catch (const std::exception& e) {
        LOG_ERROR("sd_ext_create_noise failed: %s", e.what());
        return nullptr;
    }
}

} // extern "C"
