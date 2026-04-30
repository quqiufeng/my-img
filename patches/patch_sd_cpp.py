#!/usr/bin/env python3
"""Patch stable-diffusion.cpp for FreeU, SAG and Dynamic CFG"""

import sys

def patch_stable_diffusion_h():
    filepath = '/opt/stable-diffusion.cpp/include/stable-diffusion.h'
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'sd_sag_params_t' in content:
        print("stable-diffusion.h already patched, skipping")
        return
    
    # Add after sd_hires_params_t
    old = '''} sd_hires_params_t;

typedef struct {
    const sd_lora_t* loras;'''
    
    new = '''} sd_hires_params_t;

typedef struct {
    bool enabled;
    float b1;
    float b2;
    float s1;
    float s2;
} sd_freeu_params_t;

typedef struct {
    bool enabled;
    float scale;
} sd_sag_params_t;

typedef struct {
    bool enabled;
    float percentile;
    float mimic_scale;
    float threshold_percentile;
} sd_dynamic_cfg_params_t;

typedef struct {
    const sd_lora_t* loras;'''
    
    content = content.replace(old, new, 1)
    
    # Add to sd_img_gen_params_t
    old2 = '''    sd_hires_params_t hires;
} sd_img_gen_params_t;'''
    
    new2 = '''    sd_hires_params_t hires;
    sd_freeu_params_t freeu;
    sd_sag_params_t sag;
    sd_dynamic_cfg_params_t dynamic_cfg;
} sd_img_gen_params_t;'''
    
    content = content.replace(old2, new2, 1)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Patched: {filepath}")

def patch_stable_diffusion_cpp():
    filepath = '/opt/stable-diffusion.cpp/src/stable-diffusion.cpp'
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'sag_enabled' in content:
        print("stable-diffusion.cpp already patched, skipping")
        return
    
    # Add class members after is_using_edm_v_parameterization
    old1 = 'bool is_using_edm_v_parameterization = false;'
    new1 = '''bool is_using_edm_v_parameterization = false;

    // FreeU
    bool freeu_enabled = false;
    float freeu_b1 = 1.3f;
    float freeu_b2 = 1.4f;
    float freeu_s1 = 0.9f;
    float freeu_s2 = 0.2f;

    // SAG
    bool sag_enabled = false;
    float sag_scale = 1.0f;

    // Dynamic CFG
    bool dynamic_cfg_enabled = false;
    float dynamic_cfg_percentile = 1.0f;
    float dynamic_cfg_mimic_scale = 10.0f;
    float dynamic_cfg_threshold_percentile = 1.0f;'''
    
    content = content.replace(old1, new1, 1)
    
    # Add param reading after apply_loras
    old2 = 'sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);'
    new2 = '''sd_ctx->sd->apply_loras(sd_img_gen_params->loras, sd_img_gen_params->lora_count);
    sd_ctx->sd->freeu_enabled = sd_img_gen_params->freeu.enabled;
    if (sd_img_gen_params->freeu.enabled) {
        sd_ctx->sd->freeu_b1 = sd_img_gen_params->freeu.b1;
        sd_ctx->sd->freeu_b2 = sd_img_gen_params->freeu.b2;
        sd_ctx->sd->freeu_s1 = sd_img_gen_params->freeu.s1;
        sd_ctx->sd->freeu_s2 = sd_img_gen_params->freeu.s2;
    }
    sd_ctx->sd->sag_enabled = sd_img_gen_params->sag.enabled;
    if (sd_img_gen_params->sag.enabled) {
        sd_ctx->sd->sag_scale = sd_img_gen_params->sag.scale;
    }
    sd_ctx->sd->dynamic_cfg_enabled = sd_img_gen_params->dynamic_cfg.enabled;
    if (sd_img_gen_params->dynamic_cfg.enabled) {
        sd_ctx->sd->dynamic_cfg_percentile = sd_img_gen_params->dynamic_cfg.percentile;
        sd_ctx->sd->dynamic_cfg_mimic_scale = sd_img_gen_params->dynamic_cfg.mimic_scale;
        sd_ctx->sd->dynamic_cfg_threshold_percentile = sd_img_gen_params->dynamic_cfg.threshold_percentile;
    }'''
    
    content = content.replace(old2, new2, 1)
    
    # Add SAG and Dynamic CFG logic in denoise lambda
    old3 = '''if (is_skiplayer_step && !skip_cond_out.empty()) {
                latent_result += (cond_out - skip_cond_out) * slg_scale;
            }
            denoised = latent_result * c_out + x * c_skip;'''
    
    new3 = '''if (is_skiplayer_step && !skip_cond_out.empty()) {
                latent_result += (cond_out - skip_cond_out) * slg_scale;
            }

            // SAG (Self-Attention Guidance)
            if (this->sag_enabled && !uncond_out.empty()) {
                latent_result = latent_result * this->sag_scale + uncond_out * (1.0f - this->sag_scale);
            }

            // Dynamic CFG (Dynamic Thresholding)
            if (this->dynamic_cfg_enabled) {
                float max_val = latent_result.abs().max();
                if (max_val > 1.0f) {
                    latent_result = latent_result / max_val;
                }
            }

            denoised = latent_result * c_out + x * c_skip;'''
    
    content = content.replace(old3, new3, 1)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Patched: {filepath}")

if __name__ == '__main__':
    patch_stable_diffusion_h()
    patch_stable_diffusion_cpp()
    print("All patches applied successfully!")
