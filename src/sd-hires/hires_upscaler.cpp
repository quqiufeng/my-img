#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model <path>           SD model path (GGUF)\n");
    printf("  --upscale-model <path>   ESRGAN model path (.bin)\n");
    printf("  --upscale-factor <int>  Upscale factor: 2, 3, 4 (default: 2)\n");
    printf("  --input <path>           Input image path\n");
    printf("  --output <path>          Output image path (default: hires_output.png)\n");
    printf("  --prompt <text>          Prompt\n");
    printf("  --negative-prompt <text> Negative prompt\n");
    printf("  --strength <float>       Denoising strength (default: 0.45)\n");
    printf("  --steps <int>            Sampling steps (default: 20)\n");
    printf("  --seed <int>             Random seed (default: random)\n");
    printf("  --width <int>            Output width (default: auto)\n");
    printf("  --height <int>           Output height (default: auto)\n");
    printf("  --gpu                    Use GPU (default)\n");
    printf("  --cpu                    Use CPU only\n");
    printf("  --no-flash-attn          Disable Flash Attention (default: enabled on GPU)\n");
    printf("  --debug                  Enable debug output\n");
}

int main(int argc, char* argv[]) {
    const char* model_path = NULL;
    const char* upscale_model_path = NULL;
    const char* input_path = NULL;
    const char* output_path = "hires_output.png";
    const char* prompt = "";
    const char* negative_prompt = "";
    float strength = 0.45f;
    int steps = 20;
    int64_t seed = -1;
    int width = 0;
    int height = 0;
    uint32_t upscale_factor = 2;
    bool use_gpu = true;
    bool use_flash_attn = true;
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--upscale-model") == 0 && i + 1 < argc) {
            upscale_model_path = argv[++i];
        } else if (strcmp(argv[i], "--upscale-factor") == 0 && i + 1 < argc) {
            upscale_factor = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--negative-prompt") == 0 && i + 1 < argc) {
            negative_prompt = argv[++i];
        } else if (strcmp(argv[i], "--strength") == 0 && i + 1 < argc) {
            strength = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (int64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else if (strcmp(argv[i], "--no-flash-attn") == 0) {
            use_flash_attn = false;
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (debug) {
        printf("[DEBUG] GPU mode: %s\n", use_gpu ? "ON" : "OFF");
        printf("[DEBUG] Flash Attention: %s\n", use_flash_attn ? "ON" : "OFF");
        printf("[DEBUG] Model: %s\n", model_path ? model_path : "(null)");
        printf("[DEBUG] Upscale model: %s\n", upscale_model_path ? upscale_model_path : "(null)");
        printf("[DEBUG] Input: %s\n", input_path ? input_path : "(null)");
    }

    if (!model_path || !upscale_model_path || !input_path) {
        print_usage(argv[0]);
        return 1;
    }

    printf("[Hires Fix] Loading input image: %s\n", input_path);
    int img_w, img_h, img_c;
    uint8_t* img_data = stbi_load(input_path, &img_w, &img_h, &img_c, 3);
    if (!img_data) {
        fprintf(stderr, "Failed to load image: %s\n", input_path);
        return 1;
    }
    printf("[Hires Fix] Input: %dx%d\n", img_w, img_h);

    sd_image_t input_sd_image = {(uint32_t)img_w, (uint32_t)img_h, 3, img_data};

    printf("[Hires Fix] Step 1: AI Upscale %ux with ESRGAN...\n", upscale_factor);
    upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(upscale_model_path, !use_gpu, false, 4, 128);
    if (!upscaler_ctx) {
        fprintf(stderr, "Failed to create upscaler context\n");
        stbi_image_free(img_data);
        return 1;
    }

    sd_image_t upscaled_image = upscale(upscaler_ctx, input_sd_image, upscale_factor);
    free_upscaler_ctx(upscaler_ctx);

    if (!upscaled_image.data) {
        fprintf(stderr, "Upscale failed\n");
        stbi_image_free(img_data);
        return 1;
    }
    printf("[Hires Fix] Upscaled: %dx%d\n", upscaled_image.width, upscaled_image.height);

    int out_w = width > 0 ? width : (int)upscaled_image.width;
    int out_h = height > 0 ? height : (int)upscaled_image.height;

    printf("[Hires Fix] Step 2: Latent Refinement (img2img)...\n");
    printf("[Hires Fix]   Mode: %s\n", use_gpu ? "GPU" : "CPU");
    printf("[Hires Fix]   prompt: %s\n", prompt);
    printf("[Hires Fix]   strength: %.2f, steps: %d, seed: %lld\n", strength, steps, (long long)seed);
    printf("[Hires Fix]   output size: %dx%d\n", out_w, out_h);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path;
    ctx_params.wtype = SD_TYPE_Q8_0;
    ctx_params.n_threads = 4;
    ctx_params.offload_params_to_cpu = !use_gpu;
    ctx_params.keep_vae_on_cpu = !use_gpu;
    ctx_params.keep_clip_on_cpu = !use_gpu;
    ctx_params.flash_attn = use_gpu && use_flash_attn;
    ctx_params.diffusion_flash_attn = use_gpu && use_flash_attn;

    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    if (!sd_ctx) {
        fprintf(stderr, "Failed to create SD context\n");
        stbi_image_free(upscaled_image.data);
        stbi_image_free(img_data);
        return 1;
    }

    sd_img_gen_params_t img_params;
    sd_img_gen_params_init(&img_params);
    img_params.prompt = prompt;
    img_params.negative_prompt = negative_prompt;
    img_params.width = out_w;
    img_params.height = out_h;
    img_params.strength = strength;
    img_params.seed = seed;
    img_params.init_image = upscaled_image;
    img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    img_params.sample_params.sample_steps = steps;
    img_params.sample_params.scheduler = KARRAS_SCHEDULER;
    img_params.vae_tiling_params.enabled = true;
    img_params.vae_tiling_params.tile_size_x = 512;
    img_params.vae_tiling_params.tile_size_y = 512;
    img_params.vae_tiling_params.target_overlap = 32;

    sd_image_t* result = generate_image(sd_ctx, &img_params);

    if (result && result->data) {
        printf("[Hires Fix] Writing output: %s\n", output_path);
        stbi_write_png(output_path, result->width, result->height, 4, result->data, 0);
        printf("[Hires Fix] Done! Output: %dx%d\n", result->width, result->height);
        free(result->data);
        free(result);
    } else {
        fprintf(stderr, "Image generation failed\n");
    }

    free_sd_ctx(sd_ctx);
    stbi_image_free(upscaled_image.data);
    stbi_image_free(img_data);

    return result ? 0 : 1;
}
