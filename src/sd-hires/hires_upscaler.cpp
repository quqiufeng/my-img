#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --model <path>           SD model path (GGUF)\n");
    printf("  --vae <path>             VAE path (optional, for flow models)\n");
    printf("  --llm <path>             LLM path (optional, for flow models)\n");
    printf("  --upscale-model <path>   ESRGAN model path (.bin) (optional)\n");
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
    const char* vae_path = NULL;
    const char* llm_path = NULL;
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
        } else if (strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae_path = argv[++i];
        } else if (strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            llm_path = argv[++i];
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

    if (!model_path || !input_path) {
        fprintf(stderr, "model_path: %s, input_path: %s\n", model_path ? model_path : "NULL", input_path ? input_path : "NULL");
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

    int target_w = width > 0 ? width : img_w;
    int target_h = height > 0 ? height : img_h;

    sd_image_t upscaled_image;
    upscaled_image.data = NULL;

    // 如果提供了 upscale_model_path，使用 ESRGAN 放大
    if (upscale_model_path) {
        printf("[Hires Fix] Step 1: AI Upscale %ux with ESRGAN...\n", upscale_factor);
        upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(upscale_model_path, true, false, 4, 128);
        if (!upscaler_ctx) {
            fprintf(stderr, "Failed to create upscaler context\n");
            stbi_image_free(img_data);
            return 1;
        }

        sd_image_t esrgan_result = upscale(upscaler_ctx, input_sd_image, upscale_factor);
        
        if (!esrgan_result.data) {
            fprintf(stderr, "Upscale failed\n");
            free_upscaler_ctx(upscaler_ctx);
            stbi_image_free(img_data);
            return 1;
        }
        
        // 复制 ESRGAN 结果到 upscaled_image（在释放 upscaler_ctx 之前）
        size_t esrgan_size = esrgan_result.width * esrgan_result.height * esrgan_result.channel;
        upscaled_image.data = (uint8_t*)malloc(esrgan_size);
        memcpy(upscaled_image.data, esrgan_result.data, esrgan_size);
        upscaled_image.width = esrgan_result.width;
        upscaled_image.height = esrgan_result.height;
        upscaled_image.channel = esrgan_result.channel;
        
        free(esrgan_result.data);  // 释放 ESRGAN 返回的内存
        free_upscaler_ctx(upscaler_ctx);

        printf("[Hires Fix] Upscaled: %dx%d (channels: %d)\n", upscaled_image.width, upscaled_image.height, upscaled_image.channel);
        
        // 保存到临时文件并重新加载（模拟 CLI 行为）
        char temp_upscale_path[] = "/tmp/sd_hires_upscale_XXXXXX.png";
        int fd = mkstemps(temp_upscale_path, 4);
        close(fd);
        stbi_write_png(temp_upscale_path, upscaled_image.width, upscaled_image.height, 3, upscaled_image.data, 0);
        
        // 释放原来的数据
        free(upscaled_image.data);
        
        // 重新加载图片（CLI 方式）
        int reload_w, reload_h, reload_c;
        uint8_t* reload_data = stbi_load(temp_upscale_path, &reload_w, &reload_h, &reload_c, 3);
        if (!reload_data) {
            fprintf(stderr, "Failed to reload upscale image\n");
            free_upscaler_ctx(upscaler_ctx);
            stbi_image_free(img_data);
            return 1;
        }
        upscaled_image.data = reload_data;
        upscaled_image.width = reload_w;
        upscaled_image.height = reload_h;
        upscaled_image.channel = 3;
        
        printf("[Hires Fix] Reloaded: %dx%d\n", reload_w, reload_h);
        
        target_w = upscaled_image.width;
        target_h = upscaled_image.height;
    } else {
        // 没有 ESRGAN，直接用原图
        printf("[Hires Fix] No upscale model, using original image\n");
        
        // 如果指定了目标尺寸，使用目标尺寸
        if (width > 0 || height > 0) {
            // 复制原图数据用于 img2img
            size_t orig_size = img_w * img_h * 3;
            uint8_t* orig_copy = (uint8_t*)malloc(orig_size);
            memcpy(orig_copy, img_data, orig_size);
            upscaled_image.data = orig_copy;
            upscaled_image.width = target_w;
            upscaled_image.height = target_h;
            upscaled_image.channel = 3;
            printf("[Hires Fix] Using target size: %dx%d\n", target_w, target_h);
        } else {
            upscaled_image = input_sd_image;
        }
    }

    int out_w = width > 0 ? width : (int)upscaled_image.width;
    int out_h = height > 0 ? height : (int)upscaled_image.height;

    printf("[Hires Fix] Step 2: Latent Refinement (img2img)...\n");
    printf("[Hires Fix]   Mode: %s\n", use_gpu ? "GPU" : "CPU");
    printf("[Hires Fix]   prompt: %s\n", prompt);
    printf("[Hires Fix]   strength: %.2f, steps: %d, seed: %lld\n", strength, steps, (long long)seed);

    // 不使用 mask_image，设为空
    sd_image_t mask_sd_image = {0, 0, 0, NULL};

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path;
    ctx_params.diffusion_model_path = model_path;
    ctx_params.vae_path = vae_path;
    ctx_params.llm_path = llm_path;
    ctx_params.wtype = SD_TYPE_Q8_0;
    ctx_params.n_threads = 4;
    ctx_params.vae_decode_only = false;
    ctx_params.free_params_immediately = false;
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

    // 完全按照 CLI 的方式初始化 img_params
    sd_img_gen_params_t img_params;
    memset(&img_params, 0, sizeof(img_params));
    
    img_params.loras = NULL;
    img_params.lora_count = 0;
    img_params.prompt = prompt;
    img_params.negative_prompt = negative_prompt;
    img_params.clip_skip = -1;
    img_params.init_image = upscaled_image;
    img_params.ref_images = NULL;
    img_params.ref_images_count = 0;
    img_params.auto_resize_ref_image = true;
    img_params.increase_ref_index = false;
    img_params.mask_image = mask_sd_image;  // 使用单独的 mask
    img_params.width = upscaled_image.width;
    img_params.height = upscaled_image.height;
    img_params.strength = strength;
    img_params.seed = seed;
    img_params.batch_count = 1;
    img_params.control_image.data = NULL;
    img_params.control_image.width = 0;
    img_params.control_image.height = 0;
    img_params.control_image.channel = 0;
    img_params.control_strength = 0.8f;
    img_params.sample_params.guidance.txt_cfg = 2.0f;
    img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    img_params.sample_params.sample_steps = steps;
    img_params.sample_params.scheduler = KARRAS_SCHEDULER;
    
    // 完全按照 CLI 的方式初始化 img_params
    // 不使用 VAE tiling，用更大的 tile_size 或不用
    img_params.vae_tiling_params.enabled = false;

    sd_image_t* result = generate_image(sd_ctx, &img_params);

    if (result && result->data) {
        printf("[Hires Fix] Writing output: %s\n", output_path);
        // result->channel 可能是 3 或 4
        stbi_write_png(output_path, result->width, result->height, result->channel, result->data, 0);
        printf("[Hires Fix] Done! Output: %dx%d, channels: %d\n", result->width, result->height, result->channel);
        free(result->data);
        free(result);
    } else {
        fprintf(stderr, "Image generation failed\n");
    }

    free_sd_ctx(sd_ctx);
    // 不再需要释放 mask（因为是空）
    
    // 只有当 upscaled_image.data 是从 stbi_load 来的才用 stbi_image_free
    if (upscaled_image.data != img_data && upscaled_image.data != NULL) {
        free(upscaled_image.data);
    }
    stbi_image_free(img_data);

    return result ? 0 : 1;
}
