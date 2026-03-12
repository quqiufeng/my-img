#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

void print_help() {
    printf("Usage: sd-img2img [options]\n");
    printf("\nOptions:\n");
    printf("  --model <path>          SD model path (.gguf) - main model\n");
    printf("  --diffusion-model <path> Diffusion model path (for some SD variants)\n");
    printf("  --vae <path>            VAE model path\n");
    printf("  --llm <path>            LLM model path (for Flux/Janus)\n");
    printf("  --input <path>          Input image path\n");
    printf("  --output <path>        Output image path\n");
    printf("  --prompt <text>         Positive prompt\n");
    printf("  --negative-prompt <text> Negative prompt\n");
    printf("  --strength <0.0-1.0>    Denoising strength (default: 0.45)\n");
    printf("  --steps <num>           Sampling steps (default: 20)\n");
    printf("  --seed <num>            Random seed\n");
    printf("  --width <num>           Output width\n");
    printf("  --height <num>          Output height\n");
    printf("  --scheduler <name>     Scheduler (default/karras/exponential/ays/gits/sgm_uniform/simple/smoothstep/kl_optimal/lcm/bong_tangent)\n");
    printf("  --sample-method <name>  Sample method (euler/euler_a/heun/dpm2/dpmpp_2s_a/dpmpp_2m/dpmpp_2m_v2/ipndm/ipndm_v/lcm/ddim_trailing/tcd/res_multistep/res_2s)\n");
    printf("  --gpu                   Use GPU (default)\n");
    printf("  --cpu                   Use CPU only\n");
    printf("  --no-flash-attn         Disable Flash Attention\n");
    printf("  --debug                 Print debug info\n");
    printf("  --help                  Show this help\n");
}

sd_image_t load_image(const char* path) {
    int w, h, c;
    uint8_t* data = stbi_load(path, &w, &h, &c, 3);  // Force 3 channels
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", path);
        return {0, 0, 0, nullptr};
    }
    return {(uint32_t)w, (uint32_t)h, 3, data};
}

void save_image(const sd_image_t& image, const char* path) {
    if (!stbi_write_png(path, image.width, image.height, image.channel, image.data, 0)) {
        fprintf(stderr, "Failed to save image: %s\n", path);
    }
}

void log_callback(enum sd_log_level_t level, const char* text, void* data) {
    const char* prefix = "";
    switch (level) {
        case SD_LOG_DEBUG: prefix = "[DEBUG]"; break;
        case SD_LOG_INFO: prefix = "[INFO]"; break;
        case SD_LOG_WARN: prefix = "[WARN]"; break;
        case SD_LOG_ERROR: prefix = "[ERROR]"; break;
    }
    printf("%s %s\n", prefix, text);
}

void progress_callback(int step, int steps, float time, void* data) {
    printf("\r[PROGRESS] Step %d/%d (%.2fs)", step, steps, time);
    fflush(stdout);
    if (step == steps) {
        printf("\n");
    }
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* diffusion_model_path = nullptr;
    const char* vae_path = nullptr;
    const char* llm_path = nullptr;
    const char* input_path = nullptr;
    const char* output_path = "output.png";
    const char* prompt = "";
    const char* negative_prompt = "";
    float strength = 0.45f;
    int steps = 20;
    int64_t seed = -1;
    int width = 0;
    int height = 0;
    bool use_gpu = true;
    bool use_flash_attn = true;
    bool debug = false;
    enum scheduler_t scheduler = KARRAS_SCHEDULER;
    enum sample_method_t sample_method = EULER_A_SAMPLE_METHOD;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--diffusion-model") == 0 && i + 1 < argc) {
            diffusion_model_path = argv[++i];
        } else if (strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            vae_path = argv[++i];
        } else if (strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            llm_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--negative-prompt") == 0 && i + 1 < argc) {
            negative_prompt = argv[++i];
        } else if (strcmp(argv[i], "--strength") == 0 && i + 1 < argc) {
            strength = atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (strcmp(argv[i], "--no-flash-attn") == 0) {
            use_flash_attn = false;
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (strcmp(argv[i], "--scheduler") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if (strcmp(s, "default") == 0) scheduler = DISCRETE_SCHEDULER;
            else if (strcmp(s, "karras") == 0) scheduler = KARRAS_SCHEDULER;
            else if (strcmp(s, "exponential") == 0) scheduler = EXPONENTIAL_SCHEDULER;
            else if (strcmp(s, "ays") == 0) scheduler = AYS_SCHEDULER;
            else if (strcmp(s, "gits") == 0) scheduler = GITS_SCHEDULER;
            else if (strcmp(s, "sgm_uniform") == 0) scheduler = SGM_UNIFORM_SCHEDULER;
            else if (strcmp(s, "simple") == 0) scheduler = SIMPLE_SCHEDULER;
            else if (strcmp(s, "smoothstep") == 0) scheduler = SMOOTHSTEP_SCHEDULER;
            else if (strcmp(s, "kl_optimal") == 0) scheduler = KL_OPTIMAL_SCHEDULER;
            else if (strcmp(s, "lcm") == 0) scheduler = LCM_SCHEDULER;
            else if (strcmp(s, "bong_tangent") == 0) scheduler = BONG_TANGENT_SCHEDULER;
        } else if (strcmp(argv[i], "--sample-method") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if (strcmp(s, "euler") == 0) sample_method = EULER_SAMPLE_METHOD;
            else if (strcmp(s, "euler_a") == 0) sample_method = EULER_A_SAMPLE_METHOD;
            else if (strcmp(s, "heun") == 0) sample_method = HEUN_SAMPLE_METHOD;
            else if (strcmp(s, "dpm2") == 0) sample_method = DPM2_SAMPLE_METHOD;
            else if (strcmp(s, "dpmpp_2s_a") == 0) sample_method = DPMPP2S_A_SAMPLE_METHOD;
            else if (strcmp(s, "dpmpp_2m") == 0) sample_method = DPMPP2M_SAMPLE_METHOD;
            else if (strcmp(s, "dpmpp_2m_v2") == 0) sample_method = DPMPP2Mv2_SAMPLE_METHOD;
            else if (strcmp(s, "ipndm") == 0) sample_method = IPNDM_SAMPLE_METHOD;
            else if (strcmp(s, "ipndm_v") == 0) sample_method = IPNDM_V_SAMPLE_METHOD;
            else if (strcmp(s, "lcm") == 0) sample_method = LCM_SAMPLE_METHOD;
            else if (strcmp(s, "ddim_trailing") == 0) sample_method = DDIM_TRAILING_SAMPLE_METHOD;
            else if (strcmp(s, "tcd") == 0) sample_method = TCD_SAMPLE_METHOD;
            else if (strcmp(s, "res_multistep") == 0) sample_method = RES_MULTISTEP_SAMPLE_METHOD;
            else if (strcmp(s, "res_2s") == 0) sample_method = RES_2S_SAMPLE_METHOD;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
    }

    if (!diffusion_model_path || !input_path) {
        fprintf(stderr, "Error: --diffusion-model and --input are required\n\n");
        print_help();
        return 1;
    }

    if (debug) {
        printf("[DEBUG] Model: %s\n", model_path);
        if (diffusion_model_path) printf("[DEBUG] Diffusion Model: %s\n", diffusion_model_path);
        if (vae_path) printf("[DEBUG] VAE: %s\n", vae_path);
        if (llm_path) printf("[DEBUG] LLM: %s\n", llm_path);
        printf("[DEBUG] Input: %s\n", input_path);
        printf("[DEBUG] Output: %s\n", output_path);
        printf("[DEBUG] Prompt: %s\n", prompt);
        printf("[DEBUG] Strength: %.2f\n", strength);
        printf("[DEBUG] Steps: %d\n", steps);
        printf("[DEBUG] GPU: %s\n", use_gpu ? "yes" : "no");
        printf("[DEBUG] Flash Attn: %s\n", use_flash_attn ? "yes" : "no");
    }

    sd_set_log_callback(log_callback, nullptr);
    sd_set_progress_callback(progress_callback, nullptr);

    sd_image_t input_image = load_image(input_path);
    if (!input_image.data) {
        return 1;
    }

    if (width == 0) width = input_image.width;
    if (height == 0) height = input_image.height;

    sd_image_t mask_image = {0, 0, 1, nullptr};
    mask_image.width = width;
    mask_image.height = height;
    mask_image.data = (uint8_t*)malloc(width * height);
    if (mask_image.data) {
        memset(mask_image.data, 255, width * height);
    }

    if (debug) {
        printf("[DEBUG] Image size: %dx%d\n", input_image.width, input_image.height);
        printf("[DEBUG] Output size: %dx%d\n", width, height);
    }

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path;
    if (diffusion_model_path) ctx_params.diffusion_model_path = diffusion_model_path;
    if (vae_path) ctx_params.vae_path = vae_path;
    if (llm_path) ctx_params.llm_path = llm_path;
    ctx_params.wtype = SD_TYPE_COUNT;  // Auto-detect
    ctx_params.n_threads = 4;
    ctx_params.offload_params_to_cpu = !use_gpu;
    ctx_params.keep_vae_on_cpu = !use_gpu;
    ctx_params.keep_clip_on_cpu = !use_gpu;
    ctx_params.flash_attn = use_gpu && use_flash_attn;
    ctx_params.diffusion_flash_attn = use_gpu && use_flash_attn;
    ctx_params.vae_decode_only = false;

    if (debug) {
        printf("[DEBUG] Creating SD context...\n");
    }

    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    if (!sd_ctx) {
        fprintf(stderr, "Failed to create SD context\n");
        free(input_image.data);
        return 1;
    }

    sd_sample_params_t sample_params;
    sd_sample_params_init(&sample_params);
    sample_params.sample_method = sample_method;
    sample_params.scheduler = scheduler;
    sample_params.sample_steps = steps;

    sd_img_gen_params_t img_params;
    sd_img_gen_params_init(&img_params);
    img_params.prompt = prompt;
    img_params.negative_prompt = negative_prompt;
    img_params.init_image = input_image;
    img_params.mask_image = mask_image;
    img_params.width = width;
    img_params.height = height;
    img_params.strength = strength;
    img_params.seed = seed;
    img_params.sample_params = sample_params;

    if (debug) {
        printf("[DEBUG] Generating image...\n");
    }

    sd_image_t* result = generate_image(sd_ctx, &img_params);
    if (result) {
        save_image(*result, output_path);
        printf("Image saved to: %s\n", output_path);
        free(result->data);
        free(result);
    } else {
        fprintf(stderr, "Image generation failed\n");
    }

    free_sd_ctx(sd_ctx);
    free(input_image.data);
    if (mask_image.data) free(mask_image.data);

    return result ? 0 : 1;
}
