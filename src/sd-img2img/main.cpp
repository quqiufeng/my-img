// =============================================================================
// sd-img2img: 图生图工具 - 支持 Deep HighRes Fix
// =============================================================================
//
// Deep HighRes Fix 实现（多次调用版）:
// 由于 stable-diffusion.cpp 的 API 不暴露中间 latent，我们使用多次调用策略：
//
// Phase 1: 低分辨率生成（strength=1.0，即 txt2img）
// Phase 2: 像素空间放大 + img2img
// Phase 3: 最终分辨率 img2img
//
// 用法:
//   sd-img2img \
//     --diffusion-model <path> \
//     --vae <path> \
//     --llm <path> \
//     --input <image> \
//     --output <image> \
//     --prompt <text> \
//     [--deep-hires] \
//     [--target-width 1024] \
//     [--target-height 1024]
//
// =============================================================================

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

struct Args {
    const char* diffusion_model = nullptr;
    const char* vae = nullptr;
    const char* llm = nullptr;
    const char* input = nullptr;
    const char* output = "output.png";
    const char* prompt = "";
    const char* negative_prompt = "";
    float strength = 0.45f;
    int steps = 30;
    int64_t seed = 42;
    float cfg_scale = 7.0f;
    bool deep_hires = false;
    int target_width = 0;
    int target_height = 0;
    bool vae_tiling = false;
    bool flash_attn = false;
    bool use_gpu = true;
    bool verbose = false;
};

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--diffusion-model") == 0 && i + 1 < argc) {
            args.diffusion_model = argv[++i];
        } else if (strcmp(argv[i], "--vae") == 0 && i + 1 < argc) {
            args.vae = argv[++i];
        } else if (strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            args.llm = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args.input = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            args.output = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--negative-prompt") == 0 && i + 1 < argc) {
            args.negative_prompt = argv[++i];
        } else if (strcmp(argv[i], "--strength") == 0 && i + 1 < argc) {
            args.strength = atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            args.steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--cfg-scale") == 0 && i + 1 < argc) {
            args.cfg_scale = atof(argv[++i]);
        } else if (strcmp(argv[i], "--deep-hires") == 0) {
            args.deep_hires = true;
        } else if (strcmp(argv[i], "--target-width") == 0 && i + 1 < argc) {
            args.target_width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--target-height") == 0 && i + 1 < argc) {
            args.target_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vae-tiling") == 0) {
            args.vae_tiling = true;
        } else if (strcmp(argv[i], "--flash-attn") == 0) {
            args.flash_attn = true;
        } else if (strcmp(argv[i], "--cpu") == 0) {
            args.use_gpu = false;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            args.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: sd-img2img [options]\n");
            printf("\nRequired:\n");
            printf("  --diffusion-model <path>  Path to diffusion model (.gguf)\n");
            printf("  --vae <path>              Path to VAE model\n");
            printf("  --llm <path>              Path to LLM/CLIP model\n");
            printf("  --input <path>            Input image path\n");
            printf("\nOptional:\n");
            printf("  --output <path>           Output image path (default: output.png)\n");
            printf("  --prompt <text>           Prompt for generation\n");
            printf("  --negative-prompt <text>  Negative prompt\n");
            printf("  --strength <float>        Denoising strength 0.0-1.0 (default: 0.45)\n");
            printf("  --steps <int>             Sampling steps (default: 30)\n");
            printf("  --seed <int>              Random seed (default: 42)\n");
            printf("\nDeep HighRes Fix:\n");
            printf("  --deep-hires              Enable Deep HighRes Fix\n");
            printf("  --target-width <int>      Target width\n");
            printf("  --target-height <int>     Target height\n");
            printf("\nOther:\n");
            printf("  --vae-tiling              Enable VAE tiling\n");
            printf("  --flash-attn              Enable Flash Attention\n");
            printf("  --cpu                     Use CPU instead of GPU\n");
            printf("  --verbose, -v             Verbose output\n");
            return false;
        }
    }

    if (!args.diffusion_model || !args.vae || !args.llm || !args.input) {
        printf("Error: Missing required arguments. Use --help for usage.\n");
        return false;
    }

    return true;
}

static void log_callback(enum sd_log_level_t level, const char* text, void* data) {
    Args* args = (Args*)data;
    if (!args->verbose && level > SD_LOG_INFO)
        return;
    printf("%s", text);
}

static void progress_callback(int step, int steps, float time, void* data) {
    (void)time;
    (void)data;
    printf("\r[Progress] Step %d/%d", step, steps);
    fflush(stdout);
}

static sd_image_t load_image(const char* path) {
    int w, h, c;
    uint8_t* data = stbi_load(path, &w, &h, &c, 3);
    if (!data) {
        fprintf(stderr, "[ERROR] Failed to load image: %s\n", path);
        return {0, 0, 0, nullptr};
    }
    return {(uint32_t)w, (uint32_t)h, 3, data};
}

static bool save_image(const sd_image_t& image, const char* path) {
    return stbi_write_png(path, image.width, image.height, image.channel, image.data, 0) != 0;
}

static void resize_image(const sd_image_t& src, int target_w, int target_h, std::vector<uint8_t>& dst_data,
                         sd_image_t& out_image) {
    dst_data.resize((size_t)target_w * target_h * 3);
    stbir_resize_uint8(src.data, src.width, src.height, 0, dst_data.data(), target_w, target_h, 0, 3);
    out_image = {(uint32_t)target_w, (uint32_t)target_h, 3, dst_data.data()};
}

static void free_image(sd_image_t& image) {
    if (image.data) {
        stbi_image_free(image.data);
        image.data = nullptr;
    }
}

struct SdImageDeleter {
    void operator()(sd_image_t* p) const {
        if (p) {
            if (p->data)
                std::free(p->data);
            std::free(p);
        }
    }
};
using SdImagePtr = std::unique_ptr<sd_image_t, SdImageDeleter>;

static sd_ctx_t* create_sd_context(const Args& args) {
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.diffusion_model_path = args.diffusion_model;
    ctx_params.vae_path = args.vae;
    ctx_params.llm_path = args.llm;
    ctx_params.n_threads = 4;
    ctx_params.offload_params_to_cpu = !args.use_gpu;
    ctx_params.keep_vae_on_cpu = !args.use_gpu;
    ctx_params.keep_clip_on_cpu = !args.use_gpu;
    ctx_params.flash_attn = args.use_gpu && args.flash_attn;
    ctx_params.diffusion_flash_attn = args.use_gpu && args.flash_attn;
    ctx_params.vae_decode_only = false;

    return new_sd_ctx(&ctx_params);
}

static sd_image_t* generate_single(sd_ctx_t* ctx, const sd_image_t* init_image, int width, int height, const Args& args,
                                   float strength, int steps, int64_t seed_offset, bool hires_fix = false,
                                   int hires_width = 0, int hires_height = 0, float hires_strength = 0.5f) {
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);

    params.prompt = args.prompt;
    params.negative_prompt = args.negative_prompt;
    params.width = width;
    params.height = height;
    params.strength = strength;
    params.seed = args.seed + seed_offset;
    params.sample_params.sample_steps = steps;
    params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    params.sample_params.scheduler = KARRAS_SCHEDULER;
    params.sample_params.guidance.txt_cfg = args.cfg_scale;

    if (init_image && init_image->data) {
        params.init_image = *init_image;
    }

    // HiRes Fix 参数
    params.hires_fix = hires_fix;
    if (hires_fix) {
        params.hires_width = hires_width;
        params.hires_height = hires_height;
        params.hires_strength = hires_strength;
    }

    // 自动启用 VAE tiling 避免大图显存不足和接缝问题
    // 根据目标分辨率动态调整 tile size 和 overlap
    params.vae_tiling_params.enabled = true;
    if (width >= 2048 || height >= 2048 || hires_fix) {
        // 超大图或 HiRes Fix：大 tile + 大 overlap，避免明显接缝
        params.vae_tiling_params.tile_size_x = 512;
        params.vae_tiling_params.tile_size_y = 512;
        params.vae_tiling_params.target_overlap = 256;
    } else if (width >= 1280 || height >= 1280) {
        // 大图：中等 tile + 大 overlap
        params.vae_tiling_params.tile_size_x = 512;
        params.vae_tiling_params.tile_size_y = 512;
        params.vae_tiling_params.target_overlap = 128;
    } else {
        // 小图：标准参数
        params.vae_tiling_params.tile_size_x = 128;
        params.vae_tiling_params.tile_size_y = 128;
        params.vae_tiling_params.target_overlap = 16;
    }

    return generate_image(ctx, &params);
}

static sd_image_t* deep_hires_generate(sd_ctx_t* ctx, const sd_image_t& input_image, const Args& args) {
    printf("\n[Deep HighRes Fix] Starting native hires_fix generation...\n");

    int target_w = args.target_width > 0 ? args.target_width : input_image.width;
    int target_h = args.target_height > 0 ? args.target_height : input_image.height;
    target_w = (target_w + 63) & ~63;
    target_h = (target_h + 63) & ~63;

    printf("[Deep Hires] Target resolution: %dx%d\n", target_w, target_h);

    // 计算基础分辨率（第一阶段在低分辨率运行）
    // 使用输入图尺寸或目标尺寸的一半，取较小者
    int base_w = std::min((int)input_image.width, target_w / 2);
    int base_h = std::min((int)input_image.height, target_h / 2);
    base_w = (base_w + 63) & ~63;
    base_h = (base_h + 63) & ~63;
    base_w = std::max(base_w, 64);
    base_h = std::max(base_h, 64);

    printf("[Deep Hires] Base resolution: %dx%d -> Target: %dx%d\n", base_w, base_h, target_w, target_h);
    printf("[Deep Hires] Phase 1: img2img at base resolution (strength=0.75)\n");
    printf("[Deep Hires] Phase 2: latent upscale + denoise (strength=%.2f)\n", args.strength);

    // 使用原生 hires_fix：单次调用 generate_image，内部自动处理两阶段采样
    SdImagePtr result(generate_single(ctx, &input_image, base_w, base_h, args,
                                      0.75f,        // Phase 1: 基础分辨率 img2img strength
                                      args.steps,   // 总步数（两阶段共享）
                                      0,            // seed offset
                                      true,         // hires_fix = true
                                      target_w,     // hires_width
                                      target_h,     // hires_height
                                      args.strength // Phase 2: 高分率去噪 strength
                                      ));

    if (!result || !result->data) {
        fprintf(stderr, "[ERROR] Deep HighRes Fix generation failed\n");
        return nullptr;
    }

    printf("\n[Deep HighRes Fix] Completed! Output: %dx%d\n", result->width, result->height);
    return result.release();
}

static sd_image_t* normal_img2img(sd_ctx_t* ctx, const sd_image_t& input_image, const Args& args) {
    printf("\n[Normal Img2Img] Generating...\n");

    int target_w = args.target_width > 0 ? args.target_width : input_image.width;
    int target_h = args.target_height > 0 ? args.target_height : input_image.height;
    target_w = (target_w + 63) & ~63;
    target_h = (target_h + 63) & ~63;

    sd_image_t* init_image = const_cast<sd_image_t*>(&input_image);
    std::vector<uint8_t> resized_buffer;
    sd_image_t resized = {};

    if (target_w != (int)input_image.width || target_h != (int)input_image.height) {
        printf("[Info] Resizing input from %dx%d to %dx%d\n", input_image.width, input_image.height, target_w,
               target_h);
        resize_image(input_image, target_w, target_h, resized_buffer, resized);
        init_image = &resized;
    }

    return generate_single(ctx, init_image, target_w, target_h, args, args.strength, args.steps, 0);
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    sd_set_log_callback(log_callback, &args);
    sd_set_progress_callback(progress_callback, nullptr);

    printf("[INFO] Loading input image: %s\n", args.input);
    sd_image_t input_image = load_image(args.input);
    if (!input_image.data) {
        return 1;
    }
    printf("[INFO] Input image: %dx%d\n", input_image.width, input_image.height);

    printf("[INFO] Loading model...\n");
    sd_ctx_t* ctx = create_sd_context(args);
    if (!ctx) {
        fprintf(stderr, "[ERROR] Failed to create SD context\n");
        free_image(input_image);
        return 1;
    }
    printf("[INFO] Model loaded successfully\n");

    SdImagePtr result;
    if (args.deep_hires) {
        result.reset(deep_hires_generate(ctx, input_image, args));
    } else {
        result.reset(normal_img2img(ctx, input_image, args));
    }

    printf("\n");

    if (!result || !result->data) {
        fprintf(stderr, "[ERROR] Image generation failed\n");
        free_image(input_image);
        free_sd_ctx(ctx);
        return 1;
    }

    printf("[INFO] Saving to: %s (%dx%d)\n", args.output, result->width, result->height);
    if (!save_image(*result, args.output)) {
        fprintf(stderr, "[ERROR] Failed to save output\n");
    } else {
        printf("[SUCCESS] Image saved to: %s\n", args.output);
    }

    free_image(input_image);
    free_sd_ctx(ctx);

    return 0;
}
