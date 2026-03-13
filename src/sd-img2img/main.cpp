// =============================================================================
// sd-img2img: 高清修复工具 - 3080 (10GB VRAM) 专用
// =============================================================================
// 
// 【重要调试记录 - 2025-03-13】
// 
// 问题：之前一直报错 "GGML_ASSERT(!decode_only || decode_graph)"
// 原因：误以为是官方stable-diffusion.cpp库的bug
// 
// 结论：官方库没有任何问题！问题在于调用方式。
// 
// 正确调用方式：
// 1. ctx_params.vae_decode_only = false;  // img2img需要encoder
// 2. ctx_params.keep_vae_on_cpu = true;   // 10GB VRAM优化
// 3. 使用 --diffusion-model 参数加载GGUF模型
// 4. 必须传 --vae 和 --llm 参数
//
// 调试命令（已验证可工作）：
// cd /home/dministrator/my-img && timeout 120 ./build/sd-img2img \
//   --diffusion-model /opt/image/z_image_turbo-Q6_K.gguf \
//   --vae /opt/image/ae.safetensors \
//   --llm /opt/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
//   --input /home/dministrator/hires_final_step1.png \
//   --output /home/dministrator/test_output.png \
//   --prompt "lovely lady" --strength 0.45 --steps 8 --seed 42
//
// =============================================================================

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <string>

// =============================================================================
// 1. 线性羽化拼接 (消除 Tile 物理接缝)
// =============================================================================
// 原理：相邻分块有64像素重叠区域，根据距离计算权重进行Alpha混合
// 权重计算：距离边缘越近，新像素权重越低，实现平滑过渡
static void blend_tile_to_canvas(uint8_t* canvas, int canvas_w, int canvas_h,
                                uint8_t* tile, int tile_w, int tile_h,
                                int x, int y, int overlap) {
    for (int i = 0; i < tile_h && (y + i) < canvas_h; i++) {
        for (int j = 0; j < tile_w && (x + j) < canvas_w; j++) {
            float w_x = 1.0f, w_y = 1.0f;
            if (j < overlap && x > 0) w_x = (float)j / overlap;
            if (i < overlap && y > 0) w_y = (float)i / overlap;
            float weight = w_x * w_y;
            
            int c_idx = ((y + i) * canvas_w + (x + j)) * 3;
            int t_idx = (i * tile_w + j) * 3;
            for (int c = 0; c < 3; c++) {
                canvas[c_idx + c] = (uint8_t)(canvas[c_idx + c] * (1.0f - weight) + tile[t_idx + c] * weight);
            }
        }
    }
}

// =============================================================================
// 2. 图像后期像素增强
// =============================================================================
static void apply_final_enhancement(uint8_t* data, int w, int h) {
    const float contrast = 1.05f;    // 降低对比度
    const float brightness = 0.0f;   // 移除亮度调整
    const float saturation = 1.05f;   // 降低饱和度
    const float sharpen_strength = 0.15f; // 降低锐化

    // A. 对比度与饱和度处理
    for (int i = 0; i < w * h; i++) {
        for (int c = 0; c < 3; c++) {
            float p = data[i * 3 + c];
            p = (p - 128.0f) * contrast + 128.0f + brightness;
            data[i * 3 + c] = (uint8_t)std::clamp(p, 0.0f, 255.0f);
        }
        float r = data[i * 3 + 0], g = data[i * 3 + 1], b = data[i * 3 + 2];
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        data[i * 3 + 0] = (uint8_t)std::clamp(gray + (r - gray) * saturation, 0.0f, 255.0f);
        data[i * 3 + 1] = (uint8_t)std::clamp(gray + (g - gray) * saturation, 0.0f, 255.0f);
        data[i * 3 + 2] = (uint8_t)std::clamp(gray + (b - gray) * saturation, 0.0f, 255.0f);
    }

    // B. 细节锐化 (Laplacian算子)
    std::vector<uint8_t> backup(data, data + w * h * 3);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = (y * w + x) * 3 + c;
                float center = backup[idx];
                float surround = (backup[((y-1)*w + x)*3 + c] + backup[((y+1)*w + x)*3 + c] +
                                  backup[(y*w + (x-1))*3 + c] + backup[(y*w + (x+1))*3 + c]) / 4.0f;
                float val = center + sharpen_strength * (center - surround);
                data[idx] = (uint8_t)std::clamp(val, 0.0f, 255.0f);
            }
        }
    }
}

static int align64(int v) { return (v + 63) & ~63; }

void log_cb(enum sd_log_level_t level, const char* text, void* data) {
    if (level >= SD_LOG_INFO) printf("%s\n", text);
}

void progress_cb(int step, int steps, float time, void* data) {
    printf("\r[TILE] Step %d/%d", step, steps);
    fflush(stdout);
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* diffusion_model_path = nullptr;
    const char* vae_path = nullptr;
    const char* llm_path = nullptr;
    const char* input_path = nullptr;
    const char* output_path = "output.png";
    const char* prompt = "";
    float strength = 0.35f;
    int steps = 30;
    int64_t seed = 42;
    int tile_size = 1024;
    int overlap = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "--diffusion-model") == 0) diffusion_model_path = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0) vae_path = argv[++i];
        else if (strcmp(argv[i], "--llm") == 0) llm_path = argv[++i];
        else if (strcmp(argv[i], "--input") == 0) input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0) output_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0) prompt = argv[++i];
        else if (strcmp(argv[i], "--strength") == 0) strength = atof(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0) seed = atoll(argv[++i]);
        else if (strcmp(argv[i], "--tile-size") == 0) tile_size = atoi(argv[++i]);
    }

    if ((!model_path && !diffusion_model_path) || !input_path) {
        printf("Usage: --model <path> OR --diffusion-model <path> --input <path> --prompt <text> [--strength 0.35] [--steps 30] [--tile-size 1024]\n");
        return 1;
    }

    sd_set_log_callback(log_cb, NULL);
    sd_set_progress_callback(progress_cb, NULL);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path;
    if (diffusion_model_path) ctx_params.diffusion_model_path = diffusion_model_path;
    if (vae_path) ctx_params.vae_path = vae_path;
    if (llm_path) ctx_params.llm_path = llm_path;
    ctx_params.flash_attn = false;  // 禁用可能影响多tile的选项
    ctx_params.diffusion_flash_attn = false;
    ctx_params.vae_decode_only = false;
    ctx_params.keep_vae_on_cpu = true;
    ctx_params.n_threads = 4;

    printf("[INFO] Creating context...\n");
    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) {
        printf("[ERROR] Failed to create context\n");
        return 1;
    }

    int w, h, c;
    uint8_t* raw = stbi_load(input_path, &w, &h, &c, 3);
    if (!raw) {
        printf("[ERROR] Failed to load image: %s\n", input_path);
        return 1;
    }

    uint8_t* canvas = (uint8_t*)calloc(w * h * 3, 1);
    std::string full_prompt = "masterpiece, ultra-detailed, sharp focus, 8k wallpaper, highly intricate, ";
    full_prompt += prompt;

    // 10GB VRAM: 手动分块处理 (2x2网格)
    bool use_tiling = true;
    
    if (!use_tiling) {
        // 小图：直接处理
        printf("[START] Small image (%dx%d), direct processing...\n", w, h);
        sd_image_t input_image = {(uint32_t)w, (uint32_t)h, 3, raw};
        
        sd_img_gen_params_t img_params;
        sd_img_gen_params_init(&img_params);
        img_params.prompt = full_prompt.c_str();
        img_params.init_image = input_image;
        img_params.width = w;
        img_params.height = h;
        img_params.strength = strength;
        img_params.seed = seed;
        img_params.sample_params.sample_steps = steps;
        img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        img_params.sample_params.scheduler = KARRAS_SCHEDULER;
        // 禁用VAE tiling，使用手动分块
        img_params.vae_tiling_params.enabled = false;

        sd_image_t* result = generate_image(ctx, &img_params);
        if (result && result->data) {
            apply_final_enhancement(result->data, result->width, result->height);
            stbi_write_png(output_path, result->width, result->height, 3, result->data, 0);
            printf("[SUCCESS] Saved to %s\n", output_path);
            free(result->data);
            free(result);
        }
    } else {
    // 手动分块处理: 2x2网格 - 每个tile新建ctx
    int tiles_x = 2;
    int tiles_y = 2;
    int tile_w = w / tiles_x;
    int tile_h = h / tiles_y;
    int overlap = 64;

    printf("[START] Tiled Processing: 2x2 grid (%dx%d tiles, overlap=%d)...\n", 
           tile_w, tile_h, overlap);

    for (int ty = 0; ty < tiles_y; ty++) {
        for (int tx = 0; tx < tiles_x; tx++) {
            // 每个tile新建ctx避免状态问题
            sd_ctx_t* tile_ctx = new_sd_ctx(&ctx_params);
            if (!tile_ctx) {
                printf("[ERROR] Failed to create tile ctx\n");
                continue;
            }
            
            int x = tx * tile_w;
            int y = ty * tile_h;
            int cur_w = tile_w;
            int cur_h = tile_h;
            
            uint8_t* tile_raw = (uint8_t*)calloc(cur_w * cur_h * 3, 1);
            for (int i = 0; i < cur_h; i++) {
                memcpy(tile_raw + i * cur_w * 3, raw + ((y + i) * w + x) * 3, cur_w * 3);
            }
            sd_image_t tile_img = {(uint32_t)cur_w, (uint32_t)cur_h, 3, tile_raw};

            sd_img_gen_params_t img_params;
            sd_img_gen_params_init(&img_params);
            img_params.prompt = full_prompt.c_str();
            img_params.init_image = tile_img;
            img_params.width = cur_w;
            img_params.height = cur_h;
            img_params.strength = strength;
            img_params.seed = seed;
            img_params.sample_params.sample_steps = steps;
            img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
            img_params.sample_params.scheduler = KARRAS_SCHEDULER;
            img_params.vae_tiling_params.enabled = false;

            printf("[TILE] Processing tile (%d,%d) at (%d,%d)...\n", tx, ty, x, y);
            sd_image_t* out_tile = generate_image(tile_ctx, &img_params);

            if (out_tile && out_tile->data) {
                blend_tile_to_canvas(canvas, w, h, out_tile->data, out_tile->width, out_tile->height, x, y, overlap);
                free(out_tile->data);
                free(out_tile);
                printf("[TILE] Done tile (%d,%d)\n", tx, ty);
            } else {
                printf("[ERROR] Failed to generate tile (%d,%d)\n", tx, ty);
            }
            free(tile_raw);
            free_sd_ctx(tile_ctx);
        }
    }

        // 后期增强 - 注释掉测试原始拼接效果
        // printf("[INFO] Applying contrast and sharpening...\n");
        // apply_final_enhancement(canvas, w, h);

        stbi_write_png(output_path, w, h, 3, canvas, 0);
        printf("[SUCCESS] Saved to %s\n", output_path);
    }

    stbi_image_free(raw);
    free(canvas);
    free_sd_ctx(ctx);
    return 0;
}
