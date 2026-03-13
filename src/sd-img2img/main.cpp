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

static void blend_tile_to_canvas(uint8_t* canvas, int canvas_w, int canvas_h, 
                               uint8_t* tile, int tile_w, int tile_h,
                               int x, int y, int overlap) {
    for (int i = 0; i < tile_h && (y + i) < canvas_h; i++) {
        for (int j = 0; j < tile_w && (x + j) < canvas_w; j++) {
            float weight = 1.0f;
            float w_x = 1.0f, w_y = 1.0f;
            
            if (j < overlap && x > 0) w_x = (float)j / overlap;
            if (i < overlap && y > 0) w_y = (float)i / overlap;
            weight = w_x * w_y;
            
            int canvas_idx = ((y + i) * canvas_w + (x + j)) * 3;
            int tile_idx = (i * tile_w + j) * 3;
            
            for (int c = 0; c < 3; c++) {
                canvas[canvas_idx + c] = (uint8_t)(canvas[canvas_idx + c] * (1.0f - weight) + tile[tile_idx + c] * weight);
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
    const char* taesd_path = nullptr;
    const char* llm_path = nullptr;
    const char* input_path = nullptr;
    const char* output_path = "output.png";
    const char* prompt = "";
    float strength = 0.4f;
    int steps = 30;
    int tile_size = 1024;
    int overlap = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "--diffusion-model") == 0) diffusion_model_path = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0) vae_path = argv[++i];
        else if (strcmp(argv[i], "--taesd") == 0) taesd_path = argv[++i];
        else if (strcmp(argv[i], "--llm") == 0) llm_path = argv[++i];
        else if (strcmp(argv[i], "--input") == 0) input_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0) output_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0) prompt = argv[++i];
        else if (strcmp(argv[i], "--strength") == 0) strength = atof(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0) steps = atoi(argv[++i]);
    }

    if (!model_path || !input_path) {
        printf("Usage: --model <path> --input <path> --prompt <text>\n");
        return 1;
    }

    sd_set_log_callback(log_cb, NULL);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path;
    if (diffusion_model_path) ctx_params.diffusion_model_path = diffusion_model_path;
    if (vae_path) ctx_params.vae_path = vae_path;
    if (taesd_path) ctx_params.taesd_path = taesd_path;
    if (llm_path) ctx_params.llm_path = llm_path;
    ctx_params.wtype = SD_TYPE_F16;
    ctx_params.n_threads = 4;
    ctx_params.flash_attn = true;
    ctx_params.diffusion_flash_attn = true;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create SD context\n");
        return 1;
    }

    int w, h, c;
    uint8_t* raw_img = stbi_load(input_path, &w, &h, &c, 3);
    if (!raw_img) return 1;
    uint8_t* final_canvas = (uint8_t*)calloc(w * h * 3, 1);

    std::string final_p = "masterpiece, ultra-detailed, sharp focus, 8k, " + std::string(prompt);

    printf("[START] Tiled Processing for %dx%d image...\n", w, h);

    for (int y = 0; y < h; y += (tile_size - overlap)) {
        for (int x = 0; x < w; x += (tile_size - overlap)) {
            
            int cur_w = std::min(tile_size, w - x);
            int cur_h = std::min(tile_size, h - y);
            int rw = align64(cur_w); 
            int rh = align64(cur_h);

            uint8_t* tile_raw = (uint8_t*)calloc(rw * rh * 3, 1);
            for (int i = 0; i < cur_h; i++) {
                memcpy(tile_raw + i * rw * 3, raw_img + ((y + i) * w + x) * 3, cur_w * 3);
            }
            sd_image_t tile_img = {(uint32_t)rw, (uint32_t)rh, 3, tile_raw};

            sd_set_progress_callback(progress_cb, NULL);

            sd_sample_params_t sample_params;
            sd_sample_params_init(&sample_params);
            sample_params.sample_method = EULER_A_SAMPLE_METHOD;
            sample_params.scheduler = KARRAS_SCHEDULER;
            sample_params.sample_steps = steps;

            sd_img_gen_params_t img_params;
            sd_img_gen_params_init(&img_params);
            img_params.prompt = final_p.c_str();
            img_params.negative_prompt = "";
            img_params.init_image = tile_img;
            img_params.width = rw;
            img_params.height = rh;
            img_params.strength = strength;
            img_params.seed = 42;
            img_params.sample_params = sample_params;
            img_params.vae_tiling_params.enabled = true;
            img_params.vae_tiling_params.tile_size_x = 128;
            img_params.vae_tiling_params.tile_size_y = 128;
            img_params.init_image = tile_img;
            img_params.width = rw;
            img_params.height = rh;
            img_params.strength = strength;
            img_params.seed = 42;
            img_params.sample_params = sample_params;

            sd_image_t* out_tile = generate_image(ctx, &img_params);

            if (out_tile && out_tile->data) {
                blend_tile_to_canvas(final_canvas, w, h, out_tile->data, rw, rh, x, y, overlap);
                free(out_tile->data);
                free(out_tile);
            }
            free(tile_raw);
            printf("\n[DONE] Tile at (%d,%d)\n", x, y);
        }
    }

    stbi_write_png(output_path, w, h, 3, final_canvas, 0);
    printf("[SUCCESS] Saved to %s\n", output_path);

    stbi_image_free(raw_img);
    free(final_canvas);
    free_sd_ctx(ctx);

    return 0;
}
