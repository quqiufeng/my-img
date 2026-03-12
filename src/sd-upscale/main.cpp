#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

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

void print_help() {
    printf("Usage: sd-upscale [options]\n");
    printf("\nOptions:\n");
    printf("  --model <path>        Upscale model path (.gguf)\n");
    printf("  --input <path>        Input image path\n");
    printf("  --output <path>       Output image path\n");
    printf("  --scale <num>         Upscale factor: 2 or 4 (default: 2)\n");
    printf("  --tile-size <num>     Tile size (default: 512)\n");
    printf("  --cpu                 Use CPU only (default: GPU)\n");
    printf("  --debug               Print debug info\n");
    printf("  --help                Show this help\n");
}

sd_image_t load_image(const char* path) {
    int w, h, c;
    uint8_t* data = stbi_load(path, &w, &h, &c, 3);
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

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* input_path = nullptr;
    const char* output_path = "output.png";
    int scale = 2;
    int tile_size = 512;
    bool use_gpu = true;
    bool debug = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            scale = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc) {
            tile_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else if (strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
    }

    if (!model_path || !input_path) {
        fprintf(stderr, "Error: --model and --input are required\n\n");
        print_help();
        return 1;
    }

    if (debug) {
        printf("[DEBUG] Model: %s\n", model_path);
        printf("[DEBUG] Input: %s\n", input_path);
        printf("[DEBUG] Output: %s\n", output_path);
        printf("[DEBUG] Scale: %dx\n", scale);
        printf("[DEBUG] Tile size: %d\n", tile_size);
        printf("[DEBUG] GPU: %s\n", use_gpu ? "yes" : "no");
    }

    sd_set_log_callback(log_callback, nullptr);

    sd_image_t input_image = load_image(input_path);
    if (!input_image.data) {
        return 1;
    }

    printf("Input image: %dx%d\n", input_image.width, input_image.height);

    upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(
        model_path,
        !use_gpu,
        false,
        4,
        tile_size
    );

    if (!upscaler_ctx) {
        fprintf(stderr, "Failed to create upscaler context\n");
        free(input_image.data);
        return 1;
    }

    int real_scale = get_upscale_factor(upscaler_ctx);
    printf("Upscale model scale: %dx\n", real_scale);

    if (scale != real_scale) {
        printf("Note: requested scale %dx, but model is %dx, using model scale\n", scale, real_scale);
        scale = real_scale;
    }

    printf("Upscaling...\n");

    sd_image_t result = upscale(upscaler_ctx, input_image, scale);

    if (result.data) {
        save_image(result, output_path);
        printf("Output image: %dx%d\n", result.width, result.height);
        printf("Image saved to: %s\n", output_path);
        free(result.data);
    } else {
        fprintf(stderr, "Upscale failed\n");
    }

    free_upscaler_ctx(upscaler_ctx);
    free(input_image.data);

    return result.data ? 0 : 1;
}
