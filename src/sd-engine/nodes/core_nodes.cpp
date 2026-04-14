// ============================================================================
// sd-engine/nodes/core_nodes.cpp
// ============================================================================
//
// ComfyUI 核心节点实现（Phase 2 + 真正的分离式 API + Deep HighRes Fix）
//
// 使用 stable-diffusion.cpp 扩展的 C API：
// - sd_encode_prompt()
// - sd_create_empty_latent()
// - sd_encode_image()
// - sd_sampler_run()
// - sd_decode_latent()
// - sd_set_latent_hook() / sd_set_guidance_hook() (Deep HighRes Fix)
// ============================================================================

#include "core/node.h"
#include "stable-diffusion.h"
#include "stable-diffusion-ext.h"
#include "tensor.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <cmath>

// stb_image for LoadImage/SaveImage
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"

namespace sdengine {

// ============================================================================
// CheckpointLoaderSimple - 加载模型
// ============================================================================
class CheckpointLoaderSimpleNode : public Node {
public:
    std::string get_class_type() const override { return "CheckpointLoaderSimple"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"ckpt_name", "STRING", true, std::string("")},
            {"vae_name", "STRING", false, std::string("")},
            {"clip_name", "STRING", false, std::string("")},
            {"n_threads", "INT", false, 4},
            {"use_gpu", "BOOLEAN", false, true},
            {"flash_attn", "BOOLEAN", false, false}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"MODEL", "MODEL"},
            {"CLIP", "CLIP"},
            {"VAE", "VAE"}
        };
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string ckpt_path = std::any_cast<std::string>(inputs.at("ckpt_name"));
        std::string vae_path = inputs.count("vae_name") ? 
            std::any_cast<std::string>(inputs.at("vae_name")) : "";
        std::string clip_path = inputs.count("clip_name") ?
            std::any_cast<std::string>(inputs.at("clip_name")) : "";
        int n_threads = inputs.count("n_threads") ?
            std::any_cast<int>(inputs.at("n_threads")) : 4;
        bool use_gpu = inputs.count("use_gpu") ?
            std::any_cast<bool>(inputs.at("use_gpu")) : true;
        bool flash_attn = inputs.count("flash_attn") ?
            std::any_cast<bool>(inputs.at("flash_attn")) : false;

        if (ckpt_path.empty()) {
            fprintf(stderr, "[ERROR] CheckpointLoaderSimple: ckpt_name is required\n");
            return false;
        }

        printf("[CheckpointLoaderSimple] Loading model: %s\n", ckpt_path.c_str());

        sd_ctx_params_t ctx_params;
        sd_ctx_params_init(&ctx_params);
        ctx_params.diffusion_model_path = ckpt_path.c_str();
        if (!vae_path.empty()) {
            ctx_params.vae_path = vae_path.c_str();
        }
        if (!clip_path.empty()) {
            ctx_params.llm_path = clip_path.c_str();
        }
        ctx_params.n_threads = n_threads;
        ctx_params.offload_params_to_cpu = !use_gpu;
        ctx_params.keep_vae_on_cpu = !use_gpu;
        ctx_params.keep_clip_on_cpu = !use_gpu;
        ctx_params.flash_attn = use_gpu && flash_attn;
        ctx_params.diffusion_flash_attn = use_gpu && flash_attn;
        ctx_params.vae_decode_only = false;

        sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
        if (!sd_ctx) {
            fprintf(stderr, "[ERROR] Failed to load checkpoint\n");
            return false;
        }

        printf("[CheckpointLoaderSimple] Model loaded successfully\n");

        outputs["MODEL"] = sd_ctx;
        outputs["CLIP"] = sd_ctx;
        outputs["VAE"] = sd_ctx;

        return true;
    }
};
REGISTER_NODE("CheckpointLoaderSimple", CheckpointLoaderSimpleNode);

// ============================================================================
// CLIPTextEncode - 真正的文本编码
// ============================================================================
class CLIPTextEncodeNode : public Node {
public:
    std::string get_class_type() const override { return "CLIPTextEncode"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"text", "STRING", true, std::string("")},
            {"clip", "CLIP", true, nullptr},
            {"clip_skip", "INT", false, -1}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string text = std::any_cast<std::string>(inputs.at("text"));
        sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("clip"));
        int clip_skip = inputs.count("clip_skip") ?
            std::any_cast<int>(inputs.at("clip_skip")) : -1;

        sd_conditioning_t* cond = sd_encode_prompt(sd_ctx, text.c_str(), clip_skip);
        if (!cond) {
            fprintf(stderr, "[ERROR] CLIPTextEncode: Failed to encode prompt\n");
            return false;
        }

        outputs["CONDITIONING"] = cond;
        return true;
    }
};
REGISTER_NODE("CLIPTextEncode", CLIPTextEncodeNode);

// ============================================================================
// EmptyLatentImage - 创建空 Latent
// ============================================================================
class EmptyLatentImageNode : public Node {
public:
    std::string get_class_type() const override { return "EmptyLatentImage"; }
    std::string get_category() const override { return "latent"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"width", "INT", false, 512},
            {"height", "INT", false, 512},
            {"batch_size", "INT", false, 1}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int width = std::any_cast<int>(inputs.at("width"));
        int height = std::any_cast<int>(inputs.at("height"));

        // batch_size 暂不支持（sd_create_empty_latent 返回单张）
        sd_latent_t* latent = sd_create_empty_latent(nullptr, width, height);
        if (!latent) {
            fprintf(stderr, "[ERROR] EmptyLatentImage: Failed to create latent\n");
            return false;
        }

        outputs["LATENT"] = latent;
        return true;
    }
};
REGISTER_NODE("EmptyLatentImage", EmptyLatentImageNode);

// ============================================================================
// LoadImage - 加载图像
// ============================================================================
class LoadImageNode : public Node {
public:
    std::string get_class_type() const override { return "LoadImage"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "STRING", true, std::string("")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"IMAGE", "IMAGE"},
            {"MASK", "MASK"}
        };
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string image_path = std::any_cast<std::string>(inputs.at("image"));

        if (image_path.empty()) {
            fprintf(stderr, "[ERROR] LoadImage: image path is required\n");
            return false;
        }

        printf("[LoadImage] Loading: %s\n", image_path.c_str());

        int w, h, c;
        uint8_t* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!data) {
            fprintf(stderr, "[ERROR] LoadImage: Failed to load %s\n", image_path.c_str());
            return false;
        }

        sd_image_t* image = (sd_image_t*)malloc(sizeof(sd_image_t));
        if (!image) {
            stbi_image_free(data);
            fprintf(stderr, "[ERROR] LoadImage: Out of memory\n");
            return false;
        }

        image->width = w;
        image->height = h;
        image->channel = 3;
        image->data = data;

        printf("[LoadImage] Loaded: %dx%d\n", w, h);

        outputs["IMAGE"] = image;
        outputs["MASK"] = nullptr;

        return true;
    }
};
REGISTER_NODE("LoadImage", LoadImageNode);

// ============================================================================
// VAEEncode - 真正的 VAE 编码
// ============================================================================
class VAEEncodeNode : public Node {
public:
    std::string get_class_type() const override { return "VAEEncode"; }
    std::string get_category() const override { return "latent"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"pixels", "IMAGE", true, nullptr},
            {"vae", "VAE", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_image_t* image = std::any_cast<sd_image_t*>(inputs.at("pixels"));
        sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("vae"));

        if (!image || !image->data) {
            fprintf(stderr, "[ERROR] VAEEncode: No image data\n");
            return false;
        }

        sd_latent_t* latent = sd_encode_image(sd_ctx, image);
        if (!latent) {
            fprintf(stderr, "[ERROR] VAEEncode: Failed to encode image\n");
            return false;
        }

        printf("[VAEEncode] Image encoded to latent\n");
        outputs["LATENT"] = latent;
        return true;
    }
};
REGISTER_NODE("VAEEncode", VAEEncodeNode);

// ============================================================================
// LoRALoader - 加载 LoRA
// ============================================================================
class LoRALoaderNode : public Node {
public:
    std::string get_class_type() const override { return "LoRALoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"lora_name", "STRING", true, std::string("")},
            {"strength_model", "FLOAT", false, 1.0f},
            {"strength_clip", "FLOAT", false, 1.0f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LORA", "LORA"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string lora_path = std::any_cast<std::string>(inputs.at("lora_name"));
        float strength_model = inputs.count("strength_model") ?
            std::any_cast<float>(inputs.at("strength_model")) : 1.0f;
        float strength_clip = inputs.count("strength_clip") ?
            std::any_cast<float>(inputs.at("strength_clip")) : 1.0f;

        if (lora_path.empty()) {
            fprintf(stderr, "[ERROR] LoRALoader: lora_name is required\n");
            return false;
        }

        // 存储 LoRA 信息，使用平均强度作为 multiplier
        struct LoRAInfo {
            std::string path;
            float strength;
        };

        float avg_strength = (strength_model + strength_clip) * 0.5f;
        outputs["LORA"] = LoRAInfo{lora_path, avg_strength};

        printf("[LoRALoader] Loaded LoRA: %s (strength=%.2f)\n", lora_path.c_str(), avg_strength);
        return true;
    }
};
REGISTER_NODE("LoRALoader", LoRALoaderNode);

// ============================================================================
// KSampler - 真正的采样器（调用分离式 API）
// ============================================================================
class KSamplerNode : public Node {
public:
    std::string get_class_type() const override { return "KSampler"; }
    std::string get_category() const override { return "sampling"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
            {"seed", "INT", false, 0},
            {"steps", "INT", false, 20},
            {"cfg", "FLOAT", false, 8.0f},
            {"sampler_name", "STRING", false, std::string("euler")},
            {"scheduler", "STRING", false, std::string("normal")},
            {"positive", "CONDITIONING", true, nullptr},
            {"negative", "CONDITIONING", true, nullptr},
            {"latent_image", "LATENT", true, nullptr},
            {"denoise", "FLOAT", false, 1.0f},
            {"lora_stack", "LORA", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("model"));
        int64_t seed = inputs.count("seed") ? std::any_cast<int>(inputs.at("seed")) : 0;
        int steps = inputs.count("steps") ? std::any_cast<int>(inputs.at("steps")) : 20;
        float cfg = inputs.count("cfg") ? std::any_cast<float>(inputs.at("cfg")) : 8.0f;
        float denoise = inputs.count("denoise") ? std::any_cast<float>(inputs.at("denoise")) : 1.0f;
        std::string sampler_name = inputs.count("sampler_name") ?
            std::any_cast<std::string>(inputs.at("sampler_name")) : "euler";
        std::string scheduler_name = inputs.count("scheduler") ?
            std::any_cast<std::string>(inputs.at("scheduler")) : "normal";

        sd_conditioning_t* positive = std::any_cast<sd_conditioning_t*>(inputs.at("positive"));
        sd_conditioning_t* negative = inputs.count("negative") ?
            std::any_cast<sd_conditioning_t*>(inputs.at("negative")) : nullptr;
        sd_latent_t* init_latent = std::any_cast<sd_latent_t*>(inputs.at("latent_image"));

        if (!positive || !init_latent) {
            fprintf(stderr, "[ERROR] KSampler: Missing required inputs\n");
            return false;
        }

        // 处理 LoRA
        struct LoRAInfo {
            std::string path;
            float strength;
        };

        std::vector<sd_lora_t> loras;
        if (inputs.count("lora_stack")) {
            auto lora_info = std::any_cast<LoRAInfo>(inputs.at("lora_stack"));
            sd_lora_t lora;
            lora.path = lora_info.path.c_str();
            lora.multiplier = lora_info.strength;
            lora.is_high_noise = false;
            loras.push_back(lora);
            printf("[KSampler] Applying LoRA: %s (strength=%.2f)\n", lora_info.path.c_str(), lora_info.strength);
        }

        if (!loras.empty()) {
            sd_apply_loras(sd_ctx, loras.data(), static_cast<uint32_t>(loras.size()));
        } else {
            sd_clear_loras(sd_ctx);
        }

        printf("[KSampler] Running sampler: steps=%d, seed=%ld, cfg=%.2f, denoise=%.2f\n",
               steps, (long)seed, cfg, denoise);

        sd_node_sample_params_t sample_params;
        sample_params.seed = seed;
        sample_params.steps = steps;
        sample_params.cfg_scale = cfg;
        sample_params.sample_method = str_to_sample_method(sampler_name.c_str());
        sample_params.scheduler = str_to_scheduler(scheduler_name.c_str());
        sample_params.eta = 0.0f;

        if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        }
        if (sample_params.scheduler == SCHEDULER_COUNT) {
            sample_params.scheduler = DISCRETE_SCHEDULER;
        }

        sd_latent_t* result = sd_sampler_run(
            sd_ctx, init_latent, positive, negative, &sample_params, denoise);

        if (!result) {
            fprintf(stderr, "[ERROR] KSampler: Sampling failed\n");
            return false;
        }

        printf("[KSampler] Sampling completed\n");
        outputs["LATENT"] = result;
        return true;
    }
};
REGISTER_NODE("KSampler", KSamplerNode);

// ============================================================================
// VAEDecode - 真正的 VAE 解码
// ============================================================================
class VAEDecodeNode : public Node {
public:
    std::string get_class_type() const override { return "VAEDecode"; }
    std::string get_category() const override { return "latent"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"samples", "LATENT", true, nullptr},
            {"vae", "VAE", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_latent_t* latent = std::any_cast<sd_latent_t*>(inputs.at("samples"));
        sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("vae"));

        if (!latent) {
            fprintf(stderr, "[ERROR] VAEDecode: No latent data\n");
            return false;
        }

        sd_image_t* image = sd_decode_latent(sd_ctx, latent);
        if (!image) {
            fprintf(stderr, "[ERROR] VAEDecode: Failed to decode latent\n");
            return false;
        }

        printf("[VAEDecode] Latent decoded: %dx%d\n", image->width, image->height);
        outputs["IMAGE"] = image;
        return true;
    }
};
REGISTER_NODE("VAEDecode", VAEDecodeNode);

// ============================================================================
// SaveImage - 保存图像
// ============================================================================
class SaveImageNode : public Node {
public:
    std::string get_class_type() const override { return "SaveImage"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"images", "IMAGE", true, nullptr},
            {"filename_prefix", "STRING", false, std::string("sd-engine")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        sd_image_t* image = std::any_cast<sd_image_t*>(inputs.at("images"));
        std::string prefix = inputs.count("filename_prefix") ?
            std::any_cast<std::string>(inputs.at("filename_prefix")) : "sd-engine";

        if (!image || !image->data) {
            fprintf(stderr, "[ERROR] SaveImage: No image data\n");
            return false;
        }

        std::string filename = prefix + ".png";
        printf("[SaveImage] Saving to %s (%dx%d)\n",
               filename.c_str(), image->width, image->height);

        bool success = stbi_write_png(filename.c_str(),
                                      image->width, image->height,
                                      image->channel, image->data, 0) != 0;
        if (!success) {
            fprintf(stderr, "[ERROR] SaveImage: Failed to write %s\n", filename.c_str());
            return false;
        }

        printf("[SaveImage] Saved successfully\n");
        return true;
    }
};
REGISTER_NODE("SaveImage", SaveImageNode);

// ============================================================================
// ImageScale - 图像缩放
// ============================================================================
class ImageScaleNode : public Node {
public:
    std::string get_class_type() const override { return "ImageScale"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"width", "INT", true, 0},
            {"height", "INT", true, 0},
            {"method", "STRING", false, std::string("bilinear")}  // bilinear, nearest, lanczos
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_image_t* src_image = std::any_cast<sd_image_t*>(inputs.at("image"));
        int target_width = std::any_cast<int>(inputs.at("width"));
        int target_height = std::any_cast<int>(inputs.at("height"));
        std::string method = inputs.count("method") ?
            std::any_cast<std::string>(inputs.at("method")) : "bilinear";

        if (!src_image || !src_image->data) {
            fprintf(stderr, "[ERROR] ImageScale: No source image\n");
            return false;
        }

        if (target_width <= 0 || target_height <= 0) {
            fprintf(stderr, "[ERROR] ImageScale: Invalid target size %dx%d\n", target_width, target_height);
            return false;
        }

        // 如果尺寸相同，直接返回原图
        if ((int)src_image->width == target_width && (int)src_image->height == target_height) {
            outputs["IMAGE"] = src_image;
            return true;
        }

        printf("[ImageScale] Resizing from %dx%d to %dx%d (method: %s)\n",
               src_image->width, src_image->height, target_width, target_height, method.c_str());

        // 分配输出缓冲区
        uint8_t* dst_data = (uint8_t*)malloc(target_width * target_height * src_image->channel);
        if (!dst_data) {
            fprintf(stderr, "[ERROR] ImageScale: Out of memory\n");
            return false;
        }

        // 选择插值方法
        stbir_filter filter = STBIR_FILTER_DEFAULT;
        if (method == "nearest") {
            filter = STBIR_FILTER_BOX;
        } else if (method == "bilinear") {
            filter = STBIR_FILTER_TRIANGLE;
        } else if (method == "lanczos") {
            filter = STBIR_FILTER_CATMULLROM;  // 老版本 stb_image_resize 没有 LANCZOS3
        }

        // 执行缩放（老版本 API）
        stbir_resize(
            src_image->data, src_image->width, src_image->height, 0,
            dst_data, target_width, target_height, 0,
            STBIR_TYPE_UINT8,
            src_image->channel,  // num_channels
            -1,                  // alpha_channel (no separate alpha)
            0,                   // flags
            STBIR_EDGE_CLAMP,
            STBIR_EDGE_CLAMP,
            filter,
            filter,
            STBIR_COLORSPACE_LINEAR,
            nullptr              // alloc_context
        );

        // 创建新的 sd_image_t
        sd_image_t* dst_image = (sd_image_t*)malloc(sizeof(sd_image_t));
        if (!dst_image) {
            free(dst_data);
            fprintf(stderr, "[ERROR] ImageScale: Out of memory\n");
            return false;
        }

        dst_image->width = target_width;
        dst_image->height = target_height;
        dst_image->channel = src_image->channel;
        dst_image->data = dst_data;

        outputs["IMAGE"] = dst_image;
        printf("[ImageScale] Resized successfully\n");
        return true;
    }
};
REGISTER_NODE("ImageScale", ImageScaleNode);

// ============================================================================
// ImageCrop - 图像裁剪
// ============================================================================
class ImageCropNode : public Node {
public:
    std::string get_class_type() const override { return "ImageCrop"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"x", "INT", true, 0},
            {"y", "INT", true, 0},
            {"width", "INT", true, 0},
            {"height", "INT", true, 0}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_image_t* src_image = std::any_cast<sd_image_t*>(inputs.at("image"));
        int crop_x = std::any_cast<int>(inputs.at("x"));
        int crop_y = std::any_cast<int>(inputs.at("y"));
        int crop_width = std::any_cast<int>(inputs.at("width"));
        int crop_height = std::any_cast<int>(inputs.at("height"));

        if (!src_image || !src_image->data) {
            fprintf(stderr, "[ERROR] ImageCrop: No source image\n");
            return false;
        }

        // 验证裁剪区域
        if (crop_x < 0 || crop_y < 0 ||
            crop_width <= 0 || crop_height <= 0 ||
            crop_x + crop_width > (int)src_image->width ||
            crop_y + crop_height > (int)src_image->height) {
            fprintf(stderr, "[ERROR] ImageCrop: Invalid crop region (%d,%d,%d,%d) for image %dx%d\n",
                    crop_x, crop_y, crop_width, crop_height,
                    src_image->width, src_image->height);
            return false;
        }

        printf("[ImageCrop] Cropping to (%d,%d) size %dx%d\n",
               crop_x, crop_y, crop_width, crop_height);

        // 分配输出缓冲区
        uint8_t* dst_data = (uint8_t*)malloc(crop_width * crop_height * src_image->channel);
        if (!dst_data) {
            fprintf(stderr, "[ERROR] ImageCrop: Out of memory\n");
            return false;
        }

        // 逐行复制
        int src_stride = src_image->width * src_image->channel;
        int dst_stride = crop_width * src_image->channel;
        int row_bytes = crop_width * src_image->channel;

        for (int y = 0; y < crop_height; y++) {
            uint8_t* src_row = src_image->data + (crop_y + y) * src_stride + crop_x * src_image->channel;
            uint8_t* dst_row = dst_data + y * dst_stride;
            memcpy(dst_row, src_row, row_bytes);
        }

        // 创建新的 sd_image_t
        sd_image_t* dst_image = (sd_image_t*)malloc(sizeof(sd_image_t));
        if (!dst_image) {
            free(dst_data);
            fprintf(stderr, "[ERROR] ImageCrop: Out of memory\n");
            return false;
        }

        dst_image->width = crop_width;
        dst_image->height = crop_height;
        dst_image->channel = src_image->channel;
        dst_image->data = dst_data;

        outputs["IMAGE"] = dst_image;
        printf("[ImageCrop] Cropped successfully\n");
        return true;
    }
};
REGISTER_NODE("ImageCrop", ImageCropNode);

// ============================================================================
// PreviewImage - 预览图像（只打印信息，不保存）
// ============================================================================
class PreviewImageNode : public Node {
public:
    std::string get_class_type() const override { return "PreviewImage"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"images", "IMAGE", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {};  // 无输出
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        sd_image_t* image = std::any_cast<sd_image_t*>(inputs.at("images"));

        if (!image || !image->data) {
            fprintf(stderr, "[ERROR] PreviewImage: No image data\n");
            return false;
        }

        printf("\n");
        printf("╔══════════════════════════════════════╗\n");
        printf("║         [PreviewImage]               ║\n");
        printf("║  Size: %4dx%-4d                     ║\n", image->width, image->height);
        printf("║  Channels: %d                        ║\n", image->channel);
        printf("╚══════════════════════════════════════╝\n");
        printf("\n");

        return true;
    }
};
REGISTER_NODE("PreviewImage", PreviewImageNode);

// ============================================================================
// DeepHighResFix - 原生 Deep HighRes Fix 节点
// ============================================================================
// 
// 在单次采样过程中动态改变 latent 分辨率，实现渐进式高清修复
// 核心原理：利用 sd_set_latent_hook() 在采样步骤间插值上采样 latent
// ============================================================================

struct DeepHiresNodeState {
    int phase1_steps;
    int phase2_steps;
    int phase1_w;
    int phase1_h;
    int phase2_w;
    int phase2_h;
    int target_w;
    int target_h;
    float phase1_cfg_scale;
    float phase2_cfg_scale;
    float phase3_cfg_scale;
    bool phase1_done;
    bool phase2_done;
};

// 双线性插值放大 latent
static sd::Tensor<float> upscale_latent_bilinear_node(
    const sd::Tensor<float>& latent,
    int target_w,
    int target_h,
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

// Latent hook 回调
static sd::Tensor<float> deep_hires_node_latent_hook(
    sd::Tensor<float>& latent,
    int step,
    int total_steps,
    void* user_data) {
    
    DeepHiresNodeState* state = (DeepHiresNodeState*)user_data;
    if (!state) return latent;
    
    int latent_channel = (int)latent.shape()[2];
    
    // Phase 1 -> Phase 2 过渡
    if (!state->phase1_done && step > state->phase1_steps) {
        state->phase1_done = true;
        printf("[DeepHires Hook] Step %d/%d: Upsampling %dx%d -> %dx%d\n",
               step, total_steps,
               (int)latent.shape()[0], (int)latent.shape()[1],
               state->phase2_w, state->phase2_h);
        return upscale_latent_bilinear_node(latent, state->phase2_w, state->phase2_h, latent_channel);
    }
    
    // Phase 2 -> Phase 3 过渡
    if (!state->phase2_done && step > (state->phase1_steps + state->phase2_steps)) {
        state->phase2_done = true;
        printf("[DeepHires Hook] Step %d/%d: Upsampling %dx%d -> %dx%d\n",
               step, total_steps,
               (int)latent.shape()[0], (int)latent.shape()[1],
               state->target_w, state->target_h);
        return upscale_latent_bilinear_node(latent, state->target_w, state->target_h, latent_channel);
    }
    
    return latent;
}

// Guidance hook 回调：动态调整 cfg_scale
static void deep_hires_node_guidance_hook(
    float* txt_cfg,
    float* img_cfg,
    float* distilled_guidance,
    int step,
    int total_steps,
    void* user_data) {
    
    DeepHiresNodeState* state = (DeepHiresNodeState*)user_data;
    if (!state) return;
    
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

class DeepHighResFixNode : public Node {
public:
    std::string get_class_type() const override { return "DeepHighResFix"; }
    std::string get_category() const override { return "sampling"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
            {"positive", "CONDITIONING", true, nullptr},
            {"negative", "CONDITIONING", true, nullptr},
            {"init_image", "IMAGE", false, nullptr},  // 可选，用于 img2img
            {"seed", "INT", false, 0},
            {"steps", "INT", false, 30},
            {"cfg", "FLOAT", false, 7.0f},
            {"target_width", "INT", false, 1024},
            {"target_height", "INT", false, 1024},
            {"strength", "FLOAT", false, 1.0f},  // img2img 强度
            {"phase1_cfg", "FLOAT", false, 0.0f},  // 0 = use cfg
            {"phase2_cfg", "FLOAT", false, 0.0f},
            {"phase3_cfg", "FLOAT", false, 0.0f},
            {"vae_tiling", "BOOLEAN", false, false}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("model"));
        sd_conditioning_t* positive = std::any_cast<sd_conditioning_t*>(inputs.at("positive"));
        sd_conditioning_t* negative = inputs.count("negative") ?
            std::any_cast<sd_conditioning_t*>(inputs.at("negative")) : nullptr;
        
        int64_t seed = inputs.count("seed") ? std::any_cast<int>(inputs.at("seed")) : 0;
        int total_steps = inputs.count("steps") ? std::any_cast<int>(inputs.at("steps")) : 30;
        float cfg = inputs.count("cfg") ? std::any_cast<float>(inputs.at("cfg")) : 7.0f;
        int target_width = inputs.count("target_width") ? std::any_cast<int>(inputs.at("target_width")) : 1024;
        int target_height = inputs.count("target_height") ? std::any_cast<int>(inputs.at("target_height")) : 1024;
        float strength = inputs.count("strength") ? std::any_cast<float>(inputs.at("strength")) : 1.0f;
        float phase1_cfg = inputs.count("phase1_cfg") ? std::any_cast<float>(inputs.at("phase1_cfg")) : 0.0f;
        float phase2_cfg = inputs.count("phase2_cfg") ? std::any_cast<float>(inputs.at("phase2_cfg")) : 0.0f;
        float phase3_cfg = inputs.count("phase3_cfg") ? std::any_cast<float>(inputs.at("phase3_cfg")) : 0.0f;
        bool vae_tiling = inputs.count("vae_tiling") ? std::any_cast<bool>(inputs.at("vae_tiling")) : false;

        if (!sd_ctx || !positive) {
            fprintf(stderr, "[ERROR] DeepHighResFix: Missing required inputs\n");
            return false;
        }

        // 对齐尺寸到 64 的倍数
        int target_w = (target_width + 63) & ~63;
        int target_h = (target_height + 63) & ~63;

        // 计算各阶段参数
        int phase1_steps = std::max(6, total_steps / 4);
        int phase3_steps = std::max(8, total_steps * 3 / 4);
        int phase2_steps = std::max(4, total_steps - phase1_steps - phase3_steps);

        int phase1_w = std::min(512, target_w / 2);
        int phase1_h = std::min(512, target_h / 2);
        phase1_w = (phase1_w + 63) & ~63;
        phase1_h = (phase1_h + 63) & ~63;

        int phase2_w = target_w * 3 / 4;
        int phase2_h = target_h * 3 / 4;
        phase2_w = (phase2_w + 63) & ~63;
        phase2_h = (phase2_h + 63) & ~63;

        printf("[DeepHighResFix] Target: %dx%d, Phases: %dx%d(%d) -> %dx%d(%d) -> %dx%d(%d)\n",
               target_w, target_h,
               phase1_w, phase1_h, phase1_steps,
               phase2_w, phase2_h, phase2_steps,
               target_w, target_h, phase3_steps);

        // 准备 hook 状态
        DeepHiresNodeState state = {};
        state.phase1_steps = phase1_steps;
        state.phase2_steps = phase2_steps;
        state.phase1_w = phase1_w;
        state.phase1_h = phase1_h;
        state.phase2_w = phase2_w;
        state.phase2_h = phase2_h;
        state.target_w = target_w;
        state.target_h = target_h;
        state.phase1_cfg_scale = phase1_cfg > 0 ? phase1_cfg : cfg;
        state.phase2_cfg_scale = phase2_cfg > 0 ? phase2_cfg : cfg;
        state.phase3_cfg_scale = phase3_cfg > 0 ? phase3_cfg : cfg;

        // 注册 hook
        sd_set_latent_hook(deep_hires_node_latent_hook, &state);
        sd_set_guidance_hook(deep_hires_node_guidance_hook, &state);

        // 构建生成参数
        sd_img_gen_params_t gen_params;
        sd_img_gen_params_init(&gen_params);
        
        // 从 conditioning 获取 prompt（简化处理，实际应该传递字符串）
        gen_params.prompt = "";
        gen_params.negative_prompt = "";
        gen_params.width = phase1_w;
        gen_params.height = phase1_h;
        gen_params.strength = strength;
        gen_params.seed = seed;
        gen_params.sample_params.sample_steps = total_steps;
        gen_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        gen_params.sample_params.scheduler = KARRAS_SCHEDULER;
        gen_params.sample_params.guidance.txt_cfg = cfg;

        // 处理 init_image（img2img）
        if (inputs.count("init_image")) {
            sd_image_t* init_img = std::any_cast<sd_image_t*>(inputs.at("init_image"));
            if (init_img && init_img->data) {
                gen_params.init_image = *init_img;
            }
        }

        if (vae_tiling) {
            gen_params.vae_tiling_params.enabled = true;
            gen_params.vae_tiling_params.tile_size_x = 512;
            gen_params.vae_tiling_params.tile_size_y = 512;
            gen_params.vae_tiling_params.target_overlap = 64;
        }

        // 调用生成
        sd_image_t* result = generate_image(sd_ctx, &gen_params);

        // 清除 hook
        sd_clear_latent_hook();
        sd_clear_guidance_hook();

        if (!result || !result->data) {
            fprintf(stderr, "[ERROR] DeepHighResFix: Generation failed\n");
            return false;
        }

        printf("[DeepHighResFix] Generation completed: %dx%d\n", result->width, result->height);
        outputs["IMAGE"] = result;
        return true;
    }
};
REGISTER_NODE("DeepHighResFix", DeepHighResFixNode);

// ============================================================================
// 显式初始化函数
// ============================================================================
void init_core_nodes() {
    // 确保本文件被链接进最终可执行文件
}

} // namespace sdengine
