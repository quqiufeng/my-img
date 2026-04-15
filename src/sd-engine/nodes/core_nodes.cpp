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
#include "core/sd_ptr.h"
#include "stable-diffusion.h"
#include "stable-diffusion-ext.h"
#include "tensor.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <map>
#include <cmath>
#include <cstdint>

// stb_image for LoadImage/SaveImage
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#include "face/face_detect.hpp"
#include "face/face_restore.hpp"
#include "face/face_swap.hpp"
#include "face/face_align.hpp"
#include "face/face_utils.hpp"
#include "preprocessors/lineart.hpp"
#endif

namespace sdengine {

// ============================================================================
// 公共辅助函数
// ============================================================================

/// 将 RGBA 图像转换为 RGB
static std::vector<uint8_t> convert_rgba_to_rgb(const uint8_t* src, int w, int h) {
    std::vector<uint8_t> dst(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        dst[i * 3 + 0] = src[i * 4 + 0];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
    }
    return dst;
}

/// 从已分配的 data 创建 ImagePtr（失败时释放 data 并返回错误）
static ImagePtr create_image_ptr(int w, int h, int c, uint8_t* data, sd_error_t* out_err = nullptr) {
    sd_image_t* img = acquire_image();
    if (!img) {
        free(data);
        if (out_err) *out_err = sd_error_t::ERROR_MEMORY_ALLOCATION;
        return nullptr;
    }
    img->width = w;
    img->height = h;
    img->channel = c;
    img->data = data;
    return make_image_ptr(img);
}

/// 获取 sd_ctx_t*，兼容 SDContextPtr 和裸指针
static sd_ctx_t* extract_sd_ctx(const NodeInputs& inputs, const std::string& key) {
    sd_ctx_t* sd_ctx = nullptr;
    try {
        auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at(key));
        sd_ctx = ctx_ptr.get();
    } catch (const std::bad_any_cast&) {
        try {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at(key));
        } catch (...) {}
    }
    return sd_ctx;
}

// ============================================================================
// CLIP 包装器（供 CLIPSetLastLayer / CLIPTextEncode 共享）
// ============================================================================
struct CLIPWrapper {
    sd_ctx_t* sd_ctx = nullptr;
    SDContextPtr sd_ctx_ptr;  // keep shared_ptr alive if needed
    int clip_skip = -1;
};

// ============================================================================
// LoRA 信息结构体（供 LoRALoader / LoRAStack / KSampler 共享）
// ============================================================================
struct LoRAInfo {
    std::string path;
    float strength;
};

// ============================================================================
// IPAdapter 信息结构体（供 IPAdapterLoader / IPAdapterApply / KSampler 共享）
// ============================================================================
struct IPAdapterInfo {
    std::string path;
    int cross_attention_dim;
    int num_tokens;
    int clip_embeddings_dim;
    float strength;
};

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
            {"control_net_path", "STRING", false, std::string("")},
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string ckpt_path = std::any_cast<std::string>(inputs.at("ckpt_name"));
        std::string vae_path = inputs.count("vae_name") ? 
            std::any_cast<std::string>(inputs.at("vae_name")) : "";
        std::string clip_path = inputs.count("clip_name") ?
            std::any_cast<std::string>(inputs.at("clip_name")) : "";
        std::string control_net_path = inputs.count("control_net_path") ?
            std::any_cast<std::string>(inputs.at("control_net_path")) : "";
        int n_threads = inputs.count("n_threads") ?
            std::any_cast<int>(inputs.at("n_threads")) : 4;
        bool use_gpu = inputs.count("use_gpu") ?
            std::any_cast<bool>(inputs.at("use_gpu")) : true;
        bool flash_attn = inputs.count("flash_attn") ?
            std::any_cast<bool>(inputs.at("flash_attn")) : false;

        if (ckpt_path.empty()) {
            fprintf(stderr, "[ERROR] CheckpointLoaderSimple: ckpt_name is required\n");
            return sd_error_t::ERROR_MODEL_LOADING;
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
        if (!control_net_path.empty()) {
            ctx_params.control_net_path = control_net_path.c_str();
            ctx_params.keep_control_net_on_cpu = !use_gpu;
            printf("[CheckpointLoaderSimple] Loading ControlNet: %s\n", control_net_path.c_str());
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
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        printf("[CheckpointLoaderSimple] Model loaded successfully\n");

        auto sd_ctx_ptr = make_sd_context_ptr(sd_ctx);
        outputs["MODEL"] = sd_ctx_ptr;
        outputs["CLIP"] = sd_ctx_ptr;
        outputs["VAE"] = sd_ctx_ptr;

        return sd_error_t::OK;
    }
};
REGISTER_NODE("CheckpointLoaderSimple", CheckpointLoaderSimpleNode);

// ============================================================================
// CLIPSetLastLayer - 设置 CLIP 跳过层
// ============================================================================
class CLIPSetLastLayerNode : public Node {
public:
    std::string get_class_type() const override { return "CLIPSetLastLayer"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"clip", "CLIP", true, nullptr},
            {"stop_at_clip_layer", "INT", false, -1}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CLIP", "CLIP"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = nullptr;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("clip"));
            sd_ctx = ctx_ptr.get();
        } catch (const std::bad_any_cast&) {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("clip"));
        }
        int clip_skip = inputs.count("stop_at_clip_layer") ?
            std::any_cast<int>(inputs.at("stop_at_clip_layer")) : -1;

        if (!sd_ctx) {
            fprintf(stderr, "[ERROR] CLIPSetLastLayer: Missing CLIP\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        CLIPWrapper wrapper;
        wrapper.sd_ctx = sd_ctx;
        try {
            wrapper.sd_ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("clip"));
        } catch (...) {}
        wrapper.clip_skip = clip_skip;

        outputs["CLIP"] = wrapper;
        printf("[CLIPSetLastLayer] clip_skip set to %d\n", clip_skip);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CLIPSetLastLayer", CLIPSetLastLayerNode);

// ============================================================================
// CLIPVisionEncode - CLIP Vision 图像编码
// ============================================================================
class CLIPVisionEncodeNode : public Node {
public:
    std::string get_class_type() const override { return "CLIPVisionEncode"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"clip", "CLIP", true, nullptr},
            {"image", "IMAGE", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = nullptr;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("clip"));
            sd_ctx = ctx_ptr.get();
        } catch (const std::bad_any_cast&) {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("clip"));
        }
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));

        if (!sd_ctx || !image || !image->data) {
            fprintf(stderr, "[ERROR] CLIPVisionEncode: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_clip_vision_output_t* output = sd_clip_vision_encode_image(sd_ctx, image.get(), true);
        if (!output) {
            fprintf(stderr, "[ERROR] CLIPVisionEncode: Failed to encode image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CLIP_VISION_OUTPUT"] = make_clip_vision_output_ptr(output);
        printf("[CLIPVisionEncode] Encoded image to CLIP Vision output (numel=%d)\n", output->numel);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CLIPVisionEncode", CLIPVisionEncodeNode);

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
        return {
            {"CONDITIONING", "CONDITIONING"},
            {"text", "STRING"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string text = std::any_cast<std::string>(inputs.at("text"));
        int clip_skip = inputs.count("clip_skip") ?
            std::any_cast<int>(inputs.at("clip_skip")) : -1;

        sd_ctx_t* sd_ctx = nullptr;
        try {
            // Try CLIPWrapper first (from CLIPSetLastLayer)
            CLIPWrapper wrapper = std::any_cast<CLIPWrapper>(inputs.at("clip"));
            sd_ctx = wrapper.sd_ctx;
            if (clip_skip == -1) {
                clip_skip = wrapper.clip_skip;
            }
        } catch (const std::bad_any_cast&) {
            // Fallback to SDContextPtr or raw sd_ctx_t*
            try {
                auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("clip"));
                sd_ctx = ctx_ptr.get();
            } catch (const std::bad_any_cast&) {
                sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("clip"));
            }
        }

        if (!sd_ctx) {
            fprintf(stderr, "[ERROR] CLIPTextEncode: Missing CLIP\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* cond = sd_encode_prompt(sd_ctx, text.c_str(), clip_skip);
        if (!cond) {
            fprintf(stderr, "[ERROR] CLIPTextEncode: Failed to encode prompt\n");
            return sd_error_t::ERROR_ENCODING_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(cond);
        outputs["text"] = text;
        return sd_error_t::OK;
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int width = std::any_cast<int>(inputs.at("width"));
        int height = std::any_cast<int>(inputs.at("height"));

        // batch_size 暂不支持（sd_create_empty_latent 返回单张）
        sd_latent_t* latent = sd_create_empty_latent(nullptr, width, height);
        if (!latent) {
            fprintf(stderr, "[ERROR] EmptyLatentImage: Failed to create latent\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["LATENT"] = make_latent_ptr(latent);
        return sd_error_t::OK;
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string image_path = std::any_cast<std::string>(inputs.at("image"));

        if (image_path.empty()) {
            fprintf(stderr, "[ERROR] LoadImage: image path is required\n");
            return sd_error_t::ERROR_FILE_IO;
        }

        printf("[LoadImage] Loading: %s\n", image_path.c_str());

        int w, h, c;
        uint8_t* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!data) {
            fprintf(stderr, "[ERROR] LoadImage: Failed to load %s\n", image_path.c_str());
            return sd_error_t::ERROR_FILE_IO;
        }

        sd_image_t* image = acquire_image();
        if (!image) {
            stbi_image_free(data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        image->width = w;
        image->height = h;
        image->channel = 3;
        image->data = data;

        printf("[LoadImage] Loaded: %dx%d\n", w, h);

        outputs["IMAGE"] = make_image_ptr(image);
        outputs["MASK"] = nullptr;

        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoadImage", LoadImageNode);

// ============================================================================
// LoadImageMask - 加载 Mask 图像
// ============================================================================
class LoadImageMaskNode : public Node {
public:
    std::string get_class_type() const override { return "LoadImageMask"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "STRING", true, std::string("")},
            {"channel", "STRING", false, std::string("alpha")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"MASK", "MASK"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string image_path = std::any_cast<std::string>(inputs.at("image"));
        std::string channel = inputs.count("channel") ?
            std::any_cast<std::string>(inputs.at("channel")) : "alpha";

        if (image_path.empty()) {
            fprintf(stderr, "[ERROR] LoadImageMask: image path is required\n");
            return sd_error_t::ERROR_FILE_IO;
        }

        printf("[LoadImageMask] Loading: %s (channel=%s)\n", image_path.c_str(), channel.c_str());

        int w, h, c;
        uint8_t* data = stbi_load(image_path.c_str(), &w, &h, &c, 0);
        if (!data) {
            fprintf(stderr, "[ERROR] LoadImageMask: Failed to load %s\n", image_path.c_str());
            return sd_error_t::ERROR_FILE_IO;
        }

        uint8_t* mask_data = (uint8_t*)malloc(w * h * 3);
        if (!mask_data) {
            stbi_image_free(data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        int src_channel = 0;
        if (channel == "alpha" && c == 4) {
            src_channel = 3;
        } else if (channel == "red") {
            src_channel = 0;
        } else if (channel == "green") {
            src_channel = 1;
        } else if (channel == "blue") {
            src_channel = 2;
        }

        for (int i = 0; i < w * h; i++) {
            uint8_t val = data[i * c + src_channel];
            mask_data[i * 3 + 0] = val;
            mask_data[i * 3 + 1] = val;
            mask_data[i * 3 + 2] = val;
        }
        stbi_image_free(data);

        sd_image_t* mask = acquire_image();
        if (!mask) {
            free(mask_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        mask->width = w;
        mask->height = h;
        mask->channel = 3;
        mask->data = mask_data;

        outputs["MASK"] = make_image_ptr(mask);
        printf("[LoadImageMask] Loaded mask: %dx%d\n", w, h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoadImageMask", LoadImageMaskNode);

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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_image_t image = std::any_cast<sd_image_t>(inputs.at("pixels"));
        sd_ctx_t* sd_ctx = nullptr;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("vae"));
            sd_ctx = ctx_ptr.get();
        } catch (const std::bad_any_cast&) {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("vae"));
        }

        if (!image.data) {
            fprintf(stderr, "[ERROR] VAEEncode: No image data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_latent_t* latent = sd_encode_image(sd_ctx, &image);
        if (!latent) {
            fprintf(stderr, "[ERROR] VAEEncode: Failed to encode image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("[VAEEncode] Image encoded to latent\n");
        outputs["LATENT"] = make_latent_ptr(latent);
        return sd_error_t::OK;
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string lora_path = std::any_cast<std::string>(inputs.at("lora_name"));
        float strength_model = inputs.count("strength_model") ?
            std::any_cast<float>(inputs.at("strength_model")) : 1.0f;
        float strength_clip = inputs.count("strength_clip") ?
            std::any_cast<float>(inputs.at("strength_clip")) : 1.0f;

        if (lora_path.empty()) {
            fprintf(stderr, "[ERROR] LoRALoader: lora_name is required\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        float avg_strength = (strength_model + strength_clip) * 0.5f;
        outputs["LORA"] = LoRAInfo{lora_path, avg_strength};

        printf("[LoRALoader] Loaded LoRA: %s (strength=%.2f)\n", lora_path.c_str(), avg_strength);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoRALoader", LoRALoaderNode);

// ============================================================================
// LoRAStack - 多 LoRA 堆叠
// ============================================================================
class LoRAStackNode : public Node {
public:
    std::string get_class_type() const override { return "LoRAStack"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"lora_1", "LORA", false, nullptr},
            {"lora_2", "LORA", false, nullptr},
            {"lora_3", "LORA", false, nullptr},
            {"lora_4", "LORA", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LORA_STACK", "LORA_STACK"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::vector<LoRAInfo> stack;
        for (int i = 1; i <= 4; i++) {
            std::string key = "lora_" + std::to_string(i);
            if (inputs.count(key)) {
                auto info = std::any_cast<LoRAInfo>(inputs.at(key));
                stack.push_back(info);
                printf("[LoRAStack] Stacked LoRA %d: %s (strength=%.2f)\n",
                       i, info.path.c_str(), info.strength);
            }
        }
        outputs["LORA_STACK"] = stack;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoRAStack", LoRAStackNode);

// ============================================================================
// ControlNetLoader - 加载 ControlNet 模型
// ============================================================================
// 注意：由于 stable-diffusion.cpp 的限制，ControlNet 模型实际上是在
// CheckpointLoaderSimple 创建 sd_ctx 时加载的。这个节点主要用于：
// 1. 验证模型路径有效
// 2. 在工作流中表示 ControlNet 的存在
// 3. 传递模型路径给 CheckpointLoaderSimple
// ============================================================================
class ControlNetLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "ControlNetLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"control_net_name", "STRING", true, std::string("")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"CONTROL_NET", "CONTROL_NET"},
            {"path", "STRING"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("control_net_name"));
        if (path.empty()) {
            fprintf(stderr, "[ERROR] ControlNetLoader: control_net_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        outputs["CONTROL_NET"] = path;
        outputs["path"] = path;
        printf("[ControlNetLoader] ControlNet path: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ControlNetLoader", ControlNetLoaderNode);

// ============================================================================
// ControlNetApply - 应用 ControlNet 条件
// ============================================================================
// 简化实现：将 control_image 和 strength 打包传递给 KSampler。
// 输出一个包含控制信息的结构体，下游 KSampler 会解包使用。
// ============================================================================
struct ControlNetApplyInfo {
    ImagePtr control_image;
    float strength;
};

class ControlNetApplyNode : public Node {
public:
    std::string get_class_type() const override { return "ControlNetApply"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"conditioning", "CONDITIONING", true, nullptr},
            {"control_net", "CONTROL_NET", true, nullptr},
            {"image", "IMAGE", true, nullptr},
            {"strength", "FLOAT", false, 1.0f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond = std::any_cast<ConditioningPtr>(inputs.at("conditioning"));
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        float strength = inputs.count("strength") ? std::any_cast<float>(inputs.at("strength")) : 1.0f;

        if (!cond) {
            fprintf(stderr, "[ERROR] ControlNetApply: Missing conditioning\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        printf("[ControlNetApply] Applying ControlNet with strength=%.2f, image=%dx%d\n",
               strength, image ? image->width : 0, image ? image->height : 0);

        // 简化：直接透传 conditioning，但将 control_image 和 strength 作为附加信息存储
        // 在 std::any 中。KSampler 会检查是否有这个附加信息。
        outputs["CONDITIONING"] = cond;
        outputs["_control_image"] = image;
        outputs["_control_strength"] = strength;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ControlNetApply", ControlNetApplyNode);

// ============================================================================
// IPAdapterLoader - 加载 IPAdapter 模型
// ============================================================================
class IPAdapterLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "IPAdapterLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"ipadapter_file", "STRING", true, std::string("")},
            {"cross_attention_dim", "INT", false, 768},
            {"num_tokens", "INT", false, 4},
            {"clip_embeddings_dim", "INT", false, 1024}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IPADAPTER", "IPADAPTER"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("ipadapter_file"));
        int cross_attention_dim = inputs.count("cross_attention_dim") ?
            std::any_cast<int>(inputs.at("cross_attention_dim")) : 768;
        int num_tokens = inputs.count("num_tokens") ?
            std::any_cast<int>(inputs.at("num_tokens")) : 4;
        int clip_embeddings_dim = inputs.count("clip_embeddings_dim") ?
            std::any_cast<int>(inputs.at("clip_embeddings_dim")) : 1024;

        if (path.empty()) {
            fprintf(stderr, "[ERROR] IPAdapterLoader: ipadapter_file is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        outputs["IPADAPTER"] = IPAdapterInfo{
            path, cross_attention_dim, num_tokens, clip_embeddings_dim, 1.0f};

        printf("[IPAdapterLoader] Loaded IPAdapter: %s (dim=%d, tokens=%d, clip_dim=%d)\n",
               path.c_str(), cross_attention_dim, num_tokens, clip_embeddings_dim);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("IPAdapterLoader", IPAdapterLoaderNode);

// ============================================================================
// IPAdapterApply - 应用 IPAdapter 到 conditioning
// ============================================================================
class IPAdapterApplyNode : public Node {
public:
    std::string get_class_type() const override { return "IPAdapterApply"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"conditioning", "CONDITIONING", true, nullptr},
            {"ipadapter", "IPADAPTER", true, nullptr},
            {"image", "IMAGE", true, nullptr},
            {"strength", "FLOAT", false, 1.0f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"CONDITIONING", "CONDITIONING"},
            {"IPADAPTER", "IPADAPTER"},
            {"IMAGE", "IMAGE"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond = std::any_cast<ConditioningPtr>(inputs.at("conditioning"));
        IPAdapterInfo info = std::any_cast<IPAdapterInfo>(inputs.at("ipadapter"));
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        float strength = inputs.count("strength") ?
            std::any_cast<float>(inputs.at("strength")) : 1.0f;

        if (!cond) {
            fprintf(stderr, "[ERROR] IPAdapterApply: Missing conditioning\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        info.strength = strength;
        printf("[IPAdapterApply] Applying IPAdapter strength=%.2f, image=%dx%d\n",
               strength, image ? image->width : 0, image ? image->height : 0);

        outputs["CONDITIONING"] = cond;
        outputs["IPADAPTER"] = info;
        outputs["IMAGE"] = image;
        // Also keep hidden outputs for backward compatibility with existing workflows
        outputs["_ipadapter_info"] = info;
        outputs["_ipadapter_image"] = image;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("IPAdapterApply", IPAdapterApplyNode);

// ============================================================================
// ConditioningCombine - 条件合并
// ============================================================================
class ConditioningCombineNode : public Node {
public:
    std::string get_class_type() const override { return "ConditioningCombine"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"conditioning_1", "CONDITIONING", true, nullptr},
            {"conditioning_2", "CONDITIONING", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond1 = std::any_cast<ConditioningPtr>(inputs.at("conditioning_1"));
        ConditioningPtr cond2 = std::any_cast<ConditioningPtr>(inputs.at("conditioning_2"));

        if (!cond1 || !cond2) {
            fprintf(stderr, "[ERROR] ConditioningCombine: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* combined = sd_conditioning_concat(cond1.get(), cond2.get());
        if (!combined) {
            fprintf(stderr, "[ERROR] ConditioningCombine: Failed to combine conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(combined);
        printf("[ConditioningCombine] Combined two conditionings\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningCombine", ConditioningCombineNode);

// ============================================================================
// ConditioningConcat - 条件拼接（与 Combine 行为相同，对齐 ComfyUI 命名）
// ============================================================================
class ConditioningConcatNode : public Node {
public:
    std::string get_class_type() const override { return "ConditioningConcat"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"conditioning_to", "CONDITIONING", true, nullptr},
            {"conditioning_from", "CONDITIONING", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond_to = std::any_cast<ConditioningPtr>(inputs.at("conditioning_to"));
        ConditioningPtr cond_from = std::any_cast<ConditioningPtr>(inputs.at("conditioning_from"));

        if (!cond_to || !cond_from) {
            fprintf(stderr, "[ERROR] ConditioningConcat: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* concat = sd_conditioning_concat(cond_to.get(), cond_from.get());
        if (!concat) {
            fprintf(stderr, "[ERROR] ConditioningConcat: Failed to concat conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(concat);
        printf("[ConditioningConcat] Concatenated two conditionings\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningConcat", ConditioningConcatNode);

// ============================================================================
// ConditioningAverage - 条件加权平均
// ============================================================================
class ConditioningAverageNode : public Node {
public:
    std::string get_class_type() const override { return "ConditioningAverage"; }
    std::string get_category() const override { return "conditioning"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"conditioning_to", "CONDITIONING", true, nullptr},
            {"conditioning_from", "CONDITIONING", true, nullptr},
            {"conditioning_to_strength", "FLOAT", false, 1.0f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond_to = std::any_cast<ConditioningPtr>(inputs.at("conditioning_to"));
        ConditioningPtr cond_from = std::any_cast<ConditioningPtr>(inputs.at("conditioning_from"));
        float strength = inputs.count("conditioning_to_strength") ?
            std::any_cast<float>(inputs.at("conditioning_to_strength")) : 1.0f;

        if (!cond_to || !cond_from) {
            fprintf(stderr, "[ERROR] ConditioningAverage: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* averaged = sd_conditioning_average(cond_to.get(), cond_from.get(), strength);
        if (!averaged) {
            fprintf(stderr, "[ERROR] ConditioningAverage: Failed to average conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(averaged);
        printf("[ConditioningAverage] Averaged conditionings (strength=%.2f)\n", strength);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningAverage", ConditioningAverageNode);

// ============================================================================
// RemBGModelLoader - 加载背景抠图 ONNX 模型
// ============================================================================
#ifdef HAS_ONNXRUNTIME
struct RemBGModel {
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::MemoryInfo memory_info;
    std::string path;

    RemBGModel() : env(ORT_LOGGING_LEVEL_WARNING, "rembg"), memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

class RemBGModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "RemBGModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model_path", "STRING", true, std::string("")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"REMBG_MODEL", "REMBG_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_path"));
        if (path.empty()) {
            fprintf(stderr, "[ERROR] RemBGModelLoader: model_path is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto model = std::make_shared<RemBGModel>();
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            model->session = std::make_unique<Ort::Session>(model->env, path.c_str(), session_options);
            model->path = path;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[ERROR] RemBGModelLoader: Failed to load ONNX model: %s\n", e.what());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["REMBG_MODEL"] = model;
        printf("[RemBGModelLoader] Loaded model: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("RemBGModelLoader", RemBGModelLoaderNode);

// ============================================================================
// ImageRemoveBackground - 背景抠图
// ============================================================================
class ImageRemoveBackgroundNode : public Node {
public:
    std::string get_class_type() const override { return "ImageRemoveBackground"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"model", "REMBG_MODEL", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"IMAGE", "IMAGE"},
            {"MASK", "IMAGE"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        auto model = std::any_cast<std::shared_ptr<RemBGModel>>(inputs.at("model"));

        if (!image || !image->data || !model || !model->session) {
            fprintf(stderr, "[ERROR] ImageRemoveBackground: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int src_w = (int)image->width;
        int src_h = (int)image->height;
        int src_c = (int)image->channel;

        if (src_c != 3 && src_c != 4) {
            fprintf(stderr, "[ERROR] ImageRemoveBackground: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        const int model_size = 1024;

        // Preprocess: resize to 1024x1024, normalize to [0,1], convert to NCHW
        std::vector<float> input_data(1 * 3 * model_size * model_size);
        {
            std::vector<uint8_t> resized(model_size * model_size * src_c);
            stbir_resize(
                image->data, src_w, src_h, 0,
                resized.data(), model_size, model_size, 0,
                STBIR_TYPE_UINT8, src_c, -1, 0,
                STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE,
                STBIR_COLORSPACE_LINEAR, nullptr
            );

            for (int y = 0; y < model_size; y++) {
                for (int x = 0; x < model_size; x++) {
                    int idx = (y * model_size + x) * src_c;
                    input_data[0 * 3 * model_size * model_size + 0 * model_size * model_size + y * model_size + x] = resized[idx + 0] / 255.0f;
                    input_data[0 * 3 * model_size * model_size + 1 * model_size * model_size + y * model_size + x] = resized[idx + 1] / 255.0f;
                    input_data[0 * 3 * model_size * model_size + 2 * model_size * model_size + y * model_size + x] = resized[idx + 2] / 255.0f;
                }
            }
        }

        // ONNX inference
        std::vector<int64_t> input_shape = {1, 3, model_size, model_size};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            model->memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        std::vector<Ort::Value> output_tensors;
        try {
            output_tensors = model->session->Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1);
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "[ERROR] ImageRemoveBackground: ONNX inference failed: %s\n", e.what());
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int out_h = (int)output_shape[2];
        int out_w = (int)output_shape[3];

        // Postprocess: sigmoid, resize to original size
        std::vector<uint8_t> mask_resized(src_w * src_h);
        {
            std::vector<uint8_t> mask_1024(out_w * out_h);
            for (int i = 0; i < out_w * out_h; i++) {
                float v = output_data[i];
                // sigmoid
                v = 1.0f / (1.0f + std::exp(-v));
                mask_1024[i] = (uint8_t)(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
            }

            stbir_resize(
                mask_1024.data(), out_w, out_h, 0,
                mask_resized.data(), src_w, src_h, 0,
                STBIR_TYPE_UINT8, 1, -1, 0,
                STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE,
                STBIR_COLORSPACE_LINEAR, nullptr
            );
        }

        // Create RGBA output
        uint8_t* rgba_data = (uint8_t*)malloc(src_w * src_h * 4);
        if (!rgba_data) {
            fprintf(stderr, "[ERROR] ImageRemoveBackground: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        for (int i = 0; i < src_w * src_h; i++) {
            rgba_data[i * 4 + 0] = image->data[i * src_c + 0];
            rgba_data[i * 4 + 1] = image->data[i * src_c + 1];
            rgba_data[i * 4 + 2] = image->data[i * src_c + 2];
            rgba_data[i * 4 + 3] = mask_resized[i];
        }

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            free(rgba_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = src_w;
        result_img->height = src_h;
        result_img->channel = 4;
        result_img->data = rgba_data;

        // Create mask output (grayscale)
        uint8_t* mask_data = (uint8_t*)malloc(src_w * src_h);
        if (!mask_data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(mask_data, mask_resized.data(), src_w * src_h);

        sd_image_t* mask_img = acquire_image();
        if (!mask_img) {
            free(mask_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        mask_img->width = src_w;
        mask_img->height = src_h;
        mask_img->channel = 1;
        mask_img->data = mask_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        outputs["MASK"] = make_image_ptr(mask_img);
        printf("[ImageRemoveBackground] Removed background: %dx%d -> RGBA + Mask\n", src_w, src_h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageRemoveBackground", ImageRemoveBackgroundNode);

#else // !HAS_ONNXRUNTIME

class RemBGModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "RemBGModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"REMBG_MODEL", "REMBG_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs;
        (void)outputs;
        fprintf(stderr, "[ERROR] RemBGModelLoader: ONNX Runtime is not available. Build with HAS_ONNXRUNTIME to enable.\n");
        return sd_error_t::ERROR_MODEL_LOADING;
    }
};
REGISTER_NODE("RemBGModelLoader", RemBGModelLoaderNode);

class ImageRemoveBackgroundNode : public Node {
public:
    std::string get_class_type() const override { return "ImageRemoveBackground"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"model", "REMBG_MODEL", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"IMAGE", "IMAGE"},
            {"MASK", "IMAGE"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs;
        (void)outputs;
        fprintf(stderr, "[ERROR] ImageRemoveBackground: ONNX Runtime is not available. Build with HAS_ONNXRUNTIME to enable.\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("ImageRemoveBackground", ImageRemoveBackgroundNode);

#endif // HAS_ONNXRUNTIME

// ============================================================================
// CannyEdgePreprocessor - Canny 边缘检测预处理
// ============================================================================
class CannyEdgePreprocessorNode : public Node {
public:
    std::string get_class_type() const override { return "CannyEdgePreprocessor"; }
    std::string get_category() const override { return "image/preprocessors"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"low_threshold", "INT", false, 100},
            {"high_threshold", "INT", false, 200}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src = std::any_cast<ImagePtr>(inputs.at("image"));
        int low_threshold = inputs.count("low_threshold") ? std::any_cast<int>(inputs.at("low_threshold")) : 100;
        int high_threshold = inputs.count("high_threshold") ? std::any_cast<int>(inputs.at("high_threshold")) : 200;

        if (!src || !src->data || src->channel != 3) {
            fprintf(stderr, "[ERROR] CannyEdgePreprocessor: Requires RGB image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        printf("[CannyEdgePreprocessor] Processing %dx%d (low=%d, high=%d)\n",
               src->width, src->height, low_threshold, high_threshold);

        int w = src->width;
        int h = src->height;
        size_t pixel_count = w * h;

        // 分配灰度图和边缘图
        uint8_t* gray = (uint8_t*)malloc(pixel_count);
        uint8_t* edges = (uint8_t*)malloc(pixel_count);
        uint8_t* dst_data = (uint8_t*)malloc(pixel_count * 3);
        if (!gray || !edges || !dst_data) {
            free(gray);
            free(edges);
            free(dst_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        // RGB to Grayscale
        for (size_t i = 0; i < pixel_count; i++) {
            uint8_t r = src->data[i * 3 + 0];
            uint8_t g = src->data[i * 3 + 1];
            uint8_t b = src->data[i * 3 + 2];
            gray[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        }

        // Simple Canny approximation using Sobel + threshold
        // This is a simplified version for demonstration.
        // A production version would use OpenCV.
        int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

        memset(edges, 0, pixel_count);
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int gx = 0, gy = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = (ky + 1) * 3 + (kx + 1);
                        uint8_t val = gray[(y + ky) * w + (x + kx)];
                        gx += sobel_x[idx] * val;
                        gy += sobel_y[idx] * val;
                    }
                }
                int mag = (int)sqrtf((float)(gx * gx + gy * gy));
                if (mag > high_threshold) {
                    edges[y * w + x] = 255;
                } else if (mag > low_threshold) {
                    edges[y * w + x] = 128;
                }
            }
        }

        // Convert edges to RGB
        for (size_t i = 0; i < pixel_count; i++) {
            uint8_t val = edges[i] >= 128 ? 255 : 0;
            dst_data[i * 3 + 0] = val;
            dst_data[i * 3 + 1] = val;
            dst_data[i * 3 + 2] = val;
        }

        free(gray);
        free(edges);

        sd_image_t dst_image = {};
        dst_image.width = w;
        dst_image.height = h;
        dst_image.channel = 3;
        dst_image.data = dst_data;

        sd_image_t* result = acquire_image();
        if (!result) {
            free(edges);
            free(dst_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result = dst_image;

        outputs["IMAGE"] = make_image_ptr(result);
        printf("[CannyEdgePreprocessor] Done\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CannyEdgePreprocessor", CannyEdgePreprocessorNode);

// ============================================================================
// LineArtLoader - 加载 LineArt ONNX 模型
// ============================================================================
#ifdef HAS_ONNXRUNTIME
class LineArtLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "LineArtLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model_name", "STRING", true, std::string("")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"LINEART_MODEL", "LINEART_MODEL"},
            {"path", "STRING"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_name"));
        if (path.empty()) {
            fprintf(stderr, "[ERROR] LineArtLoader: model_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto preprocessor = std::make_shared<LineArtPreprocessor>();
        if (!preprocessor->load(path)) {
            fprintf(stderr, "[ERROR] LineArtLoader: Failed to load model: %s\n", path.c_str());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["LINEART_MODEL"] = preprocessor;
        outputs["path"] = path;
        printf("[LineArtLoader] LineArt model loaded: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LineArtLoader", LineArtLoaderNode);

// ============================================================================
// LineArtPreprocessor - LineArt 线稿提取
// ============================================================================
class LineArtPreprocessorNode : public Node {
public:
    std::string get_class_type() const override { return "LineArtPreprocessor"; }
    std::string get_category() const override { return "image/preprocessors"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"lineart_model", "LINEART_MODEL", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src = std::any_cast<ImagePtr>(inputs.at("image"));
        auto preprocessor = std::any_cast<std::shared_ptr<LineArtPreprocessor>>(inputs.at("lineart_model"));

        if (!src || !src->data || src->channel != 3) {
            fprintf(stderr, "[ERROR] LineArtPreprocessor: Requires RGB image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        if (!preprocessor) {
            fprintf(stderr, "[ERROR] LineArtPreprocessor: Model not loaded\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        printf("[LineArtPreprocessor] Processing %dx%d\n",
               src->width, src->height);

        LineArtResult result = preprocessor->process(src->data, src->width, src->height);
        if (!result.success) {
            fprintf(stderr, "[ERROR] LineArtPreprocessor: Processing failed\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_image_t dst_image = {};
        dst_image.width = result.width;
        dst_image.height = result.height;
        dst_image.channel = 3;
        dst_image.data = (uint8_t*)malloc(result.data.size());
        if (!dst_image.data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(dst_image.data, result.data.data(), result.data.size());

        sd_image_t* image_result = acquire_image();
        if (!image_result) {
            free(dst_image.data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *image_result = dst_image;

        outputs["IMAGE"] = make_image_ptr(image_result);
        printf("[LineArtPreprocessor] Done\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LineArtPreprocessor", LineArtPreprocessorNode);
#endif // HAS_ONNXRUNTIME

// ============================================================================
// KSampler 通用执行逻辑
// ============================================================================
static sd_error_t run_sampler_common(
    sd_ctx_t* sd_ctx,
    const NodeInputs& inputs,
    sd_node_sample_params_t& sample_params,
    sd_latent_t** out_result)
{
    // 处理 LoRA Stack
    std::vector<sd_lora_t> loras;
    if (inputs.count("lora_stack")) {
        auto lora_stack = std::any_cast<std::vector<LoRAInfo>>(inputs.at("lora_stack"));
        for (const auto& info : lora_stack) {
            sd_lora_t lora;
            lora.path = info.path.c_str();
            lora.multiplier = info.strength;
            lora.is_high_noise = false;
            loras.push_back(lora);
            printf("[KSampler] Applying LoRA: %s (strength=%.2f)\n", info.path.c_str(), info.strength);
        }
    }

    if (!loras.empty()) {
        sd_apply_loras(sd_ctx, loras.data(), static_cast<uint32_t>(loras.size()));
    } else {
        sd_clear_loras(sd_ctx);
    }

    // 处理 ControlNet 输入
    if (inputs.count("_control_image")) {
        ImagePtr ctrl_img = std::any_cast<ImagePtr>(inputs.at("_control_image"));
        if (ctrl_img && ctrl_img->data) {
            sample_params.control_image = *ctrl_img;
            sample_params.control_strength = inputs.count("_control_strength") ?
                std::any_cast<float>(inputs.at("_control_strength")) : 1.0f;
            printf("[KSampler] Using ControlNet (from Apply): strength=%.2f, image=%dx%d\n",
                   sample_params.control_strength, ctrl_img->width, ctrl_img->height);
        }
    } else if (inputs.count("control_image")) {
        ImagePtr ctrl_img = std::any_cast<ImagePtr>(inputs.at("control_image"));
        if (ctrl_img && ctrl_img->data) {
            sample_params.control_image = *ctrl_img;
            sample_params.control_strength = inputs.count("control_strength") ?
                std::any_cast<float>(inputs.at("control_strength")) : 1.0f;
            printf("[KSampler] Using ControlNet: strength=%.2f, image=%dx%d\n",
                   sample_params.control_strength, ctrl_img->width, ctrl_img->height);
        }
    }

    // 处理 Inpaint mask 输入
    if (inputs.count("mask")) {
        ImagePtr mask = std::any_cast<ImagePtr>(inputs.at("mask"));
        if (mask && mask->data) {
            sample_params.mask_image = *mask;
            printf("[KSampler] Using Inpaint mask: %dx%d\n", mask->width, mask->height);
        }
    }

    // 处理 IPAdapter 输入
    if (inputs.count("_ipadapter_info")) {
        IPAdapterInfo info = std::any_cast<IPAdapterInfo>(inputs.at("_ipadapter_info"));
        ImagePtr ip_image = std::any_cast<ImagePtr>(inputs.at("_ipadapter_image"));
        if (!info.path.empty() && ip_image && ip_image->data) {
            printf("[KSampler] Loading IPAdapter: %s\n", info.path.c_str());
            bool loaded = sd_load_ipadapter(
                sd_ctx,
                info.path.c_str(),
                info.cross_attention_dim,
                info.num_tokens,
                info.clip_embeddings_dim
            );
            if (loaded) {
                sd_set_ipadapter_image(sd_ctx, ip_image.get(), info.strength);
                printf("[KSampler] IPAdapter applied: strength=%.2f, image=%dx%d\n",
                       info.strength, ip_image->width, ip_image->height);
            } else {
                fprintf(stderr, "[ERROR] KSampler: Failed to load IPAdapter %s\n", info.path.c_str());
                sd_clear_loras(sd_ctx);
                return sd_error_t::ERROR_MODEL_LOADING;
            }
        }
    }

    ConditioningPtr positive = std::any_cast<ConditioningPtr>(inputs.at("positive"));
    ConditioningPtr negative = inputs.count("negative") ?
        std::any_cast<ConditioningPtr>(inputs.at("negative")) : nullptr;
    LatentPtr init_latent = std::any_cast<LatentPtr>(inputs.at("latent_image"));
    float denoise = inputs.count("denoise") ? std::any_cast<float>(inputs.at("denoise")) : 1.0f;

    *out_result = sd_sampler_run(
        sd_ctx, init_latent.get(), positive.get(), negative.get(), &sample_params, denoise);

    sd_clear_loras(sd_ctx);
    sd_clear_ipadapter(sd_ctx);

    if (!*out_result) {
        fprintf(stderr, "[ERROR] KSampler: Sampling failed\n");
        return sd_error_t::ERROR_SAMPLING_FAILED;
    }

    printf("[KSampler] Sampling completed\n");
    return sd_error_t::OK;
}

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
            {"lora_stack", "LORA_STACK", false, nullptr},
            {"control_image", "IMAGE", false, nullptr},
            {"control_strength", "FLOAT", false, 1.0f},
            {"mask", "MASK", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = nullptr;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("model"));
            sd_ctx = ctx_ptr.get();
        } catch (const std::bad_any_cast&) {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("model"));
        }
        int64_t seed = inputs.count("seed") ? std::any_cast<int>(inputs.at("seed")) : 0;
        int steps = inputs.count("steps") ? std::any_cast<int>(inputs.at("steps")) : 20;
        float cfg = inputs.count("cfg") ? std::any_cast<float>(inputs.at("cfg")) : 8.0f;
        std::string sampler_name = inputs.count("sampler_name") ?
            std::any_cast<std::string>(inputs.at("sampler_name")) : "euler";
        std::string scheduler_name = inputs.count("scheduler") ?
            std::any_cast<std::string>(inputs.at("scheduler")) : "normal";

        if (!sd_ctx) {
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_node_sample_params_t sample_params = {};
        sample_params.seed = seed;
        sample_params.steps = steps;
        sample_params.cfg_scale = cfg;
        sample_params.sample_method = str_to_sample_method(sampler_name.c_str());
        sample_params.scheduler = str_to_scheduler(scheduler_name.c_str());
        sample_params.eta = 0.0f;
        sample_params.add_noise = true;

        if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        }
        if (sample_params.scheduler == SCHEDULER_COUNT) {
            sample_params.scheduler = DISCRETE_SCHEDULER;
        }

        printf("[KSampler] Running sampler: steps=%d, seed=%ld, cfg=%.2f\n",
               steps, (long)seed, cfg);

        sd_latent_t* result = nullptr;
        sd_error_t err = run_sampler_common(sd_ctx, inputs, sample_params, &result);
        if (is_error(err)) return err;

        outputs["LATENT"] = make_latent_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("KSampler", KSamplerNode);

// ============================================================================
// KSamplerAdvanced - 高级采样器
// ============================================================================
class KSamplerAdvancedNode : public Node {
public:
    std::string get_class_type() const override { return "KSamplerAdvanced"; }
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
            {"start_at_step", "INT", false, 0},
            {"end_at_step", "INT", false, 10000},
            {"add_noise", "BOOLEAN", false, true},
            {"lora_stack", "LORA_STACK", false, nullptr},
            {"control_image", "IMAGE", false, nullptr},
            {"control_strength", "FLOAT", false, 1.0f},
            {"mask", "MASK", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = nullptr;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("model"));
            sd_ctx = ctx_ptr.get();
        } catch (const std::bad_any_cast&) {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("model"));
        }
        int64_t seed = inputs.count("seed") ? std::any_cast<int>(inputs.at("seed")) : 0;
        int steps = inputs.count("steps") ? std::any_cast<int>(inputs.at("steps")) : 20;
        float cfg = inputs.count("cfg") ? std::any_cast<float>(inputs.at("cfg")) : 8.0f;
        std::string sampler_name = inputs.count("sampler_name") ?
            std::any_cast<std::string>(inputs.at("sampler_name")) : "euler";
        std::string scheduler_name = inputs.count("scheduler") ?
            std::any_cast<std::string>(inputs.at("scheduler")) : "normal";
        int start_at_step = inputs.count("start_at_step") ? std::any_cast<int>(inputs.at("start_at_step")) : 0;
        int end_at_step = inputs.count("end_at_step") ? std::any_cast<int>(inputs.at("end_at_step")) : 10000;
        bool add_noise = inputs.count("add_noise") ? std::any_cast<bool>(inputs.at("add_noise")) : true;

        if (!sd_ctx) {
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_node_sample_params_t sample_params = {};
        sample_params.seed = seed;
        sample_params.steps = steps;
        sample_params.cfg_scale = cfg;
        sample_params.sample_method = str_to_sample_method(sampler_name.c_str());
        sample_params.scheduler = str_to_scheduler(scheduler_name.c_str());
        sample_params.eta = 0.0f;
        sample_params.start_at_step = start_at_step;
        sample_params.end_at_step = end_at_step;
        sample_params.add_noise = add_noise;

        if (sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            sample_params.sample_method = EULER_A_SAMPLE_METHOD;
        }
        if (sample_params.scheduler == SCHEDULER_COUNT) {
            sample_params.scheduler = DISCRETE_SCHEDULER;
        }

        printf("[KSamplerAdvanced] Running sampler: steps=%d, seed=%ld, cfg=%.2f, start=%d, end=%d, add_noise=%s\n",
               steps, (long)seed, cfg, start_at_step, end_at_step, add_noise ? "true" : "false");

        sd_latent_t* result = nullptr;
        sd_error_t err = run_sampler_common(sd_ctx, inputs, sample_params, &result);
        if (is_error(err)) return err;

        outputs["LATENT"] = make_latent_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("KSamplerAdvanced", KSamplerAdvancedNode);

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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr latent = std::any_cast<LatentPtr>(inputs.at("samples"));
        sd_ctx_t* sd_ctx = nullptr;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("vae"));
            sd_ctx = ctx_ptr.get();
        } catch (const std::bad_any_cast&) {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("vae"));
        }

        if (!latent) {
            fprintf(stderr, "[ERROR] VAEDecode: No latent data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_image_t* image = sd_decode_latent(sd_ctx, latent.get());
        if (!image) {
            fprintf(stderr, "[ERROR] VAEDecode: Failed to decode latent\n");
            return sd_error_t::ERROR_DECODING_FAILED;
        }

        printf("[VAEDecode] Latent decoded: %dx%d\n", image->width, image->height);
        outputs["IMAGE"] = make_image_ptr(image);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("VAEDecode", VAEDecodeNode);

// ============================================================================
// UpscaleModelLoader - 加载 ESRGAN 放大模型
// ============================================================================
class UpscaleModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "UpscaleModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model_name", "STRING", true, std::string("")},
            {"use_gpu", "BOOLEAN", false, true},
            {"tile_size", "INT", false, 512}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"UPSCALE_MODEL", "UPSCALE_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string model_path = std::any_cast<std::string>(inputs.at("model_name"));
        bool use_gpu = inputs.count("use_gpu") ? std::any_cast<bool>(inputs.at("use_gpu")) : true;
        int tile_size = inputs.count("tile_size") ? std::any_cast<int>(inputs.at("tile_size")) : 512;

        if (model_path.empty()) {
            fprintf(stderr, "[ERROR] UpscaleModelLoader: model_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        printf("[UpscaleModelLoader] Loading model: %s\n", model_path.c_str());

        upscaler_ctx_t* upscaler = new_upscaler_ctx(
            model_path.c_str(),
            !use_gpu,    // w_mode
            false,       // no longer used
            4,           // threads
            tile_size
        );

        if (!upscaler) {
            fprintf(stderr, "[ERROR] UpscaleModelLoader: Failed to load model\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        int scale = get_upscale_factor(upscaler);
        printf("[UpscaleModelLoader] Model loaded, scale=%dx, tile_size=%d\n", scale, tile_size);

        outputs["UPSCALE_MODEL"] = make_upscaler_ptr(upscaler);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("UpscaleModelLoader", UpscaleModelLoaderNode);

// ============================================================================
// ImageUpscaleWithModel - 使用 ESRGAN 模型放大图像
// ============================================================================
class ImageUpscaleWithModelNode : public Node {
public:
    std::string get_class_type() const override { return "ImageUpscaleWithModel"; }
    std::string get_category() const override { return "image/upscaling"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"upscale_model", "UPSCALE_MODEL", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        UpscalerPtr upscaler = std::any_cast<UpscalerPtr>(inputs.at("upscale_model"));

        if (!image || !image->data) {
            fprintf(stderr, "[ERROR] ImageUpscaleWithModel: No image data\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        if (!upscaler) {
            fprintf(stderr, "[ERROR] ImageUpscaleWithModel: No upscale model\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int scale = get_upscale_factor(upscaler.get());
        printf("[ImageUpscaleWithModel] Upscaling %dx%d by %dx...\n",
               image->width, image->height, scale);

        sd_image_t result = upscale(upscaler.get(), *image, scale);
        if (!result.data) {
            fprintf(stderr, "[ERROR] ImageUpscaleWithModel: Upscale failed\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("[ImageUpscaleWithModel] Upscaled to %dx%d\n", result.width, result.height);

        // 将结果包装为智能指针（upscale 返回的是新分配的图像数据）
        sd_image_t* result_ptr = acquire_image();
        if (!result_ptr) {
            free(result.data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result_ptr = result;
        outputs["IMAGE"] = make_image_ptr(result_ptr);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageUpscaleWithModel", ImageUpscaleWithModelNode);

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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("images"));
        std::string prefix = inputs.count("filename_prefix") ?
            std::any_cast<std::string>(inputs.at("filename_prefix")) : "sd-engine";

        if (!image || !image->data) {
            fprintf(stderr, "[ERROR] SaveImage: No image data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        std::string filename = prefix + ".png";
        printf("[SaveImage] Saving to %s (%dx%d)\n",
               filename.c_str(), image->width, image->height);

        bool success = stbi_write_png(filename.c_str(),
                                      image->width, image->height,
                                      image->channel, image->data, 0) != 0;
        if (!success) {
            fprintf(stderr, "[ERROR] SaveImage: Failed to write %s\n", filename.c_str());
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("[SaveImage] Saved successfully\n");
        return sd_error_t::OK;
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src_image = std::any_cast<ImagePtr>(inputs.at("image"));
        int target_width = std::any_cast<int>(inputs.at("width"));
        int target_height = std::any_cast<int>(inputs.at("height"));
        std::string method = inputs.count("method") ?
            std::any_cast<std::string>(inputs.at("method")) : "bilinear";

        if (!src_image || !src_image->data) {
            fprintf(stderr, "[ERROR] ImageScale: No source image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        if (target_width <= 0 || target_height <= 0) {
            fprintf(stderr, "[ERROR] ImageScale: Invalid target size %dx%d\n", target_width, target_height);
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        // 如果尺寸相同，直接返回原图
        if ((int)src_image->width == target_width && (int)src_image->height == target_height) {
            outputs["IMAGE"] = src_image;
            return sd_error_t::OK;
        }

        printf("[ImageScale] Resizing from %dx%d to %dx%d (method: %s)\n",
               src_image->width, src_image->height, target_width, target_height, method.c_str());

        // 分配输出缓冲区
        size_t dst_size = target_width * target_height * src_image->channel;
        uint8_t* dst_data = (uint8_t*)malloc(dst_size);
        if (!dst_data) {
            fprintf(stderr, "[ERROR] ImageScale: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
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

        sd_image_t dst_image = {};
        dst_image.width = target_width;
        dst_image.height = target_height;
        dst_image.channel = src_image->channel;
        dst_image.data = dst_data;

        sd_image_t* result = acquire_image();
        if (!result) {
            free(dst_data);
            fprintf(stderr, "[ERROR] ImageScale: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result = dst_image;

        outputs["IMAGE"] = make_image_ptr(result);
        printf("[ImageScale] Resized successfully\n");
        return sd_error_t::OK;
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src_image = std::any_cast<ImagePtr>(inputs.at("image"));
        int crop_x = std::any_cast<int>(inputs.at("x"));
        int crop_y = std::any_cast<int>(inputs.at("y"));
        int crop_width = std::any_cast<int>(inputs.at("width"));
        int crop_height = std::any_cast<int>(inputs.at("height"));

        if (!src_image || !src_image->data) {
            fprintf(stderr, "[ERROR] ImageCrop: No source image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        // 验证裁剪区域
        if (crop_x < 0 || crop_y < 0 ||
            crop_width <= 0 || crop_height <= 0 ||
            crop_x + crop_width > (int)src_image->width ||
            crop_y + crop_height > (int)src_image->height) {
            fprintf(stderr, "[ERROR] ImageCrop: Invalid crop region (%d,%d,%d,%d) for image %dx%d\n",
                    crop_x, crop_y, crop_width, crop_height,
                    src_image->width, src_image->height);
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("[ImageCrop] Cropping to (%d,%d) size %dx%d\n",
               crop_x, crop_y, crop_width, crop_height);

        // 分配输出缓冲区
        uint8_t* dst_data = (uint8_t*)malloc(crop_width * crop_height * src_image->channel);
        if (!dst_data) {
            fprintf(stderr, "[ERROR] ImageCrop: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
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

        sd_image_t dst_image = {};
        dst_image.width = crop_width;
        dst_image.height = crop_height;
        dst_image.channel = src_image->channel;
        dst_image.data = dst_data;

        sd_image_t* result = acquire_image();
        if (!result) {
            free(dst_data);
            fprintf(stderr, "[ERROR] ImageCrop: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result = dst_image;

        outputs["IMAGE"] = make_image_ptr(result);
        printf("[ImageCrop] Cropped successfully\n");
        return sd_error_t::OK;
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("images"));

        if (!image || !image->data) {
            fprintf(stderr, "[ERROR] PreviewImage: No image data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("\n");
        printf("╔══════════════════════════════════════╗\n");
        printf("║         [PreviewImage]               ║\n");
        printf("║  Size: %4dx%-4d                     ║\n", image->width, image->height);
        printf("║  Channels: %d                        ║\n", image->channel);
        printf("╚══════════════════════════════════════╝\n");
        printf("\n");

        return sd_error_t::OK;
    }
};
REGISTER_NODE("PreviewImage", PreviewImageNode);

// ============================================================================
// ImageBlend - 图像混合
// ============================================================================
class ImageBlendNode : public Node {
public:
    std::string get_class_type() const override { return "ImageBlend"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image1", "IMAGE", true, nullptr},
            {"image2", "IMAGE", true, nullptr},
            {"blend_factor", "FLOAT", false, 0.5f},
            {"blend_mode", "STRING", false, std::string("normal")}  // normal, add, multiply, screen
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img1 = std::any_cast<ImagePtr>(inputs.at("image1"));
        ImagePtr img2 = std::any_cast<ImagePtr>(inputs.at("image2"));
        float blend_factor = inputs.count("blend_factor") ?
            std::any_cast<float>(inputs.at("blend_factor")) : 0.5f;
        std::string blend_mode = inputs.count("blend_mode") ?
            std::any_cast<std::string>(inputs.at("blend_mode")) : "normal";

        if (!img1 || !img1->data || !img2 || !img2->data) {
            fprintf(stderr, "[ERROR] ImageBlend: Missing input images\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int width = (int)img1->width;
        int height = (int)img1->height;
        int channels = (int)img1->channel;

        if ((int)img2->width != width || (int)img2->height != height) {
            fprintf(stderr, "[ERROR] ImageBlend: Image sizes must match (%dx%d vs %dx%d)\n",
                    width, height, (int)img2->width, (int)img2->height);
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int ch2 = (int)img2->channel;
        int out_channels = std::max(channels, ch2);

        uint8_t* dst_data = (uint8_t*)malloc(width * height * out_channels);
        if (!dst_data) {
            fprintf(stderr, "[ERROR] ImageBlend: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx1 = (y * width + x) * channels;
                int idx2 = (y * width + x) * ch2;
                int idx_dst = (y * width + x) * out_channels;

                for (int c = 0; c < out_channels; c++) {
                    float v1 = (c < channels) ? (img1->data[idx1 + c] / 255.0f) : 1.0f;
                    float v2 = (c < ch2) ? (img2->data[idx2 + c] / 255.0f) : 1.0f;
                    float result = 0.0f;

                    if (blend_mode == "normal") {
                        result = v1 * (1.0f - blend_factor) + v2 * blend_factor;
                    } else if (blend_mode == "add") {
                        result = v1 + v2 * blend_factor;
                    } else if (blend_mode == "multiply") {
                        result = v1 * (1.0f - blend_factor) + (v1 * v2) * blend_factor;
                    } else if (blend_mode == "screen") {
                        float screen = 1.0f - (1.0f - v1) * (1.0f - v2);
                        result = v1 * (1.0f - blend_factor) + screen * blend_factor;
                    } else {
                        result = v1 * (1.0f - blend_factor) + v2 * blend_factor;
                    }

                    result = std::clamp(result, 0.0f, 1.0f);
                    dst_data[idx_dst + c] = (uint8_t)(result * 255.0f + 0.5f);
                }
            }
        }

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            free(dst_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = width;
        result_img->height = height;
        result_img->channel = out_channels;
        result_img->data = dst_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        printf("[ImageBlend] Blended %dx%dx%d (mode=%s, factor=%.2f)\n",
               width, height, out_channels, blend_mode.c_str(), blend_factor);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageBlend", ImageBlendNode);

// ============================================================================
// ImageCompositeMasked - 蒙版合成
// ============================================================================
class ImageCompositeMaskedNode : public Node {
public:
    std::string get_class_type() const override { return "ImageCompositeMasked"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"destination", "IMAGE", true, nullptr},
            {"source", "IMAGE", true, nullptr},
            {"x", "INT", false, 0},
            {"y", "INT", false, 0},
            {"mask", "IMAGE", false, nullptr},
            {"resize_source", "BOOLEAN", false, false}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr dst = std::any_cast<ImagePtr>(inputs.at("destination"));
        ImagePtr src = std::any_cast<ImagePtr>(inputs.at("source"));
        int offset_x = inputs.count("x") ? std::any_cast<int>(inputs.at("x")) : 0;
        int offset_y = inputs.count("y") ? std::any_cast<int>(inputs.at("y")) : 0;
        ImagePtr mask = inputs.count("mask") ? std::any_cast<ImagePtr>(inputs.at("mask")) : nullptr;
        bool resize_source = inputs.count("resize_source") ? std::any_cast<bool>(inputs.at("resize_source")) : false;

        if (!dst || !dst->data || !src || !src->data) {
            fprintf(stderr, "[ERROR] ImageCompositeMasked: Missing input images\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int dst_w = (int)dst->width;
        int dst_h = (int)dst->height;
        int dst_c = (int)dst->channel;
        int src_w = (int)src->width;
        int src_h = (int)src->height;
        int src_c = (int)src->channel;

        // 如果需要，把 source 缩放到和 destination 一样大
        std::vector<uint8_t> src_resized;
        const uint8_t* src_data = src->data;
        int src_stride_w = src_w;
        int src_stride_h = src_h;
        if (resize_source && (src_w != dst_w || src_h != dst_h)) {
            src_resized.resize(dst_w * dst_h * src_c);
            stbir_resize(
                src->data, src_w, src_h, 0,
                src_resized.data(), dst_w, dst_h, 0,
                STBIR_TYPE_UINT8, src_c, -1, 0,
                STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE,
                STBIR_COLORSPACE_LINEAR, nullptr
            );
            src_data = src_resized.data();
            src_stride_w = dst_w;
            src_stride_h = dst_h;
        }

        // 分配输出（复制 destination）
        uint8_t* out_data = (uint8_t*)malloc(dst_w * dst_h * dst_c);
        if (!out_data) {
            fprintf(stderr, "[ERROR] ImageCompositeMasked: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(out_data, dst->data, dst_w * dst_h * dst_c);

        for (int y = 0; y < src_stride_h; y++) {
            int dst_y = offset_y + y;
            if (dst_y < 0 || dst_y >= dst_h) continue;

            for (int x = 0; x < src_stride_w; x++) {
                int dst_x = offset_x + x;
                if (dst_x < 0 || dst_x >= dst_w) continue;

                int dst_idx = (dst_y * dst_w + dst_x) * dst_c;
                int src_idx = (y * src_stride_w + x) * src_c;

                float alpha = 1.0f;
                if (mask && mask->data) {
                    int mask_x = (mask->width == 1) ? 0 : (x * (int)mask->width / src_stride_w);
                    int mask_y = (mask->height == 1) ? 0 : (y * (int)mask->height / src_stride_h);
                    mask_x = std::clamp(mask_x, 0, (int)mask->width - 1);
                    mask_y = std::clamp(mask_y, 0, (int)mask->height - 1);
                    int mask_idx = (mask_y * (int)mask->width + mask_x) * (int)mask->channel;
                    alpha = mask->data[mask_idx] / 255.0f;
                }

                for (int c = 0; c < dst_c; c++) {
                    float src_v = (c < src_c) ? (src_data[src_idx + c] / 255.0f) : 1.0f;
                    float dst_v = out_data[dst_idx + c] / 255.0f;
                    float blended = dst_v * (1.0f - alpha) + src_v * alpha;
                    out_data[dst_idx + c] = (uint8_t)(std::clamp(blended, 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
        }

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            free(out_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = dst_w;
        result_img->height = dst_h;
        result_img->channel = dst_c;
        result_img->data = out_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        printf("[ImageCompositeMasked] Composited source onto destination at (%d,%d)\n", offset_x, offset_y);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageCompositeMasked", ImageCompositeMaskedNode);

// ============================================================================
// ImageInvert - 颜色反转
// ============================================================================
class ImageInvertNode : public Node {
public:
    std::string get_class_type() const override { return "ImageInvert"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        if (!img || !img->data) {
            fprintf(stderr, "[ERROR] ImageInvert: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;
        size_t pixels = w * h * c;

        uint8_t* dst = (uint8_t*)malloc(pixels);
        if (!dst) return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            for (int ch = 0; ch < c; ch++) {
                dst[i * c + ch] = 255 - img->data[i * c + ch];
            }
        }

        sd_image_t* result = acquire_image();
        if (!result) { free(dst); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result->width = w; result->height = h; result->channel = c; result->data = dst;
        outputs["IMAGE"] = make_image_ptr(result);
        printf("[ImageInvert] Inverted %dx%dx%d\n", w, h, c);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageInvert", ImageInvertNode);

// ============================================================================
// ImageColorAdjust - 亮度/对比度/饱和度调整
// ============================================================================
class ImageColorAdjustNode : public Node {
public:
    std::string get_class_type() const override { return "ImageColorAdjust"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"brightness", "FLOAT", false, 1.0f},
            {"contrast", "FLOAT", false, 1.0f},
            {"saturation", "FLOAT", false, 1.0f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        float brightness = inputs.count("brightness") ? std::any_cast<float>(inputs.at("brightness")) : 1.0f;
        float contrast   = inputs.count("contrast")   ? std::any_cast<float>(inputs.at("contrast"))   : 1.0f;
        float saturation = inputs.count("saturation") ? std::any_cast<float>(inputs.at("saturation")) : 1.0f;

        if (!img || !img->data) {
            fprintf(stderr, "[ERROR] ImageColorAdjust: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;
        if (c != 3 && c != 4) {
            fprintf(stderr, "[ERROR] ImageColorAdjust: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        uint8_t* dst = (uint8_t*)malloc(w * h * c);
        if (!dst) return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            float r = img->data[i * c + 0] / 255.0f;
            float g = img->data[i * c + 1] / 255.0f;
            float b = img->data[i * c + 2] / 255.0f;

            // Brightness
            r *= brightness; g *= brightness; b *= brightness;

            // Contrast
            r = (r - 0.5f) * contrast + 0.5f;
            g = (g - 0.5f) * contrast + 0.5f;
            b = (b - 0.5f) * contrast + 0.5f;

            // Saturation
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            r = gray + (r - gray) * saturation;
            g = gray + (g - gray) * saturation;
            b = gray + (b - gray) * saturation;

            dst[i * c + 0] = (uint8_t)(std::clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[i * c + 1] = (uint8_t)(std::clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[i * c + 2] = (uint8_t)(std::clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
            if (c == 4) dst[i * c + 3] = img->data[i * c + 3];
        }

        sd_image_t* result = acquire_image();
        if (!result) { free(dst); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result->width = w; result->height = h; result->channel = c; result->data = dst;
        outputs["IMAGE"] = make_image_ptr(result);
        printf("[ImageColorAdjust] Adjusted %dx%dx%d (b=%.2f, c=%.2f, s=%.2f)\n", w, h, c, brightness, contrast, saturation);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageColorAdjust", ImageColorAdjustNode);

// ============================================================================
// ImageBlur - 盒式模糊（简化版高斯模糊）
// ============================================================================
class ImageBlurNode : public Node {
public:
    std::string get_class_type() const override { return "ImageBlur"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"radius", "INT", false, 3}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        int radius = inputs.count("radius") ? std::any_cast<int>(inputs.at("radius")) : 3;
        if (radius < 1) radius = 1;

        if (!img || !img->data) {
            fprintf(stderr, "[ERROR] ImageBlur: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;

        uint8_t* dst = (uint8_t*)malloc(w * h * c);
        if (!dst) return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    int sum = 0, count = 0;
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int py = std::clamp(y + dy, 0, h - 1);
                            int px = std::clamp(x + dx, 0, w - 1);
                            sum += img->data[(py * w + px) * c + ch];
                            count++;
                        }
                    }
                    dst[(y * w + x) * c + ch] = (uint8_t)(sum / count);
                }
            }
        }

        sd_image_t* result = acquire_image();
        if (!result) { free(dst); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result->width = w; result->height = h; result->channel = c; result->data = dst;
        outputs["IMAGE"] = make_image_ptr(result);
        printf("[ImageBlur] Blurred %dx%dx%d (radius=%d)\n", w, h, c, radius);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageBlur", ImageBlurNode);

// ============================================================================
// ImageGrayscale - 灰度转换
// ============================================================================
class ImageGrayscaleNode : public Node {
public:
    std::string get_class_type() const override { return "ImageGrayscale"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        if (!img || !img->data) {
            fprintf(stderr, "[ERROR] ImageGrayscale: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;
        if (c < 3) {
            fprintf(stderr, "[ERROR] ImageGrayscale: Image must have at least 3 channels\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        uint8_t* dst = (uint8_t*)malloc(w * h);
        if (!dst) return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            dst[i] = (uint8_t)(0.299f * img->data[i * c + 0] + 0.587f * img->data[i * c + 1] + 0.114f * img->data[i * c + 2] + 0.5f);
        }

        sd_image_t* result = acquire_image();
        if (!result) { free(dst); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result->width = w; result->height = h; result->channel = 1; result->data = dst;
        outputs["IMAGE"] = make_image_ptr(result);
        printf("[ImageGrayscale] Converted %dx%d to grayscale\n", w, h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageGrayscale", ImageGrayscaleNode);

// ============================================================================
// ImageThreshold - 二值化
// ============================================================================
class ImageThresholdNode : public Node {
public:
    std::string get_class_type() const override { return "ImageThreshold"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"threshold", "INT", false, 128}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        int threshold = inputs.count("threshold") ? std::any_cast<int>(inputs.at("threshold")) : 128;
        threshold = std::clamp(threshold, 0, 255);

        if (!img || !img->data) {
            fprintf(stderr, "[ERROR] ImageThreshold: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;

        uint8_t* dst = (uint8_t*)malloc(w * h * c);
        if (!dst) return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            for (int ch = 0; ch < c; ch++) {
                dst[i * c + ch] = img->data[i * c + ch] >= threshold ? 255 : 0;
            }
        }

        sd_image_t* result = acquire_image();
        if (!result) { free(dst); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result->width = w; result->height = h; result->channel = c; result->data = dst;
        outputs["IMAGE"] = make_image_ptr(result);
        printf("[ImageThreshold] Thresholded %dx%dx%d (threshold=%d)\n", w, h, c, threshold);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageThreshold", ImageThresholdNode);

// ============================================================================
// FaceDetectModelLoader - 加载人脸检测 ONNX 模型
// ============================================================================
#ifdef HAS_ONNXRUNTIME
namespace face = ::sdengine::face;

class FaceDetectModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "FaceDetectModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_DETECT_MODEL", "FACE_DETECT_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_path"));
        if (path.empty()) {
            fprintf(stderr, "[ERROR] FaceDetectModelLoader: model_path is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto detector = std::make_shared<face::FaceDetector>();
        if (!detector->load(path)) {
            fprintf(stderr, "[ERROR] FaceDetectModelLoader: Failed to load %s\n", path.c_str());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["FACE_DETECT_MODEL"] = detector;
        printf("[FaceDetectModelLoader] Loaded: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceDetectModelLoader", FaceDetectModelLoaderNode);

// ============================================================================
// FaceDetect - 人脸检测
// ============================================================================
class FaceDetectNode : public Node {
public:
    std::string get_class_type() const override { return "FaceDetect"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"model", "FACE_DETECT_MODEL", true, nullptr},
            {"confidence_threshold", "FLOAT", false, 0.5f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"IMAGE", "IMAGE"},
            {"faces", "FACE_BBOX_LIST"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        auto detector = std::any_cast<std::shared_ptr<face::FaceDetector> >(inputs.at("model"));
        float threshold = inputs.count("confidence_threshold") ?
            std::any_cast<float>(inputs.at("confidence_threshold")) : 0.5f;

        if (!image || !image->data || !detector) {
            fprintf(stderr, "[ERROR] FaceDetect: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int channels = (int)image->channel;
        if (channels != 3 && channels != 4) {
            fprintf(stderr, "[ERROR] FaceDetect: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        // Convert to RGB if RGBA
        std::vector<uint8_t> rgb_data;
        const uint8_t* input_ptr = image->data;
        if (channels == 4) {
            rgb_data.resize(image->width * image->height * 3);
            for (size_t i = 0; i < image->width * image->height; i++) {
                rgb_data[i * 3 + 0] = image->data[i * 4 + 0];
                rgb_data[i * 3 + 1] = image->data[i * 4 + 1];
                rgb_data[i * 3 + 2] = image->data[i * 4 + 2];
            }
            input_ptr = rgb_data.data();
        }

        face::FaceDetectResult detect_result = detector->detect(
            input_ptr, (int)image->width, (int)image->height, threshold);

        // Draw detection boxes on output image
        uint8_t* out_data = (uint8_t*)malloc(image->width * image->height * channels);
        if (!out_data) return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data, image->data, image->width * image->height * channels);

        auto draw_rect = [&](int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b) {
            x1 = std::max(0, x1); y1 = std::max(0, y1);
            x2 = std::min((int)image->width - 1, x2);
            y2 = std::min((int)image->height - 1, y2);
            for (int x = x1; x <= x2; x++) {
                for (int y = y1; y <= y2; y++) {
                    if (x == x1 || x == x2 || y == y1 || y == y2) {
                        size_t idx = (y * image->width + x) * channels;
                        out_data[idx + 0] = r;
                        out_data[idx + 1] = g;
                        out_data[idx + 2] = b;
                    }
                }
            }
        };

        auto draw_point = [&](int cx, int cy, uint8_t r, uint8_t g, uint8_t b) {
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int px = cx + dx, py = cy + dy;
                    if (px >= 0 && px < (int)image->width && py >= 0 && py < (int)image->height) {
                        size_t idx = (py * image->width + px) * channels;
                        out_data[idx + 0] = r;
                        out_data[idx + 1] = g;
                        out_data[idx + 2] = b;
                    }
                }
            }
        };

        for (const auto& f : detect_result.faces) {
            int x1 = (int)f.x1, y1 = (int)f.y1;
            int x2 = (int)f.x2, y2 = (int)f.y2;
            draw_rect(x1, y1, x2, y2, 0, 255, 0);
            for (int k = 0; k < 5; k++) {
                draw_point((int)f.landmarks[k * 2], (int)f.landmarks[k * 2 + 1], 0, 0, 255);
            }
        }

        sd_image_t* result_img = acquire_image();
        if (!result_img) { free(out_data); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result_img->width = image->width;
        result_img->height = image->height;
        result_img->channel = channels;
        result_img->data = out_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        outputs["faces"] = detect_result;
        printf("[FaceDetect] Detected %zu faces\n", detect_result.faces.size());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceDetect", FaceDetectNode);

// ============================================================================
// FaceRestoreModelLoader - 加载人脸修复 ONNX 模型
// ============================================================================
class FaceRestoreModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "FaceRestoreModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model_path", "STRING", true, std::string("")},
            {"model_type", "STRING", false, std::string("gfpgan")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_RESTORE_MODEL", "FACE_RESTORE_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_path"));
        std::string type_str = inputs.count("model_type") ?
            std::any_cast<std::string>(inputs.at("model_type")) : "gfpgan";

        if (path.empty()) {
            fprintf(stderr, "[ERROR] FaceRestoreModelLoader: model_path is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto type = (type_str == "codeformer") ? face::RestoreModelType::CODEFORMER :
                                                   face::RestoreModelType::GFPGAN;

        auto restorer = std::make_shared<face::FaceRestorer>();
        if (!restorer->load(path, type)) {
            fprintf(stderr, "[ERROR] FaceRestoreModelLoader: Failed to load %s\n", path.c_str());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["FACE_RESTORE_MODEL"] = restorer;
        printf("[FaceRestoreModelLoader] Loaded: %s (type=%s)\n", path.c_str(), type_str.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceRestoreModelLoader", FaceRestoreModelLoaderNode);

// ============================================================================
// FaceRestoreWithModel - 人脸修复（一键版）
// ============================================================================
class FaceRestoreWithModelNode : public Node {
public:
    std::string get_class_type() const override { return "FaceRestoreWithModel"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"face_restore_model", "FACE_RESTORE_MODEL", true, nullptr},
            {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr},
            {"codeformer_fidelity", "FLOAT", false, 0.5f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        auto restorer = std::any_cast<std::shared_ptr<face::FaceRestorer>>(inputs.at("face_restore_model"));
        float fidelity = inputs.count("codeformer_fidelity") ?
            std::any_cast<float>(inputs.at("codeformer_fidelity")) : 0.5f;

        if (!image || !image->data || !restorer) {
            fprintf(stderr, "[ERROR] FaceRestoreWithModel: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int channels = (int)image->channel;
        if (channels != 3 && channels != 4) {
            fprintf(stderr, "[ERROR] FaceRestoreWithModel: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        // 获取人脸检测器（优先使用传入的，否则尝试内部默认）
        std::shared_ptr<face::FaceDetector> detector;
        if (inputs.count("face_detect_model")) {
            detector = std::any_cast<std::shared_ptr<face::FaceDetector>>(inputs.at("face_detect_model"));
        }

        // 转换图像为 RGB
        std::vector<uint8_t> rgb_data(image->width * image->height * 3);
        if (channels == 4) {
            for (size_t i = 0; i < image->width * image->height; i++) {
                rgb_data[i * 3 + 0] = image->data[i * 4 + 0];
                rgb_data[i * 3 + 1] = image->data[i * 4 + 1];
                rgb_data[i * 3 + 2] = image->data[i * 4 + 2];
            }
        } else {
            memcpy(rgb_data.data(), image->data, image->width * image->height * 3);
        }

        // 检测人脸
        face::FaceDetectResult detect_result;
        if (detector) {
            detect_result = detector->detect(rgb_data.data(), (int)image->width, (int)image->height, 0.5f);
        }

        // 分配输出图像（复制原图）
        uint8_t* out_data = (uint8_t*)malloc(image->width * image->height * channels);
        if (!out_data) return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data, image->data, image->width * image->height * channels);

        if (detect_result.faces.empty()) {
            printf("[FaceRestoreWithModel] No faces detected, returning original image\n");
            sd_image_t* result_img = acquire_image();
            if (!result_img) { free(out_data); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
            result_img->width = image->width;
            result_img->height = image->height;
            result_img->channel = channels;
            result_img->data = out_data;
            outputs["IMAGE"] = make_image_ptr(result_img);
            return sd_error_t::OK;
        }

        // 逐个修复人脸并贴回
        for (const auto& f : detect_result.faces) {
            // 1. 裁剪人脸（使用稍大的框，确保包含完整人脸）
            float cx = (f.x1 + f.x2) * 0.5f;
            float cy = (f.y1 + f.y2) * 0.5f;
            float size = std::max(f.x2 - f.x1, f.y2 - f.y1) * 1.5f;
            int crop_x1 = (int)std::max(0.0f, cx - size * 0.5f);
            int crop_y1 = (int)std::max(0.0f, cy - size * 0.5f);
            int crop_x2 = (int)std::min((float)image->width, cx + size * 0.5f);
            int crop_y2 = (int)std::min((float)image->height, cy + size * 0.5f);
            int crop_w = crop_x2 - crop_x1;
            int crop_h = crop_y2 - crop_y1;

            if (crop_w <= 0 || crop_h <= 0) continue;

            // 提取裁剪区域
            std::vector<uint8_t> cropped(crop_w * crop_h * 3);
            for (int y = 0; y < crop_h; y++) {
                for (int x = 0; x < crop_w; x++) {
                    int src_idx = ((crop_y1 + y) * image->width + (crop_x1 + x)) * 3;
                    int dst_idx = (y * crop_w + x) * 3;
                    cropped[dst_idx + 0] = rgb_data[src_idx + 0];
                    cropped[dst_idx + 1] = rgb_data[src_idx + 1];
                    cropped[dst_idx + 2] = rgb_data[src_idx + 2];
                }
            }

            // 2. 对齐到 512x512
            float M[6], inv_M[6];
            float template_pts[10];
            face::get_standard_face_template_512(template_pts);

            if (!face::estimate_affine_transform_2d3(f.landmarks, template_pts, M)) {
                continue;
            }
            if (!face::invert_affine_transform(M, inv_M)) {
                continue;
            }

            std::vector<uint8_t> aligned_face = face::crop_face(
                cropped.data(), crop_w, crop_h, 3, f.landmarks, 512);

            // 3. ONNX 修复
            auto restore_result = restorer->restore(aligned_face.data(), fidelity);
            if (!restore_result.success) {
                fprintf(stderr, "[ERROR] FaceRestoreWithModel: Restore failed for one face\n");
                continue;
            }

            // 4. 贴回原图
            // 生成羽化蒙版
            std::vector<uint8_t> mask(512 * 512);
            face::generate_feather_mask(mask.data(), 512, 32);

            // 将修复后的人脸 warp 回裁剪空间，再贴回输出图
            // 简化处理：直接在输出图上进行逆变换采样并融合
            for (int y = 0; y < crop_h; y++) {
                for (int x = 0; x < crop_w; x++) {
                    // 裁剪空间坐标 -> 对齐空间坐标
                    float crop_x = (float)x;
                    float crop_y = (float)y;
                    float aligned_x = inv_M[0] * crop_x + inv_M[1] * crop_y + inv_M[2];
                    float aligned_y = inv_M[3] * crop_x + inv_M[4] * crop_y + inv_M[5];

                    if (aligned_x < 0 || aligned_x >= 511 || aligned_y < 0 || aligned_y >= 511) {
                        continue;
                    }

                    int ax = (int)aligned_x;
                    int ay = (int)aligned_y;
                    float fx = aligned_x - ax;
                    float fy = aligned_y - ay;

                    // 双线性采样蒙版值
                    uint8_t m00 = mask[ay * 512 + ax];
                    uint8_t m01 = mask[ay * 512 + std::min(511, ax + 1)];
                    uint8_t m10 = mask[std::min(511, ay + 1) * 512 + ax];
                    uint8_t m11 = mask[std::min(511, ay + 1) * 512 + std::min(511, ax + 1)];
                    float mask_val = (m00 * (1 - fx) * (1 - fy) +
                                      m01 * fx * (1 - fy) +
                                      m10 * (1 - fx) * fy +
                                      m11 * fx * fy) / 255.0f;

                    if (mask_val <= 0.01f) continue;

                    // 双线性采样修复后像素
                    for (int c = 0; c < 3; c++) {
                        uint8_t p00 = restore_result.restored_rgb[(ay * 512 + ax) * 3 + c];
                        uint8_t p01 = restore_result.restored_rgb[(ay * 512 + std::min(511, ax + 1)) * 3 + c];
                        uint8_t p10 = restore_result.restored_rgb[(std::min(511, ay + 1) * 512 + ax) * 3 + c];
                        uint8_t p11 = restore_result.restored_rgb[(std::min(511, ay + 1) * 512 + std::min(511, ax + 1)) * 3 + c];
                        float restored_val = (p00 * (1 - fx) * (1 - fy) +
                                              p01 * fx * (1 - fy) +
                                              p10 * (1 - fx) * fy +
                                              p11 * fx * fy);

                        int px = crop_x1 + x;
                        int py = crop_y1 + y;
                        if (px >= 0 && px < (int)image->width && py >= 0 && py < (int)image->height) {
                            size_t idx = (py * image->width + px) * channels + c;
                            float orig_val = out_data[idx];
                            out_data[idx] = (uint8_t)(orig_val * (1.0f - mask_val) + restored_val * mask_val);
                        }
                    }
                }
            }
        }

        sd_image_t* result_img = acquire_image();
        if (!result_img) { free(out_data); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result_img->width = image->width;
        result_img->height = image->height;
        result_img->channel = channels;
        result_img->data = out_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        printf("[FaceRestoreWithModel] Restored %zu faces\n", detect_result.faces.size());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceRestoreWithModel", FaceRestoreWithModelNode);

// ============================================================================
// FaceSwapModelLoader - 加载人脸换脸 ONNX 模型
// ============================================================================
class FaceSwapModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "FaceSwapModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"inswapper_path", "STRING", true, std::string("")},
            {"arcface_path", "STRING", true, std::string("")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_SWAP_MODEL", "FACE_SWAP_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string inswapper_path = std::any_cast<std::string>(inputs.at("inswapper_path"));
        std::string arcface_path = std::any_cast<std::string>(inputs.at("arcface_path"));

        if (inswapper_path.empty() || arcface_path.empty()) {
            fprintf(stderr, "[ERROR] FaceSwapModelLoader: Both inswapper_path and arcface_path are required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto swapper = std::make_shared<face::FaceSwapper>();
        if (!swapper->load(inswapper_path, arcface_path)) {
            fprintf(stderr, "[ERROR] FaceSwapModelLoader: Failed to load models\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["FACE_SWAP_MODEL"] = swapper;
        printf("[FaceSwapModelLoader] Loaded swap models\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceSwapModelLoader", FaceSwapModelLoaderNode);

// ============================================================================
// FaceSwap - 人脸换脸
// ============================================================================
class FaceSwapNode : public Node {
public:
    std::string get_class_type() const override { return "FaceSwap"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"target_image", "IMAGE", true, nullptr},
            {"source_image", "IMAGE", true, nullptr},
            {"face_swap_model", "FACE_SWAP_MODEL", true, nullptr},
            {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr target_image = std::any_cast<ImagePtr>(inputs.at("target_image"));
        ImagePtr source_image = std::any_cast<ImagePtr>(inputs.at("source_image"));
        auto swapper = std::any_cast<std::shared_ptr<face::FaceSwapper>>(inputs.at("face_swap_model"));

        if (!target_image || !target_image->data || !source_image || !source_image->data || !swapper) {
            fprintf(stderr, "[ERROR] FaceSwap: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int target_channels = (int)target_image->channel;
        int source_channels = (int)source_image->channel;
        if ((target_channels != 3 && target_channels != 4) || (source_channels != 3 && source_channels != 4)) {
            fprintf(stderr, "[ERROR] FaceSwap: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        // 获取人脸检测器
        std::shared_ptr<face::FaceDetector> detector;
        if (inputs.count("face_detect_model")) {
            detector = std::any_cast<std::shared_ptr<face::FaceDetector>>(inputs.at("face_detect_model"));
        }

        // 转换图像为 RGB
        std::vector<uint8_t> target_rgb(target_image->width * target_image->height * 3);
        std::vector<uint8_t> source_rgb(source_image->width * source_image->height * 3);

        auto convert_to_rgb = [](const uint8_t* src, uint8_t* dst, int channels, size_t pixels) {
            if (channels == 4) {
                for (size_t i = 0; i < pixels; i++) {
                    dst[i * 3 + 0] = src[i * 4 + 0];
                    dst[i * 3 + 1] = src[i * 4 + 1];
                    dst[i * 3 + 2] = src[i * 4 + 2];
                }
            } else {
                memcpy(dst, src, pixels * 3);
            }
        };

        convert_to_rgb(target_image->data, target_rgb.data(), target_channels, target_image->width * target_image->height);
        convert_to_rgb(source_image->data, source_rgb.data(), source_channels, source_image->width * source_image->height);

        // 分配输出图像（复制目标图）
        uint8_t* out_data = (uint8_t*)malloc(target_image->width * target_image->height * target_channels);
        if (!out_data) return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data, target_image->data, target_image->width * target_image->height * target_channels);

        // 检测人脸
        face::FaceDetectResult target_detect, source_detect;
        if (detector) {
            target_detect = detector->detect(target_rgb.data(), (int)target_image->width, (int)target_image->height, 0.5f);
            source_detect = detector->detect(source_rgb.data(), (int)source_image->width, (int)source_image->height, 0.5f);
        }

        if (target_detect.faces.empty() || source_detect.faces.empty()) {
            printf("[FaceSwap] No faces detected in target or source, returning original target image\n");
            sd_image_t* result_img = acquire_image();
            if (!result_img) { free(out_data); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
            result_img->width = target_image->width;
            result_img->height = target_image->height;
            result_img->channel = target_channels;
            result_img->data = out_data;
            outputs["IMAGE"] = make_image_ptr(result_img);
            return sd_error_t::OK;
        }

        // 只处理第一张人脸
        const auto& target_face = target_detect.faces[0];
        const auto& source_face = source_detect.faces[0];

        // 裁剪并对齐到 128x128
        auto align_face = [](const std::vector<uint8_t>& rgb, int width, int height, const face::FaceBBox& face) {
            float cx = (face.x1 + face.x2) * 0.5f;
            float cy = (face.y1 + face.y2) * 0.5f;
            float size = std::max(face.x2 - face.x1, face.y2 - face.y1) * 1.5f;
            int crop_x1 = (int)std::max(0.0f, cx - size * 0.5f);
            int crop_y1 = (int)std::max(0.0f, cy - size * 0.5f);
            int crop_x2 = (int)std::min((float)width, cx + size * 0.5f);
            int crop_y2 = (int)std::min((float)height, cy + size * 0.5f);
            int crop_w = crop_x2 - crop_x1;
            int crop_h = crop_y2 - crop_y1;

            std::vector<uint8_t> cropped(crop_w * crop_h * 3);
            for (int y = 0; y < crop_h; y++) {
                for (int x = 0; x < crop_w; x++) {
                    int src_idx = ((crop_y1 + y) * width + (crop_x1 + x)) * 3;
                    int dst_idx = (y * crop_w + x) * 3;
                    cropped[dst_idx + 0] = rgb[src_idx + 0];
                    cropped[dst_idx + 1] = rgb[src_idx + 1];
                    cropped[dst_idx + 2] = rgb[src_idx + 2];
                }
            }

            float M[6], inv_M[6];
            float template_pts[10];
            // inswapper 使用 128x128，但 landmark 对齐模板通常还是基于 512 模板缩放
            face::get_standard_face_template_512(template_pts);
            // 缩放到 128x128
            for (int i = 0; i < 5; i++) {
                template_pts[i * 2 + 0] *= 128.0f / 512.0f;
                template_pts[i * 2 + 1] *= 128.0f / 512.0f;
            }

            face::estimate_affine_transform_2d3(face.landmarks, template_pts, M);
            face::invert_affine_transform(M, inv_M);

            return face::crop_face(cropped.data(), crop_w, crop_h, 3, face.landmarks, 128);
        };

        std::vector<uint8_t> target_aligned = align_face(target_rgb, (int)target_image->width, (int)target_image->height, target_face);
        std::vector<uint8_t> source_aligned = align_face(source_rgb, (int)source_image->width, (int)source_image->height, source_face);

        // 执行换脸
        auto swap_result = swapper->swap(target_aligned.data(), source_aligned.data());
        if (!swap_result.success) {
            fprintf(stderr, "[ERROR] FaceSwap: Swap failed\n");
            sd_image_t* result_img = acquire_image();
            if (!result_img) { free(out_data); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
            result_img->width = target_image->width;
            result_img->height = target_image->height;
            result_img->channel = target_channels;
            result_img->data = out_data;
            outputs["IMAGE"] = make_image_ptr(result_img);
            return sd_error_t::OK;
        }

        // 贴回目标图（逆变换）
        float M[6], inv_M[6];
        float template_pts[10];
        face::get_standard_face_template_512(template_pts);
        for (int i = 0; i < 5; i++) {
            template_pts[i * 2 + 0] *= 128.0f / 512.0f;
            template_pts[i * 2 + 1] *= 128.0f / 512.0f;
        }
        face::estimate_affine_transform_2d3(target_face.landmarks, template_pts, M);
        face::invert_affine_transform(M, inv_M);

        float cx = (target_face.x1 + target_face.x2) * 0.5f;
        float cy = (target_face.y1 + target_face.y2) * 0.5f;
        float size = std::max(target_face.x2 - target_face.x1, target_face.y2 - target_face.y1) * 1.5f;
        int crop_x1 = (int)std::max(0.0f, cx - size * 0.5f);
        int crop_y1 = (int)std::max(0.0f, cy - size * 0.5f);
        int crop_x2 = (int)std::min((float)target_image->width, cx + size * 0.5f);
        int crop_y2 = (int)std::min((float)target_image->height, cy + size * 0.5f);
        int crop_w = crop_x2 - crop_x1;
        int crop_h = crop_y2 - crop_y1;

        std::vector<uint8_t> mask(128 * 128);
        face::generate_feather_mask(mask.data(), 128, 16);

        for (int y = 0; y < crop_h; y++) {
            for (int x = 0; x < crop_w; x++) {
                float crop_x = (float)x;
                float crop_y = (float)y;
                float aligned_x = inv_M[0] * crop_x + inv_M[1] * crop_y + inv_M[2];
                float aligned_y = inv_M[3] * crop_x + inv_M[4] * crop_y + inv_M[5];

                if (aligned_x < 0 || aligned_x >= 127 || aligned_y < 0 || aligned_y >= 127) {
                    continue;
                }

                int ax = (int)aligned_x;
                int ay = (int)aligned_y;
                float fx = aligned_x - ax;
                float fy = aligned_y - ay;

                uint8_t m00 = mask[ay * 128 + ax];
                uint8_t m01 = mask[ay * 128 + std::min(127, ax + 1)];
                uint8_t m10 = mask[std::min(127, ay + 1) * 128 + ax];
                uint8_t m11 = mask[std::min(127, ay + 1) * 128 + std::min(127, ax + 1)];
                float mask_val = (m00 * (1 - fx) * (1 - fy) +
                                  m01 * fx * (1 - fy) +
                                  m10 * (1 - fx) * fy +
                                  m11 * fx * fy) / 255.0f;

                if (mask_val <= 0.01f) continue;

                for (int c = 0; c < 3; c++) {
                    uint8_t p00 = swap_result.swapped_rgb[(ay * 128 + ax) * 3 + c];
                    uint8_t p01 = swap_result.swapped_rgb[(ay * 128 + std::min(127, ax + 1)) * 3 + c];
                    uint8_t p10 = swap_result.swapped_rgb[(std::min(127, ay + 1) * 128 + ax) * 3 + c];
                    uint8_t p11 = swap_result.swapped_rgb[(std::min(127, ay + 1) * 128 + std::min(127, ax + 1)) * 3 + c];
                    float swapped_val = (p00 * (1 - fx) * (1 - fy) +
                                         p01 * fx * (1 - fy) +
                                         p10 * (1 - fx) * fy +
                                         p11 * fx * fy);

                    int px = crop_x1 + x;
                    int py = crop_y1 + y;
                    if (px >= 0 && px < (int)target_image->width && py >= 0 && py < (int)target_image->height) {
                        size_t idx = (py * target_image->width + px) * target_channels + c;
                        float orig_val = out_data[idx];
                        out_data[idx] = (uint8_t)(orig_val * (1.0f - mask_val) + swapped_val * mask_val);
                    }
                }
            }
        }

        sd_image_t* result_img = acquire_image();
        if (!result_img) { free(out_data); return sd_error_t::ERROR_MEMORY_ALLOCATION; }
        result_img->width = target_image->width;
        result_img->height = target_image->height;
        result_img->channel = target_channels;
        result_img->data = out_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        printf("[FaceSwap] Swapped face successfully\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceSwap", FaceSwapNode);

#else // !HAS_ONNXRUNTIME

class FaceDetectModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "FaceDetectModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_DETECT_MODEL", "FACE_DETECT_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs; (void)outputs;
        fprintf(stderr, "[ERROR] FaceDetectModelLoader: ONNX Runtime not available\n");
        return sd_error_t::ERROR_MODEL_LOADING;
    }
};
REGISTER_NODE("FaceDetectModelLoader", FaceDetectModelLoaderNode);

class FaceDetectNode : public Node {
public:
    std::string get_class_type() const override { return "FaceDetect"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"model", "FACE_DETECT_MODEL", true, nullptr},
            {"confidence_threshold", "FLOAT", false, 0.5f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"IMAGE", "IMAGE"},
            {"faces", "FACE_BBOX_LIST"}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs; (void)outputs;
        fprintf(stderr, "[ERROR] FaceDetect: ONNX Runtime not available\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("FaceDetect", FaceDetectNode);

class FaceRestoreModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "FaceRestoreModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model_path", "STRING", true, std::string("")},
            {"model_type", "STRING", false, std::string("gfpgan")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_RESTORE_MODEL", "FACE_RESTORE_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs; (void)outputs;
        fprintf(stderr, "[ERROR] FaceRestoreModelLoader: ONNX Runtime not available\n");
        return sd_error_t::ERROR_MODEL_LOADING;
    }
};
REGISTER_NODE("FaceRestoreModelLoader", FaceRestoreModelLoaderNode);

class FaceRestoreWithModelNode : public Node {
public:
    std::string get_class_type() const override { return "FaceRestoreWithModel"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"face_restore_model", "FACE_RESTORE_MODEL", true, nullptr},
            {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr},
            {"codeformer_fidelity", "FLOAT", false, 0.5f}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs; (void)outputs;
        fprintf(stderr, "[ERROR] FaceRestoreWithModel: ONNX Runtime not available\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("FaceRestoreWithModel", FaceRestoreWithModelNode);

class FaceSwapModelLoaderNode : public Node {
public:
    std::string get_class_type() const override { return "FaceSwapModelLoader"; }
    std::string get_category() const override { return "loaders"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"inswapper_path", "STRING", true, std::string("")},
            {"arcface_path", "STRING", true, std::string("")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_SWAP_MODEL", "FACE_SWAP_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs; (void)outputs;
        fprintf(stderr, "[ERROR] FaceSwapModelLoader: ONNX Runtime not available\n");
        return sd_error_t::ERROR_MODEL_LOADING;
    }
};
REGISTER_NODE("FaceSwapModelLoader", FaceSwapModelLoaderNode);

class FaceSwapNode : public Node {
public:
    std::string get_class_type() const override { return "FaceSwap"; }
    std::string get_category() const override { return "image"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"target_image", "IMAGE", true, nullptr},
            {"source_image", "IMAGE", true, nullptr},
            {"face_swap_model", "FACE_SWAP_MODEL", true, nullptr},
            {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)inputs; (void)outputs;
        fprintf(stderr, "[ERROR] FaceSwap: ONNX Runtime not available\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("FaceSwap", FaceSwapNode);

#endif // HAS_ONNXRUNTIME

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
            {"positive_text", "STRING", false, std::string("")},
            {"negative_text", "STRING", false, std::string("")},
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

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_ctx_t* sd_ctx = nullptr;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("model"));
            sd_ctx = ctx_ptr.get();
        } catch (const std::bad_any_cast&) {
            sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("model"));
        }
        ConditioningPtr positive = std::any_cast<ConditioningPtr>(inputs.at("positive"));
        ConditioningPtr negative = inputs.count("negative") ?
            std::any_cast<ConditioningPtr>(inputs.at("negative")) : nullptr;
        
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
            return sd_error_t::ERROR_INVALID_INPUT;
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
        
        // 从字符串输入获取 prompt
        std::string positive_text = inputs.count("positive_text") ?
            std::any_cast<std::string>(inputs.at("positive_text")) : "";
        std::string negative_text = inputs.count("negative_text") ?
            std::any_cast<std::string>(inputs.at("negative_text")) : "";
        gen_params.prompt = positive_text.c_str();
        gen_params.negative_prompt = negative_text.c_str();
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
            ImagePtr init_img = std::any_cast<ImagePtr>(inputs.at("init_image"));
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
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        printf("[DeepHighResFix] Generation completed: %dx%d\n", result->width, result->height);
        outputs["IMAGE"] = make_image_ptr(result);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("DeepHighResFix", DeepHighResFixNode);

// ============================================================================
// UnloadModel - 释放模型上下文
// ============================================================================
class UnloadModelNode : public Node {
public:
    std::string get_class_type() const override { return "UnloadModel"; }
    std::string get_category() const override { return "model_management"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("model"));
            if (ctx_ptr) {
                printf("[UnloadModel] Releasing model context (ref_count=%ld)\n", ctx_ptr.use_count());
            }
        } catch (const std::bad_any_cast&) {
            sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("model"));
            if (sd_ctx) {
                printf("[UnloadModel] Releasing raw model context\n");
                free_sd_ctx(sd_ctx);
            }
        }
        return sd_error_t::OK;
    }
};
REGISTER_NODE("UnloadModel", UnloadModelNode);

// ============================================================================
// 显式初始化函数
// ============================================================================
void init_core_nodes() {
    // 确保本文件被链接进最终可执行文件
}

} // namespace sdengine
