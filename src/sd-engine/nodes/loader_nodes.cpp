// ============================================================================
// sd-engine/nodes/loader_nodes.cpp
// ============================================================================
// 加载器类节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// CheckpointLoaderSimple - 加载模型
// ============================================================================
class CheckpointLoaderSimpleNode : public Node {
  public:
    std::string get_class_type() const override {
        return "CheckpointLoaderSimple";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"ckpt_name", "STRING", true, std::string("")},
                {"vae_name", "STRING", false, std::string("")},
                {"clip_name", "STRING", false, std::string("")},
                {"control_net_path", "STRING", false, std::string("")},
                {"n_threads", "INT", false, 4},
                {"use_gpu", "BOOLEAN", false, true},
                {"flash_attn", "BOOLEAN", false, false}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"MODEL", "MODEL"}, {"CLIP", "CLIP"}, {"VAE", "VAE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string ckpt_path = std::any_cast<std::string>(inputs.at("ckpt_name"));
        std::string vae_path = inputs.count("vae_name") ? std::any_cast<std::string>(inputs.at("vae_name")) : "";
        std::string clip_path = inputs.count("clip_name") ? std::any_cast<std::string>(inputs.at("clip_name")) : "";
        std::string control_net_path =
            inputs.count("control_net_path") ? std::any_cast<std::string>(inputs.at("control_net_path")) : "";
        int n_threads = inputs.count("n_threads") ? std::any_cast<int>(inputs.at("n_threads")) : 4;
        bool use_gpu = inputs.count("use_gpu") ? std::any_cast<bool>(inputs.at("use_gpu")) : true;
        bool flash_attn = inputs.count("flash_attn") ? std::any_cast<bool>(inputs.at("flash_attn")) : false;

        if (ckpt_path.empty()) {
            LOG_ERROR("[ERROR] CheckpointLoaderSimple: ckpt_name is required\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        LOG_INFO("[CheckpointLoaderSimple] Loading model: %s\n", ckpt_path.c_str());

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
            LOG_INFO("[CheckpointLoaderSimple] Loading ControlNet: %s\n", control_net_path.c_str());
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
            LOG_ERROR("[ERROR] Failed to load checkpoint\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        LOG_INFO("[CheckpointLoaderSimple] Model loaded successfully\n");

        auto sd_ctx_ptr = make_sd_context_ptr(sd_ctx);
        outputs["MODEL"] = sd_ctx_ptr;
        outputs["CLIP"] = sd_ctx_ptr;
        outputs["VAE"] = sd_ctx_ptr;

        return sd_error_t::OK;
    }
};
REGISTER_NODE("CheckpointLoaderSimple", CheckpointLoaderSimpleNode);

// ============================================================================
// LoRALoader - 加载 LoRA
// ============================================================================
class LoRALoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LoRALoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"lora_name", "STRING", true, std::string("")},
                {"strength_model", "FLOAT", false, 1.0f},
                {"strength_clip", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LORA", "LORA"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string lora_path = std::any_cast<std::string>(inputs.at("lora_name"));
        float strength_model =
            inputs.count("strength_model") ? std::any_cast<float>(inputs.at("strength_model")) : 1.0f;
        float strength_clip = inputs.count("strength_clip") ? std::any_cast<float>(inputs.at("strength_clip")) : 1.0f;

        if (lora_path.empty()) {
            LOG_ERROR("[ERROR] LoRALoader: lora_name is required\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        float avg_strength = (strength_model + strength_clip) * 0.5f;
        outputs["LORA"] = LoRAInfo{lora_path, avg_strength};

        LOG_INFO("[LoRALoader] Loaded LoRA: %s (strength=%.2f)\n", lora_path.c_str(), avg_strength);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoRALoader", LoRALoaderNode);

// ============================================================================
// LoRAStack - 多 LoRA 堆叠
// ============================================================================
class LoRAStackNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LoRAStack";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"lora_1", "LORA", false, nullptr},
                {"lora_2", "LORA", false, nullptr},
                {"lora_3", "LORA", false, nullptr},
                {"lora_4", "LORA", false, nullptr}};
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
                LOG_INFO("[LoRAStack] Stacked LoRA %d: %s (strength=%.2f)\n", i, info.path.c_str(), info.strength);
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
class ControlNetLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ControlNetLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"control_net_name", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONTROL_NET", "CONTROL_NET"}, {"path", "STRING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("control_net_name"));
        if (path.empty()) {
            LOG_ERROR("[ERROR] ControlNetLoader: control_net_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        outputs["CONTROL_NET"] = path;
        outputs["path"] = path;
        LOG_INFO("[ControlNetLoader] ControlNet path: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ControlNetLoader", ControlNetLoaderNode);

// ============================================================================
// UpscaleModelLoader - 加载 ESRGAN 放大模型
// ============================================================================
class UpscaleModelLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "UpscaleModelLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_name", "STRING", true, std::string("")},
                {"use_gpu", "BOOLEAN", false, true},
                {"tile_size", "INT", false, 512}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"UPSCALE_MODEL", "UPSCALE_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string model_path = std::any_cast<std::string>(inputs.at("model_name"));
        bool use_gpu = inputs.count("use_gpu") ? std::any_cast<bool>(inputs.at("use_gpu")) : true;
        int tile_size = inputs.count("tile_size") ? std::any_cast<int>(inputs.at("tile_size")) : 512;

        if (model_path.empty()) {
            LOG_ERROR("[ERROR] UpscaleModelLoader: model_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[UpscaleModelLoader] Loading model: %s\n", model_path.c_str());

        upscaler_ctx_t* upscaler = new_upscaler_ctx(model_path.c_str(),
                                                    !use_gpu, // w_mode
                                                    false,    // no longer used
                                                    4,        // threads
                                                    tile_size);

        if (!upscaler) {
            LOG_ERROR("[ERROR] UpscaleModelLoader: Failed to load model\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        int scale = get_upscale_factor(upscaler);
        LOG_INFO("[UpscaleModelLoader] Model loaded, scale=%dx, tile_size=%d\n", scale, tile_size);

        outputs["UPSCALE_MODEL"] = make_upscaler_ptr(upscaler);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("UpscaleModelLoader", UpscaleModelLoaderNode);

// ============================================================================
// IPAdapterLoader - 加载 IPAdapter 模型
// ============================================================================
class IPAdapterLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "IPAdapterLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"ipadapter_file", "STRING", true, std::string("")},
                {"cross_attention_dim", "INT", false, 768},
                {"num_tokens", "INT", false, 4},
                {"clip_embeddings_dim", "INT", false, 1024}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IPADAPTER", "IPADAPTER"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("ipadapter_file"));
        int cross_attention_dim =
            inputs.count("cross_attention_dim") ? std::any_cast<int>(inputs.at("cross_attention_dim")) : 768;
        int num_tokens = inputs.count("num_tokens") ? std::any_cast<int>(inputs.at("num_tokens")) : 4;
        int clip_embeddings_dim =
            inputs.count("clip_embeddings_dim") ? std::any_cast<int>(inputs.at("clip_embeddings_dim")) : 1024;

        if (path.empty()) {
            LOG_ERROR("[ERROR] IPAdapterLoader: ipadapter_file is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        outputs["IPADAPTER"] = IPAdapterInfo{path, cross_attention_dim, num_tokens, clip_embeddings_dim, 1.0f};

        LOG_INFO("[IPAdapterLoader] Loaded IPAdapter: %s (dim=%d, tokens=%d, clip_dim=%d)\n", path.c_str(),
                 cross_attention_dim, num_tokens, clip_embeddings_dim);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("IPAdapterLoader", IPAdapterLoaderNode);

#ifdef HAS_ONNXRUNTIME

// ============================================================================
// RemBGModelLoader - 加载背景抠图 ONNX 模型
// ============================================================================
class RemBGModelLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "RemBGModelLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"REMBG_MODEL", "REMBG_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_path"));
        if (path.empty()) {
            LOG_ERROR("[ERROR] RemBGModelLoader: model_path is required\n");
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
            LOG_ERROR("[ERROR] RemBGModelLoader: Failed to load ONNX model: %s\n", e.what());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["REMBG_MODEL"] = model;
        LOG_INFO("[RemBGModelLoader] Loaded model: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("RemBGModelLoader", RemBGModelLoaderNode);

// ============================================================================
// LineArtLoader - 加载 LineArt ONNX 模型
// ============================================================================
class LineArtLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LineArtLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_name", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LINEART_MODEL", "LINEART_MODEL"}, {"path", "STRING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_name"));
        if (path.empty()) {
            LOG_ERROR("[ERROR] LineArtLoader: model_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto preprocessor = std::make_shared<LineArtPreprocessor>();
        if (!preprocessor->load(path)) {
            LOG_ERROR("[ERROR] LineArtLoader: Failed to load model: %s\n", path.c_str());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["LINEART_MODEL"] = preprocessor;
        outputs["path"] = path;
        LOG_INFO("[LineArtLoader] LineArt model loaded: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LineArtLoader", LineArtLoaderNode);

// ============================================================================
// FaceDetectModelLoader - 加载人脸检测 ONNX 模型
// ============================================================================
class FaceDetectModelLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "FaceDetectModelLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_DETECT_MODEL", "FACE_DETECT_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_path"));
        if (path.empty()) {
            LOG_ERROR("[ERROR] FaceDetectModelLoader: model_path is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto detector = std::make_shared<face::FaceDetector>();
        if (!detector->load(path)) {
            LOG_ERROR("[ERROR] FaceDetectModelLoader: Failed to load %s\n", path.c_str());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["FACE_DETECT_MODEL"] = detector;
        LOG_INFO("[FaceDetectModelLoader] Loaded: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceDetectModelLoader", FaceDetectModelLoaderNode);

// ============================================================================
// FaceRestoreModelLoader - 加载人脸修复 ONNX 模型
// ============================================================================
class FaceRestoreModelLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "FaceRestoreModelLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model_path", "STRING", true, std::string("")},
                {"model_type", "STRING", false, std::string("gfpgan")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_RESTORE_MODEL", "FACE_RESTORE_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path = std::any_cast<std::string>(inputs.at("model_path"));
        std::string type_str =
            inputs.count("model_type") ? std::any_cast<std::string>(inputs.at("model_type")) : "gfpgan";

        if (path.empty()) {
            LOG_ERROR("[ERROR] FaceRestoreModelLoader: model_path is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto type = (type_str == "codeformer") ? face::RestoreModelType::CODEFORMER : face::RestoreModelType::GFPGAN;

        auto restorer = std::make_shared<face::FaceRestorer>();
        if (!restorer->load(path, type)) {
            LOG_ERROR("[ERROR] FaceRestoreModelLoader: Failed to load %s\n", path.c_str());
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["FACE_RESTORE_MODEL"] = restorer;
        LOG_INFO("[FaceRestoreModelLoader] Loaded: %s (type=%s)\n", path.c_str(), type_str.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceRestoreModelLoader", FaceRestoreModelLoaderNode);

// ============================================================================
// FaceSwapModelLoader - 加载人脸换脸 ONNX 模型
// ============================================================================
class FaceSwapModelLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "FaceSwapModelLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"inswapper_path", "STRING", true, std::string("")}, {"arcface_path", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"FACE_SWAP_MODEL", "FACE_SWAP_MODEL"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string inswapper_path = std::any_cast<std::string>(inputs.at("inswapper_path"));
        std::string arcface_path = std::any_cast<std::string>(inputs.at("arcface_path"));

        if (inswapper_path.empty() || arcface_path.empty()) {
            LOG_ERROR("[ERROR] FaceSwapModelLoader: Both inswapper_path and arcface_path are required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto swapper = std::make_shared<face::FaceSwapper>();
        if (!swapper->load(inswapper_path, arcface_path)) {
            LOG_ERROR("[ERROR] FaceSwapModelLoader: Failed to load models\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        outputs["FACE_SWAP_MODEL"] = swapper;
        LOG_INFO("[FaceSwapModelLoader] Loaded swap models\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceSwapModelLoader", FaceSwapModelLoaderNode);

#else // !HAS_ONNXRUNTIME

static std::vector<PortDef> rembg_loader_inputs() {
    return {{"model_path", "STRING", true, std::string("")}};
}
static std::vector<PortDef> rembg_loader_outputs() {
    return {{"REMBG_MODEL", "REMBG_MODEL"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(RemBGModelLoaderNode, "RemBGModelLoader", "loaders", rembg_loader_inputs,
                             rembg_loader_outputs, sd_error_t::ERROR_MODEL_LOADING)
REGISTER_NODE("RemBGModelLoader", RemBGModelLoaderNode);

static std::vector<PortDef> lineart_loader_inputs() {
    return {{"model_name", "STRING", true, std::string("")}};
}
static std::vector<PortDef> lineart_loader_outputs() {
    return {{"LINEART_MODEL", "LINEART_MODEL"}, {"path", "STRING"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(LineArtLoaderNode, "LineArtLoader", "loaders", lineart_loader_inputs,
                             lineart_loader_outputs, sd_error_t::ERROR_MODEL_LOADING)
REGISTER_NODE("LineArtLoader", LineArtLoaderNode);

static std::vector<PortDef> face_detect_loader_inputs() {
    return {{"model_path", "STRING", true, std::string("")}};
}
static std::vector<PortDef> face_detect_loader_outputs() {
    return {{"FACE_DETECT_MODEL", "FACE_DETECT_MODEL"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(FaceDetectModelLoaderNode, "FaceDetectModelLoader", "loaders", face_detect_loader_inputs,
                             face_detect_loader_outputs, sd_error_t::ERROR_MODEL_LOADING)
REGISTER_NODE("FaceDetectModelLoader", FaceDetectModelLoaderNode);

static std::vector<PortDef> face_restore_loader_inputs() {
    return {{"model_path", "STRING", true, std::string("")}, {"model_type", "STRING", false, std::string("gfpgan")}};
}
static std::vector<PortDef> face_restore_loader_outputs() {
    return {{"FACE_RESTORE_MODEL", "FACE_RESTORE_MODEL"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(FaceRestoreModelLoaderNode, "FaceRestoreModelLoader", "loaders",
                             face_restore_loader_inputs, face_restore_loader_outputs, sd_error_t::ERROR_MODEL_LOADING)
REGISTER_NODE("FaceRestoreModelLoader", FaceRestoreModelLoaderNode);

static std::vector<PortDef> face_swap_loader_inputs() {
    return {{"inswapper_path", "STRING", true, std::string("")}, {"arcface_path", "STRING", true, std::string("")}};
}
static std::vector<PortDef> face_swap_loader_outputs() {
    return {{"FACE_SWAP_MODEL", "FACE_SWAP_MODEL"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(FaceSwapModelLoaderNode, "FaceSwapModelLoader", "loaders", face_swap_loader_inputs,
                             face_swap_loader_outputs, sd_error_t::ERROR_MODEL_LOADING)
REGISTER_NODE("FaceSwapModelLoader", FaceSwapModelLoaderNode);

#endif // HAS_ONNXRUNTIME

// ============================================================================
// UnloadModel - 释放模型上下文
// ============================================================================
class UnloadModelNode : public Node {
  public:
    std::string get_class_type() const override {
        return "UnloadModel";
    }
    std::string get_category() const override {
        return "model_management";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model", "MODEL", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        try {
            auto ctx_ptr = std::any_cast<SDContextPtr>(inputs.at("model"));
            if (ctx_ptr) {
                LOG_INFO("[UnloadModel] Releasing model context (ref_count=%ld)\n", ctx_ptr.use_count());
            }
        } catch (const std::bad_any_cast&) {
            sd_ctx_t* sd_ctx = std::any_cast<sd_ctx_t*>(inputs.at("model"));
            if (sd_ctx) {
                LOG_INFO("[UnloadModel] Releasing raw model context\n");
                free_sd_ctx(sd_ctx);
            }
        }
        return sd_error_t::OK;
    }
};
REGISTER_NODE("UnloadModel", UnloadModelNode);

void init_loader_nodes() {
    // 空函数，仅确保本翻译单元被链接
}

} // namespace sdengine
