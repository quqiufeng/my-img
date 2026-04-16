// ============================================================================
// sd-engine/nodes/onnx_loader_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

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
        std::string path;
        if (sd_error_t err = get_input(inputs, "model_path", path); is_error(err)) {
            return err;
        }
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
        std::string path;
        if (sd_error_t err = get_input(inputs, "model_name", path); is_error(err)) {
            return err;
        }
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
        std::string path;
        if (sd_error_t err = get_input(inputs, "model_path", path); is_error(err)) {
            return err;
        }
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
        std::string path;
        if (sd_error_t err = get_input(inputs, "model_path", path); is_error(err)) {
            return err;
        }
        std::string type_str = get_input_opt<std::string>(inputs, "model_type", "gfpgan");

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
        std::string inswapper_path;
        if (sd_error_t err = get_input(inputs, "inswapper_path", inswapper_path); is_error(err)) {
            return err;
        }
        std::string arcface_path;
        if (sd_error_t err = get_input(inputs, "arcface_path", arcface_path); is_error(err)) {
            return err;
        }

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

void init_onnx_loader_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
