// ============================================================================
// sd-engine/nodes/preprocessor_onnx_nodes.cpp
// ============================================================================
// ONNX 图像预处理器节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

#ifdef HAS_ONNXRUNTIME

// ============================================================================
// ImageRemoveBackground - 背景抠图
// ============================================================================
class ImageRemoveBackgroundNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageRemoveBackground";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}, {"model", "REMBG_MODEL", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}, {"MASK", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        auto model = std::any_cast<std::shared_ptr<RemBGModel>>(inputs.at("model"));

        if (!image || !image->data || !model || !model->session) {
            LOG_ERROR("[ERROR] ImageRemoveBackground: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int src_w = (int)image->width;
        int src_h = (int)image->height;
        int src_c = (int)image->channel;

        if (src_c != 3 && src_c != 4) {
            LOG_ERROR("[ERROR] ImageRemoveBackground: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        const int model_size = 1024;

        std::vector<float> input_data(1 * 3 * model_size * model_size);
        {
            std::vector<uint8_t> resized(model_size * model_size * src_c);
            stbir_resize(image->data, src_w, src_h, 0, resized.data(), model_size, model_size, 0, STBIR_TYPE_UINT8,
                         src_c, -1, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE,
                         STBIR_COLORSPACE_LINEAR, nullptr);

            for (int y = 0; y < model_size; y++) {
                for (int x = 0; x < model_size; x++) {
                    int idx = (y * model_size + x) * src_c;
                    input_data[0 * 3 * model_size * model_size + 0 * model_size * model_size + y * model_size + x] =
                        resized[idx + 0] / 255.0f;
                    input_data[0 * 3 * model_size * model_size + 1 * model_size * model_size + y * model_size + x] =
                        resized[idx + 1] / 255.0f;
                    input_data[0 * 3 * model_size * model_size + 2 * model_size * model_size + y * model_size + x] =
                        resized[idx + 2] / 255.0f;
                }
            }
        }

        std::vector<int64_t> input_shape = {1, 3, model_size, model_size};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            model->memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        std::vector<Ort::Value> output_tensors;
        try {
            output_tensors =
                model->session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        } catch (const Ort::Exception& e) {
            LOG_ERROR("[ERROR] ImageRemoveBackground: ONNX inference failed: %s\n", e.what());
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int out_h = (int)output_shape[2];
        int out_w = (int)output_shape[3];

        std::vector<uint8_t> mask_resized(src_w * src_h);
        {
            std::vector<uint8_t> mask_1024(out_w * out_h);
            for (int i = 0; i < out_w * out_h; i++) {
                float v = output_data[i];
                v = 1.0f / (1.0f + std::exp(-v));
                mask_1024[i] = (uint8_t)(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
            }

            stbir_resize(mask_1024.data(), out_w, out_h, 0, mask_resized.data(), src_w, src_h, 0, STBIR_TYPE_UINT8, 1,
                         -1, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE,
                         STBIR_COLORSPACE_LINEAR, nullptr);
        }

        auto rgba_data = make_malloc_buffer(src_w * src_h * 4);
        if (!rgba_data) {
            LOG_ERROR("[ERROR] ImageRemoveBackground: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        for (int i = 0; i < src_w * src_h; i++) {
            rgba_data[i * 4 + 0] = image->data[i * src_c + 0];
            rgba_data[i * 4 + 1] = image->data[i * src_c + 1];
            rgba_data[i * 4 + 2] = image->data[i * src_c + 2];
            rgba_data[i * 4 + 3] = mask_resized[i];
        }

        auto result_img = create_image_ptr(src_w, src_h, 4, std::move(rgba_data));
        if (!result_img) {
            LOG_ERROR("[ERROR] ImageRemoveBackground: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        auto mask_mb = make_malloc_buffer(mask_resized.size());
        if (!mask_mb) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(mask_mb.get(), mask_resized.data(), mask_resized.size());
        auto mask_img = create_image_ptr(src_w, src_h, 1, std::move(mask_mb));
        if (!mask_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = result_img;
        outputs["MASK"] = mask_img;
        LOG_INFO("[ImageRemoveBackground] Removed background: %dx%d -> RGBA + Mask\n", src_w, src_h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageRemoveBackground", ImageRemoveBackgroundNode);

// ============================================================================
// LineArtPreprocessor - LineArt 线稿提取
// ============================================================================
class LineArtPreprocessorNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LineArtPreprocessor";
    }
    std::string get_category() const override {
        return "image/preprocessors";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}, {"lineart_model", "LINEART_MODEL", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src = std::any_cast<ImagePtr>(inputs.at("image"));
        auto preprocessor = std::any_cast<std::shared_ptr<LineArtPreprocessor>>(inputs.at("lineart_model"));

        if (!src || !src->data || src->channel != 3) {
            LOG_ERROR("[ERROR] LineArtPreprocessor: Requires RGB image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        if (!preprocessor) {
            LOG_ERROR("[ERROR] LineArtPreprocessor: Model not loaded\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[LineArtPreprocessor] Processing %dx%d\n", src->width, src->height);

        LineArtResult result = preprocessor->process(src->data, src->width, src->height);
        if (!result.success) {
            LOG_ERROR("[ERROR] LineArtPreprocessor: Processing failed\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto out_img = create_image_ptr(result.width, result.height, 3, std::move(result.data));
        if (!out_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = out_img;
        LOG_INFO("[LineArtPreprocessor] Done\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LineArtPreprocessor", LineArtPreprocessorNode);

#else // !HAS_ONNXRUNTIME

static std::vector<PortDef> rembg_inputs() {
    return {{"image", "IMAGE", true, nullptr}, {"model", "REMBG_MODEL", true, nullptr}};
}
static std::vector<PortDef> rembg_outputs() {
    return {{"IMAGE", "IMAGE"}, {"MASK", "IMAGE"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(ImageRemoveBackgroundNode, "ImageRemoveBackground", "image", rembg_inputs, rembg_outputs,
                             sd_error_t::ERROR_EXECUTION_FAILED)
REGISTER_NODE("ImageRemoveBackground", ImageRemoveBackgroundNode);

static std::vector<PortDef> lineart_inputs() {
    return {{"image", "IMAGE", true, nullptr}, {"lineart_model", "LINEART_MODEL", true, nullptr}};
}
static std::vector<PortDef> lineart_outputs() {
    return {{"IMAGE", "IMAGE"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(LineArtPreprocessorNode, "LineArtPreprocessor", "image/preprocessors", lineart_inputs,
                             lineart_outputs, sd_error_t::ERROR_EXECUTION_FAILED)
REGISTER_NODE("LineArtPreprocessor", LineArtPreprocessorNode);

#endif // HAS_ONNXRUNTIME

void init_preprocessor_onnx_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
