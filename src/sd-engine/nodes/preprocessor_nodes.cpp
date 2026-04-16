// ============================================================================
// sd-engine/nodes/preprocessor_nodes.cpp
// ============================================================================
// 图像预处理器节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// CannyEdgePreprocessor - Canny 边缘检测预处理
// ============================================================================
class CannyEdgePreprocessorNode : public Node {
  public:
    std::string get_class_type() const override {
        return "CannyEdgePreprocessor";
    }
    std::string get_category() const override {
        return "image/preprocessors";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr},
                {"low_threshold", "INT", false, 100},
                {"high_threshold", "INT", false, 200}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src = std::any_cast<ImagePtr>(inputs.at("image"));
        int low_threshold = inputs.count("low_threshold") ? std::any_cast<int>(inputs.at("low_threshold")) : 100;
        int high_threshold = inputs.count("high_threshold") ? std::any_cast<int>(inputs.at("high_threshold")) : 200;

        if (!src || !src->data || src->channel != 3) {
            LOG_ERROR("[ERROR] CannyEdgePreprocessor: Requires RGB image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[CannyEdgePreprocessor] Processing %dx%d (low=%d, high=%d)\n", src->width, src->height, low_threshold,
                 high_threshold);

        int w = src->width;
        int h = src->height;
        size_t pixel_count = w * h;

        std::vector<uint8_t> gray(pixel_count);
        std::vector<uint8_t> edges(pixel_count, 0);
        std::vector<uint8_t> dst_data(pixel_count * 3);

        for (size_t i = 0; i < pixel_count; i++) {
            uint8_t r = src->data[i * 3 + 0];
            uint8_t g = src->data[i * 3 + 1];
            uint8_t b = src->data[i * 3 + 2];
            gray[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
        }

        int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

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

        for (size_t i = 0; i < pixel_count; i++) {
            uint8_t val = edges[i] >= 128 ? 255 : 0;
            dst_data[i * 3 + 0] = val;
            dst_data[i * 3 + 1] = val;
            dst_data[i * 3 + 2] = val;
        }

        uint8_t* final_data = (uint8_t*)malloc(dst_data.size());
        if (!final_data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_data, dst_data.data(), dst_data.size());

        sd_image_t dst_image = {};
        dst_image.width = w;
        dst_image.height = h;
        dst_image.channel = 3;
        dst_image.data = final_data;

        sd_image_t* result = acquire_image();
        if (!result) {
            free(final_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result = dst_image;

        outputs["IMAGE"] = make_image_ptr(result);
        LOG_INFO("[CannyEdgePreprocessor] Done\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CannyEdgePreprocessor", CannyEdgePreprocessorNode);

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

        std::vector<uint8_t> rgba_data(src_w * src_h * 4);
        for (int i = 0; i < src_w * src_h; i++) {
            rgba_data[i * 4 + 0] = image->data[i * src_c + 0];
            rgba_data[i * 4 + 1] = image->data[i * src_c + 1];
            rgba_data[i * 4 + 2] = image->data[i * src_c + 2];
            rgba_data[i * 4 + 3] = mask_resized[i];
        }

        uint8_t* final_rgba = (uint8_t*)malloc(rgba_data.size());
        if (!final_rgba) {
            LOG_ERROR("[ERROR] ImageRemoveBackground: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_rgba, rgba_data.data(), rgba_data.size());

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            free(final_rgba);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = src_w;
        result_img->height = src_h;
        result_img->channel = 4;
        result_img->data = final_rgba;

        uint8_t* final_mask = (uint8_t*)malloc(src_w * src_h);
        if (!final_mask) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_mask, mask_resized.data(), src_w * src_h);

        sd_image_t* mask_img = acquire_image();
        if (!mask_img) {
            free(final_mask);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        mask_img->width = src_w;
        mask_img->height = src_h;
        mask_img->channel = 1;
        mask_img->data = final_mask;

        outputs["IMAGE"] = make_image_ptr(result_img);
        outputs["MASK"] = make_image_ptr(mask_img);
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

        uint8_t* final_data = (uint8_t*)malloc(result.data.size());
        if (!final_data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_data, result.data.data(), result.data.size());

        sd_image_t dst_image = {};
        dst_image.width = result.width;
        dst_image.height = result.height;
        dst_image.channel = 3;
        dst_image.data = final_data;

        sd_image_t* image_result = acquire_image();
        if (!image_result) {
            free(final_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *image_result = dst_image;

        outputs["IMAGE"] = make_image_ptr(image_result);
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

void init_preprocessor_nodes() {
    // 空函数，仅确保本翻译单元被链接
}

} // namespace sdengine
