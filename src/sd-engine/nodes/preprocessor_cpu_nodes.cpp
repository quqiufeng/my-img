// ============================================================================
// sd-engine/nodes/preprocessor_cpu_nodes.cpp
// ============================================================================
// CPU 图像预处理器节点实现
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
        ImagePtr src;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", src));
        int low_threshold = get_input_opt<int>(inputs, "low_threshold", 100);
        int high_threshold = get_input_opt<int>(inputs, "high_threshold", 200);

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
        auto dst_data = make_malloc_buffer(pixel_count * 3);
        if (!dst_data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

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

        auto out_img = create_image_ptr(w, h, 3, std::move(dst_data));
        if (!out_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = out_img;
        LOG_INFO("[CannyEdgePreprocessor] Done\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("CannyEdgePreprocessor", CannyEdgePreprocessorNode);

// ============================================================================
// MiDaS-DepthMapPreprocessor - MiDaS 深度图预处理（占位符，需 ONNX 模型支持）
// ============================================================================
class MiDaSDepthMapPreprocessorNode : public Node {
  public:
    std::string get_class_type() const override {
        return "MiDaS-DepthMapPreprocessor";
    }
    std::string get_category() const override {
        return "image/preprocessors";
    }

    bool is_placeholder() const override {
        return true;
    }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr}, {"a", "FLOAT", false, 6.28f}, {"bg_threshold", "FLOAT", false, 0.1f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", src));
        float a = get_input_opt<float>(inputs, "a", 6.28f);
        float bg_threshold = get_input_opt<float>(inputs, "bg_threshold", 0.1f);
        (void)a;
        (void)bg_threshold;

        if (!src || !src->data || src->channel != 3) {
            LOG_ERROR("[ERROR] MiDaS-DepthMapPreprocessor: Requires RGB image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_ERROR("[ERROR] MiDaS-DepthMapPreprocessor: Real MiDaS depth estimation requires ONNX model. "
                  "Placeholder is disabled. Please build with HAS_ONNXRUNTIME and provide a MiDaS ONNX model.\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("MiDaS-DepthMapPreprocessor", MiDaSDepthMapPreprocessorNode);

// ============================================================================
// OpenPosePreprocessor - OpenPose 姿态预处理（占位符，需 ONNX 模型支持）
// ============================================================================
class OpenPosePreprocessorNode : public Node {
  public:
    std::string get_class_type() const override {
        return "OpenPosePreprocessor";
    }
    std::string get_category() const override {
        return "image/preprocessors";
    }

    bool is_placeholder() const override {
        return true;
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", src));

        if (!src || !src->data || src->channel != 3) {
            LOG_ERROR("[ERROR] OpenPosePreprocessor: Requires RGB image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_ERROR("[ERROR] OpenPosePreprocessor: Real OpenPose pose detection requires ONNX model. "
                  "Placeholder is disabled. Please build with HAS_ONNXRUNTIME and provide an OpenPose ONNX model.\n");
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }
};
REGISTER_NODE("OpenPosePreprocessor", OpenPosePreprocessorNode);

void init_preprocessor_cpu_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
