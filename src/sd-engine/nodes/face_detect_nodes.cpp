// ============================================================================
// sd-engine/nodes/face_detect_nodes.cpp
// ============================================================================
// 人脸检测节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

#ifdef HAS_ONNXRUNTIME

// ============================================================================
// FaceDetect - 人脸检测
// ============================================================================
class FaceDetectNode : public Node {
  public:
    std::string get_class_type() const override {
        return "FaceDetect";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr},
                {"model", "FACE_DETECT_MODEL", true, nullptr},
                {"confidence_threshold", "FLOAT", false, 0.5f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}, {"faces", "FACE_BBOX_LIST"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image;
        if (sd_error_t err = get_input(inputs, "image", image); is_error(err)) {
            return err;
        }
        std::shared_ptr<face::FaceDetector> detector;
        if (sd_error_t err = get_input(inputs, "model", detector); is_error(err)) {
            return err;
        }
        float threshold =
            get_input_opt<float>(inputs, "confidence_threshold", 0.5f);

        if (!image || !image->data || !detector) {
            LOG_ERROR("[ERROR] FaceDetect: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int channels = (int)image->channel;
        if (channels != 3 && channels != 4) {
            LOG_ERROR("[ERROR] FaceDetect: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

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

        face::FaceDetectResult detect_result =
            detector->detect(input_ptr, (int)image->width, (int)image->height, threshold);

        auto out_data = make_malloc_buffer(image->width * image->height * channels);
        if (!out_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data.get(), image->data, image->width * image->height * channels);

        auto draw_rect = [&](int x1, int y1, int x2, int y2, uint8_t r, uint8_t g, uint8_t b) {
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
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

        auto result_img = create_image_ptr((int)image->width, (int)image->height, channels, std::move(out_data));
        if (!result_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = result_img;
        outputs["faces"] = detect_result;
        LOG_INFO("[FaceDetect] Detected %zu faces\n", detect_result.faces.size());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceDetect", FaceDetectNode);

#else // !HAS_ONNXRUNTIME

static std::vector<PortDef> face_detect_inputs() {
    return {{"image", "IMAGE", true, nullptr},
            {"model", "FACE_DETECT_MODEL", true, nullptr},
            {"confidence_threshold", "FLOAT", false, 0.5f}};
}
static std::vector<PortDef> face_detect_outputs() {
    return {{"IMAGE", "IMAGE"}, {"faces", "FACE_BBOX_LIST"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(FaceDetectNode, "FaceDetect", "image", face_detect_inputs, face_detect_outputs,
                             sd_error_t::ERROR_EXECUTION_FAILED)
REGISTER_NODE("FaceDetect", FaceDetectNode);

#endif // HAS_ONNXRUNTIME

void init_face_detect_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
