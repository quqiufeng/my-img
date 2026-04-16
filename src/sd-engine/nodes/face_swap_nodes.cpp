// ============================================================================
// sd-engine/nodes/face_swap_nodes.cpp
// ============================================================================
// 人脸换脸节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

#ifdef HAS_ONNXRUNTIME

// ============================================================================
// FaceSwap - 人脸换脸
// ============================================================================
class FaceSwapNode : public Node {
  public:
    std::string get_class_type() const override {
        return "FaceSwap";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"target_image", "IMAGE", true, nullptr},
                {"source_image", "IMAGE", true, nullptr},
                {"face_swap_model", "FACE_SWAP_MODEL", true, nullptr},
                {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr target_image;
        if (sd_error_t err = get_input(inputs, "target_image", target_image); is_error(err)) {
            return err;
        }
        ImagePtr source_image;
        if (sd_error_t err = get_input(inputs, "source_image", source_image); is_error(err)) {
            return err;
        }
        std::shared_ptr<face::FaceSwapper> swapper;
        if (sd_error_t err = get_input(inputs, "face_swap_model", swapper); is_error(err)) {
            return err;
        }

        if (!target_image || !target_image->data || !source_image || !source_image->data || !swapper) {
            LOG_ERROR("[ERROR] FaceSwap: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int target_channels = (int)target_image->channel;
        int source_channels = (int)source_image->channel;
        if ((target_channels != 3 && target_channels != 4) || (source_channels != 3 && source_channels != 4)) {
            LOG_ERROR("[ERROR] FaceSwap: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        std::shared_ptr<face::FaceDetector> detector;
        if (inputs.count("face_detect_model")) {
            if (sd_error_t err = get_input(inputs, "face_detect_model", detector); is_error(err)) {
                return err;
            }
        }

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

        convert_to_rgb(target_image->data, target_rgb.data(), target_channels,
                       target_image->width * target_image->height);
        convert_to_rgb(source_image->data, source_rgb.data(), source_channels,
                       source_image->width * source_image->height);

        auto out_data = make_malloc_buffer(target_image->width * target_image->height * target_channels);
        if (!out_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data.get(), target_image->data, target_image->width * target_image->height * target_channels);

        face::FaceDetectResult target_detect, source_detect;
        if (detector) {
            target_detect =
                detector->detect(target_rgb.data(), (int)target_image->width, (int)target_image->height, 0.5f);
            source_detect =
                detector->detect(source_rgb.data(), (int)source_image->width, (int)source_image->height, 0.5f);
        }

        if (target_detect.faces.empty() || source_detect.faces.empty()) {
            LOG_INFO("[FaceSwap] No faces detected in target or source, returning original target image\n");
            auto result_img = create_image_ptr((int)target_image->width, (int)target_image->height, target_channels,
                                               std::move(out_data));
            if (!result_img) {
                return sd_error_t::ERROR_MEMORY_ALLOCATION;
            }
            outputs["IMAGE"] = result_img;
            return sd_error_t::OK;
        }

        const auto& target_face = target_detect.faces[0];
        const auto& source_face = source_detect.faces[0];

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
            face::get_standard_face_template_512(template_pts);
            for (int i = 0; i < 5; i++) {
                template_pts[i * 2 + 0] *= 128.0f / 512.0f;
                template_pts[i * 2 + 1] *= 128.0f / 512.0f;
            }

            face::estimate_affine_transform_2d3(face.landmarks, template_pts, M);
            face::invert_affine_transform(M, inv_M);

            return face::crop_face(cropped.data(), crop_w, crop_h, 3, face.landmarks, 128);
        };

        std::vector<uint8_t> target_aligned =
            align_face(target_rgb, (int)target_image->width, (int)target_image->height, target_face);
        std::vector<uint8_t> source_aligned =
            align_face(source_rgb, (int)source_image->width, (int)source_image->height, source_face);

        auto swap_result = swapper->swap(target_aligned.data(), source_aligned.data());
        if (!swap_result.success) {
            LOG_ERROR("[ERROR] FaceSwap: Swap failed\n");
            auto result_img = create_image_ptr((int)target_image->width, (int)target_image->height, target_channels,
                                               std::move(out_data));
            if (!result_img) {
                return sd_error_t::ERROR_MEMORY_ALLOCATION;
            }
            outputs["IMAGE"] = result_img;
            return sd_error_t::OK;
        }

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
                float mask_val =
                    (m00 * (1 - fx) * (1 - fy) + m01 * fx * (1 - fy) + m10 * (1 - fx) * fy + m11 * fx * fy) / 255.0f;

                if (mask_val <= 0.01f)
                    continue;

                for (int c = 0; c < 3; c++) {
                    uint8_t p00 = swap_result.swapped_rgb[(ay * 128 + ax) * 3 + c];
                    uint8_t p01 = swap_result.swapped_rgb[(ay * 128 + std::min(127, ax + 1)) * 3 + c];
                    uint8_t p10 = swap_result.swapped_rgb[(std::min(127, ay + 1) * 128 + ax) * 3 + c];
                    uint8_t p11 =
                        swap_result.swapped_rgb[(std::min(127, ay + 1) * 128 + std::min(127, ax + 1)) * 3 + c];
                    float swapped_val =
                        (p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy) + p10 * (1 - fx) * fy + p11 * fx * fy);

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

        auto result_img =
            create_image_ptr((int)target_image->width, (int)target_image->height, target_channels, std::move(out_data));
        if (!result_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = result_img;
        LOG_INFO("[FaceSwap] Swapped face successfully\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceSwap", FaceSwapNode);

#else // !HAS_ONNXRUNTIME

static std::vector<PortDef> face_swap_inputs() {
    return {{"target_image", "IMAGE", true, nullptr},
            {"source_image", "IMAGE", true, nullptr},
            {"face_swap_model", "FACE_SWAP_MODEL", true, nullptr},
            {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr}};
}
static std::vector<PortDef> face_swap_outputs() {
    return {{"IMAGE", "IMAGE"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(FaceSwapNode, "FaceSwap", "image", face_swap_inputs, face_swap_outputs,
                             sd_error_t::ERROR_EXECUTION_FAILED)
REGISTER_NODE("FaceSwap", FaceSwapNode);

#endif // HAS_ONNXRUNTIME

void init_face_swap_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
