// ============================================================================
// sd-engine/nodes/face_restore_nodes.cpp
// ============================================================================
// 人脸修复节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"
#include <future>
#include <mutex>

namespace sdengine {

#ifdef HAS_ONNXRUNTIME

// ============================================================================
// FaceRestoreWithModel - 人脸修复（一键版）
// ============================================================================
class FaceRestoreWithModelNode : public Node {
  public:
    std::string get_class_type() const override {
        return "FaceRestoreWithModel";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr},
                {"face_restore_model", "FACE_RESTORE_MODEL", true, nullptr},
                {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr},
                {"codeformer_fidelity", "FLOAT", false, 0.5f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image;
        if (sd_error_t err = get_input(inputs, "image", image); is_error(err)) {
            return err;
        }
        std::shared_ptr<face::FaceRestorer> restorer;
        if (sd_error_t err = get_input(inputs, "face_restore_model", restorer); is_error(err)) {
            return err;
        }
        float fidelity = get_input_opt<float>(inputs, "codeformer_fidelity", 0.5f);

        if (!image || !image->data || !restorer) {
            LOG_ERROR("[ERROR] FaceRestoreWithModel: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int channels = (int)image->channel;
        if (channels != 3 && channels != 4) {
            LOG_ERROR("[ERROR] FaceRestoreWithModel: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        std::shared_ptr<face::FaceDetector> detector;
        if (inputs.count("face_detect_model")) {
            if (sd_error_t err = get_input(inputs, "face_detect_model", detector); is_error(err)) {
                return err;
            }
        }

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

        face::FaceDetectResult detect_result;
        if (detector) {
            detect_result = detector->detect(rgb_data.data(), (int)image->width, (int)image->height, 0.5f);
        }

        auto out_data = make_malloc_buffer(image->width * image->height * channels);
        if (!out_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data.get(), image->data, image->width * image->height * channels);

        if (detect_result.faces.empty()) {
            LOG_INFO("[FaceRestoreWithModel] No faces detected, returning original image\n");
            auto result_img = create_image_ptr((int)image->width, (int)image->height, channels, std::move(out_data));
            if (!result_img) {
                return sd_error_t::ERROR_MEMORY_ALLOCATION;
            }
            outputs["IMAGE"] = result_img;
            return sd_error_t::OK;
        }

        struct FaceRestoreJob {
            int crop_x1, crop_y1, crop_w, crop_h;
            float inv_M[6];
            face::RestoreResult restore_result;
            bool success;
        };

        std::vector<FaceRestoreJob> jobs;
        jobs.reserve(detect_result.faces.size());
        std::vector<std::future<void>> futures;
        futures.reserve(detect_result.faces.size());

        for (const auto& f : detect_result.faces) {
            futures.push_back(std::async(std::launch::async, [&, f]() {
                float cx = (f.x1 + f.x2) * 0.5f;
                float cy = (f.y1 + f.y2) * 0.5f;
                float size = std::max(f.x2 - f.x1, f.y2 - f.y1) * 1.5f;
                int crop_x1 = (int)std::max(0.0f, cx - size * 0.5f);
                int crop_y1 = (int)std::max(0.0f, cy - size * 0.5f);
                int crop_x2 = (int)std::min((float)image->width, cx + size * 0.5f);
                int crop_y2 = (int)std::min((float)image->height, cy + size * 0.5f);
                int crop_w = crop_x2 - crop_x1;
                int crop_h = crop_y2 - crop_y1;

                if (crop_w <= 0 || crop_h <= 0)
                    return;

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

                float M[6], inv_M[6];
                float template_pts[10];
                face::get_standard_face_template_512(template_pts);

                if (!face::estimate_affine_transform_2d3(f.landmarks, template_pts, M)) {
                    return;
                }
                if (!face::invert_affine_transform(M, inv_M)) {
                    return;
                }

                std::vector<uint8_t> aligned_face =
                    face::crop_face(cropped.data(), crop_w, crop_h, 3, f.landmarks, 512);

                auto restore_result = restorer->restore(aligned_face.data(), fidelity);
                if (!restore_result.success) {
                    LOG_ERROR("[ERROR] FaceRestoreWithModel: Restore failed for one face\n");
                    return;
                }

                FaceRestoreJob job;
                job.crop_x1 = crop_x1;
                job.crop_y1 = crop_y1;
                job.crop_w = crop_w;
                job.crop_h = crop_h;
                memcpy(job.inv_M, inv_M, sizeof(inv_M));
                job.restore_result = std::move(restore_result);
                job.success = true;

                static std::mutex jobs_mutex;
                std::lock_guard<std::mutex> lock(jobs_mutex);
                jobs.push_back(std::move(job));
            }));
        }

        for (auto& fut : futures) {
            fut.wait();
        }

        for (const auto& job : jobs) {
            if (!job.success)
                continue;

            std::vector<uint8_t> mask(512 * 512);
            face::generate_feather_mask(mask.data(), 512, 32);

            for (int y = 0; y < job.crop_h; y++) {
                for (int x = 0; x < job.crop_w; x++) {
                    float crop_x = (float)x;
                    float crop_y = (float)y;
                    float aligned_x = job.inv_M[0] * crop_x + job.inv_M[1] * crop_y + job.inv_M[2];
                    float aligned_y = job.inv_M[3] * crop_x + job.inv_M[4] * crop_y + job.inv_M[5];

                    if (aligned_x < 0 || aligned_x >= 511 || aligned_y < 0 || aligned_y >= 511) {
                        continue;
                    }

                    int ax = (int)aligned_x;
                    int ay = (int)aligned_y;
                    float fx = aligned_x - ax;
                    float fy = aligned_y - ay;

                    uint8_t m00 = mask[ay * 512 + ax];
                    uint8_t m01 = mask[ay * 512 + std::min(511, ax + 1)];
                    uint8_t m10 = mask[std::min(511, ay + 1) * 512 + ax];
                    uint8_t m11 = mask[std::min(511, ay + 1) * 512 + std::min(511, ax + 1)];
                    float mask_val =
                        (m00 * (1 - fx) * (1 - fy) + m01 * fx * (1 - fy) + m10 * (1 - fx) * fy + m11 * fx * fy) /
                        255.0f;

                    if (mask_val <= 0.01f)
                        continue;

                    for (int c = 0; c < 3; c++) {
                        uint8_t p00 = job.restore_result.restored_rgb[(ay * 512 + ax) * 3 + c];
                        uint8_t p01 = job.restore_result.restored_rgb[(ay * 512 + std::min(511, ax + 1)) * 3 + c];
                        uint8_t p10 = job.restore_result.restored_rgb[(std::min(511, ay + 1) * 512 + ax) * 3 + c];
                        uint8_t p11 = job.restore_result
                                          .restored_rgb[(std::min(511, ay + 1) * 512 + std::min(511, ax + 1)) * 3 + c];
                        float restored_val =
                            (p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy) + p10 * (1 - fx) * fy + p11 * fx * fy);

                        int px = job.crop_x1 + x;
                        int py = job.crop_y1 + y;
                        if (px >= 0 && px < (int)image->width && py >= 0 && py < (int)image->height) {
                            size_t idx = (py * image->width + px) * channels + c;
                            float orig_val = out_data[idx];
                            out_data[idx] = (uint8_t)(orig_val * (1.0f - mask_val) + restored_val * mask_val);
                        }
                    }
                }
            }
        }

        auto result_img = create_image_ptr((int)image->width, (int)image->height, channels, std::move(out_data));
        if (!result_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = result_img;
        LOG_INFO("[FaceRestoreWithModel] Restored %zu faces\n", detect_result.faces.size());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceRestoreWithModel", FaceRestoreWithModelNode);

#else // !HAS_ONNXRUNTIME

static std::vector<PortDef> face_restore_inputs() {
    return {{"image", "IMAGE", true, nullptr},
            {"face_restore_model", "FACE_RESTORE_MODEL", true, nullptr},
            {"face_detect_model", "FACE_DETECT_MODEL", false, nullptr},
            {"codeformer_fidelity", "FLOAT", false, 0.5f}};
}
static std::vector<PortDef> face_restore_outputs() {
    return {{"IMAGE", "IMAGE"}};
}
DEFINE_ONNX_PLACEHOLDER_NODE(FaceRestoreWithModelNode, "FaceRestoreWithModel", "image", face_restore_inputs,
                             face_restore_outputs, sd_error_t::ERROR_EXECUTION_FAILED)
REGISTER_NODE("FaceRestoreWithModel", FaceRestoreWithModelNode);

#endif // HAS_ONNXRUNTIME

void init_face_restore_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
