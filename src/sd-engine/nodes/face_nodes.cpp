// ============================================================================
// sd-engine/nodes/face_nodes.cpp
// ============================================================================
// 人脸检测/修复/换脸节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"
#include <future>
#include <mutex>

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
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        auto detector = std::any_cast<std::shared_ptr<face::FaceDetector>>(inputs.at("model"));
        float threshold =
            inputs.count("confidence_threshold") ? std::any_cast<float>(inputs.at("confidence_threshold")) : 0.5f;

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

        uint8_t* out_data = (uint8_t*)malloc(image->width * image->height * channels);
        if (!out_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data, image->data, image->width * image->height * channels);

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

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            free(out_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = image->width;
        result_img->height = image->height;
        result_img->channel = channels;
        result_img->data = out_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        outputs["faces"] = detect_result;
        LOG_INFO("[FaceDetect] Detected %zu faces\n", detect_result.faces.size());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceDetect", FaceDetectNode);

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
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        auto restorer = std::any_cast<std::shared_ptr<face::FaceRestorer>>(inputs.at("face_restore_model"));
        float fidelity =
            inputs.count("codeformer_fidelity") ? std::any_cast<float>(inputs.at("codeformer_fidelity")) : 0.5f;

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
            detector = std::any_cast<std::shared_ptr<face::FaceDetector>>(inputs.at("face_detect_model"));
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

        uint8_t* out_data = (uint8_t*)malloc(image->width * image->height * channels);
        if (!out_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data, image->data, image->width * image->height * channels);

        if (detect_result.faces.empty()) {
            LOG_INFO("[FaceRestoreWithModel] No faces detected, returning original image\n");
            sd_image_t* result_img = acquire_image();
            if (!result_img) {
                free(out_data);
                return sd_error_t::ERROR_MEMORY_ALLOCATION;
            }
            result_img->width = image->width;
            result_img->height = image->height;
            result_img->channel = channels;
            result_img->data = out_data;
            outputs["IMAGE"] = make_image_ptr(result_img);
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

                // Synchronize access to jobs vector
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

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            free(out_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = image->width;
        result_img->height = image->height;
        result_img->channel = channels;
        result_img->data = out_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        LOG_INFO("[FaceRestoreWithModel] Restored %zu faces\n", detect_result.faces.size());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceRestoreWithModel", FaceRestoreWithModelNode);

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
        ImagePtr target_image = std::any_cast<ImagePtr>(inputs.at("target_image"));
        ImagePtr source_image = std::any_cast<ImagePtr>(inputs.at("source_image"));
        auto swapper = std::any_cast<std::shared_ptr<face::FaceSwapper>>(inputs.at("face_swap_model"));

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
            detector = std::any_cast<std::shared_ptr<face::FaceDetector>>(inputs.at("face_detect_model"));
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

        uint8_t* out_data = (uint8_t*)malloc(target_image->width * target_image->height * target_channels);
        if (!out_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(out_data, target_image->data, target_image->width * target_image->height * target_channels);

        face::FaceDetectResult target_detect, source_detect;
        if (detector) {
            target_detect =
                detector->detect(target_rgb.data(), (int)target_image->width, (int)target_image->height, 0.5f);
            source_detect =
                detector->detect(source_rgb.data(), (int)source_image->width, (int)source_image->height, 0.5f);
        }

        if (target_detect.faces.empty() || source_detect.faces.empty()) {
            LOG_INFO("[FaceSwap] No faces detected in target or source, returning original target image\n");
            sd_image_t* result_img = acquire_image();
            if (!result_img) {
                free(out_data);
                return sd_error_t::ERROR_MEMORY_ALLOCATION;
            }
            result_img->width = target_image->width;
            result_img->height = target_image->height;
            result_img->channel = target_channels;
            result_img->data = out_data;
            outputs["IMAGE"] = make_image_ptr(result_img);
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
            sd_image_t* result_img = acquire_image();
            if (!result_img) {
                free(out_data);
                return sd_error_t::ERROR_MEMORY_ALLOCATION;
            }
            result_img->width = target_image->width;
            result_img->height = target_image->height;
            result_img->channel = target_channels;
            result_img->data = out_data;
            outputs["IMAGE"] = make_image_ptr(result_img);
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

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            free(out_data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = target_image->width;
        result_img->height = target_image->height;
        result_img->channel = target_channels;
        result_img->data = out_data;

        outputs["IMAGE"] = make_image_ptr(result_img);
        LOG_INFO("[FaceSwap] Swapped face successfully\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceSwap", FaceSwapNode);

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

void init_face_nodes() {
    // 空函数，仅确保本翻译单元被链接
}

} // namespace sdengine
