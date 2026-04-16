// ============================================================================
// face_detect.cpp
// ============================================================================
// YuNet 人脸检测 ONNX 推理实现
// ============================================================================

#ifdef HAS_ONNXRUNTIME

#include "face_detect.hpp"
#include <cstring>
#include "core/log.h"

namespace sdengine {
namespace face {

bool FaceDetector::load(const std::string& model_path) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[ERROR] FaceDetector: Failed to load model: %s\n", e.what());
        return false;
    }
    return true;
}

FaceDetectResult FaceDetector::detect(const uint8_t* rgb_data, int width, int height, float confidence_threshold) {
    FaceDetectResult result;
    result.img_w = width;
    result.img_h = height;

    if (!session_) {
        LOG_ERROR("[ERROR] FaceDetector: Model not loaded\n");
        return result;
    }

    input_size_ = 640;

    // Calculate resize scale with aspect ratio preservation
    float scale = std::min((float)input_size_ / width, (float)input_size_ / height);
    int new_w = (int)(width * scale);
    int new_h = (int)(height * scale);
    int pad_w = (input_size_ - new_w) / 2;
    int pad_h = (input_size_ - new_h) / 2;

    // Preprocess: resize + pad + normalize
    std::vector<float> input_data(1 * 3 * input_size_ * input_size_, 0.0f);
    std::vector<uint8_t> resized(new_w * new_h * 3);

    // Simple nearest neighbor resize
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_x = (int)(x / scale);
            int src_y = (int)(y / scale);
            src_x = std::min(src_x, width - 1);
            src_y = std::min(src_y, height - 1);
            for (int c = 0; c < 3; c++) {
                resized[(y * new_w + x) * 3 + c] = rgb_data[(src_y * width + src_x) * 3 + c];
            }
        }
    }

    // Fill padded input (NCHW)
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            for (int c = 0; c < 3; c++) {
                int dst_y = y + pad_h;
                int dst_x = x + pad_w;
                float val = resized[(y * new_w + x) * 3 + c];
                input_data[c * input_size_ * input_size_ + dst_y * input_size_ + dst_x] = (val - mean_[c]) / std_[c];
            }
        }
    }

    // ONNX inference
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<const char*> input_names;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_names;

    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        input_name_ptrs.push_back(session_->GetInputNameAllocated(i, allocator));
        input_names.push_back(input_name_ptrs.back().get());
    }

    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        output_name_ptrs.push_back(session_->GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_ptrs.back().get());
    }

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, input_names.size(),
            output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[ERROR] FaceDetector: ONNX inference failed: %s\n", e.what());
        return result;
    }

    // Parse YuNet outputs: cls_8/16/32, obj_8/16/32, bbox_8/16/32, kps_8/16/32
    std::vector<FaceBBox> all_boxes;

    // Map output names to tensors
    struct OutputGroup {
        const float* cls = nullptr;
        const float* obj = nullptr;
        const float* bbox = nullptr;
        const float* kps = nullptr;
        int num_anchors = 0;
        int stride = 0;
    };

    std::vector<OutputGroup> groups;
    for (size_t i = 0; i < output_names.size(); i++) {
        std::string name(output_names[i]);
        auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3) continue;

        int num_anchors = (int)shape[1];
        int stride = 0;
        if (name.find("_8") != std::string::npos) stride = 8;
        else if (name.find("_16") != std::string::npos) stride = 16;
        else if (name.find("_32") != std::string::npos) stride = 32;

        if (stride == 0) continue;

        // Find or create group
        bool found = false;
        for (auto& g : groups) {
            if (g.stride == stride) {
                if (name.find("cls_") == 0) g.cls = output_tensors[i].GetTensorMutableData<float>();
                else if (name.find("obj_") == 0) g.obj = output_tensors[i].GetTensorMutableData<float>();
                else if (name.find("bbox_") == 0) g.bbox = output_tensors[i].GetTensorMutableData<float>();
                else if (name.find("kps_") == 0) g.kps = output_tensors[i].GetTensorMutableData<float>();
                found = true;
                break;
            }
        }
        if (!found) {
            OutputGroup g;
            g.num_anchors = num_anchors;
            g.stride = stride;
            if (name.find("cls_") == 0) g.cls = output_tensors[i].GetTensorMutableData<float>();
            else if (name.find("obj_") == 0) g.obj = output_tensors[i].GetTensorMutableData<float>();
            else if (name.find("bbox_") == 0) g.bbox = output_tensors[i].GetTensorMutableData<float>();
            else if (name.find("kps_") == 0) g.kps = output_tensors[i].GetTensorMutableData<float>();
            groups.push_back(g);
        }
    }

    float scale_w = (float)width / input_size_;
    float scale_h = (float)height / input_size_;

    for (const auto& g : groups) {
        if (!g.cls || !g.obj || !g.bbox || !g.kps) continue;

        int feat_h = input_size_ / g.stride;
        int feat_w = input_size_ / g.stride;

        auto boxes = generate_bboxes(g.cls, g.obj, g.bbox, g.kps, g.stride, feat_h, feat_w,
                                     confidence_threshold, scale_w, scale_h);
        all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
    }

    // Adjust for padding
    float pad_scale_w = (float)input_size_ / new_w;
    float pad_scale_h = (float)input_size_ / new_h;
    for (auto& box : all_boxes) {
        box.x1 = (box.x1 - pad_w) * pad_scale_w * scale;
        box.y1 = (box.y1 - pad_h) * pad_scale_h * scale;
        box.x2 = (box.x2 - pad_w) * pad_scale_w * scale;
        box.y2 = (box.y2 - pad_h) * pad_scale_h * scale;
        for (int i = 0; i < 10; i += 2) {
            box.landmarks[i] = (box.landmarks[i] - pad_w) * pad_scale_w * scale;
            box.landmarks[i + 1] = (box.landmarks[i + 1] - pad_h) * pad_scale_h * scale;
        }
    }

    result.faces = nms(all_boxes, 0.4f);
    printf("[FaceDetector] Detected %zu faces\n", result.faces.size());
    return result;
}

std::vector<FaceBBox> FaceDetector::generate_bboxes(const float* cls, const float* obj, const float* bboxes, const float* kps,
                                                     int stride, int feat_h, int feat_w, float threshold,
                                                     float scale_w, float scale_h) {
    std::vector<FaceBBox> boxes;
    (void)scale_w; (void)scale_h;

    int num_anchors = feat_h * feat_w;

    for (int idx = 0; idx < num_anchors; idx++) {
        float score = cls[idx] * obj[idx];

        if (score < threshold) continue;

        int a = idx % num_anchors; // anchor index
        int y = a / feat_w;
        int x = a % feat_w;

        FaceBBox box;
        box.score = score;

        // YuNet bbox format: [x1, y1, x2, y2] relative to stride
        float bx1 = bboxes[idx * 4 + 0];
        float by1 = bboxes[idx * 4 + 1];
        float bx2 = bboxes[idx * 4 + 2];
        float by2 = bboxes[idx * 4 + 3];

        box.x1 = (x + bx1) * stride;
        box.y1 = (y + by1) * stride;
        box.x2 = (x + bx2) * stride;
        box.y2 = (y + by2) * stride;

        // Decode landmarks
        for (int k = 0; k < 5; k++) {
            box.landmarks[k * 2 + 0] = (x + kps[idx * 10 + k * 2 + 0]) * stride;
            box.landmarks[k * 2 + 1] = (y + kps[idx * 10 + k * 2 + 1]) * stride;
        }

        boxes.push_back(box);
    }
    return boxes;
}

std::vector<FaceBBox> FaceDetector::nms(std::vector<FaceBBox>& boxes, float nms_threshold) {
    std::sort(boxes.begin(), boxes.end(), [](const FaceBBox& a, const FaceBBox& b) {
        return a.score > b.score;
    });

    std::vector<FaceBBox> result;
    std::vector<bool> suppressed(boxes.size(), false);

    auto iou = [](const FaceBBox& a, const FaceBBox& b) {
        float x1 = std::max(a.x1, b.x1);
        float y1 = std::max(a.y1, b.y1);
        float x2 = std::min(a.x2, b.x2);
        float y2 = std::min(a.y2, b.y2);
        float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        float union_area = area_a + area_b - inter;
        return union_area > 0 ? inter / union_area : 0.0f;
    };

    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            if (iou(boxes[i], boxes[j]) > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

} // namespace face
} // namespace sdengine

#endif // HAS_ONNXRUNTIME
