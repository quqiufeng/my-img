// ============================================================================
// face_detect.hpp
// ============================================================================
// SCRFD 人脸检测 ONNX 推理封装
// ============================================================================

#pragma once

#ifdef HAS_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace sdengine {
namespace face {

struct FaceBBox {
    float x1, y1, x2, y2;
    float score;
    float landmarks[10];  // 5 points * 2 (x, y)
};

struct FaceDetectResult {
    std::vector<FaceBBox> faces;
    int img_w = 0;
    int img_h = 0;
};

class FaceDetector {
public:
    bool load(const std::string& model_path);
    FaceDetectResult detect(const uint8_t* rgb_data, int width, int height, float confidence_threshold = 0.5f);

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "scrfd"};
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    int input_size_ = 640;
    float mean_[3] = {127.5f, 127.5f, 127.5f};
    float std_[3]  = {128.0f, 128.0f, 128.0f};

    std::vector<FaceBBox> generate_bboxes(const float* cls, const float* obj, const float* bboxes, const float* kps,
                                             int stride, int feat_h, int feat_w, float threshold,
                                             float scale_w, float scale_h);
    std::vector<FaceBBox> nms(std::vector<FaceBBox>& boxes, float nms_threshold = 0.4f);
};

} // namespace face
} // namespace sdengine

#endif // HAS_ONNXRUNTIME
