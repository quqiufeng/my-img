// ============================================================================
// face_swap.hpp
// ============================================================================
// InsightFace inswapper_128 + ArcFace 人脸换脸 ONNX 推理封装
// ============================================================================

#pragma once

#ifdef HAS_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace sdengine {
namespace face {

struct SwapResult {
    std::vector<uint8_t> swapped_rgb;  // 128x128x3
    bool success = false;
};

class FaceSwapper {
public:
    // 加载 inswapper_128 和 arcface 模型
    bool load(const std::string& inswapper_path, const std::string& arcface_path);

    // 输入：target 和 source 都是 128x128 RGB 对齐后的人脸图像
    // 输出：128x128 换脸结果
    SwapResult swap(const uint8_t* target_rgb_128, const uint8_t* source_rgb_128);

private:
    std::unique_ptr<Ort::Session> inswapper_session_;
    std::unique_ptr<Ort::Session> arcface_session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "face_swap"};
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    // 从 source 人脸提取 512 维 embedding
    std::vector<float> extract_embedding(const uint8_t* rgb_128);

    static constexpr int face_size_ = 128;
    static constexpr int arcface_size_ = 112;
};

} // namespace face
} // namespace sdengine

#endif // HAS_ONNXRUNTIME
