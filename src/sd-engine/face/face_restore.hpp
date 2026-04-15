// ============================================================================
// face_restore.hpp
// ============================================================================
// GFPGAN / CodeFormer 人脸修复 ONNX 推理封装
// ============================================================================

#pragma once

#ifdef HAS_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

namespace sdengine {
namespace face {

enum class RestoreModelType {
    GFPGAN,
    CODEFORMER
};

struct RestoreResult {
    std::vector<uint8_t> restored_rgb;  // 512x512x3
    bool success = false;
};

class FaceRestorer {
public:
    bool load(const std::string& model_path, RestoreModelType type);
    
    // 输入：512x512 RGB 图像（已对齐）
    // codeformer_fidelity: CodeFormer 的 fidelity 参数，范围 [0.0, 1.0]，默认 0.5
    RestoreResult restore(const uint8_t* rgb_512, float codeformer_fidelity = 0.5f);

    RestoreModelType get_model_type() const { return model_type_; }

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "face_restore"};
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    RestoreModelType model_type_ = RestoreModelType::GFPGAN;

    // 输入/输出维度
    static constexpr int input_size_ = 512;
    static constexpr int output_size_ = 512;
};

} // namespace face
} // namespace sdengine

#endif // HAS_ONNXRUNTIME
