// ============================================================================
// face_restore.cpp
// ============================================================================
// GFPGAN / CodeFormer 人脸修复 ONNX 推理实现
// ============================================================================

#include "face_restore.hpp"

#ifdef HAS_ONNXRUNTIME

#include <cstdio>
#include <cstring>
#include <algorithm>
#include "core/log.h"

namespace sdengine {
namespace face {

bool FaceRestorer::load(const std::string& model_path, RestoreModelType type) {
    model_type_ = type;
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[ERROR] FaceRestorer::load: %s\n", e.what());
        return false;
    }

    printf("[FaceRestorer] Loaded model: %s (type=%s)\n",
           model_path.c_str(),
           type == RestoreModelType::GFPGAN ? "GFPGAN" : "CodeFormer");
    return true;
}

RestoreResult FaceRestorer::restore(const uint8_t* rgb_512, float codeformer_fidelity) {
    RestoreResult result;
    if (!session_) {
        LOG_ERROR("[ERROR] FaceRestorer::restore: Model not loaded\n");
        return result;
    }

    // ONNX 输入: [1, 3, 512, 512] NCHW, float32
    // 归一化到 [-1, 1] 或 [0, 1]（GFPGAN/CodeFormer 通常使用 [-1, 1]）
    std::vector<float> input_tensor_values(1 * 3 * 512 * 512);
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float std[3]  = {0.5f, 0.5f, 0.5f};

    for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 512; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = c * 512 * 512 + y * 512 + x;
                float val = rgb_512[(y * 512 + x) * 3 + c] / 255.0f;
                input_tensor_values[idx] = (val - mean[c]) / std[c];
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, 512, 512};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // 获取输入/输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<const char*> input_names;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_names;

    size_t num_inputs = session_->GetInputCount();
    size_t num_outputs = session_->GetOutputCount();

    for (size_t i = 0; i < num_inputs; i++) {
        input_name_ptrs.push_back(session_->GetInputNameAllocated(i, allocator));
        input_names.push_back(input_name_ptrs.back().get());
    }
    for (size_t i = 0; i < num_outputs; i++) {
        output_name_ptrs.push_back(session_->GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_ptrs.back().get());
    }

    // CodeFormer 部分 ONNX 导出包含 fidelity 参数 w
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    // 如果输入数量为 2，假设第二个输入是 w (CodeFormer)
    // 注意：不同 ONNX 导出版本的 w 类型可能不同（float 或 double）
    if (num_inputs >= 2 && model_type_ == RestoreModelType::CODEFORMER) {
        // 先尝试 double，失败则 fallback 到 float
        std::vector<int64_t> w_shape = {1};
        std::vector<double> w_values_double = {(double)codeformer_fidelity};
        Ort::Value w_tensor = Ort::Value::CreateTensor<double>(
            memory_info_, w_values_double.data(), w_values_double.size(), w_shape.data(), w_shape.size());
        input_tensors.push_back(std::move(w_tensor));
    }

    // 运行推理
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "[ERROR] FaceRestorer::restore: ONNX Runtime error: %s\n", e.what());
        return result;
    }

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        LOG_ERROR("[ERROR] FaceRestorer::restore: Invalid output\n");
        return result;
    }

    // 解析输出
    Ort::Value& output_tensor = output_tensors[0];
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 3 ||
        output_shape[2] != 512 || output_shape[3] != 512) {
        LOG_ERROR("[ERROR] FaceRestorer::restore: Unexpected output shape [%ld, %ld, %ld, %ld]\n",
                output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
        return result;
    }

    const float* output_data = output_tensor.GetTensorData<float>();
    result.restored_rgb.resize(512 * 512 * 3);

    for (int y = 0; y < 512; y++) {
        for (int x = 0; x < 512; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = c * 512 * 512 + y * 512 + x;
                float val = output_data[idx] * std[c] + mean[c];
                val = std::max(0.0f, std::min(1.0f, val));
                result.restored_rgb[(y * 512 + x) * 3 + c] = (uint8_t)(val * 255.0f);
            }
        }
    }

    result.success = true;
    LOG_INFO("[FaceRestorer] Restored face successfully\n");
    return result;
}

} // namespace face
} // namespace sdengine

#endif // HAS_ONNXRUNTIME
