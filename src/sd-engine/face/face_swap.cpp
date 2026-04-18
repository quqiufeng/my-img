// ============================================================================
// face_swap.cpp
// ============================================================================
// InsightFace inswapper_128 + ArcFace 人脸换脸 ONNX 推理实现
// ============================================================================

#include "face_swap.hpp"

#ifdef HAS_ONNXRUNTIME

#include "core/log.h"
#include <algorithm>
#include <cstring>

namespace sdengine {
namespace face {

bool FaceSwapper::load(const std::string& inswapper_path, const std::string& arcface_path) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        inswapper_session_ = std::make_unique<Ort::Session>(env_, inswapper_path.c_str(), session_options);
        arcface_session_ = std::make_unique<Ort::Session>(env_, arcface_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("[ERROR] FaceSwapper::load: %s\n", e.what());
        return false;
    }

    LOG_INFO("[FaceSwapper] Loaded inswapper: %s\n", inswapper_path.c_str());
    LOG_INFO("[FaceSwapper] Loaded arcface: %s\n", arcface_path.c_str());
    return true;
}

std::vector<float> FaceSwapper::extract_embedding(const uint8_t* rgb_128) {
    std::vector<float> embedding;
    if (!arcface_session_) {
        LOG_ERROR("[ERROR] FaceSwapper::extract_embedding: ArcFace not loaded\n");
        return embedding;
    }

    // ArcFace 输入: [1, 112, 112, 3] NHWC, float32
    // 将 128x128 缩放到 112x112，归一化到 [-1, 1] 或 [0, 1]
    std::vector<float> input_tensor_values(1 * 112 * 112 * 3);
    for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
            // 双线性插值从 128x128 缩放到 112x112
            float src_x = (x + 0.5f) * 128.0f / 112.0f - 0.5f;
            float src_y = (y + 0.5f) * 128.0f / 112.0f - 0.5f;
            int x0 = (int)std::floor(src_x);
            int y0 = (int)std::floor(src_y);
            int x1 = std::min(127, x0 + 1);
            int y1 = std::min(127, y0 + 1);
            float fx = src_x - x0;
            float fy = src_y - y0;

            for (int c = 0; c < 3; c++) {
                float p00 = rgb_128[(y0 * 128 + x0) * 3 + c] / 255.0f;
                float p01 = rgb_128[(y0 * 128 + x1) * 3 + c] / 255.0f;
                float p10 = rgb_128[(y1 * 128 + x0) * 3 + c] / 255.0f;
                float p11 = rgb_128[(y1 * 128 + x1) * 3 + c] / 255.0f;
                float val = p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy) + p10 * (1 - fx) * fy + p11 * fx * fy;
                // 归一化到 [-1, 1]
                input_tensor_values[(y * 112 + x) * 3 + c] = (val - 0.5f) / 0.5f;
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 112, 112, 3};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<const char*> input_names;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_names;

    size_t num_inputs = arcface_session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        input_name_ptrs.push_back(arcface_session_->GetInputNameAllocated(i, allocator));
        input_names.push_back(input_name_ptrs.back().get());
    }

    size_t num_outputs = arcface_session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        output_name_ptrs.push_back(arcface_session_->GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_ptrs.back().get());
    }

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = arcface_session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                                               output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        LOG_ERROR("[ERROR] FaceSwapper::extract_embedding: ONNX Runtime error: %s\n", e.what());
        return embedding;
    }

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        LOG_ERROR("[ERROR] FaceSwapper::extract_embedding: Invalid output\n");
        return embedding;
    }

    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 2 || output_shape[0] != 1 || output_shape[1] != 512) {
        LOG_ERROR("[ERROR] FaceSwapper::extract_embedding: Unexpected output shape [%ld, %ld]\n", output_shape[0],
                  output_shape[1]);
        return embedding;
    }

    const float* output_data = output_tensors[0].GetTensorData<float>();
    embedding.assign(output_data, output_data + 512);
    return embedding;
}

SwapResult FaceSwapper::swap(const uint8_t* target_rgb_128, const uint8_t* source_rgb_128) {
    SwapResult result;
    if (!inswapper_session_ || !arcface_session_) {
        LOG_ERROR("[ERROR] FaceSwapper::swap: Models not loaded\n");
        return result;
    }

    // 1. 提取 source 的 embedding
    std::vector<float> embedding = extract_embedding(source_rgb_128);
    if (embedding.empty()) {
        LOG_ERROR("[ERROR] FaceSwapper::swap: Failed to extract embedding\n");
        return result;
    }

    // 2. 准备 target 输入 [1, 3, 128, 128] NCHW
    std::vector<float> target_tensor_values(1 * 3 * 128 * 128);
    for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = c * 128 * 128 + y * 128 + x;
                float val = target_rgb_128[(y * 128 + x) * 3 + c] / 255.0f;
                target_tensor_values[idx] = (val - 0.5f) / 0.5f;
            }
        }
    }

    std::vector<int64_t> target_shape = {1, 3, 128, 128};
    Ort::Value target_tensor =
        Ort::Value::CreateTensor<float>(memory_info_, target_tensor_values.data(), target_tensor_values.size(),
                                        target_shape.data(), target_shape.size());

    // 3. 准备 source embedding 输入 [1, 512]
    std::vector<int64_t> source_shape = {1, 512};
    Ort::Value source_tensor = Ort::Value::CreateTensor<float>(memory_info_, embedding.data(), embedding.size(),
                                                               source_shape.data(), source_shape.size());

    // 4. 获取输入/输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<const char*> input_names;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_names;

    size_t num_inputs = inswapper_session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        input_name_ptrs.push_back(inswapper_session_->GetInputNameAllocated(i, allocator));
        input_names.push_back(input_name_ptrs.back().get());
    }

    size_t num_outputs = inswapper_session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        output_name_ptrs.push_back(inswapper_session_->GetOutputNameAllocated(i, allocator));
        output_names.push_back(output_name_ptrs.back().get());
    }

    // 5. 运行推理
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(target_tensor));
    input_tensors.push_back(std::move(source_tensor));

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = inswapper_session_->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                                                 input_tensors.size(), output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        LOG_ERROR("[ERROR] FaceSwapper::swap: ONNX Runtime error: %s\n", e.what());
        return result;
    }

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        LOG_ERROR("[ERROR] FaceSwapper::swap: Invalid output\n");
        return result;
    }

    // 6. 解析输出 [1, 3, 128, 128]
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 4 || output_shape[0] != 1 || output_shape[1] != 3 || output_shape[2] != 128 ||
        output_shape[3] != 128) {
        LOG_ERROR("[ERROR] FaceSwapper::swap: Unexpected output shape [%ld, %ld, %ld, %ld]\n", output_shape[0],
                  output_shape[1], output_shape[2], output_shape[3]);
        return result;
    }

    const float* output_data = output_tensors[0].GetTensorData<float>();
    result.swapped_rgb.resize(128 * 128 * 3);

    for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = c * 128 * 128 + y * 128 + x;
                float val = output_data[idx] * 0.5f + 0.5f;
                val = std::max(0.0f, std::min(1.0f, val));
                result.swapped_rgb[(y * 128 + x) * 3 + c] = (uint8_t)(val * 255.0f);
            }
        }
    }

    result.success = true;
    LOG_INFO("[FaceSwapper] Swapped face successfully\n");
    return result;
}

} // namespace face
} // namespace sdengine

#endif // HAS_ONNXRUNTIME
