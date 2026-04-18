// ============================================================================
// lineart.cpp
// ============================================================================
// LineArt 线稿提取 ONNX 推理实现
// ============================================================================

#include "lineart.hpp"

#ifdef HAS_ONNXRUNTIME

#include "core/log.h"
#include <algorithm>
#include <cmath>

namespace sdengine {

bool LineArtPreprocessor::load(const std::string& model_path) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("[ERROR] LineArtPreprocessor: Failed to load model: %s\n", e.what());
        return false;
    }

    // Try to infer input size from model metadata
    try {
        size_t num_inputs = session_->GetInputCount();
        if (num_inputs > 0) {
            Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = tensor_info.GetShape();
            if (shape.size() >= 4) {
                // shape: [batch, channels, height, width]
                if (shape[2] > 0)
                    input_height_ = static_cast<int>(shape[2]);
                if (shape[3] > 0)
                    input_width_ = static_cast<int>(shape[3]);
            }
        }
    } catch (...) {
        // fallback to default 512x512
    }

    LOG_INFO("[LineArtPreprocessor] Loaded model: %s (input_size=%dx%d)\n", model_path.c_str(), input_width_,
             input_height_);
    return true;
}

std::vector<uint8_t> LineArtPreprocessor::resize_rgb(const uint8_t* src, int src_w, int src_h, int dst_w, int dst_h) {
    std::vector<uint8_t> dst(dst_w * dst_h * 3);
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int src_x = std::min((int)(x * (float)src_w / dst_w), src_w - 1);
            int src_y = std::min((int)(y * (float)src_h / dst_h), src_h - 1);
            for (int c = 0; c < 3; c++) {
                dst[(y * dst_w + x) * 3 + c] = src[(src_y * src_w + src_x) * 3 + c];
            }
        }
    }
    return dst;
}

LineArtResult LineArtPreprocessor::process(const uint8_t* rgb_data, int width, int height) {
    LineArtResult result;
    if (!session_) {
        LOG_ERROR("[ERROR] LineArtPreprocessor: Model not loaded\n");
        return result;
    }

    // Resize to model input size
    std::vector<uint8_t> resized = resize_rgb(rgb_data, width, height, input_width_, input_height_);

    // Normalize to [0, 1] and convert to NCHW
    std::vector<float> input_tensor_values(1 * 3 * input_height_ * input_width_);
    for (int y = 0; y < input_height_; y++) {
        for (int x = 0; x < input_width_; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = c * input_height_ * input_width_ + y * input_width_ + x;
                input_tensor_values[idx] = resized[(y * input_width_ + x) * 3 + c] / 255.0f;
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Get input/output names
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

    // Run inference
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, num_inputs,
                                       output_names.data(), num_outputs);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("[ERROR] LineArtPreprocessor: ONNX inference failed: %s\n", e.what());
        return result;
    }

    // Process output
    try {
        Ort::TypeInfo type_info = output_tensors[0].GetTypeInfo();
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = tensor_info.GetShape();

        int out_h = input_height_;
        int out_w = input_width_;
        int out_c = 1;

        if (out_shape.size() >= 3) {
            out_h = static_cast<int>(out_shape[out_shape.size() - 2]);
            out_w = static_cast<int>(out_shape[out_shape.size() - 1]);
        }
        if (out_shape.size() >= 4) {
            out_c = static_cast<int>(out_shape[out_shape.size() - 3]);
        }

        float* out_data = output_tensors[0].GetTensorMutableData<float>();

        // Convert output to RGB
        result.width = out_w;
        result.height = out_h;
        result.data.resize(out_w * out_h * 3);

        // Find min/max for normalization
        size_t pixel_count = out_w * out_h;
        float min_val = out_data[0];
        float max_val = out_data[0];
        for (size_t i = 1; i < pixel_count * out_c; i++) {
            min_val = std::min(min_val, out_data[i]);
            max_val = std::max(max_val, out_data[i]);
        }
        float range = max_val - min_val;
        if (range < 1e-5f)
            range = 1.0f;

        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int idx = y * out_w + x;
                float val = out_data[idx]; // assume first channel if multi-channel
                if (out_c > 1) {
                    // average channels
                    val = 0.0f;
                    for (int c = 0; c < out_c; c++) {
                        val += out_data[c * out_h * out_w + idx];
                    }
                    val /= out_c;
                }
                uint8_t gray = static_cast<uint8_t>(std::clamp((val - min_val) / range * 255.0f, 0.0f, 255.0f));
                result.data[idx * 3 + 0] = gray;
                result.data[idx * 3 + 1] = gray;
                result.data[idx * 3 + 2] = gray;
            }
        }

        result.success = true;
    } catch (const std::exception& e) {
        LOG_ERROR("[ERROR] LineArtPreprocessor: Output processing failed: %s\n", e.what());
    }

    return result;
}

} // namespace sdengine

#endif // HAS_ONNXRUNTIME
