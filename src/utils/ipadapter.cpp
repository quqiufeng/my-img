#include "utils/ipadapter.h"
#include "utils/log.h"

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace myimg {

// ============================================================
// PIMPL: ONNX Runtime 实现细节
// ============================================================
struct IPAdapter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "IPAdapter"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> clip_session;
    std::unique_ptr<Ort::Session> ipa_session;
    std::unique_ptr<Ort::Session> proj_session;  // 768→2560 线性投影

    // CLIP Vision 输入输出名
    std::string clip_input_name;
    std::string clip_output_name;

    // IPAdapter MLP 输入输出名
    std::string ipa_input_name;
    std::string ipa_output_name;

    // 投影层输入输出名
    std::string proj_input_name;
    std::string proj_output_name;

    Impl() {
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    ~Impl() = default;

    // 禁止拷贝
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    bool load_clip_vision(const std::string& path) {
        try {
            clip_session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);

            // 获取输入输出名
            auto input_count = clip_session->GetInputCount();
            auto output_count = clip_session->GetOutputCount();

            if (input_count < 1 || output_count < 1) {
                LOG_ERROR("IPAdapter: CLIP Vision has no inputs/outputs");
                return false;
            }

            auto input_name_ptr = clip_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            clip_input_name = input_name_ptr.get();

            auto output_name_ptr = clip_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            clip_output_name = output_name_ptr.get();

            LOG_INFO("IPAdapter: CLIP Vision ONNX model loaded: %s", path.c_str());
            LOG_INFO("  Input: %s", clip_input_name.c_str());
            LOG_INFO("  Output: %s", clip_output_name.c_str());

            // 获取输入 shape 信息
            auto input_type_info = clip_session->GetInputTypeInfo(0);
            auto input_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
            std::string shape_str = "[";
            for (size_t i = 0; i < input_shape.size(); i++) {
                if (i > 0) shape_str += ", ";
                if (input_shape[i] == -1) {
                    shape_str += "N";
                } else {
                    shape_str += std::to_string(input_shape[i]);
                }
            }
            shape_str += "]";
            LOG_INFO("  Input shape: %s", shape_str.c_str());

            return true;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: Failed to load CLIP Vision: %s", e.what());
            return false;
        }
    }

    bool load_ipadapter(const std::string& path) {
        try {
            ipa_session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);

            auto input_name_ptr = ipa_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            ipa_input_name = input_name_ptr.get();

            auto output_name_ptr = ipa_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            ipa_output_name = output_name_ptr.get();

            LOG_INFO("IPAdapter: IPAdapter MLP ONNX model loaded: %s", path.c_str());
            LOG_INFO("  Input: %s", ipa_input_name.c_str());
            LOG_INFO("  Output: %s", ipa_output_name.c_str());

            return true;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: Failed to load IPAdapter MLP: %s", e.what());
            return false;
        }
    }

    // 运行 CLIP Vision 推理
    // input: [1, 3, 224, 224] float32 normalized image
    // output: [1, 1024] float32 image embedding
    std::vector<float> run_clip_vision(const std::vector<float>& input_image) {
        if (!clip_session) {
            LOG_ERROR("IPAdapter: CLIP Vision session not loaded");
            return {};
        }

        try {
            // 构建 input tensor
            std::vector<int64_t> input_shape = {1, 3, 224, 224};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(input_image.data()),
                input_image.size(), input_shape.data(), input_shape.size());

            // 运行推理
            const char* input_names[] = {clip_input_name.c_str()};
            const char* output_names[] = {clip_output_name.c_str()};

            auto output_tensors = clip_session->Run(Ort::RunOptions{nullptr},
                                                     input_names, &input_tensor, 1,
                                                     output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                LOG_ERROR("IPAdapter: CLIP Vision inference returned no output");
                return {};
            }

            // 读取输出
            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto output_shape = output_info.GetShape();
            size_t num_elements = output_info.GetElementCount();

            if (num_elements == 0) {
                LOG_ERROR("IPAdapter: CLIP Vision output is empty");
                return {};
            }

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            LOG_INFO("IPAdapter: CLIP Vision output shape [%zu x %zu]",
                     output_shape.size() >= 1 ? (size_t)output_shape[0] : 0,
                     output_shape.size() >= 2 ? (size_t)output_shape[1] : num_elements);

            return result;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: CLIP Vision inference failed: %s", e.what());
            return {};
        }
    }

    bool load_projection(const std::string& path) {
        try {
            proj_session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);

            auto input_name_ptr = proj_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            proj_input_name = input_name_ptr.get();

            auto output_name_ptr = proj_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            proj_output_name = output_name_ptr.get();

            LOG_INFO("IPAdapter: Projection ONNX model loaded: %s", path.c_str());
            LOG_INFO("  Input: %s", proj_input_name.c_str());
            LOG_INFO("  Output: %s", proj_output_name.c_str());

            return true;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: Failed to load projection: %s", e.what());
            return false;
        }
    }

    // 运行投影层 768→2560
    // input: [1, 768] float32 IPAdapter output
    // output: [1, 2560] float32 projected to Z-Image context space
    std::vector<float> run_projection(const std::vector<float>& ipa_tokens) {
        if (!proj_session) {
            LOG_WARN("IPAdapter: projection session not loaded, using zero-pad");
            // Fall back to zero-padding: first 768 dims = tokens, rest = 0
            std::vector<float> result(2560, 0.0f);
            size_t copy_n = std::min(ipa_tokens.size(), (size_t)768);
            std::copy(ipa_tokens.begin(), ipa_tokens.begin() + copy_n, result.begin());
            return result;
        }

        try {
            std::vector<int64_t> input_shape = {1, 768};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            size_t input_size = std::min(ipa_tokens.size(), (size_t)768);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(ipa_tokens.data()),
                input_size, input_shape.data(), input_shape.size());

            const char* input_names[] = {proj_input_name.c_str()};
            const char* output_names[] = {proj_output_name.c_str()};

            auto output_tensors = proj_session->Run(Ort::RunOptions{nullptr},
                                                     input_names, &input_tensor, 1,
                                                     output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                LOG_ERROR("IPAdapter: Projection inference returned no output");
                return {};
            }

            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t num_elements = output_info.GetElementCount();
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            LOG_INFO("IPAdapter: Projection output size: %zu floats (768→2560)", num_elements);
            return result;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: Projection inference failed: %s", e.what());
            return {};
        }
    }

    // 运行 IPAdapter MLP 推理
    // input: [1, 1024] float32 CLIP image embedding
    // output: [1, 768] float32 image tokens
    std::vector<float> run_ipadapter(const std::vector<float>& clip_embedding) {
        if (!ipa_session) {
            LOG_ERROR("IPAdapter: IPAdapter session not loaded");
            return {};
        }

        try {
            // 构建 input tensor
            std::vector<int64_t> input_shape = {1, 1024};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(clip_embedding.data()),
                clip_embedding.size(), input_shape.data(), input_shape.size());

            // 运行推理
            const char* input_names[] = {ipa_input_name.c_str()};
            const char* output_names[] = {ipa_output_name.c_str()};

            auto output_tensors = ipa_session->Run(Ort::RunOptions{nullptr},
                                                    input_names, &input_tensor, 1,
                                                    output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                LOG_ERROR("IPAdapter: IPAdapter MLP inference returned no output");
                return {};
            }

            // 读取输出
            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t num_elements = output_info.GetElementCount();

            if (num_elements == 0) {
                LOG_ERROR("IPAdapter: IPAdapter MLP output is empty");
                return {};
            }

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            LOG_INFO("IPAdapter: IPAdapter MLP output size: %zu floats", num_elements);

            return result;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: IPAdapter MLP inference failed: %s", e.what());
            return {};
        }
    }
};

// ============================================================
// IPAdapter 公共接口实现
// ============================================================

IPAdapter::IPAdapter()
    : impl_(std::make_unique<Impl>()) {
}

IPAdapter::IPAdapter(const IPAdapterConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>()) {
    if (!config_.model_path.empty() && !config_.clip_vision_path.empty()) {
        load_model(config_.model_path, config_.clip_vision_path);
    }
    if (!config_.image_path.empty()) {
        load_reference_image(config_.image_path);
    }
}

IPAdapter::~IPAdapter() = default;

IPAdapter::IPAdapter(IPAdapter&&) noexcept = default;
IPAdapter& IPAdapter::operator=(IPAdapter&&) noexcept = default;

bool IPAdapter::load_model(const std::string& model_path, const std::string& clip_vision_path) {
    LOG_INFO("IPAdapter: loading models...");
    LOG_INFO("  IPAdapter MLP: %s", model_path.c_str());
    LOG_INFO("  CLIP Vision: %s", clip_vision_path.c_str());

    // 先加载 CLIP Vision（更大，2.4GB）
    bool clip_ok = impl_->load_clip_vision(clip_vision_path);

    // 再加载 IPAdapter MLP（较小，5.4MB）
    bool ipa_ok = impl_->load_ipadapter(model_path);

    // 可选：加载线性投影层 768→2560
    if (!config_.projection_path.empty()) {
        LOG_INFO("  Projection: %s", config_.projection_path.c_str());
        impl_->load_projection(config_.projection_path);
    } else {
        LOG_INFO("  Projection: not provided, using zero-pad fallback");
    }

    model_loaded_ = clip_ok && ipa_ok;

    if (!model_loaded_) {
        LOG_WARN("IPAdapter: model loading %s",
                 clip_ok ? "partially failed (IPAdapter MLP)" :
                 ipa_ok ? "partially failed (CLIP Vision)" :
                          "failed (both models)");
    } else {
        LOG_INFO("IPAdapter: all models loaded successfully");
    }

    return model_loaded_;
}

bool IPAdapter::load_reference_image(const std::string& image_path) {
    LOG_INFO("IPAdapter: processing reference image: %s", image_path.c_str());

    if (!model_loaded_) {
        LOG_WARN("IPAdapter: models not loaded, cannot process image");
        return false;
    }

    // 1. 加载图像 (OpenCV)
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        LOG_ERROR("IPAdapter: failed to load image: %s", image_path.c_str());
        return false;
    }
    LOG_INFO("IPAdapter: loaded image %dx%d", img.cols, img.rows);

    // 2. 预处理: BGR→RGB, resize 224x224, float32, normalize
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    // Convert to float32 and normalize
    // CLIP normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_val[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> input_data(3 * 224 * 224);
    for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            // CHW format
            input_data[0 * 224 * 224 + y * 224 + x] = (pixel[0] / 255.0f - mean[0]) / std_val[0];  // R
            input_data[1 * 224 * 224 + y * 224 + x] = (pixel[1] / 255.0f - mean[1]) / std_val[1];  // G
            input_data[2 * 224 * 224 + y * 224 + x] = (pixel[2] / 255.0f - mean[2]) / std_val[2];  // B
        }
    }

    LOG_INFO("IPAdapter: preprocessed image to [1, 3, 224, 224] float32 CHW");

    // 3. 运行 CLIP Vision 推理
    auto clip_embedding = impl_->run_clip_vision(input_data);
    if (clip_embedding.empty()) {
        LOG_ERROR("IPAdapter: CLIP Vision inference failed");
        return false;
    }
    LOG_INFO("IPAdapter: CLIP Vision embedding size: %zu", clip_embedding.size());

    // 验证 shape 正确
    if (clip_embedding.size() != 1024) {
        LOG_WARN("IPAdapter: CLIP Vision output size %zu, expected 1024",
                 clip_embedding.size());
        // 仍然继续，让 IPAdapter MLP 决定
    }

    // 4. 运行 IPAdapter MLP 推理
    auto raw_tokens = impl_->run_ipadapter(clip_embedding);
    if (raw_tokens.empty()) {
        LOG_ERROR("IPAdapter: IPAdapter MLP inference failed");
        return false;
    }
    LOG_INFO("IPAdapter: IPAdapter MLP output size: %zu (768-dim)", raw_tokens.size());

    // 5. 运行线性投影层 768→2560
    auto projected = impl_->run_projection(raw_tokens);
    if (projected.empty()) {
        LOG_ERROR("IPAdapter: projection failed");
        return false;
    }

    image_tokens_ = std::move(projected);
    LOG_INFO("IPAdapter: projected tokens size: %zu (2560-dim, Z-Image context space)",
             image_tokens_.size());

    // 打印统计信息
    float min_val = *std::min_element(image_tokens_.begin(), image_tokens_.end());
    float max_val = *std::max_element(image_tokens_.begin(), image_tokens_.end());
    float mean_val = std::accumulate(image_tokens_.begin(), image_tokens_.end(), 0.0f) / image_tokens_.size();
    LOG_INFO("IPAdapter: projected tokens stats: min=%.4f, max=%.4f, mean=%.4f",
             min_val, max_val, mean_val);

    return true;
}

} // namespace myimg
