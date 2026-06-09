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
    std::unique_ptr<Ort::Session> sdxl_plus_session;  // SDXL IPAdapter Plus (257x1280 -> 16x2048)

    // CLIP Vision 输入输出名
    std::string clip_input_name;
    std::string clip_output_name;

    // SDXL Plus 输入输出名
    std::string sdxl_plus_input_name;
    std::string sdxl_plus_output_name;

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

    bool load_sdxl_plus(const std::string& path) {
        try {
            sdxl_plus_session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);

            auto input_name_ptr = sdxl_plus_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            sdxl_plus_input_name = input_name_ptr.get();

            auto output_name_ptr = sdxl_plus_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            sdxl_plus_output_name = output_name_ptr.get();

            LOG_INFO("IPAdapter: SDXL Plus ONNX model loaded: %s", path.c_str());
            LOG_INFO("  Input: %s", sdxl_plus_input_name.c_str());
            LOG_INFO("  Output: %s", sdxl_plus_output_name.c_str());

            return true;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: Failed to load SDXL Plus: %s", e.what());
            return false;
        }
    }

    // 运行 CLIP Vision 推理
    // input: [1, 3, 224, 224] float32 normalized image
    // output: [1, 257, 1280] float32 hidden states (ViT-H/14)
    std::vector<float> run_clip_vision(const std::vector<float>& input_image) {
        if (!clip_session) {
            LOG_ERROR("IPAdapter: CLIP Vision session not loaded");
            return {};
        }

        try {
            std::vector<int64_t> input_shape = {1, 3, 224, 224};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(input_image.data()),
                input_image.size(), input_shape.data(), input_shape.size());

            const char* input_names[] = {clip_input_name.c_str()};
            const char* output_names[] = {clip_output_name.c_str()};

            auto output_tensors = clip_session->Run(Ort::RunOptions{nullptr},
                                                     input_names, &input_tensor, 1,
                                                     output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                LOG_ERROR("IPAdapter: CLIP Vision inference returned no output");
                return {};
            }

            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto output_shape = output_info.GetShape();
            size_t num_elements = output_info.GetElementCount();

            if (num_elements == 0) {
                LOG_ERROR("IPAdapter: CLIP Vision output is empty");
                return {};
            }

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            if (output_shape.size() == 3) {
                LOG_INFO("IPAdapter: CLIP Vision output shape [%zu x %zu x %zu] (hidden states)",
                         (size_t)output_shape[0], (size_t)output_shape[1], (size_t)output_shape[2]);
            } else {
                LOG_INFO("IPAdapter: CLIP Vision output shape [%zu x %zu] (pooled embedding)",
                         output_shape.size() >= 1 ? (size_t)output_shape[0] : 0,
                         output_shape.size() >= 2 ? (size_t)output_shape[1] : num_elements);
            }

            return result;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: CLIP Vision inference failed: %s", e.what());
            return {};
        }
    }

    // 运行 SDXL Plus MLP 推理
    // input: [1, 257, 1280] float32 CLIP Vision hidden states
    // output: [1, 16, 2048] float32 image tokens
    std::vector<float> run_sdxl_plus(const std::vector<float>& clip_embedding) {
        if (!sdxl_plus_session) {
            LOG_ERROR("IPAdapter: SDXL Plus session not loaded");
            return {};
        }

        try {
            std::vector<int64_t> input_shape = {1, 257, 1280};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(clip_embedding.data()),
                clip_embedding.size(), input_shape.data(), input_shape.size());

            const char* input_names[] = {sdxl_plus_input_name.c_str()};
            const char* output_names[] = {sdxl_plus_output_name.c_str()};

            auto output_tensors = sdxl_plus_session->Run(Ort::RunOptions{nullptr},
                                                          input_names, &input_tensor, 1,
                                                          output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                LOG_ERROR("IPAdapter: SDXL Plus inference returned no output");
                return {};
            }

            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t num_elements = output_info.GetElementCount();

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            LOG_INFO("IPAdapter: SDXL Plus output: %zu floats (16x2048)", num_elements);
            return result;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: SDXL Plus inference failed: %s", e.what());
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
    LOG_INFO("  IPAdapter: %s", model_path.c_str());
    LOG_INFO("  CLIP Vision: %s", clip_vision_path.c_str());

    bool clip_ok = impl_->load_clip_vision(clip_vision_path);
    if (!clip_ok) {
        LOG_WARN("IPAdapter: CLIP Vision loading failed");
        model_loaded_ = false;
        return false;
    }

    bool ipa_ok = impl_->load_sdxl_plus(model_path);
    if (!ipa_ok) {
        LOG_WARN("IPAdapter: SDXL Plus loading failed");
        model_loaded_ = false;
        return false;
    }

    model_loaded_ = true;
    LOG_INFO("IPAdapter: all models loaded successfully");
    return true;
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

    // 2. 预处理: BGR->RGB, resize 224x224, float32, normalize
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

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

    if (clip_embedding.size() != 257 * 1280) {
        LOG_WARN("IPAdapter: CLIP Vision output size %zu, expected %d for hidden states",
                 clip_embedding.size(), 257 * 1280);
    }

    // 4. 运行 SDXL Plus MLP (257x1280 -> 16x2048)
    auto sdxl_tokens = impl_->run_sdxl_plus(clip_embedding);
    if (sdxl_tokens.empty()) {
        LOG_ERROR("IPAdapter: SDXL Plus inference failed");
        return false;
    }
    LOG_INFO("IPAdapter: SDXL Plus output size: %zu (16x2048)", sdxl_tokens.size());

    image_tokens_ = std::move(sdxl_tokens);
    num_tokens_ = 16;

    // 打印统计信息
    float min_val = *std::min_element(image_tokens_.begin(), image_tokens_.end());
    float max_val = *std::max_element(image_tokens_.begin(), image_tokens_.end());
    float mean_val = std::accumulate(image_tokens_.begin(), image_tokens_.end(), 0.0f) / image_tokens_.size();
    LOG_INFO("IPAdapter: tokens stats: min=%.4f, max=%.4f, mean=%.4f",
             min_val, max_val, mean_val);

    return true;
}

} // namespace myimg
