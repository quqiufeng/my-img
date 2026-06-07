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
    std::unique_ptr<Ort::Session> sdxl_plus_session;  // SDXL IPAdapter Plus (1024→16×2048)
    std::unique_ptr<Ort::Session> sdxl_proj_session;  // 2048→2560 投影

    // CLIP Vision 输入输出名
    std::string clip_input_name;
    std::string clip_output_name;

    // IPAdapter MLP 输入输出名
    std::string ipa_input_name;
    std::string ipa_output_name;

    // 投影层输入输出名
    std::string proj_input_name;
    std::string proj_output_name;

    // SDXL Plus 输入输出名
    std::string sdxl_plus_input_name;
    std::string sdxl_plus_output_name;

    // SDXL 投影输入输出名
    std::string sdxl_proj_input_name;
    std::string sdxl_proj_output_name;

    bool use_sdxl_plus = false;  // 是否使用 SDXL Plus 模式

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

    bool load_sdxl_projection(const std::string& path) {
        try {
            sdxl_proj_session = std::make_unique<Ort::Session>(env, path.c_str(), session_options);

            auto input_name_ptr = sdxl_proj_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            sdxl_proj_input_name = input_name_ptr.get();

            auto output_name_ptr = sdxl_proj_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
            sdxl_proj_output_name = output_name_ptr.get();

            LOG_INFO("IPAdapter: SDXL projection (2048→2560) loaded: %s", path.c_str());
            LOG_INFO("  Input: %s", sdxl_proj_input_name.c_str());
            LOG_INFO("  Output: %s", sdxl_proj_output_name.c_str());

            return true;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: Failed to load SDXL projection: %s", e.what());
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
    std::vector<float> run_sdxl_plus(const std::vector<float>& clip_embedding) {
        if (!sdxl_plus_session) {
            LOG_ERROR("IPAdapter: SDXL Plus session not loaded");
            return {};
        }

        try {
            // Input: [1, 257, 1280] - CLIP hidden states (ViT-H/14 patch tokens)
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

    std::vector<float> run_sdxl_projection(const std::vector<float>& sdxl_tokens) {
        if (!sdxl_proj_session) {
            LOG_DEBUG("IPAdapter: SDXL projection not loaded, using zero-pad fallback");
            return {};
        }

        try {
            // Input: [1, 16, 2048]
            std::vector<int64_t> input_shape = {1, 16, 2048};
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(sdxl_tokens.data()),
                sdxl_tokens.size(), input_shape.data(), input_shape.size());

            const char* input_names[] = {sdxl_proj_input_name.c_str()};
            const char* output_names[] = {sdxl_proj_output_name.c_str()};

            auto output_tensors = sdxl_proj_session->Run(Ort::RunOptions{nullptr},
                                                          input_names, &input_tensor, 1,
                                                          output_names, 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                LOG_ERROR("IPAdapter: SDXL projection returned no output");
                return {};
            }

            auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t num_elements = output_info.GetElementCount();

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            std::vector<float> result(output_data, output_data + num_elements);

            LOG_INFO("IPAdapter: SDXL projection output: %zu floats (16x2560)", num_elements);
            return result;
        } catch (const Ort::Exception& e) {
            LOG_ERROR("IPAdapter: SDXL projection inference failed: %s", e.what());
            return {};
        }
    }

    std::vector<float> zero_pad_2048_to_2560(const std::vector<float>& sdxl_tokens) {
        // Fallback: zero-pad each 2048-dim token to 2560-dim
        // sdxl_tokens: [1, 16, 2048] = 32768 floats
        std::vector<float> result;
        result.reserve(16 * 2560);

        for (size_t t = 0; t < 16; t++) {
            for (size_t i = 0; i < 2048; i++) {
                result.push_back(sdxl_tokens[t * 2048 + i]);
            }
            for (size_t i = 0; i < 512; i++) {
                result.push_back(0.0f);  // zero padding
            }
        }

        LOG_INFO("IPAdapter: zero-padded %zu tokens from 2048 to 2560", 16);
        return result;
    }

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
    LOG_INFO("  IPAdapter: %s", model_path.c_str());
    LOG_INFO("  CLIP Vision: %s", clip_vision_path.c_str());

    // 先加载 CLIP Vision（更大，2.4GB）
    bool clip_ok = impl_->load_clip_vision(clip_vision_path);
    if (!clip_ok) {
        LOG_WARN("IPAdapter: CLIP Vision loading failed");
        model_loaded_ = false;
        return false;
    }

    // 检测模型类型：SDXL Plus 或 SD1.5
    bool is_sdxl_plus = (model_path.find("sdxl") != std::string::npos) ||
                        (model_path.find("plus") != std::string::npos) ||
                        (model_path.find("2048") != std::string::npos);

    bool ipa_ok = false;
    if (is_sdxl_plus) {
        impl_->use_sdxl_plus = true;
        LOG_INFO("  Mode: SDXL IPAdapter Plus (16 tokens, 2048-dim)");
        ipa_ok = impl_->load_sdxl_plus(model_path);

        // SDXL Plus 需要 2048→2560 投影
        if (!config_.projection_path.empty()) {
            LOG_INFO("  Projection: %s", config_.projection_path.c_str());
            impl_->load_sdxl_projection(config_.projection_path);
        } else {
            LOG_INFO("  Projection: not provided, using zero-pad fallback (2048→2560)");
        }
    } else {
        impl_->use_sdxl_plus = false;
        LOG_INFO("  Mode: SD1.5 IPAdapter (1 token, 768-dim)");
        ipa_ok = impl_->load_ipadapter(model_path);

        // SD1.5 可选 768→2560 投影
        if (!config_.projection_path.empty()) {
            LOG_INFO("  Projection: %s", config_.projection_path.c_str());
            impl_->load_projection(config_.projection_path);
        } else {
            LOG_INFO("  Projection: not provided, using zero-pad fallback");
        }
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

    // 验证 shape: pooled=[1024], hidden_states=[257*1280=329760]
    if (impl_->use_sdxl_plus && clip_embedding.size() != 257 * 1280) {
        LOG_WARN("IPAdapter: CLIP Vision output size %zu, expected %d for SDXL Plus hidden states",
                 clip_embedding.size(), 257 * 1280);
    } else if (!impl_->use_sdxl_plus && clip_embedding.size() != 1024) {
        LOG_WARN("IPAdapter: CLIP Vision output size %zu, expected 1024 for SD1.5",
                 clip_embedding.size());
    }

    // 4. 运行 IPAdapter 推理
    std::vector<float> final_tokens;
    size_t num_tokens = 0;

    if (impl_->use_sdxl_plus) {
        // SDXL Plus 模式: 1024 → 16×2048 → 16×2560
        auto sdxl_tokens = impl_->run_sdxl_plus(clip_embedding);
        if (sdxl_tokens.empty()) {
            LOG_ERROR("IPAdapter: SDXL Plus inference failed");
            return false;
        }
        LOG_INFO("IPAdapter: SDXL Plus output size: %zu (16×2048)", sdxl_tokens.size());

        // 运行 2048→2560 投影
        auto projected = impl_->run_sdxl_projection(sdxl_tokens);
        if (projected.empty()) {
            LOG_ERROR("IPAdapter: SDXL projection failed, using zero-pad fallback");
            // Fallback: zero-pad 2048→2560
            projected = impl_->zero_pad_2048_to_2560(sdxl_tokens);
        }

        final_tokens = std::move(projected);
        num_tokens = 16;
        LOG_INFO("IPAdapter: SDXL projected tokens size: %zu (%zu×2560)",
                 final_tokens.size(), num_tokens);
    } else {
        // SD1.5 模式: 1024 → 768 → 2560
        auto raw_tokens = impl_->run_ipadapter(clip_embedding);
        if (raw_tokens.empty()) {
            LOG_ERROR("IPAdapter: IPAdapter MLP inference failed");
            return false;
        }
        LOG_INFO("IPAdapter: IPAdapter MLP output size: %zu (768-dim)", raw_tokens.size());

        // 运行线性投影层 768→2560
        auto projected = impl_->run_projection(raw_tokens);
        if (projected.empty()) {
            LOG_ERROR("IPAdapter: projection failed");
            return false;
        }

        final_tokens = std::move(projected);
        num_tokens = 1;
        LOG_INFO("IPAdapter: projected tokens size: %zu (2560-dim)",
                 final_tokens.size());
    }

    image_tokens_ = std::move(final_tokens);
    num_tokens_ = static_cast<int>(num_tokens);

    // 打印统计信息
    float min_val = *std::min_element(image_tokens_.begin(), image_tokens_.end());
    float max_val = *std::max_element(image_tokens_.begin(), image_tokens_.end());
    float mean_val = std::accumulate(image_tokens_.begin(), image_tokens_.end(), 0.0f) / image_tokens_.size();
    LOG_INFO("IPAdapter: projected tokens stats: min=%.4f, max=%.4f, mean=%.4f",
             min_val, max_val, mean_val);

    return true;
}

} // namespace myimg
