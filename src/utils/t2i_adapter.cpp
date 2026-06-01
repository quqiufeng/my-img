#include "utils/t2i_adapter.h"
#include "utils/log.h"
#include "utils/image_utils.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>

namespace myimg {

T2IAdapter::T2IAdapter(const T2IAdapterConfig& config) 
    : config_(config) {
    if (!config_.model_path.empty()) {
        load_model(config_.model_path);
    }
    if (!config_.condition_image.empty()) {
        load_condition_image(config_.condition_image);
    }
}

bool T2IAdapter::load_model(const std::string& model_path) {
    LOG_INFO("T2IAdapter: loading model from %s", model_path.c_str());
    
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("T2IAdapter: model not found: %s", model_path.c_str());
        return false;
    }
    
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            LOG_ERROR("T2IAdapter: failed to load ONNX model");
            return false;
        }
        
        // 尝试使用 CUDA 加速
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        
        LOG_INFO("T2IAdapter: ONNX model loaded successfully");
        model_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("T2IAdapter: model loading failed: %s", e.what());
        return false;
    }
}

bool T2IAdapter::load_condition_image(const std::string& image_path) {
    LOG_INFO("T2IAdapter: loading condition image from %s", image_path.c_str());
    
    if (!std::filesystem::exists(image_path)) {
        LOG_ERROR("T2IAdapter: image not found: %s", image_path.c_str());
        return false;
    }
    
    try {
        auto image = load_image(image_path);
        condition_features_ = extract_features(image);
        return condition_features_.defined();
    } catch (const std::exception& e) {
        LOG_ERROR("T2IAdapter: failed to load condition image: %s", e.what());
        return false;
    }
}

torch::Tensor T2IAdapter::extract_features(const torch::Tensor& condition_image) {
    if (!model_loaded_) {
        LOG_WARN("T2IAdapter: model not loaded");
        return torch::Tensor();
    }
    
    try {
        auto img_data = tensor_to_image_data(condition_image);
        cv::Mat img(img_data.height, img_data.width, CV_8UC3, (void*)img_data.data.data());
        
        // 预处理：resize 到 512x512，归一化
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(512, 512));
        
        cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(512, 512),
                                              cv::Scalar(0, 0, 0), false, false);
        
        cv::dnn::Net net = cv::dnn::readNetFromONNX(config_.model_path);
        net.setInput(blob);
        cv::Mat features = net.forward();
        
        // 转换为 torch tensor
        // features shape: [1, C, H, W]
        auto sizes = features.size;
        return torch::from_blob(features.data, {sizes[0], sizes[1], sizes[2], sizes[3]}, torch::kFloat32).clone();
    } catch (const std::exception& e) {
        LOG_ERROR("T2IAdapter: feature extraction failed: %s", e.what());
        return torch::Tensor();
    }
}

torch::Tensor T2IAdapter::apply(const torch::Tensor& latent, int step, int total_steps) {
    if (!model_loaded_ || !condition_features_.defined()) {
        return latent;
    }
    
    // 检查步数范围
    int effective_end = (config_.end_step < 0) ? total_steps : config_.end_step;
    if (step < config_.start_step || step > effective_end) {
        return latent;
    }
    
    // T2I-Adapter 特征注入
    // 简化版：将条件特征与 latent 在通道维度拼接
    try {
        auto scaled_features = condition_features_ * config_.strength;
        
        // 如果尺寸不匹配，进行插值
        if (scaled_features.sizes() != latent.sizes()) {
            scaled_features = torch::nn::functional::interpolate(
                scaled_features,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{latent.size(2), latent.size(3)})
                    .mode(torch::kBilinear)
                    .align_corners(false)
            );
            
            // 调整通道数
            if (scaled_features.size(1) != latent.size(1)) {
                scaled_features = scaled_features.narrow(1, 0, std::min(scaled_features.size(1), latent.size(1)));
                if (scaled_features.size(1) < latent.size(1)) {
                    auto padding = torch::zeros({scaled_features.size(0), 
                                                  latent.size(1) - scaled_features.size(1),
                                                  scaled_features.size(2), 
                                                  scaled_features.size(3)}, 
                                                 scaled_features.options());
                    scaled_features = torch::cat({scaled_features, padding}, 1);
                }
            }
        }
        
        // 简单的特征融合
        return latent + scaled_features * 0.1f;
    } catch (const std::exception& e) {
        LOG_WARN("T2IAdapter: apply failed: %s", e.what());
        return latent;
    }
}

torch::Tensor T2IAdapter::extract_t2i_features(const torch::Tensor& image) {
    (void)image;
    return torch::Tensor();
}

torch::Tensor T2IAdapter::inject_into_unet(const torch::Tensor& latent, const torch::Tensor& features) {
    (void)latent;
    (void)features;
    return torch::Tensor();
}

} // namespace myimg
