#include "utils/ipadapter.h"
#include "utils/log.h"
#include "utils/image_utils.h"
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace myimg {

IPAdapter::IPAdapter(const IPAdapterConfig& config) 
    : config_(config) {
    if (!config_.model_path.empty() && !config_.clip_vision_path.empty()) {
        load_model(config_.model_path, config_.clip_vision_path);
    }
    if (!config_.image_path.empty()) {
        load_reference_image(config_.image_path);
    }
}

bool IPAdapter::load_model(const std::string& model_path, const std::string& clip_vision_path) {
    LOG_INFO("IPAdapter: loading IPAdapter model from %s", model_path.c_str());
    LOG_INFO("IPAdapter: loading CLIP Vision model from %s", clip_vision_path.c_str());
    
    try {
        // 加载 IPAdapter ONNX 模型
        if (std::filesystem::exists(model_path)) {
            cv::dnn::Net ip_net = cv::dnn::readNetFromONNX(model_path);
            if (!ip_net.empty()) {
                ip_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                ip_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                LOG_INFO("IPAdapter: IPAdapter ONNX model loaded");
                model_loaded_ = true;
            }
        }
        
        // 加载 CLIP Vision ONNX 模型
        if (std::filesystem::exists(clip_vision_path)) {
            cv::dnn::Net clip_net = cv::dnn::readNetFromONNX(clip_vision_path);
            if (!clip_net.empty()) {
                clip_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                clip_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                LOG_INFO("IPAdapter: CLIP Vision ONNX model loaded");
                clip_vision_loaded_ = true;
            }
        }
        
        if (!model_loaded_) {
            LOG_WARN("IPAdapter: model loading failed");
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("IPAdapter: model loading failed: %s", e.what());
        return false;
    }
}

bool IPAdapter::load_reference_image(const std::string& image_path) {
    LOG_INFO("IPAdapter: loading reference image from %s", image_path.c_str());
    
    if (!std::filesystem::exists(image_path)) {
        LOG_ERROR("IPAdapter: image not found: %s", image_path.c_str());
        return false;
    }
    
    try {
        auto image = load_image(image_path);
        image_features_ = extract_image_features(image);
        return image_features_.defined();
    } catch (const std::exception& e) {
        LOG_ERROR("IPAdapter: failed to load reference image: %s", e.what());
        return false;
    }
}

torch::Tensor IPAdapter::extract_image_features(const torch::Tensor& image) {
    if (!clip_vision_loaded_) {
        LOG_WARN("IPAdapter: CLIP Vision model not loaded");
        return torch::Tensor();
    }
    
    try {
        auto img_data = tensor_to_image_data(image);
        cv::Mat img(img_data.height, img_data.width, CV_8UC3, (void*)img_data.data.data());
        
        // CLIP 预处理：Resize 224x224，归一化
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(224, 224));
        cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(224, 224),
                                              cv::Scalar(0.485, 0.456, 0.406), true, false);
        
        cv::dnn::Net clip_net = cv::dnn::readNetFromONNX(config_.clip_vision_path);
        clip_net.setInput(blob);
        cv::Mat features = clip_net.forward();
        
        auto sizes = features.size;
        auto tensor = torch::from_blob(features.data, {sizes[0], sizes[1]}, torch::kFloat32).clone();
        
        LOG_INFO("IPAdapter: extracted image features, shape: [%ld, %ld]", tensor.size(0), tensor.size(1));
        return tensor;
    } catch (const std::exception& e) {
        LOG_ERROR("IPAdapter: feature extraction failed: %s", e.what());
        return torch::Tensor();
    }
}

torch::Tensor IPAdapter::apply_ipadapter(const torch::Tensor& latent, int step, int total_steps) {
    if (!is_loaded() || !image_features_.defined()) {
        return latent;
    }
    
    float progress = static_cast<float>(step) / total_steps;
    
    // 检查是否在注入范围内
    if (progress < config_.start_at || progress > config_.end_at) {
        return latent;
    }
    
    // 计算当前步数的有效权重
    float effective_weight = config_.weight;
    if (progress < config_.start_at + 0.1f) {
        effective_weight *= (progress - config_.start_at) / 0.1f;
    } else if (progress > config_.end_at - 0.1f) {
        effective_weight *= (config_.end_at - progress) / 0.1f;
    }
    
    try {
        // 通过 IPAdapter 模型投影图像特征
        cv::dnn::Net ip_net = cv::dnn::readNetFromONNX(config_.model_path);
        
        // 准备输入
        auto features_cpu = image_features_.to(torch::kCPU);
        cv::Mat img_features(image_features_.size(0), image_features_.size(1), CV_32F, features_cpu.data_ptr<float>());
        cv::Mat blob = cv::dnn::blobFromImage(img_features, 1.0, cv::Size(), cv::Scalar(), false, false);
        
        ip_net.setInput(blob);
        cv::Mat projected = ip_net.forward();
        
        // 转换为 torch tensor
        auto sizes = projected.size;
        auto projected_tensor = torch::from_blob(projected.data, {sizes[0], sizes[1]}, torch::kFloat32).clone();
        
        // 应用到 latent（简化版）
        return inject_attention(latent, projected_tensor * effective_weight);
    } catch (const std::exception& e) {
        LOG_WARN("IPAdapter: apply failed: %s", e.what());
        return latent;
    }
}

torch::Tensor IPAdapter::clip_vision_encode(const torch::Tensor& image) {
    (void)image;
    return torch::Tensor();
}

torch::Tensor IPAdapter::inject_attention(const torch::Tensor& latent, const torch::Tensor& image_features) {
    try {
        // 简化版注意力注入
        // 将图像特征扩展为与 latent 兼容的形状
        auto expanded_features = image_features.unsqueeze(-1).unsqueeze(-1);
        expanded_features = expanded_features.expand({latent.size(0), -1, latent.size(2), latent.size(3)});
        
        // 调整通道数
        if (expanded_features.size(1) != latent.size(1)) {
            auto target_channels = latent.size(1);
            auto current_channels = expanded_features.size(1);
            
            if (current_channels < target_channels) {
                auto padding = torch::zeros({expanded_features.size(0), 
                                            target_channels - current_channels,
                                            expanded_features.size(2), 
                                            expanded_features.size(3)}, 
                                           expanded_features.options());
                expanded_features = torch::cat({expanded_features, padding}, 1);
            } else {
                expanded_features = expanded_features.narrow(1, 0, target_channels);
            }
        }
        
        // 特征融合
        return latent * (1.0f + expanded_features * 0.05f);
    } catch (const std::exception& e) {
        LOG_WARN("IPAdapter: attention injection failed: %s", e.what());
        return latent;
    }
}

} // namespace myimg
