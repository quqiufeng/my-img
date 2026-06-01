#include "utils/style_transfer.h"
#include "utils/log.h"
#include "utils/image_utils.h"
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

namespace myimg {

StyleTransfer::StyleTransfer(const StyleTransferConfig& config) 
    : config_(config) {
    if (!config_.style_model_path.empty()) {
        load_model(config_.style_model_path);
    }
    if (!config_.style_image_path.empty()) {
        load_style_image(config_.style_image_path);
    }
}

bool StyleTransfer::load_model(const std::string& model_path) {
    LOG_INFO("StyleTransfer: loading model from %s", model_path.c_str());
    
    try {
        // 加载风格迁移 ONNX 模型（如 AdaIN, InstantStyle 等）
        if (std::filesystem::exists(model_path)) {
            cv::dnn::Net style_net = cv::dnn::readNetFromONNX(model_path);
            if (!style_net.empty()) {
                style_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                style_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                LOG_INFO("StyleTransfer: ONNX model loaded");
                model_loaded_ = true;
                return true;
            }
        }
        
        LOG_WARN("StyleTransfer: model not found at %s", model_path.c_str());
        return false;
        
    } catch (const std::exception& e) {
        LOG_ERROR("StyleTransfer: failed to load model: %s", e.what());
        return false;
    }
}

bool StyleTransfer::load_style_image(const std::string& image_path) {
    LOG_INFO("StyleTransfer: loading style image from %s", image_path.c_str());
    
    try {
        if (!std::filesystem::exists(image_path)) {
            LOG_WARN("StyleTransfer: image not found at %s", image_path.c_str());
            return false;
        }
        
        // 加载图像
        ImageData img = load_image_from_file(image_path);
        if (img.data.empty()) {
            LOG_WARN("StyleTransfer: failed to load image");
            return false;
        }
        
        // 提取风格特征
        torch::Tensor image_tensor = image_data_to_tensor(img).unsqueeze(0);
        style_features_ = extract_style_features(image_tensor);
        
        auto sizes = style_features_.sizes();
        std::string shape_str;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i > 0) shape_str += "x";
            shape_str += std::to_string(sizes[i]);
        }
        LOG_INFO("StyleTransfer: style features extracted, shape: %s", shape_str.c_str());
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("StyleTransfer: failed to load style image: %s", e.what());
        return false;
    }
}

torch::Tensor StyleTransfer::extract_style_features(const torch::Tensor& image) {
    LOG_DEBUG("StyleTransfer: extracting style features");
    
    // 归一化到 [0, 1]
    torch::Tensor normalized = image.to(torch::kFloat32) / 255.0f;
    
    // 使用 VGG 特征提取器提取多层特征
    // 这里使用简单的卷积层模拟 VGG 风格特征提取
    auto conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1));
    auto conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1));
    auto conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1));
    
    torch::Tensor feat1 = torch::relu(conv1->forward(normalized));
    torch::Tensor feat2 = torch::relu(conv2->forward(feat1));
    torch::Tensor feat3 = torch::relu(conv3->forward(feat2));
    
    // 计算各层 Gram Matrix 作为风格特征
    torch::Tensor gram1 = compute_gram_matrix(feat1);
    torch::Tensor gram2 = compute_gram_matrix(feat2);
    torch::Tensor gram3 = compute_gram_matrix(feat3);
    
    // 拼接所有层特征
    torch::Tensor style_features = torch::cat({gram1.flatten(), gram2.flatten(), gram3.flatten()});
    
    return style_features;
}

torch::Tensor StyleTransfer::compute_gram_matrix(const torch::Tensor& features) {
    // features: [B, C, H, W]
    auto b = features.size(0);
    auto c = features.size(1);
    auto h = features.size(2);
    auto w = features.size(3);
    
    // reshape to [B, C, H*W]
    auto features_flat = features.view({b, c, h * w});
    
    // Gram Matrix: [B, C, C]
    auto gram = torch::bmm(features_flat, features_flat.transpose(1, 2));
    
    // normalize by H*W
    gram = gram / (h * w);
    
    return gram;
}

torch::Tensor StyleTransfer::adain(const torch::Tensor& content, const torch::Tensor& style) {
    // Adaptive Instance Normalization
    // content, style: [B, C, H, W]
    
    auto content_mean = content.mean({2, 3}, true);
    auto content_std = content.std({2, 3}, true) + 1e-5;
    
    auto style_mean = style.mean({2, 3}, true);
    auto style_std = style.std({2, 3}, true) + 1e-5;
    
    // normalize content
    auto normalized = (content - content_mean) / content_std;
    
    // scale with style statistics
    auto styled = normalized * style_std + style_mean;
    
    return styled;
}

torch::Tensor StyleTransfer::apply_style_transfer(const torch::Tensor& content_latent, 
                                                int step, int total_steps) {
    if (!style_features_.defined()) {
        LOG_WARN("StyleTransfer: style features not loaded");
        return content_latent;
    }
    
    float progress = static_cast<float>(step) / total_steps;
    
    // 根据配置决定注入强度
    float strength = config_.strength;
    if (progress < 0.2f) {
        // 早期步骤: 降低风格影响以保留结构
        strength *= 0.5f;
    }
    
    // 简单的风格注入: 将风格特征融合到 latent
    // 实际实现需要更复杂的注意力注入机制
    auto style_expanded = style_features_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    auto style_broadcasted = style_expanded.expand_as(content_latent);
    
    // 混合 content 和 style
    torch::Tensor result = content_latent * (1.0f - strength) + style_broadcasted * strength;
    
    return result;
}

} // namespace myimg
