#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include "utils/image_utils.h"

namespace myimg {

// 风格迁移配置
struct StyleTransferConfig {
    std::string style_model_path;    // 风格模型路径 (ONNX)
    std::string style_image_path;    // 风格参考图路径
    float strength = 1.0f;           // 风格强度 (0.0-2.0)
    int style_block = 1;             // 注入层: 0=early, 1=mid, 2=late
    bool content_preserve = true;    // 是否保留内容结构
};

// 风格迁移: 提取风格特征并注入到扩散模型
class StyleTransfer {
public:
    StyleTransfer() = default;
    explicit StyleTransfer(const StyleTransferConfig& config);
    
    // 加载风格模型
    bool load_model(const std::string& model_path);
    
    // 加载风格参考图
    bool load_style_image(const std::string& image_path);
    
    // 提取风格特征
    torch::Tensor extract_style_features(const torch::Tensor& image);
    
    // 应用风格迁移
    torch::Tensor apply_style_transfer(const torch::Tensor& content_latent, 
                                       int step, int total_steps);
    
    // 是否已加载
    bool is_loaded() const { return model_loaded_ && style_features_.defined(); }
    
    const StyleTransferConfig& config() const { return config_; }
    
private:
    StyleTransferConfig config_;
    bool model_loaded_ = false;
    torch::Tensor style_features_;   // 缓存的风格特征
    
    // 风格特征提取 (Gram Matrix)
    torch::Tensor compute_gram_matrix(const torch::Tensor& features);
    
    // AdaIN 风格化
    torch::Tensor adain(const torch::Tensor& content, const torch::Tensor& style);
};

} // namespace myimg
