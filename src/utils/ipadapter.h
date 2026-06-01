#pragma once

#include <string>
#include <torch/torch.h>
#include "utils/image_utils.h"

namespace myimg {

// IPAdapter: 图像提示词
// 通过 CLIP Vision 提取图像特征，注入到扩散模型的注意力中
struct IPAdapterConfig {
    std::string model_path;          // IPAdapter 模型路径 (.safetensors/.bin)
    std::string clip_vision_path;    // CLIP Vision 模型路径
    std::string image_path;          // 参考图像路径
    float weight = 1.0f;             // 注入权重 (0.0-1.0)
    float start_at = 0.0f;           // 开始注入的步数比例 (0.0-1.0)
    float end_at = 1.0f;             // 结束注入的步数比例 (0.0-1.0)
    bool faceid = false;             // 是否使用 FaceID 模式
};

class IPAdapter {
public:
    IPAdapter() = default;
    explicit IPAdapter(const IPAdapterConfig& config);
    
    // 加载模型
    bool load_model(const std::string& model_path, const std::string& clip_vision_path);
    
    // 加载参考图像
    bool load_reference_image(const std::string& image_path);
    
    // 提取图像特征
    torch::Tensor extract_image_features(const torch::Tensor& image);
    
    // 注入注意力（在采样过程中调用）
    // latent: 当前 latent
    // step: 当前步数
    // total_steps: 总步数
    torch::Tensor apply_ipadapter(const torch::Tensor& latent, int step, int total_steps);
    
    // 是否已加载
    bool is_loaded() const { return model_loaded_ && clip_vision_loaded_; }
    
    const IPAdapterConfig& config() const { return config_; }
    
private:
    IPAdapterConfig config_;
    bool model_loaded_ = false;
    bool clip_vision_loaded_ = false;
    torch::Tensor image_features_;   // 缓存的图像特征
    
    // CLIP Vision 推理
    torch::Tensor clip_vision_encode(const torch::Tensor& image);
    
    // IPAdapter 注意力注入
    torch::Tensor inject_attention(const torch::Tensor& latent, const torch::Tensor& image_features);
};

} // namespace myimg
