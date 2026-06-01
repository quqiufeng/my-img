#pragma once

#include <string>
#include <torch/torch.h>

namespace myimg {

// T2I-Adapter: 轻量级条件控制
// 比 ControlNet 更省显存，适合 Sketch/Keypose/Segmentation
struct T2IAdapterConfig {
    std::string model_path;     // T2I-Adapter 模型路径
    std::string condition_image; // 条件图像路径（sketch/keypose/seg）
    float strength = 1.0f;      // 控制强度 (0.0-1.0)
    int start_step = 0;         // 开始控制的步数
    int end_step = -1;          // 结束控制的步数 (-1 = 到最后)
};

class T2IAdapter {
public:
    T2IAdapter() = default;
    explicit T2IAdapter(const T2IAdapterConfig& config);
    
    // 加载模型
    bool load_model(const std::string& model_path);
    
    // 加载条件图像
    bool load_condition_image(const std::string& image_path);
    
    // 提取条件特征
    torch::Tensor extract_features(const torch::Tensor& condition_image);
    
    // 应用控制（在采样过程中调用）
    torch::Tensor apply(const torch::Tensor& latent, int step, int total_steps);
    
    bool is_loaded() const { return model_loaded_; }
    const T2IAdapterConfig& config() const { return config_; }
    
private:
    T2IAdapterConfig config_;
    bool model_loaded_ = false;
    torch::Tensor condition_features_;
    
    // T2I-Adapter 特征提取
    torch::Tensor extract_t2i_features(const torch::Tensor& image);
    
    // 注入到 UNet
    torch::Tensor inject_into_unet(const torch::Tensor& latent, const torch::Tensor& features);
};

} // namespace myimg
