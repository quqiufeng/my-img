#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include "utils/image_utils.h"

namespace myimg {

// 前向声明
struct GenerationParams;

// PhotoMaker: 个性化图像生成
// 通过多张参考图像学习人物特征，生成保持人物一致性的图像
struct PhotoMakerConfig {
    std::string model_path;           // PhotoMaker 模型路径
    std::vector<std::string> id_images; // ID 参考图像路径列表
    float id_weight = 1.0f;           // ID 权重 (0.0-1.0)
    float style_weight = 0.5f;        // 风格权重
    bool use_stack = true;            // 是否堆叠多张 ID 图像
    int stack_images = 1;             // 堆叠的图像数量
};

class PhotoMaker {
public:
    PhotoMaker() = default;
    explicit PhotoMaker(const PhotoMakerConfig& config);
    
    // 加载模型
    bool load_model(const std::string& model_path);
    
    // 加载 ID 图像
    bool load_id_images(const std::vector<std::string>& image_paths);
    
    // 提取 ID 特征
    torch::Tensor extract_id_features(const std::vector<torch::Tensor>& id_images);
    
    // 注入到文本编码（将 ID 特征注入到 prompt 编码中）
    // 返回修改后的文本嵌入
    torch::Tensor inject_into_text_embedding(const torch::Tensor& text_embedding,
                                              const torch::Tensor& id_features);
    
    // 应用 PhotoMaker（在生成过程中调用）
    // 此函数在适配器层中调用，修改 GenerationParams
    bool apply(GenerationParams& params);
    
    bool is_loaded() const { return model_loaded_; }
    bool has_id_images() const { return id_features_.defined() && id_features_.numel() > 0; }
    const PhotoMakerConfig& config() const { return config_; }
    
private:
    PhotoMakerConfig config_;
    bool model_loaded_ = false;
    torch::Tensor id_features_;       // 聚合的 ID 特征
    std::vector<torch::Tensor> id_image_tensors_; // 加载的 ID 图像
    
    // 编码单张 ID 图像
    torch::Tensor encode_id_image(const torch::Tensor& image);
    
    // 聚合多张 ID 图像特征
    torch::Tensor aggregate_id_features(const std::vector<torch::Tensor>& features);
};

} // namespace myimg
