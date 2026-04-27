#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <optional>
#include "backend/model.h"

namespace myimg {

// Z-Image 模型参数
struct ZImageParams {
    int patch_size             = 2;
    int64_t hidden_size        = 3840;
    int64_t in_channels        = 16;
    int64_t out_channels       = 16;
    int64_t num_layers         = 30;
    int64_t num_refiner_layers = 2;
    int64_t head_dim           = 128;
    int64_t num_heads          = 30;
    int64_t num_kv_heads       = 30;
    int64_t multiple_of        = 256;
    float ffn_dim_multiplier   = 8.0f / 3.0f;
    float norm_eps             = 1e-5f;
    bool qk_norm               = true;
    int64_t cap_feat_dim       = 2560;
    int theta                  = 256;
    std::vector<int> axes_dim  = {32, 48, 48};
    int64_t axes_dim_sum       = 128;
};

// RMSNorm 层
class RMSNormImpl : public torch::nn::Module {
public:
    RMSNormImpl(int64_t dim, float eps = 1e-5f);
    torch::Tensor forward(torch::Tensor x);
private:
    float eps_;
    torch::Tensor weight_;
};
TORCH_MODULE(RMSNorm);

// 时间步嵌入器
class TimestepEmbedderImpl : public torch::nn::Module {
public:
    TimestepEmbedderImpl(int64_t hidden_size, int64_t freq_dim = 256, int64_t max_period = 256);
    torch::Tensor forward(torch::Tensor timesteps);
private:
    int64_t hidden_size_;
    int64_t freq_dim_;
    int64_t max_period_;
    torch::nn::Linear mlp_0_{nullptr};
    torch::nn::Linear mlp_2_{nullptr};
};
TORCH_MODULE(TimestepEmbedder);

// 联合注意力层（Joint Attention）
class JointAttentionImpl : public torch::nn::Module {
public:
    JointAttentionImpl(int64_t hidden_size, int64_t head_dim, int64_t num_heads, int64_t num_kv_heads, bool qk_norm);
    torch::Tensor forward(torch::Tensor x, torch::Tensor pe, torch::optional<torch::Tensor> mask = torch::nullopt);
private:
    int64_t head_dim_;
    int64_t num_heads_;
    int64_t num_kv_heads_;
    bool qk_norm_;
    
    torch::nn::Linear qkv_proj_{nullptr};
    torch::nn::Linear out_proj_{nullptr};
    RMSNorm q_norm_{nullptr};
    RMSNorm k_norm_{nullptr};
};
TORCH_MODULE(JointAttention);

// FeedForward（SwiGLU）
class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(int64_t dim, int64_t hidden_dim, int64_t multiple_of, float ffn_dim_multiplier = 0.f);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Linear w1_{nullptr};
    torch::nn::Linear w2_{nullptr};
    torch::nn::Linear w3_{nullptr};
};
TORCH_MODULE(FeedForward);

// Joint Transformer Block
class JointTransformerBlockImpl : public torch::nn::Module {
public:
    JointTransformerBlockImpl(int layer_id, int64_t hidden_size, int64_t head_dim, int64_t num_heads, 
                               int64_t num_kv_heads, int64_t multiple_of, float ffn_dim_multiplier,
                               float norm_eps, bool qk_norm, bool modulation = true);
    torch::Tensor forward(torch::Tensor x, torch::Tensor pe, 
                          torch::optional<torch::Tensor> mask = torch::nullopt,
                          torch::optional<torch::Tensor> adaln_input = torch::nullopt);
private:
    bool modulation_;
    JointAttention attention_{nullptr};
    FeedForward feed_forward_{nullptr};
    RMSNorm attention_norm1_{nullptr};
    RMSNorm ffn_norm1_{nullptr};
    RMSNorm attention_norm2_{nullptr};
    RMSNorm ffn_norm2_{nullptr};
    torch::nn::Linear adaLN_modulation_0_{nullptr};
};
TORCH_MODULE(JointTransformerBlock);

// Final Layer
class FinalLayerImpl : public torch::nn::Module {
public:
    FinalLayerImpl(int64_t hidden_size, int64_t patch_size, int64_t out_channels);
    torch::Tensor forward(torch::Tensor x, torch::Tensor c);
private:
    torch::nn::Linear linear_{nullptr};
    torch::nn::Linear adaLN_modulation_1_{nullptr};
    torch::nn::LayerNorm norm_final_{nullptr};
};
TORCH_MODULE(FinalLayer);

// RoPE 位置编码工具
class RoPE {
public:
    static torch::Tensor gen_z_image_pe(int H, int W, int patch_size, int N, int n_txt_token, int seq_multi_of,
                                        int theta, const std::vector<int>& axes_dim);
};

// ZImage DiT 模型
class ZImageDiTImpl : public torch::nn::Module {
public:
    ZImageDiTImpl(ZImageParams params = ZImageParams{});
    
    // 加载权重
    bool load_weights(const std::map<std::string, torch::Tensor>& weights);
    
    // 前向传播
    torch::Tensor forward(torch::Tensor x, torch::Tensor timestep, torch::Tensor context);
    
    // 获取模型参数
    const ZImageParams& get_params() const { return params_; }

private:
    ZImageParams params_;
    
    torch::nn::Linear x_embedder_{nullptr};
    TimestepEmbedder t_embedder_{nullptr};
    RMSNorm cap_embedder_0_{nullptr};
    torch::nn::Linear cap_embedder_1_{nullptr};
    
    std::vector<JointTransformerBlock> context_refiner_blocks_;
    std::vector<JointTransformerBlock> noise_refiner_blocks_;
    std::vector<JointTransformerBlock> layers_;
    
    FinalLayer final_layer_{nullptr};
    
    torch::Tensor cap_pad_token_;
    torch::Tensor x_pad_token_;
    
    torch::Tensor patchify(torch::Tensor x);
    torch::Tensor unpatchify(torch::Tensor x, int H, int W);
    int64_t bound_mod(int64_t n, int64_t m);
};
TORCH_MODULE(ZImageDiT);

// ZImage 模型包装类（兼容 Model 接口）
class ZImageModel : public Model {
public:
    ZImageModel();
    
    bool load(const std::string& path) override;
    
    torch::Tensor forward(
        torch::Tensor sample,
        torch::Tensor timestep,
        torch::Tensor encoder_hidden_states,
        std::optional<torch::Tensor> cross_attention_kwargs = std::nullopt
    );
    
    ZImageDiT get_dit() { return dit_; }

private:
    ZImageDiT dit_{nullptr};
    bool loaded_ = false;
};

} // namespace myimg
