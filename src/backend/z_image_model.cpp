#include "backend/z_image_model.h"
#include "utils/gguf_loader.h"
#include <gguf.h>
#include <cmath>
#include <iostream>

namespace myimg {

// ============================================================
// RMSNorm
// ============================================================
RMSNormImpl::RMSNormImpl(int64_t dim, float eps) : eps_(eps) {
    weight_ = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
    auto input_dtype = x.scalar_type();
    x = x.to(torch::kFloat32);
    auto variance = x.pow(2).mean(-1, true);
    x = x * torch::rsqrt(variance + eps_);
    return (weight_ * x).to(input_dtype);
}

// ============================================================
// TimestepEmbedder
// ============================================================
TimestepEmbedderImpl::TimestepEmbedderImpl(int64_t hidden_size, int64_t freq_dim, int64_t max_period)
    : hidden_size_(hidden_size), freq_dim_(freq_dim), max_period_(max_period) {
    mlp_0_ = register_module("mlp_0", torch::nn::Linear(freq_dim, hidden_size));
    mlp_2_ = register_module("mlp_2", torch::nn::Linear(hidden_size, hidden_size));
}

torch::Tensor TimestepEmbedderImpl::forward(torch::Tensor timesteps) {
    // timesteps: [N]
    auto device = timesteps.device();
    auto dtype = timesteps.dtype();
    
    // 生成频率
    auto half_dim = freq_dim_ / 2;
    auto freqs = torch::exp(
        -torch::log(torch::tensor(static_cast<float>(max_period_))) *
        torch::arange(0, half_dim, torch::dtype(dtype).device(device)) / half_dim
    );
    
    // 计算正弦和余弦编码
    auto args = timesteps.unsqueeze(1) * freqs.unsqueeze(0);  // [N, half_dim]
    auto embedding = torch::cat({torch::cos(args), torch::sin(args)}, -1);  // [N, freq_dim]
    
    // MLP
    embedding = torch::silu(mlp_0_->forward(embedding));
    embedding = mlp_2_->forward(embedding);
    
    return embedding;  // [N, hidden_size]
}

// ============================================================
// JointAttention
// ============================================================
JointAttentionImpl::JointAttentionImpl(int64_t hidden_size, int64_t head_dim, int64_t num_heads, int64_t num_kv_heads, bool qk_norm)
    : head_dim_(head_dim), num_heads_(num_heads), num_kv_heads_(num_kv_heads), qk_norm_(qk_norm) {
    int64_t qkv_dim = (num_heads + num_kv_heads * 2) * head_dim;
    qkv_proj_ = register_module("qkv", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, qkv_dim).bias(false)));
    out_proj_ = register_module("out", torch::nn::Linear(torch::nn::LinearOptions(num_heads * head_dim, hidden_size).bias(false)));
    
    if (qk_norm_) {
        q_norm_ = register_module("q_norm", RMSNorm(head_dim));
        k_norm_ = register_module("k_norm", RMSNorm(head_dim));
    }
}

torch::Tensor JointAttentionImpl::forward(torch::Tensor x, torch::Tensor pe, torch::optional<torch::Tensor> mask) {
    // x: [N, n_token, hidden_size]
    // pe: [pos_len, d_head/2, 2, 2] - RoPE位置编码
    auto N = x.size(0);
    auto n_token = x.size(1);
    
    // QKV投影
    auto qkv = qkv_proj_->forward(x);  // [N, n_token, (num_heads + num_kv_heads*2)*head_dim]
    qkv = qkv.reshape({N, n_token, num_heads_ + num_kv_heads_ * 2, head_dim_});  // [N, n_token, num_qkv_heads, head_dim]
    
    // 分割Q, K, V
    auto q = qkv.slice(2, 0, num_heads_);  // [N, n_token, num_heads, head_dim]
    auto k = qkv.slice(2, num_heads_, num_heads_ + num_kv_heads_);  // [N, n_token, num_kv_heads, head_dim]
    auto v = qkv.slice(2, num_heads_ + num_kv_heads_, num_heads_ + num_kv_heads_ * 2);  // [N, n_token, num_kv_heads, head_dim]
    
    // QK Norm
    if (qk_norm_) {
        auto q_norm_module = dynamic_cast<RMSNormImpl*>(q_norm_.ptr().get());
        auto k_norm_module = dynamic_cast<RMSNormImpl*>(k_norm_.ptr().get());
        q = q_norm_module->forward(q);
        k = k_norm_module->forward(k);
    }
    
    // 应用RoPE位置编码
    // pe: [pos_len, d_head/2, 2, 2]
    auto pos_len = pe.size(0);
    auto d_half = pe.size(1);
    
    // 重塑pe为复数形式
    pe = pe.reshape({pos_len, d_half, 4});  // [pos_len, d_half, 4]
    
    // 分割为实部和虚部
    auto cos_pe = pe.slice(2, 0, 2).mean(2);  // [pos_len, d_half]
    auto sin_pe = pe.slice(2, 2, 4).mean(2);  // [pos_len, d_half]
    
    // 应用旋转编码
    auto q_half = q.reshape({N, n_token, q.size(2), d_half, 2});  // [N, n_token, num_heads, d_half, 2]
    auto q_real = q_half.slice(-1, 0, 1).squeeze(-1);  // [N, n_token, num_heads, d_half]
    auto q_imag = q_half.slice(-1, 1, 2).squeeze(-1);  // [N, n_token, num_heads, d_half]
    
    auto q_rotated_real = q_real * cos_pe.unsqueeze(0).unsqueeze(2) - q_imag * sin_pe.unsqueeze(0).unsqueeze(2);
    auto q_rotated_imag = q_real * sin_pe.unsqueeze(0).unsqueeze(2) + q_imag * cos_pe.unsqueeze(0).unsqueeze(2);
    
    q = torch::stack({q_rotated_real, q_rotated_imag}, -1).flatten(-2);  // [N, n_token, num_heads, head_dim]
    
    auto k_half = k.reshape({N, n_token, k.size(2), d_half, 2});
    auto k_real = k_half.slice(-1, 0, 1).squeeze(-1);
    auto k_imag = k_half.slice(-1, 1, 2).squeeze(-1);
    
    auto k_rotated_real = k_real * cos_pe.unsqueeze(0).unsqueeze(2) - k_imag * sin_pe.unsqueeze(0).unsqueeze(2);
    auto k_rotated_imag = k_real * sin_pe.unsqueeze(0).unsqueeze(2) + k_imag * cos_pe.unsqueeze(0).unsqueeze(2);
    
    k = torch::stack({k_rotated_real, k_rotated_imag}, -1).flatten(-2);
    
    // 注意力计算
    q = q.transpose(1, 2);  // [N, num_heads, n_token, head_dim]
    k = k.transpose(1, 2);  // [N, num_kv_heads, n_token, head_dim]
    v = v.transpose(1, 2);  // [N, num_kv_heads, n_token, head_dim]
    
    // 重复K,V以匹配Q的头数（GQA）
    if (num_kv_heads_ < num_heads_) {
        auto repeat_factor = num_heads_ / num_kv_heads_;
        k = k.repeat_interleave(repeat_factor, 1);
        v = v.repeat_interleave(repeat_factor, 1);
    }
    
    // 使用torch的scaled_dot_product_attention（内存高效）
    auto attn_output = at::scaled_dot_product_attention(
        q, k, v, mask, 0.0, false, 1.0f / 128.0f
    );  // [N, num_heads, n_token, head_dim]
    
    attn_output = attn_output.transpose(1, 2).reshape({N, n_token, num_heads_ * head_dim_});
    attn_output = out_proj_->forward(attn_output);
    
    return attn_output;
}

// ============================================================
// FeedForward (SwiGLU)
// ============================================================
FeedForwardImpl::FeedForwardImpl(int64_t dim, int64_t hidden_dim, int64_t multiple_of, float ffn_dim_multiplier) {
    if (ffn_dim_multiplier > 0.f) {
        hidden_dim = static_cast<int64_t>(ffn_dim_multiplier * hidden_dim);
    }
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
    
    w1_ = register_module("w1", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
    w2_ = register_module("w2", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim).bias(false)));
    w3_ = register_module("w3", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
    auto x1 = w1_->forward(x);
    auto x3 = w3_->forward(x);
    // SwiGLU: silu(x1) * x3
    x = torch::silu(x1) * x3;
    x = w2_->forward(x);
    return x;
}

// ============================================================
// JointTransformerBlock
// ============================================================
JointTransformerBlockImpl::JointTransformerBlockImpl(int layer_id, int64_t hidden_size, int64_t head_dim, 
                                                       int64_t num_heads, int64_t num_kv_heads, int64_t multiple_of,
                                                       float ffn_dim_multiplier, float norm_eps, bool qk_norm, bool modulation)
    : modulation_(modulation) {
    attention_ = register_module("attention", JointAttention(hidden_size, head_dim, num_heads, num_kv_heads, qk_norm));
    feed_forward_ = register_module("feed_forward", FeedForward(hidden_size, hidden_size, multiple_of, ffn_dim_multiplier));
    attention_norm1_ = register_module("attention_norm1", RMSNorm(hidden_size, norm_eps));
    ffn_norm1_ = register_module("ffn_norm1", RMSNorm(hidden_size, norm_eps));
    attention_norm2_ = register_module("attention_norm2", RMSNorm(hidden_size, norm_eps));
    ffn_norm2_ = register_module("ffn_norm2", RMSNorm(hidden_size, norm_eps));
    
    if (modulation_) {
        adaLN_modulation_0_ = register_module("adaLN_modulation_0", 
            torch::nn::Linear(std::min(hidden_size, static_cast<int64_t>(256)), 4 * hidden_size));
    }
}

torch::Tensor JointTransformerBlockImpl::forward(torch::Tensor x, torch::Tensor pe,
                                                   torch::optional<torch::Tensor> mask,
                                                   torch::optional<torch::Tensor> adaln_input) {
    if (modulation_) {
        TORCH_CHECK(adaln_input.has_value(), "adaln_input is required when modulation is enabled");
        auto m = adaLN_modulation_0_->forward(torch::silu(adaln_input.value()));  // [N, 4 * hidden_size]
        
        auto chunks = m.chunk(4, -1);
        auto scale_msa = chunks[0];
        auto gate_msa = chunks[1];
        auto scale_mlp = chunks[2];
        auto gate_mlp = chunks[3];
        
        auto residual = x;
        x = attention_norm1_->forward(x);
        // modulate
        x = x * (1 + scale_msa.unsqueeze(1));
        x = attention_->forward(x, pe, mask);
        x = attention_norm2_->forward(x);
        x = x * torch::tanh(gate_msa.unsqueeze(1));
        x = residual + x;
        
        residual = x;
        x = ffn_norm1_->forward(x);
        x = x * (1 + scale_mlp.unsqueeze(1));
        x = feed_forward_->forward(x);
        x = ffn_norm2_->forward(x);
        x = x * torch::tanh(gate_mlp.unsqueeze(1));
        x = residual + x;
    } else {
        auto residual = x;
        x = attention_norm1_->forward(x);
        x = attention_->forward(x, pe, mask);
        x = attention_norm2_->forward(x);
        x = residual + x;
        
        residual = x;
        x = ffn_norm1_->forward(x);
        x = feed_forward_->forward(x);
        x = ffn_norm2_->forward(x);
        x = residual + x;
    }
    
    return x;
}

// ============================================================
// FinalLayer
// ============================================================
FinalLayerImpl::FinalLayerImpl(int64_t hidden_size, int64_t patch_size, int64_t out_channels) {
    norm_final_ = register_module("norm_final", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({hidden_size}).eps(1e-6).elementwise_affine(false)));
    linear_ = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, patch_size * patch_size * out_channels).bias(true)));
    adaLN_modulation_1_ = register_module("adaLN_modulation_1", 
        torch::nn::Linear(std::min(hidden_size, static_cast<int64_t>(256)), hidden_size));
}

torch::Tensor FinalLayerImpl::forward(torch::Tensor x, torch::Tensor c) {
    // x: [N, n_token, hidden_size]
    // c: [N, hidden_size]
    auto scale = adaLN_modulation_1_->forward(torch::silu(c));  // [N, hidden_size]
    x = norm_final_->forward(x);
    x = x * (1 + scale.unsqueeze(1));
    x = linear_->forward(x);
    return x;
}

// ============================================================
// RoPE Position Encoding
// ============================================================
torch::Tensor RoPE::gen_z_image_pe(int H, int W, int patch_size, int N, int n_txt_token, int seq_multi_of,
                                    int theta, const std::vector<int>& axes_dim) {
    // 计算图像token数
    int n_img_token = (H / patch_size) * (W / patch_size);
    
    // padding到seq_multi_of的倍数
    auto bound_mod = [](int n, int m) {
        return (m - (n % m)) % m;
    };
    
    int n_txt_pad = bound_mod(n_txt_token, seq_multi_of);
    int n_img_pad = bound_mod(n_img_token, seq_multi_of);
    
    int total_len = n_txt_token + n_txt_pad + n_img_token + n_img_pad;
    
    // 生成3D位置编码 (参考stable-diffusion.cpp的Rope实现)
    // 简化实现：使用2D位置编码
    std::vector<float> pe_data;
    pe_data.reserve(total_len * 128 * 2);  // axes_dim_sum = 128, 每个位置2个值
    
    // 这里简化处理，使用正弦位置编码
    // 实际应该使用stable-diffusion.cpp中的Rope::gen_z_image_pe算法
    for (int pos = 0; pos < total_len; pos++) {
        for (int d = 0; d < 128; d++) {
            float freq = 1.0f / std::pow(theta, static_cast<float>(d) / 64.0f);
            float angle = pos * freq;
            pe_data.push_back(std::cos(angle));
            pe_data.push_back(std::sin(angle));
        }
    }
    
    return torch::from_blob(pe_data.data(), {total_len, 64, 2, 2}, torch::kFloat32).clone();
}

// ============================================================
// ZImageDiT
// ============================================================
ZImageDiTImpl::ZImageDiTImpl(ZImageParams params) : params_(params) {
    std::cout << "[ZImageDiT] Starting model creation..." << std::endl;
    
    x_embedder_ = register_module("x_embedder", 
        torch::nn::Linear(params.patch_size * params.patch_size * params.in_channels, params.hidden_size));
    std::cout << "[ZImageDiT] x_embedder created" << std::endl;
    
    t_embedder_ = register_module("t_embedder", 
        TimestepEmbedder(std::min(params.hidden_size, static_cast<int64_t>(1024)), 256, 256));
    std::cout << "[ZImageDiT] t_embedder created" << std::endl;
    
    cap_embedder_0_ = register_module("cap_embedder_0", RMSNorm(params.cap_feat_dim, params.norm_eps));
    cap_embedder_1_ = register_module("cap_embedder_1", 
        torch::nn::Linear(params.cap_feat_dim, params.hidden_size));
    std::cout << "[ZImageDiT] cap_embedder created" << std::endl;
    
    // Context refiner blocks (带modulation=false)
    std::cout << "[ZImageDiT] Creating context_refiner blocks..." << std::endl;
    for (int i = 0; i < params.num_refiner_layers; i++) {
        auto block = JointTransformerBlock(i, params.hidden_size, params.head_dim, params.num_heads,
                                           params.num_kv_heads, params.multiple_of, params.ffn_dim_multiplier,
                                           params.norm_eps, params.qk_norm, false);
        context_refiner_blocks_.push_back(register_module("context_refiner_" + std::to_string(i), block));
    }
    std::cout << "[ZImageDiT] context_refiner blocks created" << std::endl;
    
    // Noise refiner blocks (带modulation=true)
    std::cout << "[ZImageDiT] Creating noise_refiner blocks..." << std::endl;
    for (int i = 0; i < params.num_refiner_layers; i++) {
        auto block = JointTransformerBlock(i, params.hidden_size, params.head_dim, params.num_heads,
                                           params.num_kv_heads, params.multiple_of, params.ffn_dim_multiplier,
                                           params.norm_eps, params.qk_norm, true);
        noise_refiner_blocks_.push_back(register_module("noise_refiner_" + std::to_string(i), block));
    }
    std::cout << "[ZImageDiT] noise_refiner blocks created" << std::endl;
    
    // Main layers
    std::cout << "[ZImageDiT] Creating main layers (" << params.num_layers << ")..." << std::endl;
    for (int i = 0; i < params.num_layers; i++) {
        if (i % 5 == 0) {
            std::cout << "[ZImageDiT]   Layer " << i << "/" << params.num_layers << std::endl;
        }
        auto block = JointTransformerBlock(i, params.hidden_size, params.head_dim, params.num_heads,
                                           params.num_kv_heads, params.multiple_of, params.ffn_dim_multiplier,
                                           params.norm_eps, params.qk_norm, true);
        layers_.push_back(register_module("layers_" + std::to_string(i), block));
    }
    std::cout << "[ZImageDiT] main layers created" << std::endl;
    
    final_layer_ = register_module("final_layer", 
        FinalLayer(params.hidden_size, params.patch_size, params.out_channels));
    std::cout << "[ZImageDiT] final_layer created" << std::endl;
    
    // Pad tokens
    cap_pad_token_ = register_parameter("cap_pad_token", torch::zeros({params.hidden_size}));
    x_pad_token_ = register_parameter("x_pad_token", torch::zeros({params.hidden_size}));
    std::cout << "[ZImageDiT] Model creation completed!" << std::endl;
}

int64_t ZImageDiTImpl::bound_mod(int64_t n, int64_t m) {
    return (m - (n % m)) % m;
}

torch::Tensor ZImageDiTImpl::patchify(torch::Tensor x) {
    // x: [N, C, H, W]
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto p = params_.patch_size;
    
    // 使用unfold提取patch
    x = x.unfold(2, p, p).unfold(3, p, p);  // [N, C, H/p, W/p, p, p]
    x = x.permute({0, 2, 3, 1, 4, 5}).contiguous();  // [N, H/p, W/p, C, p, p]
    x = x.reshape({N, (H/p) * (W/p), C * p * p});  // [N, n_token, C*p*p]
    
    return x;
}

torch::Tensor ZImageDiTImpl::unpatchify(torch::Tensor x, int H, int W) {
    // x: [N, n_token, C*p*p]
    auto N = x.size(0);
    auto C = params_.out_channels;
    auto p = params_.patch_size;
    auto h = H / p;
    auto w = W / p;
    
    x = x.reshape({N, h, w, C, p, p});
    x = x.permute({0, 3, 1, 4, 2, 5}).contiguous();  // [N, C, h, p, w, p]
    x = x.reshape({N, C, H, W});
    
    return x;
}

torch::Tensor ZImageDiTImpl::forward(torch::Tensor x, torch::Tensor timestep, torch::Tensor context) {
    // x: [N, C, H, W] - latent
    // timestep: [N] - 时间步
    // context: [N, L, D] - 文本条件
    // return: [N, C, H, W] - 预测
    
    auto N = x.size(0);
    auto H = x.size(2);
    auto W = x.size(3);
    auto n_txt_token = context.size(1);
    
    // 时间步嵌入
    auto t_emb = t_embedder_->forward(timestep);  // [N, hidden_size]
    
    // 文本编码
    auto txt = cap_embedder_0_->forward(context);  // [N, L, cap_feat_dim]
    txt = cap_embedder_1_->forward(txt);  // [N, L, hidden_size]
    
    // 图像patch化
    auto img = patchify(x);  // [N, n_img_token, C*p*p]
    img = x_embedder_->forward(img);  // [N, n_img_token, hidden_size]
    auto n_img_token = img.size(1);
    
    // Padding
    auto n_txt_pad = bound_mod(n_txt_token, 32);  // SEQ_MULTI_OF = 32
    if (n_txt_pad > 0) {
        auto txt_pad = cap_pad_token_.unsqueeze(0).unsqueeze(0).expand({N, n_txt_pad, -1});
        txt = torch::cat({txt, txt_pad}, 1);
    }
    
    auto n_img_pad = bound_mod(n_img_token, 32);
    if (n_img_pad > 0) {
        auto img_pad = x_pad_token_.unsqueeze(0).unsqueeze(0).expand({N, n_img_pad, -1});
        img = torch::cat({img, img_pad}, 1);
    }
    
    // 生成位置编码
    auto pe = RoPE::gen_z_image_pe(H, W, params_.patch_size, static_cast<int>(N), 
                                    static_cast<int>(n_txt_token + n_txt_pad), 32,
                                    params_.theta, params_.axes_dim);
    pe = pe.to(x.device());
    
    // 分割PE为文本和图像部分
    auto txt_pe = pe.slice(0, 0, n_txt_token + n_txt_pad);
    auto img_pe = pe.slice(0, n_txt_token + n_txt_pad, n_txt_token + n_txt_pad + n_img_token + n_img_pad);
    auto full_pe = pe;  // 用于主layers
    
    // Context refiner blocks
    for (auto& block : context_refiner_blocks_) {
        txt = block->forward(txt, txt_pe, torch::nullopt, torch::nullopt);
    }
    
    // Noise refiner blocks
    for (auto& block : noise_refiner_blocks_) {
        img = block->forward(img, img_pe, torch::nullopt, t_emb);
    }
    
    // 拼接文本和图像
    auto txt_img = torch::cat({txt, img}, 1);  // [N, total_len, hidden_size]
    
    // Main layers
    for (auto& block : layers_) {
        txt_img = block->forward(txt_img, full_pe, torch::nullopt, t_emb);
    }
    
    // Final layer
    txt_img = final_layer_->forward(txt_img, t_emb);  // [N, total_len, p*p*C]
    
    // 提取图像部分
    img = txt_img.slice(1, n_txt_token + n_txt_pad, n_txt_token + n_txt_pad + n_img_token);
    
    // Unpatchify
    img = unpatchify(img, H, W);  // [N, C, H, W]
    
    // 缩放（参考stable-diffusion.cpp的ggml_ext_scale with -1.f）
    img = img * -1.0f;
    
    return img;
}

bool ZImageDiTImpl::load_weights(const std::map<std::string, torch::Tensor>& weights) {
    try {
        auto model_params = this->named_parameters();
        auto model_buffers = this->named_buffers();
        
        int loaded = 0;
        int total = 0;
        
        for (const auto& [name, tensor] : weights) {
            total++;
            
            // 转换GGUF名称到libtorch名称
            std::string torch_name = name;
            
            // 移除前缀
            if (torch_name.find("model.diffusion_model.") == 0) {
                torch_name = torch_name.substr(22);  // 移除 "model.diffusion_model."
            }
            
            // 替换名称映射（参考name_conversion.cpp）
            // 转换Z-Image特定名称
            if (torch_name.find("all_x_embedder.2-1.") == 0) {
                torch_name = "x_embedder." + torch_name.substr(19);
            } else if (torch_name.find("all_final_layer.2-1.") == 0) {
                torch_name = "final_layer." + torch_name.substr(20);
            }
            
            // 在参数中查找
            if (model_params.contains(torch_name)) {
                auto param = model_params[torch_name];
                if (param.sizes() == tensor.sizes()) {
                    param.copy_(tensor);
                    loaded++;
                } else {
                    std::cerr << "[ZImage] Shape mismatch for " << torch_name 
                              << ": expected " << param.sizes() 
                              << ", got " << tensor.sizes() << std::endl;
                }
            } else if (model_buffers.contains(torch_name)) {
                auto buffer = model_buffers[torch_name];
                if (buffer.sizes() == tensor.sizes()) {
                    buffer.copy_(tensor);
                    loaded++;
                } else {
                    std::cerr << "[ZImage] Shape mismatch for buffer " << torch_name << std::endl;
                }
            } else {
                // 尝试其他名称变体
                bool found = false;
                for (const auto& item : model_params) {
                    const auto& param_name = item.key();
                    const auto& param = item.value();
                    if (param_name.find(torch_name) != std::string::npos || 
                        torch_name.find(param_name) != std::string::npos) {
                        if (param.sizes() == tensor.sizes()) {
                            param.copy_(tensor);
                            loaded++;
                            found = true;
                            break;
                        }
                    }
                }
                if (!found) {
                    std::cerr << "[ZImage] Weight not found: " << name << " (converted: " << torch_name << ")" << std::endl;
                }
            }
        }
        
        std::cout << "[ZImage] Loaded " << loaded << "/" << total << " weights" << std::endl;
        return loaded > 0;
    } catch (const std::exception& e) {
        std::cerr << "[ZImage] Failed to load weights: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================
// ZImageModel (Model接口包装)
// ============================================================
ZImageModel::ZImageModel() {
    dit_ = std::make_shared<ZImageDiTImpl>();
}

bool ZImageModel::load(const std::string& path) {
    std::cout << "[ZImageModel] Loading from: " << path << std::endl;
    
    if (path.find(".gguf") != std::string::npos) {
        auto tensors = GGUFLoder::load(path);
        if (tensors.empty()) {
            std::cerr << "[ZImageModel] Failed to load GGUF" << std::endl;
            return false;
        }
        
        loaded_ = dit_->load_weights(tensors);
        if (loaded_) {
            std::cout << "[ZImageModel] Model loaded successfully" << std::endl;
        }
        return loaded_;
    }
    
    std::cerr << "[ZImageModel] Unsupported format: " << path << std::endl;
    return false;
}

torch::Tensor ZImageModel::forward(
    torch::Tensor sample,
    torch::Tensor timestep,
    torch::Tensor encoder_hidden_states,
    std::optional<torch::Tensor> cross_attention_kwargs) {
    if (!loaded_) {
        std::cerr << "[ZImageModel] Model not loaded!" << std::endl;
        return sample;
    }
    
    return dit_->forward(sample, timestep, encoder_hidden_states);
}

} // namespace myimg
