#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>

namespace myimg {

// Base model class
class Model {
public:
    virtual ~Model() = default;
    
    virtual bool load(const std::string& path) = 0;
    void to(torch::Device device) { device_ = device; }
    torch::Device get_device() const { return device_; }
    
protected:
    torch::Device device_ = torch::kCPU;
};

// UNet/Flux diffusion model
class UNetModel : public Model {
public:
    bool load(const std::string& path) override;
    
    torch::Tensor forward(
        torch::Tensor sample,
        torch::Tensor timestep,
        torch::Tensor encoder_hidden_states,
        std::optional<torch::Tensor> cross_attention_kwargs = std::nullopt
    );
};

// VAE model
class VAEModel : public Model {
public:
    bool load(const std::string& path) override;
    
    torch::Tensor encode(torch::Tensor image);
    torch::Tensor decode(torch::Tensor latent);
};

// CLIP text encoder
class CLIPModel : public Model {
public:
    bool load(const std::string& path) override;
    
    torch::Tensor encode_text(const std::string& text);
    std::vector<int> tokenize(const std::string& text);
};

} // namespace myimg
