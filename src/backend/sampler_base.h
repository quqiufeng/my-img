#pragma once
#include "backend/model.h"
#include <string>

namespace myimg {

class Sampler {
public:
    virtual ~Sampler() = default;
    virtual std::string get_name() const = 0;
    
    virtual torch::Tensor sample(
        UNetModel* model,
        torch::Tensor x,
        torch::Tensor positive_cond,
        torch::Tensor negative_cond,
        int steps,
        float cfg_scale
    ) = 0;
    
protected:
    torch::Tensor get_sigmas(int steps, const std::string& scheduler = "normal");
};

} // namespace myimg
