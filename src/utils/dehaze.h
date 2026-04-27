#pragma once

#include <torch/torch.h>

namespace myimg {

// Dehaze using Dark Channel Prior
// strength: 0.0-1.0 (dehazing strength)
torch::Tensor dehaze(const torch::Tensor& image, float strength = 0.8f);

} // namespace myimg
