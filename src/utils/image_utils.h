#pragma once
#include <torch/torch.h>
#include <string>

namespace myimg {

// Image utilities
torch::Tensor load_image(const std::string& path);
void save_image(const torch::Tensor& image, const std::string& path);

} // namespace myimg
