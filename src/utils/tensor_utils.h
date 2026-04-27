#pragma once
#include <torch/torch.h>
#include <string>

namespace myimg {

// Tensor utilities
void print_tensor_info(const torch::Tensor& tensor);
torch::Tensor load_tensor_from_file(const std::string& path);

} // namespace myimg
