#include "utils/tensor_utils.h"
#include <iostream>

namespace myimg {

void print_tensor_info(const torch::Tensor& tensor) {
    std::cout << "Tensor shape: ";
    for (auto s : tensor.sizes()) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    std::cout << "Device: " << tensor.device() << std::endl;
}

torch::Tensor load_tensor_from_file(const std::string& path) {
    // TODO: Implement
    return torch::zeros({1});
}

} // namespace myimg
