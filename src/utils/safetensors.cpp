#include "utils/safetensors.h"
#include <iostream>

namespace myimg {

std::map<std::string, torch::Tensor> SafetensorsLoader::load(const std::string& path) {
    std::map<std::string, torch::Tensor> tensors;
    std::cout << "[Safetensors] Loading: " << path << std::endl;
    // TODO: Implement actual safetensors loading
    return tensors;
}

} // namespace myimg
