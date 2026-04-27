#pragma once
#include <torch/torch.h>
#include <string>
#include <map>

namespace myimg {

class GGUFLoder {
public:
    static std::map<std::string, torch::Tensor> load(const std::string& path);
};

} // namespace myimg
