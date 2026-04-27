#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

namespace myimg {

// 3D LUT structure
struct LUT3D {
    int size = 0;  // LUT size (e.g., 33 for 33x33x33)
    std::vector<float> data;  // RGB values, size = size^3 * 3
    std::string title;
    
    bool empty() const { return data.empty(); }
    
    // Load from .cube file
    bool load_from_file(const std::string& path);
    
    // Apply LUT to image tensor (CHW, float32, 0-1)
    torch::Tensor apply(const torch::Tensor& image) const;
};

} // namespace myimg
