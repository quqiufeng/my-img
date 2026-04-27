#pragma once
#include <torch/torch.h>
#include <cstring>
#include <vector>
#include <string>
#include <cstdint>

namespace myimg {

// Simple image structure for I/O
struct ImageData {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;
    
    bool empty() const { return data.empty(); }
    size_t size() const { return width * height * channels; }
};

// Image utilities
torch::Tensor load_image(const std::string& path);
void save_image(const torch::Tensor& image, const std::string& path);

// Load image from file to ImageData struct (for img2img)
ImageData load_image_from_file(const std::string& path);

// Convert ImageData <-> torch::Tensor (CHW, float32, 0-1)
torch::Tensor image_data_to_tensor(const ImageData& img);
ImageData tensor_to_image_data(const torch::Tensor& tensor);

} // namespace myimg
