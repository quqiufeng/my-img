#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "utils/image_utils.h"
#include <stb_image.h>
#include <stb_image_write.h>
#include <iostream>

namespace myimg {

torch::Tensor load_image(const std::string& path) {
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return torch::zeros({1, 3, 1, 1});
    }
    
    auto tensor = torch::from_blob(data, {height, width, 3}, torch::kUInt8);
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).to(torch::kFloat32) / 255.0f;
    
    stbi_image_free(data);
    return tensor;
}

void save_image(const torch::Tensor& image, const std::string& path) {
    auto img = image.squeeze().permute({1, 2, 0}).to(torch::kFloat32) * 255.0f;
    img = img.clamp(0, 255).to(torch::kUInt8);
    
    int width = img.size(1);
    int height = img.size(0);
    int channels = img.size(2);
    
    stbi_write_png(path.c_str(), width, height, channels, img.data_ptr<uint8_t>(), width * channels);
}

} // namespace myimg
