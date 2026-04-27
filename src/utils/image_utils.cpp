#include "utils/image_utils.h"
#include <stb_image.h>
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

ImageData load_image_from_file(const std::string& path) {
    ImageData img;
    int width, height, channels;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::cerr << "[Image] Failed to load image: " << path << std::endl;
        return img;
    }
    
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.data.resize(width * height * 3);
    std::memcpy(img.data.data(), data, width * height * 3);
    
    stbi_image_free(data);
    return img;
}

torch::Tensor image_data_to_tensor(const ImageData& img) {
    auto tensor = torch::from_blob(
        const_cast<uint8_t*>(img.data.data()),
        {img.height, img.width, img.channels},
        torch::kUInt8
    ).clone();
    tensor = tensor.permute({2, 0, 1}).to(torch::kFloat32) / 255.0f;
    return tensor;
}

ImageData tensor_to_image_data(const torch::Tensor& tensor) {
    ImageData img;
    auto t = tensor.detach().cpu().clamp(0, 1);
    img.channels = t.size(0);
    img.height = t.size(1);
    img.width = t.size(2);
    img.data.resize(img.width * img.height * img.channels);
    
    auto t_uint8 = (t * 255.0f).to(torch::kUInt8);
    std::memcpy(img.data.data(), t_uint8.data_ptr<uint8_t>(), img.data.size());
    return img;
}

} // namespace myimg
