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

ImageData resize_image(const ImageData& img, int new_width, int new_height, const std::string& mode) {
    auto tensor = image_data_to_tensor(img);
    
    auto options = torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>({new_height, new_width}));
    
    if (mode == "nearest") {
        options = options.mode(torch::kNearest);
    } else if (mode == "bicubic") {
        options = options.mode(torch::kBicubic).align_corners(true);
    } else {
        options = options.mode(torch::kBilinear).align_corners(true);
    }
    
    auto resized = torch::nn::functional::interpolate(tensor.unsqueeze(0), options).squeeze(0);
    
    return tensor_to_image_data(resized);
}

ImageData crop_image(const ImageData& img, int x, int y, int w, int h) {
    auto tensor = image_data_to_tensor(img);
    auto cropped = tensor.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(y, std::min(y + h, img.height)),
        torch::indexing::Slice(x, std::min(x + w, img.width))
    });
    return tensor_to_image_data(cropped);
}

ImageData flip_image(const ImageData& img, bool horizontal) {
    auto tensor = image_data_to_tensor(img);
    auto flipped = horizontal ? tensor.flip({2}) : tensor.flip({1});
    return tensor_to_image_data(flipped);
}

ImageData rotate_image(const ImageData& img, int degrees) {
    auto tensor = image_data_to_tensor(img);
    auto rotated = tensor;
    
    // k: number of 90-degree counter-clockwise rotations
    int k = ((360 - degrees) / 90) % 4;
    rotated = torch::rot90(tensor, k, {1, 2});
    
    return tensor_to_image_data(rotated);
}

ImageData convert_channels(const ImageData& img, const std::string& from_to) {
    if (from_to == "rgb_to_bgr" || from_to == "bgr_to_rgb") {
        ImageData result = img;
        for (size_t i = 0; i < img.data.size(); i += 3) {
            std::swap(result.data[i], result.data[i + 2]);
        }
        return result;
    }
    return img;
}

std::pair<ImageData, ImageData> create_outpaint_canvas(
    const ImageData& original,
    int top, int bottom, int left, int right
) {
    int new_width = original.width + left + right;
    int new_height = original.height + top + bottom;
    
    ImageData canvas;
    canvas.width = new_width;
    canvas.height = new_height;
    canvas.channels = original.channels;
    canvas.data.assign(new_width * new_height * original.channels, 0);
    
    ImageData mask;
    mask.width = new_width;
    mask.height = new_height;
    mask.channels = 3;
    mask.data.assign(new_width * new_height * 3, 255); // white = generate
    
    // Copy original image to canvas
    for (int y = 0; y < original.height; ++y) {
        for (int x = 0; x < original.width; ++x) {
            int src_idx = (y * original.width + x) * original.channels;
            int dst_idx = ((y + top) * new_width + (x + left)) * original.channels;
            for (int c = 0; c < original.channels; ++c) {
                canvas.data[dst_idx + c] = original.data[src_idx + c];
            }
        }
    }
    
    // Mark original area as black in mask (keep)
    for (int y = 0; y < original.height; ++y) {
        for (int x = 0; x < original.width; ++x) {
            int mask_idx = ((y + top) * new_width + (x + left)) * 3;
            mask.data[mask_idx + 0] = 0;
            mask.data[mask_idx + 1] = 0;
            mask.data[mask_idx + 2] = 0;
        }
    }
    
    return {std::move(canvas), std::move(mask)};
}

} // namespace myimg
