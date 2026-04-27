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

// Outpainting: expand canvas and create mask
// top/bottom/left/right: pixels to expand in each direction
// Returns: {expanded_image, mask} where mask white = generate, black = keep
std::pair<ImageData, ImageData> create_outpaint_canvas(
    const ImageData& original,
    int top, int bottom, int left, int right
);

// Image transformations using libtorch
// Resize image using specified interpolation
// mode: "nearest", "bilinear", "bicubic"
ImageData resize_image(const ImageData& img, int new_width, int new_height, const std::string& mode = "bilinear");

// Crop image (x, y, width, height)
ImageData crop_image(const ImageData& img, int x, int y, int w, int h);

// Flip image (horizontal or vertical)
ImageData flip_image(const ImageData& img, bool horizontal);

// Rotate image by 90/180/270 degrees
ImageData rotate_image(const ImageData& img, int degrees); // 90, 180, 270

// Convert image format (just reorders channels if needed)
// Currently supports RGB <-> BGR conversion
ImageData convert_channels(const ImageData& img, const std::string& from_to);

} // namespace myimg
