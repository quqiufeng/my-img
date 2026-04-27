#include "utils/dehaze.h"
#include <cmath>
#include <iostream>

namespace myimg {

torch::Tensor dehaze(const torch::Tensor& image, float strength) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    int h = img.size(1);
    int w = img.size(2);
    
    // Step 1: Compute dark channel
    // Dark channel = min over color channels
    auto min_vals = torch::min(img, 0); // Returns tuple (values, indices)
    auto min_channel = std::get<0>(min_vals); // [H, W]
    
    // Apply min filter (approximate with avg_pool)
    int patch_size = 15;
    int pad = patch_size / 2;
    auto min_pooled = torch::nn::functional::avg_pool2d(
        min_channel.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::AvgPool2dFuncOptions(patch_size).stride(1).padding(pad)
    ).squeeze();
    
    // Step 2: Estimate atmospheric light
    // Top 0.1% brightest pixels in dark channel
    auto flat_dark = min_pooled.flatten();
    int num_pixels = flat_dark.numel();
    int top_k = std::max(1, static_cast<int>(num_pixels * 0.001f));
    auto top_vals = torch::topk(flat_dark, top_k);
    auto top_values = std::get<0>(top_vals);
    float A = top_values.mean().item().toFloat();
    A = std::min(A, 0.95f); // Cap atmospheric light
    
    // Step 3: Estimate transmission
    auto transmission = 1.0f - strength * min_pooled / A;
    transmission = transmission.clamp(0.1f, 1.0f);
    
    // Step 4: Refine transmission with guided filter (simplified)
    // Apply box filter for smoothing
    transmission = torch::nn::functional::avg_pool2d(
        transmission.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::AvgPool2dFuncOptions(31).stride(1).padding(15)
    ).squeeze();
    
    // Step 5: Recover scene radiance
    // J = (I - A) / t + A
    auto t = transmission.unsqueeze(0); // [1, H, W]
    auto result = (img - A) / t + A;
    
    // Auto contrast adjustment for dehazed image
    result = result.clamp(0, 1);
    
    // Slight saturation boost to compensate for haze removal
    auto gray = result.mean(0, true);
    result = gray + (result - gray) * 1.1f;
    
    return result.clamp(0, 1);
}

} // namespace myimg
