#pragma once

#include <torch/torch.h>

namespace myimg {

// Temperature: -1.0 (cold/blue) to 1.0 (warm/yellow)
torch::Tensor adjust_temperature(const torch::Tensor& image, float temperature);

// Brightness: -1.0 to 1.0
torch::Tensor adjust_brightness(const torch::Tensor& image, float brightness);

// Contrast: -1.0 to 1.0, 0 = no change
torch::Tensor adjust_contrast(const torch::Tensor& image, float contrast);

// Saturation: -1.0 to 1.0
torch::Tensor adjust_saturation(const torch::Tensor& image, float saturation);

// Exposure: EV -5.0 to 5.0
torch::Tensor adjust_exposure(const torch::Tensor& image, float ev);

// Highlights: -100 to 100
torch::Tensor adjust_highlights(const torch::Tensor& image, float highlights);

// Shadows: -100 to 100
torch::Tensor adjust_shadows(const torch::Tensor& image, float shadows);

// Auto enhance (one-click fix)
torch::Tensor auto_enhance(const torch::Tensor& image);

// USM Sharpening
// amount: 0.0-3.0 (strength)
// radius: 1-5 (blur radius in pixels)
// threshold: 0-255 (only sharpen pixels that differ by more than threshold)
torch::Tensor usm_sharpen(const torch::Tensor& image, float amount, int radius, float threshold);

// Basic denoising (gaussian blur based)
// strength: 0.0-1.0
torch::Tensor denoise(const torch::Tensor& image, float strength);

// Smart denoise: preserve edges while reducing noise
// strength: 0.0-1.0
torch::Tensor smart_denoise(const torch::Tensor& image, float strength);

} // namespace myimg
