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

// Vibrance: smart saturation that protects skin tones
// strength: -1.0 to 1.0
torch::Tensor adjust_vibrance(const torch::Tensor& image, float strength);

// Clarity/Texture enhancement: boost mid-frequency details
// strength: 0.0-1.0
torch::Tensor enhance_clarity(const torch::Tensor& image, float strength);

// Split toning: color highlights and shadows separately
// highlight_color, shadow_color: hex strings like "#FFE4C4" or "FF0000"
// strength: 0.0-1.0
torch::Tensor split_tone(const torch::Tensor& image, const std::string& highlight_color, 
                         const std::string& shadow_color, float strength);

// Tint: green/magenta shift
// strength: -1.0 to 1.0 (positive = green, negative = magenta)
torch::Tensor adjust_tint(const torch::Tensor& image, float strength);

// Auto white balance
torch::Tensor auto_white_balance(const torch::Tensor& image);

// Black/White levels adjustment
// blacks: -100 to 100 (negative = crush blacks, positive = lift blacks)
// whites: -100 to 100 (negative = reduce whites, positive = increase whites)
torch::Tensor adjust_levels(const torch::Tensor& image, float blacks, float whites);

// Brightness curves: apply curve to luminance channel only
// curves_str: "in,out;in,out" format (0-255)
torch::Tensor apply_brightness_curves(const torch::Tensor& image, const std::string& curves_str);

// Per-channel RGB curves
// r_curves, g_curves, b_curves: "in,out;in,out" format (0-255)
torch::Tensor apply_channel_curves(const torch::Tensor& image, const std::string& r_curves, 
                                    const std::string& g_curves, const std::string& b_curves);

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

// Edge-mask sharpening: sharpen only non-edge areas to avoid halos
// amount: 0.0-3.0 (strength)
// radius: 1-5 (blur radius)
// threshold: 0-1 (edge detection threshold, lower = more edges excluded)
torch::Tensor edge_mask_sharpen(const torch::Tensor& image, float amount, int radius, float threshold);

// Luminance noise reduction: reduce noise in brightness while preserving edges
// strength: 0.0-1.0
torch::Tensor luminance_denoise(const torch::Tensor& image, float strength);

// Color noise reduction: reduce chroma noise while preserving luminance
// strength: 0.0-1.0
torch::Tensor color_denoise(const torch::Tensor& image, float strength);

// Portrait retouching
// Whitening: 0.0-1.0
torch::Tensor whiten(const torch::Tensor& image, float strength);

// Skin smoothing: simple gaussian blur with edge preservation
// strength: 0.0-1.0
torch::Tensor skin_smooth(const torch::Tensor& image, float strength);

// RGB Curves adjustment
// curves: vector of control points in format "input,output;input,output;..."
// input/output values are 0-255
torch::Tensor apply_curves(const torch::Tensor& image, const std::string& curves);

// Built-in filter presets
// name: "vintage", "bw", "film", "japanese", "warm", "cool", "dramatic"
torch::Tensor apply_preset(const torch::Tensor& image, const std::string& name);

// Vignette effect
// strength: 0.0-1.0 (darken edges)
// radius: 0.0-1.0 (0.5 = half image size)
torch::Tensor vignette(const torch::Tensor& image, float strength, float radius = 0.75f);

// Radial filter: apply adjustments within a circular region
// cx, cy: center coordinates (0.0-1.0, relative to image size)
// radius: radius (0.0-1.0, relative to image size)
// exposure, contrast, saturation: adjustment values
torch::Tensor radial_filter(const torch::Tensor& image, float cx, float cy, float radius,
                            float exposure, float contrast, float saturation);

// Graduated filter: apply adjustments along a linear gradient
// angle: gradient angle in degrees (0 = top to bottom, 90 = left to right)
// position: gradient center position (0.0-1.0)
// width: gradient width (0.0-1.0)
// exposure, contrast, saturation: adjustment values
torch::Tensor graduated_filter(const torch::Tensor& image, float angle, float position, float width,
                               float exposure, float contrast, float saturation);

// Smart sharpen: edge-aware sharpening
// strength: 0.0-3.0 (sharpening strength)
// radius: 1-5 (blur radius for edge detection)
torch::Tensor smart_sharpen(const torch::Tensor& image, float strength, int radius = 2);

} // namespace myimg
