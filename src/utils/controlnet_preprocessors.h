#pragma once
#include "image_utils.h"
#include <string>

namespace myimg {

// ControlNet preprocessor types
enum class ControlNetPreprocessor {
    Canny,
    Depth,
    Lineart,
    OpenPose,
    NormalMap,
    Scribble,
    None
};

// Canny edge detection
// low_threshold, high_threshold: Canny thresholds (default: 100, 200)
ImageData canny_edges(const ImageData& img, int low_threshold = 100, int high_threshold = 200);

// Lineart extraction (Canny-based)
ImageData lineart(const ImageData& img, int threshold = 100);

// Normal map estimation (using Sobel gradients)
ImageData normal_map(const ImageData& img);

// Scribble (simplified line detection)
ImageData scribble(const ImageData& img, int threshold = 100);

// Depth estimation using MiDaS/DPT model (libtorch)
// model_path: path to .pt model file
ImageData depth_map(const ImageData& img, const std::string& model_path = "");

// OpenPose body pose detection (libtorch)
// model_path: path to .pth model file
ImageData openpose(const ImageData& img, const std::string& model_path = "");

// ONNX-based depth estimation (fallback)
ImageData depth_map_onnx(const ImageData& img, const std::string& model_path = "");

// ONNX-based OpenPose detection (fallback)
ImageData openpose_onnx(const ImageData& img, const std::string& model_path = "");

// Apply preprocessor by name
// Returns empty ImageData if preprocessor is not available
// model_path: optional path to the model file (for depth/openpose)
ImageData apply_preprocessor(const ImageData& img, const std::string& name, int param1 = 0, int param2 = 0, const std::string& model_path = "");

} // namespace myimg
