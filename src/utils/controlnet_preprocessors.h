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

// Depth estimation placeholder (requires MiDaS/DPT model)
ImageData depth_map(const ImageData& img);

// OpenPose placeholder (requires OpenPose model)
ImageData openpose(const ImageData& img);

// Apply preprocessor by name
// Returns empty ImageData if preprocessor is not available
ImageData apply_preprocessor(const ImageData& img, const std::string& name, int param1 = 0, int param2 = 0);

} // namespace myimg
