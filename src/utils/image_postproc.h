#pragma once

#include "adapters/sdcpp_adapter.h"

namespace myimg {

/**
 * @brief 图像后处理参数
 */
struct PostProcessParams {
    // Clarity (局部对比度增强) 0.0 ~ 1.0
    float clarity = 0.0f;

    // Unsharp Mark sharpen 0.0 ~ 3.0
    float sharpen_amount = 0.0f;
    int sharpen_radius = 1;       // 1-5
    float sharpen_threshold = 0;  // 0-255

    // Smart sharpen (edge-aware) 0.0 ~ 3.0
    float smart_sharpen_strength = 0.0f;
    int smart_sharpen_radius = 2;

    // Edge-mask sharpen 0.0 ~ 3.0
    float edge_sharpen_amount = 0.0f;
    int edge_sharpen_radius = 2;
    float edge_sharpen_threshold = 0.3f;
};

/**
 * @brief 对生成图像进行后处理
 * 
 * 按顺序应用：Clarity → USM Sharpen → Smart Sharpen → Edge Sharpen
 * 每个步骤仅在 > 0 时执行。
 * 
 * @param img 输入/输出图像，RGBA uint8
 * @param params 后处理参数
 * @return true 处理成功
 */
bool apply_image_postprocessing(Image& img, const PostProcessParams& params);

} // namespace myimg
