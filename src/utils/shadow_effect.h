#pragma once

#include <string>

#include "image_utils.h"

namespace myimg {

struct DropShadowConfig {
    int offset_x = 10;           // 阴影水平偏移
    int offset_y = 10;           // 阴影垂直偏移
    int blur_radius = 10;        // 模糊半径
    float opacity = 0.5f;        // 透明度
    uint32_t color = 0x000000FF; // 阴影颜色 (RGBA)
};

struct ReflectionConfig {
    float height_ratio = 0.3f;   // 倒影高度占原图比例
    float opacity = 0.4f;        // 倒影透明度
    int blur_radius = 3;         // 底部模糊
    int gap = 5;                 // 原图与倒影间隙
};

class ShadowEffect {
public:
    // 添加投影阴影（Drop Shadow）
    // 效果：图像后方有模糊阴影，产生悬浮感
    static ImageData add_drop_shadow(const ImageData& image, const DropShadowConfig& config);

    // 添加倒影（Reflection）
    // 效果：图像下方有镜像倒影，类似水面/桌面反射
    static ImageData add_reflection(const ImageData& image, const ReflectionConfig& config);

    // 添加地面接触阴影（Contact Shadow）
    // 效果：图像底部有椭圆阴影，产生站立感
    static ImageData add_contact_shadow(const ImageData& image, float opacity, int blur_radius);

    // 组合效果：投影 + 接触阴影
    static ImageData add_product_shadow(const ImageData& image,
                                        const DropShadowConfig& drop_config,
                                        const ReflectionConfig& reflection_config);
};

} // namespace myimg
