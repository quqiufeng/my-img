// ============================================================================
// face_utils.hpp
// ============================================================================
// 人脸处理公共工具
// ============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <cmath>

namespace sdengine {
namespace face {

// 高斯羽化：在边缘区域创建 alpha 渐变，用于无缝贴回
// mask: 输出灰度蒙版，size x size
void generate_feather_mask(uint8_t* mask, int size, int feather_radius);

// 将修复后的人脸融合回原图
// dst_image: 原图缓冲区（会被修改）
// restored_face: 修复后的人脸图像（对齐后尺寸）
// inv_M: 从对齐空间到原图空间的逆仿射矩阵
// mask: 羽化蒙版
void blend_face_back(uint8_t* dst_image, int img_w, int img_h, int channels,
                     const uint8_t* restored_face, int face_size,
                     const float inv_M[6], const uint8_t* mask);

// 简单颜色校正：匹配目标区域均值和方差
void color_transfer(const uint8_t* src_face, const uint8_t* dst_face,
                    uint8_t* output, int size, int channels);

} // namespace face
} // namespace sdengine
