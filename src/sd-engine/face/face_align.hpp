// ============================================================================
// face_align.hpp
// ============================================================================
// 人脸对齐：5 点仿射变换 + 双线性 warp
// ============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <cmath>

namespace sdengine {
namespace face {

// 标准 512x512 人脸模板（GFPGAN/CodeFormer 通用）
// 左眼、右眼、鼻尖、嘴左、嘴右
inline void get_standard_face_template_512(float template_points[10]) {
    template_points[0] = 192.0f; template_points[1] = 240.0f;  // 左眼
    template_points[2] = 320.0f; template_points[3] = 240.0f;  // 右眼
    template_points[4] = 256.0f; template_points[5] = 320.0f;  // 鼻尖
    template_points[6] = 220.0f; template_points[7] = 400.0f;  // 嘴左
    template_points[8] = 292.0f; template_points[9] = 400.0f;  // 嘴右
}

// 计算 2x3 仿射矩阵：从源 landmark 到目标 landmark
// 使用最小二乘法求解
bool estimate_affine_transform_2d3(const float src_points[10], const float dst_points[10], float matrix[6]);

// 计算逆仿射矩阵
bool invert_affine_transform(const float M[6], float inv_M[6]);

// 双线性插值 warp
// src: 源图像数据，src_w x src_h x src_c
// dst: 目标图像缓冲区，dst_w x dst_h x src_c
// M: 2x3 仿射矩阵，将目标坐标映射到源坐标
void warp_affine_bilinear(const uint8_t* src, int src_w, int src_h, int src_c,
                          uint8_t* dst, int dst_w, int dst_h,
                          const float M[6]);

// 裁剪人脸区域（带边界检查）
std::vector<uint8_t> crop_face(const uint8_t* image, int img_w, int img_h, int channels,
                               const float landmarks[10], int out_size = 512);

} // namespace face
} // namespace sdengine
