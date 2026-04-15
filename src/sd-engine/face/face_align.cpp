// ============================================================================
// face_align.cpp
// ============================================================================

#include "face_align.hpp"
#include <algorithm>

namespace sdengine {
namespace face {

bool estimate_affine_transform_2d3(const float src_points[10], const float dst_points[10], float matrix[6]) {
    // Solve for 2x3 affine matrix using 5 point pairs
    // We use a simplified approach: compute scale, rotation, and translation
    // based on the 3 main points (eyes + nose)

    float src_center_x = (src_points[0] + src_points[2] + src_points[4]) / 3.0f;
    float src_center_y = (src_points[1] + src_points[3] + src_points[5]) / 3.0f;
    float dst_center_x = (dst_points[0] + dst_points[2] + dst_points[4]) / 3.0f;
    float dst_center_y = (dst_points[1] + dst_points[3] + dst_points[5]) / 3.0f;

    float src_dx = src_points[2] - src_points[0];
    float src_dy = src_points[3] - src_points[1];
    float dst_dx = dst_points[2] - dst_points[0];
    float dst_dy = dst_points[3] - dst_points[1];

    float src_len = std::sqrt(src_dx * src_dx + src_dy * src_dy);
    float dst_len = std::sqrt(dst_dx * dst_dx + dst_dy * dst_dy);

    if (src_len < 1e-6f) return false;

    float scale = dst_len / src_len;

    float src_angle = std::atan2(src_dy, src_dx);
    float dst_angle = std::atan2(dst_dy, dst_dx);
    float angle = dst_angle - src_angle;

    float cos_a = std::cos(angle) * scale;
    float sin_a = std::sin(angle) * scale;

    matrix[0] = cos_a;
    matrix[1] = -sin_a;
    matrix[2] = dst_center_x - (cos_a * src_center_x - sin_a * src_center_y);
    matrix[3] = sin_a;
    matrix[4] = cos_a;
    matrix[5] = dst_center_y - (sin_a * src_center_x + cos_a * src_center_y);

    return true;
}

bool invert_affine_transform(const float M[6], float inv_M[6]) {
    float det = M[0] * M[4] - M[1] * M[3];
    if (std::abs(det) < 1e-6f) return false;

    float inv_det = 1.0f / det;
    inv_M[0] =  M[4] * inv_det;
    inv_M[1] = -M[1] * inv_det;
    inv_M[3] = -M[3] * inv_det;
    inv_M[4] =  M[0] * inv_det;
    inv_M[2] = -(inv_M[0] * M[2] + inv_M[1] * M[5]);
    inv_M[5] = -(inv_M[3] * M[2] + inv_M[4] * M[5]);
    return true;
}

void warp_affine_bilinear(const uint8_t* src, int src_w, int src_h, int src_c,
                          uint8_t* dst, int dst_w, int dst_h,
                          const float M[6]) {
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float src_x = M[0] * x + M[1] * y + M[2];
            float src_y = M[3] * x + M[4] * y + M[5];

            int x0 = (int)std::floor(src_x);
            int y0 = (int)std::floor(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float fx = src_x - x0;
            float fy = src_y - y0;

            for (int c = 0; c < src_c; c++) {
                float v00 = (x0 >= 0 && x0 < src_w && y0 >= 0 && y0 < src_h) ? src[(y0 * src_w + x0) * src_c + c] : 0;
                float v01 = (x1 >= 0 && x1 < src_w && y0 >= 0 && y0 < src_h) ? src[(y0 * src_w + x1) * src_c + c] : 0;
                float v10 = (x0 >= 0 && x0 < src_w && y1 >= 0 && y1 < src_h) ? src[(y1 * src_w + x0) * src_c + c] : 0;
                float v11 = (x1 >= 0 && x1 < src_w && y1 >= 0 && y1 < src_h) ? src[(y1 * src_w + x1) * src_c + c] : 0;

                float val = v00 * (1 - fx) * (1 - fy)
                          + v01 * fx * (1 - fy)
                          + v10 * (1 - fx) * fy
                          + v11 * fx * fy;

                dst[(y * dst_w + x) * src_c + c] = (uint8_t)(std::max(0.0f, std::min(255.0f, val)) + 0.5f);
            }
        }
    }
}

std::vector<uint8_t> crop_face(const uint8_t* image, int img_w, int img_h, int channels,
                               const float landmarks[10], int out_size) {
    float template_points[10];
    get_standard_face_template_512(template_points);

    // Scale template to out_size
    float scale = out_size / 512.0f;
    for (int i = 0; i < 10; i++) {
        template_points[i] *= scale;
    }

    float M[6];
    estimate_affine_transform_2d3(landmarks, template_points, M);

    std::vector<uint8_t> output(out_size * out_size * channels);
    warp_affine_bilinear(image, img_w, img_h, channels, output.data(), out_size, out_size, M);
    return output;
}

} // namespace face
} // namespace sdengine
