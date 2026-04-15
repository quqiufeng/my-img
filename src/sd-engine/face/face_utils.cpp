// ============================================================================
// face_utils.cpp
// ============================================================================

#include "face_utils.hpp"
#include "face_align.hpp"
#include <algorithm>

namespace sdengine {
namespace face {

void generate_feather_mask(uint8_t* mask, int size, int feather_radius) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dist_to_edge = std::min({x, y, size - 1 - x, size - 1 - y});
            if (dist_to_edge >= feather_radius) {
                mask[y * size + x] = 255;
            } else {
                mask[y * size + x] = (uint8_t)(255.0f * dist_to_edge / feather_radius);
            }
        }
    }
}

void blend_face_back(uint8_t* dst_image, int img_w, int img_h, int channels,
                     const uint8_t* restored_face, int face_size,
                     const float inv_M[6], const uint8_t* mask) {
    for (int y = 0; y < face_size; y++) {
        for (int x = 0; x < face_size; x++) {
            float src_x = inv_M[0] * x + inv_M[1] * y + inv_M[2];
            float src_y = inv_M[3] * x + inv_M[4] * y + inv_M[5];

            int px = (int)(src_x + 0.5f);
            int py = (int)(src_y + 0.5f);

            if (px < 0 || px >= img_w || py < 0 || py >= img_h) continue;

            float alpha = mask ? (mask[y * face_size + x] / 255.0f) : 1.0f;
            int dst_idx = (py * img_w + px) * channels;
            int src_idx = (y * face_size + x) * channels;

            for (int c = 0; c < channels && c < 3; c++) {
                float blended = dst_image[dst_idx + c] * (1.0f - alpha) + restored_face[src_idx + c] * alpha;
                dst_image[dst_idx + c] = (uint8_t)(std::max(0.0f, std::min(255.0f, blended)) + 0.5f);
            }
        }
    }
}

void color_transfer(const uint8_t* src_face, const uint8_t* dst_face,
                    uint8_t* output, int size, int channels) {
    // Compute mean and std for both images (RGB only)
    float src_mean[3] = {0}, dst_mean[3] = {0};
    float src_var[3] = {0}, dst_var[3] = {0};
    int pixels = size * size;

    for (int i = 0; i < pixels; i++) {
        for (int c = 0; c < 3 && c < channels; c++) {
            src_mean[c] += src_face[i * channels + c];
            dst_mean[c] += dst_face[i * channels + c];
        }
    }
    for (int c = 0; c < 3; c++) {
        src_mean[c] /= pixels;
        dst_mean[c] /= pixels;
    }

    for (int i = 0; i < pixels; i++) {
        for (int c = 0; c < 3 && c < channels; c++) {
            float sd = src_face[i * channels + c] - src_mean[c];
            float dd = dst_face[i * channels + c] - dst_mean[c];
            src_var[c] += sd * sd;
            dst_var[c] += dd * dd;
        }
    }
    for (int c = 0; c < 3; c++) {
        src_var[c] = std::sqrt(src_var[c] / pixels);
        dst_var[c] = std::sqrt(dst_var[c] / pixels);
    }

    for (int i = 0; i < pixels; i++) {
        for (int c = 0; c < channels; c++) {
            if (c < 3) {
                float scale = (src_var[c] > 1e-6f) ? (dst_var[c] / src_var[c]) : 1.0f;
                float val = (src_face[i * channels + c] - src_mean[c]) * scale + dst_mean[c];
                output[i * channels + c] = (uint8_t)(std::max(0.0f, std::min(255.0f, val)) + 0.5f);
            } else {
                output[i * channels + c] = src_face[i * channels + c];
            }
        }
    }
}

} // namespace face
} // namespace sdengine
