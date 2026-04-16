// ============================================================================
// sd-engine/nodes/image_effect_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// ImageBlur - 盒式模糊
// ============================================================================
class ImageBlurNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageBlur";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}, {"radius", "INT", false, 3}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", img));
        int radius = get_input_opt<int>(inputs, "radius", 3);
        if (radius < 1)
            radius = 1;

        if (!img || !img->data) {
            LOG_ERROR("[ERROR] ImageBlur: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;

        auto dst = make_malloc_buffer(w * h * c);
        if (!dst)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;

        // 使用分离核优化：先水平模糊，再垂直模糊
        // 复杂度从 O(w*h*r^2) 降到 O(w*h*r)
        std::vector<uint8_t> tmp(w * h * c);

        for (int ch = 0; ch < c; ch++) {
            // 水平方向
            for (int y = 0; y < h; y++) {
                int sum = 0;
                int count = 0;
                // 初始化窗口
                for (int dx = -radius; dx <= radius; dx++) {
                    int px = std::clamp(dx, 0, w - 1);
                    sum += img->data[(y * w + px) * c + ch];
                    count++;
                }
                tmp[(y * w + 0) * c + ch] = (uint8_t)(sum / count);

                // 滑动窗口
                for (int x = 1; x < w; x++) {
                    int left_x = std::clamp(x - radius - 1, 0, w - 1);
                    int right_x = std::clamp(x + radius, 0, w - 1);
                    sum -= img->data[(y * w + left_x) * c + ch];
                    sum += img->data[(y * w + right_x) * c + ch];
                    tmp[(y * w + x) * c + ch] = (uint8_t)(sum / count);
                }
            }

            // 垂直方向
            for (int x = 0; x < w; x++) {
                int sum = 0;
                int count = 0;
                // 初始化窗口
                for (int dy = -radius; dy <= radius; dy++) {
                    int py = std::clamp(dy, 0, h - 1);
                    sum += tmp[(py * w + x) * c + ch];
                    count++;
                }
                dst[(0 * w + x) * c + ch] = (uint8_t)(sum / count);

                // 滑动窗口
                for (int y = 1; y < h; y++) {
                    int top_y = std::clamp(y - radius - 1, 0, h - 1);
                    int bottom_y = std::clamp(y + radius, 0, h - 1);
                    sum -= tmp[(top_y * w + x) * c + ch];
                    sum += tmp[(bottom_y * w + x) * c + ch];
                    dst[(y * w + x) * c + ch] = (uint8_t)(sum / count);
                }
            }
        }

        auto out_img = create_image_ptr(w, h, c, std::move(dst));
        if (!out_img)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageBlur] Blurred %dx%dx%d (radius=%d)\n", w, h, c, radius);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageBlur", ImageBlurNode);

// ============================================================================
// ImageGrayscale - 灰度转换
// ============================================================================
class ImageGrayscaleNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageGrayscale";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", img));
        if (!img || !img->data) {
            LOG_ERROR("[ERROR] ImageGrayscale: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;
        if (c < 3) {
            LOG_ERROR("[ERROR] ImageGrayscale: Image must have at least 3 channels\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto dst = make_malloc_buffer(w * h);
        if (!dst)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            dst[i] = (uint8_t)(0.299f * img->data[i * c + 0] + 0.587f * img->data[i * c + 1] +
                               0.114f * img->data[i * c + 2] + 0.5f);
        }

        auto out_img = create_image_ptr(w, h, 1, std::move(dst));
        if (!out_img)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageGrayscale] Converted %dx%d to grayscale\n", w, h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageGrayscale", ImageGrayscaleNode);

// ============================================================================
// ImageThreshold - 二值化
// ============================================================================
class ImageThresholdNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageThreshold";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}, {"threshold", "INT", false, 128}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", img));
        int threshold = get_input_opt<int>(inputs, "threshold", 128);
        threshold = std::clamp(threshold, 0, 255);

        if (!img || !img->data) {
            LOG_ERROR("[ERROR] ImageThreshold: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;

        auto dst = make_malloc_buffer(w * h * c);
        if (!dst)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            for (int ch = 0; ch < c; ch++) {
                dst[i * c + ch] = img->data[i * c + ch] >= threshold ? 255 : 0;
            }
        }

        auto out_img = create_image_ptr(w, h, c, std::move(dst));
        if (!out_img)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageThreshold] Thresholded %dx%dx%d (threshold=%d)\n", w, h, c, threshold);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageThreshold", ImageThresholdNode);

void init_image_effect_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
