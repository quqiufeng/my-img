// ============================================================================
// sd-engine/nodes/image_adjust_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// ImageInvert - 颜色反转
// ============================================================================
class ImageInvertNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageInvert";
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
            LOG_ERROR("[ERROR] ImageInvert: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;
        size_t pixels = w * h * c;

        auto dst = make_malloc_buffer(pixels);
        if (!dst)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            for (int ch = 0; ch < c; ch++) {
                dst[i * c + ch] = 255 - img->data[i * c + ch];
            }
        }

        auto out_img = create_image_ptr(w, h, c, std::move(dst));
        if (!out_img)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageInvert] Inverted %dx%dx%d\n", w, h, c);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageInvert", ImageInvertNode);

// ============================================================================
// ImageColorAdjust - 亮度/对比度/饱和度调整
// ============================================================================
class ImageColorAdjustNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageColorAdjust";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr},
                {"brightness", "FLOAT", false, 1.0f},
                {"contrast", "FLOAT", false, 1.0f},
                {"saturation", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", img));
        float brightness = get_input_opt<float>(inputs, "brightness", 1.0f);
        float contrast = get_input_opt<float>(inputs, "contrast", 1.0f);
        float saturation = get_input_opt<float>(inputs, "saturation", 1.0f);

        if (!img || !img->data) {
            LOG_ERROR("[ERROR] ImageColorAdjust: Missing input image\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int w = (int)img->width;
        int h = (int)img->height;
        int c = (int)img->channel;
        if (c != 3 && c != 4) {
            LOG_ERROR("[ERROR] ImageColorAdjust: Only 3 or 4 channel images supported\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        auto dst = make_malloc_buffer(w * h * c);
        if (!dst)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;

        for (int i = 0; i < w * h; i++) {
            float r = img->data[i * c + 0] / 255.0f;
            float g = img->data[i * c + 1] / 255.0f;
            float b = img->data[i * c + 2] / 255.0f;

            r *= brightness;
            g *= brightness;
            b *= brightness;

            r = (r - 0.5f) * contrast + 0.5f;
            g = (g - 0.5f) * contrast + 0.5f;
            b = (b - 0.5f) * contrast + 0.5f;

            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            r = gray + (r - gray) * saturation;
            g = gray + (g - gray) * saturation;
            b = gray + (b - gray) * saturation;

            dst[i * c + 0] = (uint8_t)(std::clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[i * c + 1] = (uint8_t)(std::clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[i * c + 2] = (uint8_t)(std::clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
            if (c == 4)
                dst[i * c + 3] = img->data[i * c + 3];
        }

        auto out_img = create_image_ptr(w, h, c, std::move(dst));
        if (!out_img)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageColorAdjust] Adjusted %dx%dx%d (b=%.2f, c=%.2f, s=%.2f)\n", w, h, c, brightness, contrast,
                 saturation);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageColorAdjust", ImageColorAdjustNode);

void init_image_adjust_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
