// ============================================================================
// sd-engine/nodes/image_filter_nodes.cpp
// ============================================================================
// 图像滤镜/效果节点实现（混合、合成、反转、调整、模糊、灰度、二值化）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// ImageBlend - 图像混合
// ============================================================================
class ImageBlendNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageBlend";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image1", "IMAGE", true, nullptr},
                {"image2", "IMAGE", true, nullptr},
                {"blend_factor", "FLOAT", false, 0.5f},
                {"blend_mode", "STRING", false, std::string("normal")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr img1 = std::any_cast<ImagePtr>(inputs.at("image1"));
        ImagePtr img2 = std::any_cast<ImagePtr>(inputs.at("image2"));
        float blend_factor = inputs.count("blend_factor") ? std::any_cast<float>(inputs.at("blend_factor")) : 0.5f;
        std::string blend_mode =
            inputs.count("blend_mode") ? std::any_cast<std::string>(inputs.at("blend_mode")) : "normal";

        if (!img1 || !img1->data || !img2 || !img2->data) {
            LOG_ERROR("[ERROR] ImageBlend: Missing input images\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int width = (int)img1->width;
        int height = (int)img1->height;
        int channels = (int)img1->channel;

        if ((int)img2->width != width || (int)img2->height != height) {
            LOG_ERROR("[ERROR] ImageBlend: Image sizes must match (%dx%d vs %dx%d)\n", width, height, (int)img2->width,
                      (int)img2->height);
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int ch2 = (int)img2->channel;
        int out_channels = std::max(channels, ch2);

        auto dst_data = make_malloc_buffer(width * height * out_channels);
        if (!dst_data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx1 = (y * width + x) * channels;
                int idx2 = (y * width + x) * ch2;
                int idx_dst = (y * width + x) * out_channels;

                for (int c = 0; c < out_channels; c++) {
                    float v1 = (c < channels) ? (img1->data[idx1 + c] / 255.0f) : 1.0f;
                    float v2 = (c < ch2) ? (img2->data[idx2 + c] / 255.0f) : 1.0f;
                    float result = 0.0f;

                    if (blend_mode == "normal") {
                        result = v1 * (1.0f - blend_factor) + v2 * blend_factor;
                    } else if (blend_mode == "add") {
                        result = v1 + v2 * blend_factor;
                    } else if (blend_mode == "multiply") {
                        result = v1 * (1.0f - blend_factor) + (v1 * v2) * blend_factor;
                    } else if (blend_mode == "screen") {
                        float screen = 1.0f - (1.0f - v1) * (1.0f - v2);
                        result = v1 * (1.0f - blend_factor) + screen * blend_factor;
                    } else {
                        result = v1 * (1.0f - blend_factor) + v2 * blend_factor;
                    }

                    result = std::clamp(result, 0.0f, 1.0f);
                    dst_data[idx_dst + c] = (uint8_t)(result * 255.0f + 0.5f);
                }
            }
        }

        auto out_img = create_image_ptr(width, height, out_channels, std::move(dst_data));
        if (!out_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageBlend] Blended %dx%dx%d (mode=%s, factor=%.2f)\n", width, height, out_channels,
                 blend_mode.c_str(), blend_factor);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageBlend", ImageBlendNode);

// ============================================================================
// ImageCompositeMasked - 蒙版合成
// ============================================================================
class ImageCompositeMaskedNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageCompositeMasked";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"destination", "IMAGE", true, nullptr},
                {"source", "IMAGE", true, nullptr},
                {"x", "INT", false, 0},
                {"y", "INT", false, 0},
                {"mask", "IMAGE", false, nullptr},
                {"resize_source", "BOOLEAN", false, false}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr dst = std::any_cast<ImagePtr>(inputs.at("destination"));
        ImagePtr src = std::any_cast<ImagePtr>(inputs.at("source"));
        int offset_x = inputs.count("x") ? std::any_cast<int>(inputs.at("x")) : 0;
        int offset_y = inputs.count("y") ? std::any_cast<int>(inputs.at("y")) : 0;
        ImagePtr mask = inputs.count("mask") ? std::any_cast<ImagePtr>(inputs.at("mask")) : nullptr;
        bool resize_source = inputs.count("resize_source") ? std::any_cast<bool>(inputs.at("resize_source")) : false;

        if (!dst || !dst->data || !src || !src->data) {
            LOG_ERROR("[ERROR] ImageCompositeMasked: Missing input images\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int dst_w = (int)dst->width;
        int dst_h = (int)dst->height;
        int dst_c = (int)dst->channel;
        int src_w = (int)src->width;
        int src_h = (int)src->height;
        int src_c = (int)src->channel;

        std::vector<uint8_t> src_resized;
        const uint8_t* src_data = src->data;
        int src_stride_w = src_w;
        int src_stride_h = src_h;
        if (resize_source && (src_w != dst_w || src_h != dst_h)) {
            src_resized.resize(dst_w * dst_h * src_c);
            stbir_resize(src->data, src_w, src_h, 0, src_resized.data(), dst_w, dst_h, 0, STBIR_TYPE_UINT8, src_c, -1,
                         0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE,
                         STBIR_COLORSPACE_LINEAR, nullptr);
            src_data = src_resized.data();
            src_stride_w = dst_w;
            src_stride_h = dst_h;
        }

        auto out_data = make_malloc_buffer(dst_w * dst_h * dst_c);
        if (!out_data) {
            LOG_ERROR("[ERROR] ImageCompositeMasked: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(out_data.get(), dst->data, dst_w * dst_h * dst_c);

        for (int y = 0; y < src_stride_h; y++) {
            int dst_y = offset_y + y;
            if (dst_y < 0 || dst_y >= dst_h)
                continue;

            for (int x = 0; x < src_stride_w; x++) {
                int dst_x = offset_x + x;
                if (dst_x < 0 || dst_x >= dst_w)
                    continue;

                int dst_idx = (dst_y * dst_w + dst_x) * dst_c;
                int src_idx = (y * src_stride_w + x) * src_c;

                float alpha = 1.0f;
                if (mask && mask->data) {
                    int mask_x = (mask->width == 1) ? 0 : (x * (int)mask->width / src_stride_w);
                    int mask_y = (mask->height == 1) ? 0 : (y * (int)mask->height / src_stride_h);
                    mask_x = std::clamp(mask_x, 0, (int)mask->width - 1);
                    mask_y = std::clamp(mask_y, 0, (int)mask->height - 1);
                    int mask_idx = (mask_y * (int)mask->width + mask_x) * (int)mask->channel;
                    alpha = mask->data[mask_idx] / 255.0f;
                }

                for (int c = 0; c < dst_c; c++) {
                    float src_v = (c < src_c) ? (src_data[src_idx + c] / 255.0f) : 1.0f;
                    float dst_v = out_data[dst_idx + c] / 255.0f;
                    float blended = dst_v * (1.0f - alpha) + src_v * alpha;
                    out_data[dst_idx + c] = (uint8_t)(std::clamp(blended, 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
        }

        auto out_img = create_image_ptr(dst_w, dst_h, dst_c, std::move(out_data));
        if (!out_img) {
            LOG_ERROR("[ERROR] ImageCompositeMasked: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageCompositeMasked] Composited source onto destination at (%d,%d)\n", offset_x, offset_y);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageCompositeMasked", ImageCompositeMaskedNode);

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
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
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
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        float brightness = inputs.count("brightness") ? std::any_cast<float>(inputs.at("brightness")) : 1.0f;
        float contrast = inputs.count("contrast") ? std::any_cast<float>(inputs.at("contrast")) : 1.0f;
        float saturation = inputs.count("saturation") ? std::any_cast<float>(inputs.at("saturation")) : 1.0f;

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
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        int radius = inputs.count("radius") ? std::any_cast<int>(inputs.at("radius")) : 3;
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
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
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
        ImagePtr img = std::any_cast<ImagePtr>(inputs.at("image"));
        int threshold = inputs.count("threshold") ? std::any_cast<int>(inputs.at("threshold")) : 128;
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

void init_image_filter_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
