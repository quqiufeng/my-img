// ============================================================================
// sd-engine/nodes/image_blend_nodes.cpp
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
        ImagePtr img1;
        SD_RETURN_IF_ERROR(get_input(inputs, "image1", img1));
        ImagePtr img2;
        SD_RETURN_IF_ERROR(get_input(inputs, "image2", img2));
        float blend_factor = get_input_opt<float>(inputs, "blend_factor", 0.5f);
        std::string blend_mode =
            get_input_opt<std::string>(inputs, "blend_mode", "normal");

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
        ImagePtr dst;
        SD_RETURN_IF_ERROR(get_input(inputs, "destination", dst));
        ImagePtr src;
        SD_RETURN_IF_ERROR(get_input(inputs, "source", src));
        int offset_x = get_input_opt<int>(inputs, "x", 0);
        int offset_y = get_input_opt<int>(inputs, "y", 0);
        ImagePtr mask = get_input_opt<ImagePtr>(inputs, "mask", nullptr);
        bool resize_source = get_input_opt<bool>(inputs, "resize_source", false);

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

void init_image_blend_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
