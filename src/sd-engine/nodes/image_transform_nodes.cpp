// ============================================================================
// sd-engine/nodes/image_transform_nodes.cpp
// ============================================================================
// 图像几何变换节点实现（缩放、裁剪、超分）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// ImageScale - 图像缩放
// ============================================================================
class ImageScaleNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageScale";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr},
                {"width", "INT", true, 0},
                {"height", "INT", true, 0},
                {"method", "STRING", false, std::string("bilinear")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src_image;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", src_image));
        int target_width;
        SD_RETURN_IF_ERROR(get_input(inputs, "width", target_width));
        int target_height;
        SD_RETURN_IF_ERROR(get_input(inputs, "height", target_height));
        std::string method = get_input_opt<std::string>(inputs, "method", "bilinear");

        if (!src_image || !src_image->data) {
            LOG_ERROR("[ERROR] ImageScale: No source image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        if (target_width <= 0 || target_height <= 0) {
            LOG_ERROR("[ERROR] ImageScale: Invalid target size %dx%d\n", target_width, target_height);
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        if ((int)src_image->width == target_width && (int)src_image->height == target_height) {
            outputs["IMAGE"] = src_image;
            return sd_error_t::OK;
        }

        LOG_INFO("[ImageScale] Resizing from %dx%d to %dx%d (method: %s)\n", src_image->width, src_image->height,
                 target_width, target_height, method.c_str());

        size_t dst_size = target_width * target_height * src_image->channel;
        auto dst_data = make_malloc_buffer(dst_size);
        if (!dst_data) {
            LOG_ERROR("[ERROR] ImageScale: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        stbir_filter filter = STBIR_FILTER_DEFAULT;
        if (method == "nearest") {
            filter = STBIR_FILTER_BOX;
        } else if (method == "bilinear") {
            filter = STBIR_FILTER_TRIANGLE;
        } else if (method == "lanczos") {
            filter = STBIR_FILTER_CATMULLROM;
        }

        stbir_resize(src_image->data, src_image->width, src_image->height, 0, dst_data.get(), target_width,
                     target_height, 0, STBIR_TYPE_UINT8, src_image->channel, -1, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     filter, filter, STBIR_COLORSPACE_LINEAR, nullptr);

        auto out_img = create_image_ptr(target_width, target_height, src_image->channel, std::move(dst_data));
        if (!out_img) {
            LOG_ERROR("[ERROR] ImageScale: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageScale] Resized successfully\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageScale", ImageScaleNode);

// ============================================================================
// ImageCrop - 图像裁剪
// ============================================================================
class ImageCropNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageCrop";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr},
                {"x", "INT", true, 0},
                {"y", "INT", true, 0},
                {"width", "INT", true, 0},
                {"height", "INT", true, 0}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src_image;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", src_image));
        int crop_x;
        SD_RETURN_IF_ERROR(get_input(inputs, "x", crop_x));
        int crop_y;
        SD_RETURN_IF_ERROR(get_input(inputs, "y", crop_y));
        int crop_width;
        SD_RETURN_IF_ERROR(get_input(inputs, "width", crop_width));
        int crop_height;
        SD_RETURN_IF_ERROR(get_input(inputs, "height", crop_height));

        if (!src_image || !src_image->data) {
            LOG_ERROR("[ERROR] ImageCrop: No source image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        if (crop_x < 0 || crop_y < 0 || crop_width <= 0 || crop_height <= 0 ||
            crop_x + crop_width > (int)src_image->width || crop_y + crop_height > (int)src_image->height) {
            LOG_ERROR("[ERROR] ImageCrop: Invalid crop region (%d,%d,%d,%d) for image %dx%d\n", crop_x, crop_y,
                      crop_width, crop_height, src_image->width, src_image->height);
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        LOG_INFO("[ImageCrop] Cropping to (%d,%d) size %dx%d\n", crop_x, crop_y, crop_width, crop_height);

        auto dst_data = make_malloc_buffer(crop_width * crop_height * src_image->channel);
        if (!dst_data) {
            LOG_ERROR("[ERROR] ImageCrop: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        int src_stride = src_image->width * src_image->channel;
        int dst_stride = crop_width * src_image->channel;
        int row_bytes = crop_width * src_image->channel;

        for (int y = 0; y < crop_height; y++) {
            uint8_t* src_row = src_image->data + (crop_y + y) * src_stride + crop_x * src_image->channel;
            uint8_t* dst_row = dst_data.get() + y * dst_stride;
            memcpy(dst_row, src_row, row_bytes);
        }

        auto out_img = create_image_ptr(crop_width, crop_height, src_image->channel, std::move(dst_data));
        if (!out_img) {
            LOG_ERROR("[ERROR] ImageCrop: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["IMAGE"] = out_img;
        LOG_INFO("[ImageCrop] Cropped successfully\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageCrop", ImageCropNode);

// ============================================================================
// ImageUpscaleWithModel - 使用 ESRGAN 模型放大图像
// ============================================================================
class ImageUpscaleWithModelNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ImageUpscaleWithModel";
    }
    std::string get_category() const override {
        return "image/upscaling";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "IMAGE", true, nullptr}, {"upscale_model", "UPSCALE_MODEL", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", image));
        UpscalerPtr upscaler;
        SD_RETURN_IF_ERROR(get_input(inputs, "upscale_model", upscaler));

        if (!image || !image->data) {
            LOG_ERROR("[ERROR] ImageUpscaleWithModel: No image data\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        if (!upscaler) {
            LOG_ERROR("[ERROR] ImageUpscaleWithModel: No upscale model\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int scale = get_upscale_factor(upscaler.get());
        LOG_INFO("[ImageUpscaleWithModel] Upscaling %dx%d by %dx...\n", image->width, image->height, scale);

        sd_image_t result = upscale(upscaler.get(), *image, scale);
        if (!result.data) {
            LOG_ERROR("[ERROR] ImageUpscaleWithModel: Upscale failed\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        LOG_INFO("[ImageUpscaleWithModel] Upscaled to %dx%d\n", result.width, result.height);

        auto result_buffer = make_malloc_buffer(result.width * result.height * result.channel);
        if (!result_buffer) {
            std::free(result.data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(result_buffer.get(), result.data, result.width * result.height * result.channel);
        std::free(result.data);

        sd_image_t* result_ptr = acquire_image();
        if (!result_ptr) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result_ptr = result;
        result_ptr->data = result_buffer.release();
        outputs["IMAGE"] = make_image_ptr(result_ptr);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageUpscaleWithModel", ImageUpscaleWithModelNode);

void init_image_transform_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
