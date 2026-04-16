// ============================================================================
// sd-engine/nodes/image_nodes.cpp
// ============================================================================
// 图像处理节点实现
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// LoadImage - 加载图像
// ============================================================================
class LoadImageNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LoadImage";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}, {"MASK", "MASK"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string image_path = std::any_cast<std::string>(inputs.at("image"));

        if (image_path.empty()) {
            LOG_ERROR("[ERROR] LoadImage: image path is required\n");
            return sd_error_t::ERROR_FILE_IO;
        }

        LOG_INFO("[LoadImage] Loading: %s\n", image_path.c_str());

        int w, h, c;
        uint8_t* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!data) {
            LOG_ERROR("[ERROR] LoadImage: Failed to load %s\n", image_path.c_str());
            return sd_error_t::ERROR_FILE_IO;
        }

        sd_image_t* image = acquire_image();
        if (!image) {
            stbi_image_free(data);
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        image->width = w;
        image->height = h;
        image->channel = 3;
        image->data = data;

        LOG_INFO("[LoadImage] Loaded: %dx%d\n", w, h);

        outputs["IMAGE"] = make_image_ptr(image);
        outputs["MASK"] = nullptr;

        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoadImage", LoadImageNode);

// ============================================================================
// LoadImageMask - 加载 Mask 图像
// ============================================================================
class LoadImageMaskNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LoadImageMask";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"image", "STRING", true, std::string("")}, {"channel", "STRING", false, std::string("alpha")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"MASK", "MASK"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string image_path = std::any_cast<std::string>(inputs.at("image"));
        std::string channel = inputs.count("channel") ? std::any_cast<std::string>(inputs.at("channel")) : "alpha";

        if (image_path.empty()) {
            LOG_ERROR("[ERROR] LoadImageMask: image path is required\n");
            return sd_error_t::ERROR_FILE_IO;
        }

        LOG_INFO("[LoadImageMask] Loading: %s (channel=%s)\n", image_path.c_str(), channel.c_str());

        int w, h, c;
        uint8_t* data = stbi_load(image_path.c_str(), &w, &h, &c, 0);
        if (!data) {
            LOG_ERROR("[ERROR] LoadImageMask: Failed to load %s\n", image_path.c_str());
            return sd_error_t::ERROR_FILE_IO;
        }

        std::vector<uint8_t> mask_data(w * h * 3);

        int src_channel = 0;
        if (channel == "alpha" && c == 4) {
            src_channel = 3;
        } else if (channel == "red") {
            src_channel = 0;
        } else if (channel == "green") {
            src_channel = 1;
        } else if (channel == "blue") {
            src_channel = 2;
        }

        for (int i = 0; i < w * h; i++) {
            uint8_t val = data[i * c + src_channel];
            mask_data[i * 3 + 0] = val;
            mask_data[i * 3 + 1] = val;
            mask_data[i * 3 + 2] = val;
        }
        stbi_image_free(data);

        auto final_data = make_malloc_buffer(mask_data.size());
        if (!final_data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_data.get(), mask_data.data(), mask_data.size());

        sd_image_t* mask = acquire_image();
        if (!mask) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        mask->width = w;
        mask->height = h;
        mask->channel = 3;
        mask->data = final_data.release();

        outputs["MASK"] = make_image_ptr(mask);
        LOG_INFO("[LoadImageMask] Loaded mask: %dx%d\n", w, h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoadImageMask", LoadImageMaskNode);

// ============================================================================
// SaveImage - 保存图像
// ============================================================================
class SaveImageNode : public Node {
  public:
    std::string get_class_type() const override {
        return "SaveImage";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"images", "IMAGE", true, nullptr}, {"filename_prefix", "STRING", false, std::string("sd-engine")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("images"));
        std::string prefix =
            inputs.count("filename_prefix") ? std::any_cast<std::string>(inputs.at("filename_prefix")) : "sd-engine";

        if (!image || !image->data) {
            LOG_ERROR("[ERROR] SaveImage: No image data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        std::string filename = prefix + ".png";
        LOG_INFO("[SaveImage] Saving to %s (%dx%d)\n", filename.c_str(), image->width, image->height);

        bool success =
            stbi_write_png(filename.c_str(), image->width, image->height, image->channel, image->data, 0) != 0;
        if (!success) {
            LOG_ERROR("[ERROR] SaveImage: Failed to write %s\n", filename.c_str());
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        LOG_INFO("[SaveImage] Saved successfully\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("SaveImage", SaveImageNode);

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
        ImagePtr src_image = std::any_cast<ImagePtr>(inputs.at("image"));
        int target_width = std::any_cast<int>(inputs.at("width"));
        int target_height = std::any_cast<int>(inputs.at("height"));
        std::string method = inputs.count("method") ? std::any_cast<std::string>(inputs.at("method")) : "bilinear";

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
        std::vector<uint8_t> dst_data(dst_size);

        stbir_filter filter = STBIR_FILTER_DEFAULT;
        if (method == "nearest") {
            filter = STBIR_FILTER_BOX;
        } else if (method == "bilinear") {
            filter = STBIR_FILTER_TRIANGLE;
        } else if (method == "lanczos") {
            filter = STBIR_FILTER_CATMULLROM;
        }

        stbir_resize(src_image->data, src_image->width, src_image->height, 0, dst_data.data(), target_width,
                     target_height, 0, STBIR_TYPE_UINT8, src_image->channel, -1, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
                     filter, filter, STBIR_COLORSPACE_LINEAR, nullptr);

        auto final_data = make_malloc_buffer(dst_data.size());
        if (!final_data) {
            LOG_ERROR("[ERROR] ImageScale: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_data.get(), dst_data.data(), dst_data.size());

        sd_image_t dst_image = {};
        dst_image.width = target_width;
        dst_image.height = target_height;
        dst_image.channel = src_image->channel;
        dst_image.data = final_data.release();

        sd_image_t* result = acquire_image();
        if (!result) {
            LOG_ERROR("[ERROR] ImageScale: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result = dst_image;

        outputs["IMAGE"] = make_image_ptr(result);
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
        ImagePtr src_image = std::any_cast<ImagePtr>(inputs.at("image"));
        int crop_x = std::any_cast<int>(inputs.at("x"));
        int crop_y = std::any_cast<int>(inputs.at("y"));
        int crop_width = std::any_cast<int>(inputs.at("width"));
        int crop_height = std::any_cast<int>(inputs.at("height"));

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

        std::vector<uint8_t> dst_data(crop_width * crop_height * src_image->channel);

        int src_stride = src_image->width * src_image->channel;
        int dst_stride = crop_width * src_image->channel;
        int row_bytes = crop_width * src_image->channel;

        for (int y = 0; y < crop_height; y++) {
            uint8_t* src_row = src_image->data + (crop_y + y) * src_stride + crop_x * src_image->channel;
            uint8_t* dst_row = dst_data.data() + y * dst_stride;
            memcpy(dst_row, src_row, row_bytes);
        }

        auto final_data = make_malloc_buffer(dst_data.size());
        if (!final_data) {
            LOG_ERROR("[ERROR] ImageCrop: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_data.get(), dst_data.data(), dst_data.size());

        sd_image_t dst_image = {};
        dst_image.width = crop_width;
        dst_image.height = crop_height;
        dst_image.channel = src_image->channel;
        dst_image.data = final_data.release();

        sd_image_t* result = acquire_image();
        if (!result) {
            LOG_ERROR("[ERROR] ImageCrop: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        *result = dst_image;

        outputs["IMAGE"] = make_image_ptr(result);
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
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("image"));
        UpscalerPtr upscaler = std::any_cast<UpscalerPtr>(inputs.at("upscale_model"));

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

// ============================================================================
// PreviewImage - 预览图像
// ============================================================================
class PreviewImageNode : public Node {
  public:
    std::string get_class_type() const override {
        return "PreviewImage";
    }
    std::string get_category() const override {
        return "image";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"images", "IMAGE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        ImagePtr image = std::any_cast<ImagePtr>(inputs.at("images"));

        if (!image || !image->data) {
            LOG_ERROR("[ERROR] PreviewImage: No image data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        LOG_INFO("\n");
        LOG_INFO("╔══════════════════════════════════════╗\n");
        LOG_INFO("║         [PreviewImage]               ║\n");
        LOG_INFO("║  Size: %4dx%-4d                     ║\n", image->width, image->height);
        LOG_INFO("║  Channels: %d                        ║\n", image->channel);
        LOG_INFO("╚══════════════════════════════════════╝\n");
        LOG_INFO("\n");

        return sd_error_t::OK;
    }
};
REGISTER_NODE("PreviewImage", PreviewImageNode);

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

        std::vector<uint8_t> dst_data(width * height * out_channels);

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

        auto final_data = make_malloc_buffer(dst_data.size());
        if (!final_data) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_data.get(), dst_data.data(), dst_data.size());

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = width;
        result_img->height = height;
        result_img->channel = out_channels;
        result_img->data = final_data.release();

        outputs["IMAGE"] = make_image_ptr(result_img);
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

        std::vector<uint8_t> out_data(dst_w * dst_h * dst_c);
        memcpy(out_data.data(), dst->data, dst_w * dst_h * dst_c);

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

        auto final_data = make_malloc_buffer(out_data.size());
        if (!final_data) {
            LOG_ERROR("[ERROR] ImageCompositeMasked: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        memcpy(final_data.get(), out_data.data(), out_data.size());

        sd_image_t* result_img = acquire_image();
        if (!result_img) {
            LOG_ERROR("[ERROR] ImageCompositeMasked: Out of memory\n");
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result_img->width = dst_w;
        result_img->height = dst_h;
        result_img->channel = dst_c;
        result_img->data = final_data.release();

        outputs["IMAGE"] = make_image_ptr(result_img);
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

        std::vector<uint8_t> dst(pixels);

        for (int i = 0; i < w * h; i++) {
            for (int ch = 0; ch < c; ch++) {
                dst[i * c + ch] = 255 - img->data[i * c + ch];
            }
        }

        auto final_data = make_malloc_buffer(dst.size());
        if (!final_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(final_data.get(), dst.data(), dst.size());

        sd_image_t* result = acquire_image();
        if (!result) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result->width = w;
        result->height = h;
        result->channel = c;
        result->data = final_data.release();
        outputs["IMAGE"] = make_image_ptr(result);
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

        std::vector<uint8_t> dst(w * h * c);

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

        auto final_data = make_malloc_buffer(dst.size());
        if (!final_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(final_data.get(), dst.data(), dst.size());

        sd_image_t* result = acquire_image();
        if (!result) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result->width = w;
        result->height = h;
        result->channel = c;
        result->data = final_data.release();
        outputs["IMAGE"] = make_image_ptr(result);
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

        std::vector<uint8_t> dst(w * h * c);

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

        auto final_data = make_malloc_buffer(dst.size());
        if (!final_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(final_data.get(), dst.data(), dst.size());

        sd_image_t* result = acquire_image();
        if (!result) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result->width = w;
        result->height = h;
        result->channel = c;
        result->data = final_data.release();
        outputs["IMAGE"] = make_image_ptr(result);
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

        std::vector<uint8_t> dst(w * h);

        for (int i = 0; i < w * h; i++) {
            dst[i] = (uint8_t)(0.299f * img->data[i * c + 0] + 0.587f * img->data[i * c + 1] +
                               0.114f * img->data[i * c + 2] + 0.5f);
        }

        auto final_data = make_malloc_buffer(dst.size());
        if (!final_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(final_data.get(), dst.data(), dst.size());

        sd_image_t* result = acquire_image();
        if (!result) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result->width = w;
        result->height = h;
        result->channel = c;
        result->data = final_data.release();
        outputs["IMAGE"] = make_image_ptr(result);
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

        std::vector<uint8_t> dst(w * h * c);

        for (int i = 0; i < w * h; i++) {
            for (int ch = 0; ch < c; ch++) {
                dst[i * c + ch] = img->data[i * c + ch] >= threshold ? 255 : 0;
            }
        }

        auto final_data = make_malloc_buffer(dst.size());
        if (!final_data)
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        memcpy(final_data.get(), dst.data(), dst.size());

        sd_image_t* result = acquire_image();
        if (!result) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        result->width = w;
        result->height = h;
        result->channel = c;
        result->data = final_data.release();
        outputs["IMAGE"] = make_image_ptr(result);
        LOG_INFO("[ImageThreshold] Thresholded %dx%dx%d (threshold=%d)\n", w, h, c, threshold);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageThreshold", ImageThresholdNode);

void init_image_nodes() {
    // 空函数，仅确保本翻译单元被链接
}

} // namespace sdengine
