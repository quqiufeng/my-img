// ============================================================================
// sd-engine/nodes/image_io_nodes.cpp
// ============================================================================
// 图像 I/O 节点实现（加载、保存、预览）
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
        std::string image_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", image_path));

        if (image_path.empty()) {
            LOG_ERROR("[ERROR] LoadImage: image path is required\n");
            return sd_error_t::ERROR_FILE_IO;
        }

        if (!is_path_safe(image_path)) {
            LOG_ERROR("[ERROR] LoadImage: Illegal path detected: %s\n", image_path.c_str());
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[LoadImage] Loading: %s\n", image_path.c_str());

        int w, h, c;
        uint8_t* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
        if (!data) {
            LOG_ERROR("[ERROR] LoadImage: Failed to load %s\n", image_path.c_str());
            return sd_error_t::ERROR_FILE_IO;
        }

        sd_image_t* image = new sd_image_t();
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
        std::string image_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", image_path));
        std::string channel = get_input_opt<std::string>(inputs, "channel", "alpha");

        if (image_path.empty()) {
            LOG_ERROR("[ERROR] LoadImageMask: image path is required\n");
            return sd_error_t::ERROR_FILE_IO;
        }

        if (!is_path_safe(image_path)) {
            LOG_ERROR("[ERROR] LoadImageMask: Illegal path detected: %s\n", image_path.c_str());
            return sd_error_t::ERROR_INVALID_INPUT;
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

        auto mask_img = create_image_ptr(w, h, 3, std::move(mask_data));
        if (!mask_img) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }

        outputs["MASK"] = mask_img;
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
        ImagePtr image;
        SD_RETURN_IF_ERROR(get_input(inputs, "images", image));
        std::string prefix =
            get_input_opt<std::string>(inputs, "filename_prefix", "sd-engine");

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
        ImagePtr image;
        SD_RETURN_IF_ERROR(get_input(inputs, "images", image));

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
        return {
            {"image", "IMAGE", true, nullptr},
            {"width", "INT", true, 512},
            {"height", "INT", true, 512},
            {"method", "STRING", false, std::string("bilinear")}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr src;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", src));
        SD_RETURN_IF_NULL(src.get(), sd_error_t::ERROR_INVALID_INPUT);

        int target_w = get_input_opt<int>(inputs, "width", 512);
        int target_h = get_input_opt<int>(inputs, "height", 512);
        std::string method = get_input_opt<std::string>(inputs, "method", "bilinear");

        if (target_w <= 0 || target_h <= 0) {
            LOG_ERROR("[ImageScale] Invalid target size: %dx%d\n", target_w, target_h);
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[ImageScale] Resizing %dx%d -> %dx%d (method=%s)\n",
                 src->width, src->height, target_w, target_h, method.c_str());

        auto dst = resize_image(src.get(), target_w, target_h);
        if (!dst) {
            LOG_ERROR("[ImageScale] Resize failed\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["IMAGE"] = dst;
        LOG_INFO("[ImageScale] Resized to %dx%d\n", target_w, target_h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ImageScale", ImageScaleNode);

void init_image_io_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
