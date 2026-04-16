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

void init_image_io_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
