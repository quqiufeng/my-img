// ============================================================================
// sd-engine/nodes/vae_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// VAEEncode - 真正的 VAE 编码
// ============================================================================
class VAEEncodeNode : public Node {
  public:
    std::string get_class_type() const override {
        return "VAEEncode";
    }
    std::string get_category() const override {
        return "latent";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"pixels", "IMAGE", true, nullptr}, {"vae", "VAE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        sd_image_t image;
        SD_RETURN_IF_ERROR(get_input(inputs, "pixels", image));
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "vae");

        if (!image.data) {
            LOG_ERROR("[ERROR] VAEEncode: No image data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_latent_t* latent = sd_encode_image(sd_ctx, &image);
        if (!latent) {
            LOG_ERROR("[ERROR] VAEEncode: Failed to encode image\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        LOG_INFO("[VAEEncode] Image encoded to latent\n");
        outputs["LATENT"] = make_latent_ptr(latent);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("VAEEncode", VAEEncodeNode);

// ============================================================================
// VAEDecode - 真正的 VAE 解码
// ============================================================================
class VAEDecodeNode : public Node {
  public:
    std::string get_class_type() const override {
        return "VAEDecode";
    }
    std::string get_category() const override {
        return "latent";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"samples", "LATENT", true, nullptr}, {"vae", "VAE", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr latent;
        SD_RETURN_IF_ERROR(get_input(inputs, "samples", latent));
        sd_ctx_t* sd_ctx = extract_sd_ctx(inputs, "vae");

        if (!latent) {
            LOG_ERROR("[ERROR] VAEDecode: No latent data\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        sd_image_t* image = sd_decode_latent(sd_ctx, latent.get());
        if (!image) {
            LOG_ERROR("[ERROR] VAEDecode: Failed to decode latent\n");
            return sd_error_t::ERROR_DECODING_FAILED;
        }

        LOG_INFO("[VAEDecode] Latent decoded: %dx%d\n", image->width, image->height);
        outputs["IMAGE"] = make_image_ptr(image);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("VAEDecode", VAEDecodeNode);

void init_vae_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
