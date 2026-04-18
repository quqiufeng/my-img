// ============================================================================
// sd-engine/nodes/latent_nodes.cpp
// ============================================================================
// Latent Nodes 节点入口（薄封装，实际实现分散在子模块中）
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

extern void init_empty_latent_nodes();
extern void init_vae_nodes();
extern void init_sampler_nodes();
extern void init_deep_hires_nodes();

// ============================================================================
// LatentUpscale - Latent 空间上采样（使用最近邻插值）
// ============================================================================
class LatentUpscaleNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LatentUpscale";
    }
    std::string get_category() const override {
        return "latent";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"samples", "LATENT", true, nullptr},
                {"upscale_method", "STRING", false, std::string("nearest-exact")},
                {"width", "INT", true, 0},
                {"height", "INT", true, 0},
                {"crop", "STRING", false, std::string("disabled")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr latent;
        SD_RETURN_IF_ERROR(get_input(inputs, "samples", latent));
        std::string method = get_input_opt<std::string>(inputs, "upscale_method", "nearest-exact");
        int target_width;
        SD_RETURN_IF_ERROR(get_input(inputs, "width", target_width));
        int target_height;
        SD_RETURN_IF_ERROR(get_input(inputs, "height", target_height));

        if (!latent) {
            LOG_ERROR("[ERROR] LatentUpscale: No latent data\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int current_w, current_h, channels;
        sd_latent_get_shape(latent.get(), &current_w, &current_h, &channels);

        if (current_w == 0 || current_h == 0) {
            LOG_ERROR("[ERROR] LatentUpscale: Invalid latent shape\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int target_w = target_width > 0 ? target_width : current_w;
        int target_h = target_height > 0 ? target_height : current_h;

        if (current_w == target_w && current_h == target_h) {
            outputs["LATENT"] = latent;
            LOG_INFO("[LatentUpscale] No resize needed (%dx%d)\n", target_w, target_h);
            return sd_error_t::OK;
        }

        LOG_INFO("[LatentUpscale] Upscaling latent %dx%d -> %dx%d (method=%s, channels=%d)\n", current_w, current_h,
                 target_w, target_h, method.c_str(), channels);

        sd::Tensor<float> result({target_w, target_h, channels, 1});

        if (method == "nearest-exact" || method == "nearest") {
            for (int y = 0; y < target_h; y++) {
                int src_y = (y * current_h) / target_h;
                src_y = std::min(src_y, current_h - 1);
                for (int x = 0; x < target_w; x++) {
                    int src_x = (x * current_w) / target_w;
                    src_x = std::min(src_x, current_w - 1);
                    for (int c = 0; c < channels; c++) {
                        result.data()[((y * target_w + x) * channels + c)] =
                            latent->tensor.data()[((src_y * current_w + src_x) * channels + c)];
                    }
                }
            }
        } else {
            float scale_x = (float)current_w / target_w;
            float scale_y = (float)current_h / target_h;

            for (int y = 0; y < target_h; y++) {
                float fy = y * scale_y;
                int y0 = (int)fy;
                int y1 = std::min(y0 + 1, current_h - 1);
                float dy = fy - y0;

                for (int x = 0; x < target_w; x++) {
                    float fx = x * scale_x;
                    int x0 = (int)fx;
                    int x1 = std::min(x0 + 1, current_w - 1);
                    float dx = fx - x0;

                    for (int c = 0; c < channels; c++) {
                        float v00 = latent->tensor.data()[((y0 * current_w + x0) * channels + c)];
                        float v01 = latent->tensor.data()[((y0 * current_w + x1) * channels + c)];
                        float v10 = latent->tensor.data()[((y1 * current_w + x0) * channels + c)];
                        float v11 = latent->tensor.data()[((y1 * current_w + x1) * channels + c)];

                        float v0 = v00 * (1.0f - dx) + v01 * dx;
                        float v1 = v10 * (1.0f - dx) + v11 * dx;
                        float v = v0 * (1.0f - dy) + v1 * dy;

                        result.data()[((y * target_w + x) * channels + c)] = v;
                    }
                }
            }
        }

        // sd_latent_t is a complete type in stable-diffusion-ext.h with public tensor member
        sd_latent_t* out_latent = nullptr;
        try {
            out_latent = new sd_latent_t;
        } catch (const std::bad_alloc&) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        out_latent->tensor = std::move(result);

        outputs["LATENT"] = make_latent_ptr(out_latent);
        LOG_INFO("[LatentUpscale] Upscaled to %dx%d\n", target_w, target_h);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LatentUpscale", LatentUpscaleNode);

// ============================================================================
// LatentComposite - 在 Latent 空间合成两个 latent
// ============================================================================
class LatentCompositeNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LatentComposite";
    }
    std::string get_category() const override {
        return "latent";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"destination", "LATENT", true, nullptr},
                {"source", "LATENT", true, nullptr},
                {"x", "INT", false, 0},
                {"y", "INT", false, 0},
                {"feather", "INT", false, 0}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        LatentPtr dst;
        SD_RETURN_IF_ERROR(get_input(inputs, "destination", dst));
        LatentPtr src;
        SD_RETURN_IF_ERROR(get_input(inputs, "source", src));
        int offset_x = get_input_opt<int>(inputs, "x", 0);
        int offset_y = get_input_opt<int>(inputs, "y", 0);
        int feather = get_input_opt<int>(inputs, "feather", 0);

        if (!dst || !src) {
            LOG_ERROR("[ERROR] LatentComposite: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        int dst_w, dst_h, dst_c;
        int src_w, src_h, src_c;
        sd_latent_get_shape(dst.get(), &dst_w, &dst_h, &dst_c);
        sd_latent_get_shape(src.get(), &src_w, &src_h, &src_c);

        if (dst_c != src_c) {
            LOG_ERROR("[ERROR] LatentComposite: Channel mismatch (%d vs %d)\n", dst_c, src_c);
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        LOG_INFO("[LatentComposite] Compositing %dx%d onto %dx%d at (%d,%d) feather=%d\n", src_w, src_h, dst_w, dst_h,
                 offset_x, offset_y, feather);

        // sd_latent_t is a complete type in stable-diffusion-ext.h with public tensor member
        sd_latent_t* result = nullptr;
        try {
            result = new sd_latent_t;
        } catch (const std::bad_alloc&) {
            return sd_error_t::ERROR_MEMORY_ALLOCATION;
        }
        // Deep copy the destination tensor
        result->tensor = dst->tensor;

        int channels = dst_c;

        for (int y = 0; y < src_h; y++) {
            int dst_y = offset_y + y;
            if (dst_y < 0 || dst_y >= dst_h)
                continue;

            for (int x = 0; x < src_w; x++) {
                int dst_x = offset_x + x;
                if (dst_x < 0 || dst_x >= dst_w)
                    continue;

                float alpha = 1.0f;
                if (feather > 0) {
                    int dist_x = std::min({x, src_w - 1 - x, dst_x, dst_w - 1 - dst_x});
                    int dist_y = std::min({y, src_h - 1 - y, dst_y, dst_h - 1 - dst_y});
                    int dist = std::min(dist_x, dist_y);
                    if (dist < feather) {
                        alpha = (float)dist / feather;
                    }
                }

                for (int c = 0; c < channels; c++) {
                    float src_val = src->tensor.data()[((y * src_w + x) * channels + c)];
                    float dst_val = result->tensor.data()[((dst_y * dst_w + dst_x) * channels + c)];
                    result->tensor.data()[((dst_y * dst_w + dst_x) * channels + c)] =
                        dst_val * (1.0f - alpha) + src_val * alpha;
                }
            }
        }

        outputs["LATENT"] = make_latent_ptr(result);
        LOG_INFO("[LatentComposite] Done\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LatentComposite", LatentCompositeNode);

void init_latent_nodes() {
    init_empty_latent_nodes();
    init_vae_nodes();
    init_sampler_nodes();
    init_deep_hires_nodes();
}

} // namespace sdengine
