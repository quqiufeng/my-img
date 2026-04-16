// ============================================================================
// sd-engine/nodes/ipadapter_loader_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// IPAdapterLoader - 加载 IPAdapter 模型
// ============================================================================
class IPAdapterLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "IPAdapterLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"ipadapter_file", "STRING", true, std::string("")},
                {"cross_attention_dim", "INT", false, 768},
                {"num_tokens", "INT", false, 4},
                {"clip_embeddings_dim", "INT", false, 1024}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IPADAPTER", "IPADAPTER"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path;
        if (sd_error_t err = get_input(inputs, "ipadapter_file", path); is_error(err)) {
            return err;
        }
        int cross_attention_dim = get_input_opt<int>(inputs, "cross_attention_dim", 768);
        int num_tokens = get_input_opt<int>(inputs, "num_tokens", 4);
        int clip_embeddings_dim = get_input_opt<int>(inputs, "clip_embeddings_dim", 1024);

        if (path.empty()) {
            LOG_ERROR("[ERROR] IPAdapterLoader: ipadapter_file is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        outputs["IPADAPTER"] = IPAdapterInfo{path, cross_attention_dim, num_tokens, clip_embeddings_dim, 1.0f};

        LOG_INFO("[IPAdapterLoader] Loaded IPAdapter: %s (dim=%d, tokens=%d, clip_dim=%d)\n", path.c_str(),
                 cross_attention_dim, num_tokens, clip_embeddings_dim);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("IPAdapterLoader", IPAdapterLoaderNode);

void init_ipadapter_loader_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
