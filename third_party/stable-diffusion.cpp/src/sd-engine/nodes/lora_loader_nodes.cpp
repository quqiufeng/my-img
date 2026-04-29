// ============================================================================
// sd-engine/nodes/lora_loader_nodes.cpp
// ============================================================================

#include "adapter/sd_adapter.h"
#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// LoRALoader - 加载 LoRA
// ============================================================================
class LoRALoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LoRALoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"lora_name", "STRING", true, std::string("")},
                {"strength_model", "FLOAT", false, 1.0f},
                {"strength_clip", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LORA", "LORA"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string lora_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "lora_name", lora_path));
        float strength_model = get_input_opt<float>(inputs, "strength_model", 1.0f);
        float strength_clip = get_input_opt<float>(inputs, "strength_clip", 1.0f);

        if (lora_path.empty()) {
            LOG_ERROR("[ERROR] LoRALoader: lora_name is required\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        if (!is_path_safe(lora_path)) {
            LOG_ERROR("[ERROR] LoRALoader: Illegal path detected: %s\n", lora_path.c_str());
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        float avg_strength = (strength_model + strength_clip) * 0.5f;
        outputs["LORA"] = LoRAInfo{lora_path, avg_strength};

        LOG_INFO("[LoRALoader] Loaded LoRA: %s (strength=%.2f)\n", lora_path.c_str(), avg_strength);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoRALoader", LoRALoaderNode);

// ============================================================================
// LoRAStack - 多 LoRA 堆叠
// ============================================================================
class LoRAStackNode : public Node {
  public:
    std::string get_class_type() const override {
        return "LoRAStack";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"lora_1", "LORA", false, nullptr},
                {"lora_2", "LORA", false, nullptr},
                {"lora_3", "LORA", false, nullptr},
                {"lora_4", "LORA", false, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"LORA_STACK", "LORA_STACK"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::vector<LoRAInfo> stack;
        for (int i = 1; i <= 4; i++) {
            std::string key = "lora_" + std::to_string(i);
            if (inputs.count(key)) {
                LoRAInfo info;
                if (sd_error_t err = get_input(inputs, key, info); is_error(err)) {
                    return err;
                }
                stack.push_back(info);
                LOG_INFO("[LoRAStack] Stacked LoRA %d: %s (strength=%.2f)\n", i, info.path.c_str(), info.strength);
            }
        }
        outputs["LORA_STACK"] = stack;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("LoRAStack", LoRAStackNode);

// ============================================================================
// ControlNetLoader - 加载 ControlNet 模型
// ============================================================================
class ControlNetLoaderNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ControlNetLoader";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"control_net_name", "STRING", true, std::string("")}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONTROL_NET", "CONTROL_NET"}, {"path", "STRING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string path;
        SD_RETURN_IF_ERROR(get_input(inputs, "control_net_name", path));
        if (path.empty()) {
            LOG_ERROR("[ERROR] ControlNetLoader: control_net_name is required\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        outputs["CONTROL_NET"] = path;
        outputs["path"] = path;
        LOG_INFO("[ControlNetLoader] ControlNet path: %s\n", path.c_str());
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ControlNetLoader", ControlNetLoaderNode);

void init_lora_loader_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
