// ============================================================================
// sd-engine/nodes/conditioning_combine_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"

namespace sdengine {

// ============================================================================
// ConditioningCombine - 条件合并
// ============================================================================
class ConditioningCombineNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ConditioningCombine";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning_1", "CONDITIONING", true, nullptr}, {"conditioning_2", "CONDITIONING", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond1;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning_1", cond1));
        ConditioningPtr cond2;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning_2", cond2));

        if (!cond1 || !cond2) {
            LOG_ERROR("[ERROR] ConditioningCombine: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* combined = sd_conditioning_concat(cond1.get(), cond2.get());
        if (!combined) {
            LOG_ERROR("[ERROR] ConditioningCombine: Failed to combine conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(combined);
        LOG_INFO("[ConditioningCombine] Combined two conditionings\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningCombine", ConditioningCombineNode);

// ============================================================================
// ConditioningConcat - 条件拼接（与 Combine 行为相同，对齐 ComfyUI 命名）
// ============================================================================
class ConditioningConcatNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ConditioningConcat";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning_to", "CONDITIONING", true, nullptr},
                {"conditioning_from", "CONDITIONING", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond_to;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning_to", cond_to));
        ConditioningPtr cond_from;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning_from", cond_from));

        if (!cond_to || !cond_from) {
            LOG_ERROR("[ERROR] ConditioningConcat: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* concat = sd_conditioning_concat(cond_to.get(), cond_from.get());
        if (!concat) {
            LOG_ERROR("[ERROR] ConditioningConcat: Failed to concat conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(concat);
        LOG_INFO("[ConditioningConcat] Concatenated two conditionings\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningConcat", ConditioningConcatNode);

// ============================================================================
// ConditioningAverage - 条件加权平均
// ============================================================================
class ConditioningAverageNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ConditioningAverage";
    }
    std::string get_category() const override {
        return "conditioning";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"conditioning_to", "CONDITIONING", true, nullptr},
                {"conditioning_from", "CONDITIONING", true, nullptr},
                {"conditioning_to_strength", "FLOAT", false, 1.0f}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ConditioningPtr cond_to;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning_to", cond_to));
        ConditioningPtr cond_from;
        SD_RETURN_IF_ERROR(get_input(inputs, "conditioning_from", cond_from));
        float strength = get_input_opt<float>(inputs, "conditioning_to_strength", 1.0f);

        if (!cond_to || !cond_from) {
            LOG_ERROR("[ERROR] ConditioningAverage: Missing inputs\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        sd_conditioning_t* averaged = sd_conditioning_average(cond_to.get(), cond_from.get(), strength);
        if (!averaged) {
            LOG_ERROR("[ERROR] ConditioningAverage: Failed to average conditionings\n");
            return sd_error_t::ERROR_EXECUTION_FAILED;
        }

        outputs["CONDITIONING"] = make_conditioning_ptr(averaged);
        LOG_INFO("[ConditioningAverage] Averaged conditionings (strength=%.2f)\n", strength);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConditioningAverage", ConditioningAverageNode);

void init_conditioning_combine_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
