// ============================================================================
// sd-engine/nodes/test_nodes.cpp
// ============================================================================
//
// 测试用节点，用于验证 sd-engine 核心功能
// ============================================================================

#include "core/node.h"
#include "nodes/node_utils.h"
#include <cstdio>
#include <sstream>

namespace sdengine {

// 常量输出节点
class ConstantIntNode : public Node {
  public:
    std::string get_class_type() const override {
        return "ConstantInt";
    }
    std::string get_category() const override {
        return "test";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"value", "INT", false, 42}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"value", "INT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int value;
        if (sd_error_t err = get_input(inputs, "value", value); is_error(err)) {
            return err;
        }
        outputs["value"] = value;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("ConstantInt", ConstantIntNode);

// 加法节点
class AddIntNode : public Node {
  public:
    std::string get_class_type() const override {
        return "AddInt";
    }
    std::string get_category() const override {
        return "test";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"a", "INT", true, 0}, {"b", "INT", true, 0}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"result", "INT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int a;
        if (sd_error_t err = get_input(inputs, "a", a); is_error(err)) {
            return err;
        }
        int b;
        if (sd_error_t err = get_input(inputs, "b", b); is_error(err)) {
            return err;
        }
        outputs["result"] = a + b;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("AddInt", AddIntNode);

// 乘法节点
class MultiplyIntNode : public Node {
  public:
    std::string get_class_type() const override {
        return "MultiplyInt";
    }
    std::string get_category() const override {
        return "test";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"a", "INT", true, 0}, {"b", "INT", true, 0}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"result", "INT"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int a;
        if (sd_error_t err = get_input(inputs, "a", a); is_error(err)) {
            return err;
        }
        int b;
        if (sd_error_t err = get_input(inputs, "b", b); is_error(err)) {
            return err;
        }
        outputs["result"] = a * b;
        return sd_error_t::OK;
    }
};
REGISTER_NODE("MultiplyInt", MultiplyIntNode);

// 打印输出节点
class PrintIntNode : public Node {
  public:
    std::string get_class_type() const override {
        return "PrintInt";
    }
    std::string get_category() const override {
        return "test";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"value", "INT", true, 0}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        int value;
        if (sd_error_t err = get_input(inputs, "value", value); is_error(err)) {
            return err;
        }
        LOG_INFO("[PrintInt] Result: %d\n", value);
        return sd_error_t::OK;
    }
};
REGISTER_NODE("PrintInt", PrintIntNode);

// 显式初始化函数，确保链接器保留本文件
void init_test_nodes() {
    // REGISTER_NODE 宏中的静态变量会在程序启动时自动注册节点。
    // 但静态库中的对象文件可能被链接器优化掉。
    // 通过提供一个被显式调用的函数，确保 test_nodes.cpp 被链接进最终可执行文件。
}

} // namespace sdengine
