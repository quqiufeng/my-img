// ============================================================================
// sd-engine/nodes/test_nodes.cpp
// ============================================================================
// 
// 测试用节点，用于验证 sd-engine 核心功能
// ============================================================================

#include "core/node.h"
#include <cstdio>
#include <sstream>

namespace sdengine {

// 常量输出节点
class ConstantIntNode : public Node {
public:
    std::string get_class_type() const override { return "ConstantInt"; }
    std::string get_category() const override { return "test"; }
    
    std::vector<PortDef> get_inputs() const override {
        return {{"value", "INT", false, 42}};
    }
    
    std::vector<PortDef> get_outputs() const override {
        return {{"value", "INT"}};
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int value = std::any_cast<int>(inputs.at("value"));
        outputs["value"] = value;
        return true;
    }
};
REGISTER_NODE("ConstantInt", ConstantIntNode);

// 加法节点
class AddIntNode : public Node {
public:
    std::string get_class_type() const override { return "AddInt"; }
    std::string get_category() const override { return "test"; }
    
    std::vector<PortDef> get_inputs() const override {
        return {
            {"a", "INT", true, 0},
            {"b", "INT", true, 0}
        };
    }
    
    std::vector<PortDef> get_outputs() const override {
        return {{"result", "INT"}};
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int a = std::any_cast<int>(inputs.at("a"));
        int b = std::any_cast<int>(inputs.at("b"));
        outputs["result"] = a + b;
        return true;
    }
};
REGISTER_NODE("AddInt", AddIntNode);

// 乘法节点
class MultiplyIntNode : public Node {
public:
    std::string get_class_type() const override { return "MultiplyInt"; }
    std::string get_category() const override { return "test"; }
    
    std::vector<PortDef> get_inputs() const override {
        return {
            {"a", "INT", true, 0},
            {"b", "INT", true, 0}
        };
    }
    
    std::vector<PortDef> get_outputs() const override {
        return {{"result", "INT"}};
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int a = std::any_cast<int>(inputs.at("a"));
        int b = std::any_cast<int>(inputs.at("b"));
        outputs["result"] = a * b;
        return true;
    }
};
REGISTER_NODE("MultiplyInt", MultiplyIntNode);

// 打印输出节点
class PrintIntNode : public Node {
public:
    std::string get_class_type() const override { return "PrintInt"; }
    std::string get_category() const override { return "test"; }
    
    std::vector<PortDef> get_inputs() const override {
        return {{"value", "INT", true, 0}};
    }
    
    std::vector<PortDef> get_outputs() const override {
        return {};
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        int value = std::any_cast<int>(inputs.at("value"));
        printf("[PrintInt] Result: %d\n", value);
        return true;
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
