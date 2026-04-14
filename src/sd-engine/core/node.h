// ============================================================================
// sd-engine/core/node.h
// ============================================================================
// 
// 节点基类定义
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <any>
#include <functional>
#include <unordered_map>

namespace sdengine {

// 端口定义
struct PortDef {
    std::string name;
    std::string type;
    bool required = true;
    std::any default_value;
};

// 节点输入/输出
using NodeInputs  = std::map<std::string, std::any>;
using NodeOutputs = std::map<std::string, std::any>;

// 节点连接关系
struct Link {
    std::string src_node_id;
    int src_slot = 0;
    std::string dst_node_id;
    int dst_slot = 0;
};

// 节点基类
class Node {
public:
    virtual ~Node() = default;
    
    void set_id(const std::string& id) { id_ = id; }
    std::string get_id() const { return id_; }
    
    // 节点元信息
    virtual std::string get_class_type() const = 0;
    virtual std::string get_category() const { return "general"; }
    
    // 输入输出定义
    virtual std::vector<PortDef> get_inputs() const = 0;
    virtual std::vector<PortDef> get_outputs() const = 0;
    
    // 执行
    virtual bool execute(const NodeInputs& inputs, NodeOutputs& outputs) = 0;
    
    // 计算缓存哈希
    virtual std::string compute_hash(const NodeInputs& inputs) const;

protected:
    std::string id_;
};

// 节点工厂
using NodeCreator = std::function<std::unique_ptr<Node>()>;

class NodeRegistry {
public:
    static NodeRegistry& instance();
    
    void register_node(const std::string& class_type, NodeCreator creator);
    std::unique_ptr<Node> create(const std::string& class_type) const;
    std::vector<std::string> get_supported_nodes() const;
    bool has_node(const std::string& class_type) const;

private:
    std::unordered_map<std::string, NodeCreator> creators_;
};

// 注册宏
#define REGISTER_NODE(class_type, node_class) \
    static bool _registered_##node_class = []() { \
        sdengine::NodeRegistry::instance().register_node(class_type, []() { \
            return std::make_unique<node_class>(); \
        }); \
        return true; \
    }();

} // namespace sdengine
