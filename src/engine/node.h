#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <any>
#include <optional>
#include <functional>

namespace myimg {

// Forward declarations
class Node;
class Workflow;
class Executor;

// Port definition
struct PortDef {
    std::string name;
    std::string type;
    bool required;
    std::any default_value;
};

// Node inputs/outputs
using NodeInputs = std::map<std::string, std::any>;
using NodeOutputs = std::map<std::string, std::any>;

// Node base class
class Node {
public:
    virtual ~Node() = default;
    
    virtual std::string get_class_type() const = 0;
    virtual std::string get_category() const = 0;
    
    virtual std::vector<PortDef> get_inputs() const = 0;
    virtual std::vector<PortDef> get_outputs() const = 0;
    
    virtual NodeOutputs execute(const NodeInputs& inputs) = 0;
    
    virtual std::string compute_hash(const NodeInputs& inputs) const {
        return get_class_type();
    }
};

// Node factory
class NodeRegistry {
public:
    using CreatorFunc = std::function<std::unique_ptr<Node>()>;
    
    static void register_node(const std::string& name, CreatorFunc creator);
    static std::unique_ptr<Node> create_node(const std::string& name);
    static std::vector<std::string> get_registered_nodes();
};

// Auto-registration macro
#define REGISTER_NODE(name, class_type) \
    static struct class_type##Registrar { \
        class_type##Registrar() { \
            myimg::NodeRegistry::register_node(name, []() -> std::unique_ptr<myimg::Node> { \
                return std::make_unique<class_type>(); \
            }); \
        } \
    } class_type##Instance;

} // namespace myimg
