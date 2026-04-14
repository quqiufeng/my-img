// ============================================================================
// sd-engine/core/node.cpp
// ============================================================================

#include "node.h"
#include <sstream>
#include <iomanip>

namespace sdengine {

std::string Node::compute_hash(const NodeInputs& inputs) const {
    std::stringstream ss;
    ss << get_class_type();
    
    for (const auto& [name, value] : inputs) {
        ss << "|" << name << "=";
        
        // 基础类型的哈希
        if (value.type() == typeid(int)) {
            ss << std::any_cast<int>(value);
        } else if (value.type() == typeid(int64_t)) {
            ss << std::any_cast<int64_t>(value);
        } else if (value.type() == typeid(float)) {
            ss << std::fixed << std::setprecision(6) << std::any_cast<float>(value);
        } else if (value.type() == typeid(double)) {
            ss << std::fixed << std::setprecision(6) << std::any_cast<double>(value);
        } else if (value.type() == typeid(std::string)) {
            ss << std::any_cast<std::string>(value);
        } else if (value.type() == typeid(bool)) {
            ss << (std::any_cast<bool>(value) ? "1" : "0");
        } else {
            ss << "[complex]";
        }
    }
    
    return ss.str();
}

NodeRegistry& NodeRegistry::instance() {
    static NodeRegistry registry;
    return registry;
}

void NodeRegistry::register_node(const std::string& class_type, NodeCreator creator) {
    creators_[class_type] = creator;
}

std::unique_ptr<Node> NodeRegistry::create(const std::string& class_type) const {
    auto it = creators_.find(class_type);
    if (it != creators_.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> NodeRegistry::get_supported_nodes() const {
    std::vector<std::string> result;
    for (const auto& [type, _] : creators_) {
        result.push_back(type);
    }
    return result;
}

bool NodeRegistry::has_node(const std::string& class_type) const {
    return creators_.find(class_type) != creators_.end();
}

} // namespace sdengine
