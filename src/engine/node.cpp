#include "engine/node.h"
#include <iostream>

namespace myimg {

// NodeRegistry implementation
static std::map<std::string, NodeRegistry::CreatorFunc>& get_registry() {
    static std::map<std::string, NodeRegistry::CreatorFunc> registry;
    return registry;
}

void NodeRegistry::register_node(const std::string& name, CreatorFunc creator) {
    get_registry()[name] = creator;
    std::cout << "[NodeRegistry] Registered node: " << name << std::endl;
}

std::unique_ptr<Node> NodeRegistry::create_node(const std::string& name) {
    auto& registry = get_registry();
    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> NodeRegistry::get_registered_nodes() {
    std::vector<std::string> names;
    for (const auto& [name, _] : get_registry()) {
        names.push_back(name);
    }
    return names;
}

} // namespace myimg
