#include "engine/workflow.h"
#include <queue>
#include <stdexcept>

namespace myimg {

Workflow Workflow::from_json(const nlohmann::json& json) {
    Workflow workflow;
    
    for (const auto& [id, node_json] : json.items()) {
        NodeDef node;
        node.id = id;
        node.class_type = node_json.value("class_type", "");
        
        if (node_json.contains("inputs")) {
            for (const auto& [key, value] : node_json["inputs"].items()) {
                // Simplified: store as string for now
                node.inputs[key] = value.dump();
            }
        }
        
        workflow.nodes_[id] = node;
    }
    
    // Build edges
    for (const auto& [id, node] : workflow.nodes_) {
        for (const auto& [key, value] : node.inputs) {
            // Check if input is a reference [node_id, output_index]
            // This is simplified
        }
    }
    
    return workflow;
}

std::vector<std::string> Workflow::get_execution_order() const {
    // Topological sort (Kahn's algorithm)
    std::map<std::string, int> in_degree;
    for (const auto& [id, _] : nodes_) {
        in_degree[id] = 0;
    }
    
    // Calculate in-degrees
    for (const auto& [id, edges] : edges_) {
        for (const auto& target : edges) {
            in_degree[target]++;
        }
    }
    
    std::queue<std::string> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(id);
        }
    }
    
    std::vector<std::string> order;
    while (!queue.empty()) {
        std::string id = queue.front();
        queue.pop();
        order.push_back(id);
        
        auto it = edges_.find(id);
        if (it != edges_.end()) {
            for (const auto& target : it->second) {
                if (--in_degree[target] == 0) {
                    queue.push(target);
                }
            }
        }
    }
    
    return order;
}

const NodeDef& Workflow::get_node(const std::string& id) const {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) {
        throw std::runtime_error("Node not found: " + id);
    }
    return it->second;
}

bool Workflow::validate(std::string& error_msg) const {
    // Check for cycles
    auto order = get_execution_order();
    if (order.size() != nodes_.size()) {
        error_msg = "Workflow contains cycles";
        return false;
    }
    
    // Check node count limit
    if (nodes_.size() > 100) {
        error_msg = "Workflow too large: " + std::to_string(nodes_.size()) + " nodes";
        return false;
    }
    
    return true;
}

} // namespace myimg
