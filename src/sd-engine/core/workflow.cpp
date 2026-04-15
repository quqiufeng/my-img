// ============================================================================
// sd-engine/core/workflow.cpp
// ============================================================================

#include "workflow.h"
#include "node.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <queue>
#include <set>
#include <stdexcept>

namespace sdengine {

using json = nlohmann::json;

bool Workflow::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string json_str((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
    return load_from_string(json_str);
}

bool Workflow::load_from_string(const std::string& json_str) {
    clear();
    return parse_comfyui_json(json_str);
}

// 辅助函数：将 json 值转换为 std::any
static std::any json_to_any(const json& j) {
    if (j.is_number_integer()) {
        return j.get<int>();
    } else if (j.is_number_float()) {
        return j.get<double>();
    } else if (j.is_string()) {
        return j.get<std::string>();
    } else if (j.is_boolean()) {
        return j.get<bool>();
    }
    return std::any();
}

bool Workflow::parse_comfyui_json(const std::string& json_str) {
    try {
        json j = json::parse(json_str);
        
        // 解析每个节点
        for (auto& [node_id, node_data] : j.items()) {
            if (!node_data.is_object()) continue;
            
            // 获取节点类型
            std::string class_type = node_data.value("class_type", "");
            if (class_type.empty()) {
                continue;
            }
            
            // 创建节点实例
            auto node = NodeRegistry::instance().create(class_type);
            if (!node) {
                // 不支持的节点类型，跳过
                continue;
            }
            
            // 在 move 之前先获取输入定义
            auto input_defs = node->get_inputs();
            
            node->set_id(node_id);
            nodes_[node_id] = std::move(node);
            
            // 解析输入连接和字面量值
            if (node_data.contains("inputs") && node_data["inputs"].is_object()) {
                for (auto& [input_name, input_value] : node_data["inputs"].items()) {
                    // 检查是否是连接引用 [node_id, slot_index]
                    if (input_value.is_array() && input_value.size() == 2 && 
                        input_value[0].is_string() && input_value[1].is_number()) {
                        
                        Link link;
                        link.src_node_id = input_value[0].get<std::string>();
                        link.src_slot = input_value[1].get<int>();
                        link.dst_node_id = node_id;
                        // 根据输入名称推断 dst_slot
                        link.dst_slot = -1;
                        for (int i = 0; i < (int)input_defs.size(); i++) {
                            if (input_defs[i].name == input_name) {
                                link.dst_slot = i;
                                break;
                            }
                        }

                        input_links_[node_id].push_back(link);
                        output_links_[link.src_node_id].push_back(link);
                    } else {
                        // 字面量值
                        auto val = json_to_any(input_value);
                        if (val.has_value()) {
                            input_values_[node_id][input_name] = val;
                        }
                    }
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

Node* Workflow::get_node(const std::string& id) {
    auto it = nodes_.find(id);
    return it != nodes_.end() ? it->second.get() : nullptr;
}

const Node* Workflow::get_node(const std::string& id) const {
    auto it = nodes_.find(id);
    return it != nodes_.end() ? it->second.get() : nullptr;
}

std::vector<Node*> Workflow::get_all_nodes() {
    std::vector<Node*> result;
    for (auto& [_, node] : nodes_) {
        result.push_back(node.get());
    }
    return result;
}

std::vector<std::string> Workflow::get_all_node_ids() const {
    std::vector<std::string> result;
    for (const auto& [id, _] : nodes_) {
        result.push_back(id);
    }
    return result;
}

std::vector<Link> Workflow::get_input_links(const std::string& node_id) const {
    auto it = input_links_.find(node_id);
    return it != input_links_.end() ? it->second : std::vector<Link>{};
}

std::vector<Link> Workflow::get_output_links(const std::string& node_id) const {
    auto it = output_links_.find(node_id);
    return it != output_links_.end() ? it->second : std::vector<Link>{};
}

std::map<std::string, std::any> Workflow::get_input_values(const std::string& node_id) const {
    auto it = input_values_.find(node_id);
    return it != input_values_.end() ? it->second : std::map<std::string, std::any>{};
}

std::vector<std::string> Workflow::topological_sort() const {
    std::vector<std::string> result;
    std::map<std::string, int> in_degree;
    std::queue<std::string> queue;
    
    // 初始化入度
    for (const auto& [id, _] : nodes_) {
        in_degree[id] = 0;
    }
    
    // 计算入度
    for (const auto& [id, links] : input_links_) {
        in_degree[id] = links.size();
    }
    
    // 找到入度为 0 的节点
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(id);
        }
    }
    
    // Kahn 算法
    while (!queue.empty()) {
        std::string u = queue.front();
        queue.pop();
        result.push_back(u);
        
        // 找到所有 u 的下游节点
        auto it = output_links_.find(u);
        if (it != output_links_.end()) {
            for (const auto& link : it->second) {
                std::string v = link.dst_node_id;
                in_degree[v]--;
                if (in_degree[v] == 0) {
                    queue.push(v);
                }
            }
        }
    }
    
    return result;
}

bool Workflow::validate(std::string& error_msg) const {
    // 检查是否有环
    auto sorted = topological_sort();
    if (sorted.size() != nodes_.size()) {
        error_msg = "Workflow contains cycles";
        return false;
    }
    
    // 检查所有引用的节点是否存在
    for (const auto& [id, links] : input_links_) {
        for (const auto& link : links) {
            if (nodes_.find(link.src_node_id) == nodes_.end()) {
                error_msg = "Referenced node not found: " + link.src_node_id;
                return false;
            }
        }
    }
    
    return true;
}

void Workflow::clear() {
    nodes_.clear();
    input_links_.clear();
    output_links_.clear();
    input_values_.clear();
}

} // namespace sdengine
