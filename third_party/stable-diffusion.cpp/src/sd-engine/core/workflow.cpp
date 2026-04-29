// ============================================================================
// sd-engine/core/workflow.cpp
// ============================================================================

#include "workflow.h"
#include "nlohmann/json.hpp"
#include "node.h"
#include <fstream>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace sdengine {

using json = nlohmann::json;

static size_t g_max_nodes = 100; // 默认最大 100 个节点

void Workflow::set_max_nodes(size_t max_nodes) {
    g_max_nodes = max_nodes;
}

size_t Workflow::get_max_nodes() {
    return g_max_nodes;
}

bool Workflow::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return load_from_string(json_str);
}

bool Workflow::load_from_string(const std::string& json_str) {
    clear();
    try {
        json j = json::parse(json_str);
        if (j.contains("format") && j["format"] == "sd-engine") {
            return parse_native_json(json_str);
        }
        if (j.contains("steps") && j["steps"].is_array()) {
            return parse_native_json(json_str);
        }
        return parse_comfyui_json(json_str);
    } catch (const std::exception& e) {
        return false;
    }
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
            if (!node_data.is_object())
                continue;

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
                    if (input_value.is_array() && input_value.size() == 2 && input_value[0].is_string() &&
                        input_value[1].is_number()) {
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

NodeInputs Workflow::get_input_values(const std::string& node_id) const {
    auto it = input_values_.find(node_id);
    return it != input_values_.end() ? it->second : NodeInputs{};
}

std::vector<std::string> Workflow::topological_sort() const {
    std::vector<std::string> result;
    std::unordered_map<std::string, int> in_degree;
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
    // 检查节点数量上限（防止恶意构造超大 DAG）
    if (nodes_.size() > g_max_nodes) {
        error_msg = "Workflow too large: " + std::to_string(nodes_.size()) + " nodes exceeds limit of " +
                    std::to_string(g_max_nodes);
        return false;
    }

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

    // 检查必填端口是否都有值/连接
    for (const auto& [node_id, node_ptr] : nodes_) {
        auto input_defs = node_ptr->get_inputs();
        auto links = get_input_links(node_id);
        auto values = get_input_values(node_id);

        for (const auto& def : input_defs) {
            if (!def.required)
                continue;

            bool has_link = false;
            for (const auto& link : links) {
                if (link.dst_slot >= 0 && link.dst_slot < (int)input_defs.size()) {
                    if (input_defs[link.dst_slot].name == def.name) {
                        has_link = true;
                        break;
                    }
                }
            }

            bool has_value = values.find(def.name) != values.end();

            if (!has_link && !has_value) {
                error_msg =
                    "Node " + node_id + " (" + node_ptr->get_class_type() + ") missing required input: " + def.name;
                return false;
            }
        }
    }

    // 检查端口类型匹配
    for (const auto& [dst_node_id, links] : input_links_) {
        auto dst_node = get_node(dst_node_id);
        if (!dst_node)
            continue;
        auto dst_inputs = dst_node->get_inputs();

        for (const auto& link : links) {
            auto src_node = get_node(link.src_node_id);
            if (!src_node)
                continue;
            auto src_outputs = src_node->get_outputs();

            if (link.src_slot < 0 || link.src_slot >= (int)src_outputs.size()) {
                error_msg = "Invalid output slot " + std::to_string(link.src_slot) + " on node " + link.src_node_id;
                return false;
            }
            if (link.dst_slot < 0 || link.dst_slot >= (int)dst_inputs.size()) {
                error_msg = "Invalid input slot " + std::to_string(link.dst_slot) + " on node " + link.dst_node_id;
                return false;
            }

            const std::string& src_type = src_outputs[link.src_slot].type;
            const std::string& dst_type = dst_inputs[link.dst_slot].type;

            // 允许 MODEL/CLIP/VAE 之间的通用连接（它们都基于 sd_ctx_t）
            bool is_generic_model = (src_type == "MODEL" || src_type == "CLIP" || src_type == "VAE") &&
                                    (dst_type == "MODEL" || dst_type == "CLIP" || dst_type == "VAE");

            if (src_type != dst_type && !is_generic_model) {
                error_msg = "Type mismatch: node " + link.src_node_id + " outputs " + src_type + " but node " +
                            link.dst_node_id + " expects " + dst_type;
                return false;
            }
        }
    }

    return true;
}

bool Workflow::parse_native_json(const std::string& json_str) {
    try {
        json j = json::parse(json_str);
        
        if (!j.contains("steps") || !j["steps"].is_array()) {
            return false;
        }
        
        // 第一遍：创建所有节点并建立 id -> slot_name -> slot_index 映射
        std::unordered_map<std::string, std::unordered_map<std::string, int>> node_output_slots;
        
        for (const auto& step : j["steps"]) {
            if (!step.contains("id") || !step.contains("type")) {
                continue;
            }
            
            std::string node_id = step["id"];
            std::string node_type = step["type"];
            
            auto node = NodeRegistry::instance().create(node_type);
            if (!node) {
                continue;
            }
            
            auto output_defs = node->get_outputs();
            for (int i = 0; i < (int)output_defs.size(); i++) {
                node_output_slots[node_id][output_defs[i].name] = i;
            }
            
            node->set_id(node_id);
            nodes_[node_id] = std::move(node);
        }
        
        // 第二遍：解析输入（字面量值和引用）
        for (const auto& step : j["steps"]) {
            if (!step.contains("id") || !step.contains("type")) {
                continue;
            }
            
            std::string node_id = step["id"];
            auto node = get_node(node_id);
            if (!node) continue;
            
            auto input_defs = node->get_inputs();
            std::unordered_map<std::string, int> input_name_to_slot;
            for (int i = 0; i < (int)input_defs.size(); i++) {
                input_name_to_slot[input_defs[i].name] = i;
            }
            
            if (step.contains("inputs") && step["inputs"].is_object()) {
                for (auto& [input_name, input_value] : step["inputs"].items()) {
                    if (input_value.is_string()) {
                        std::string val_str = input_value.get<std::string>();
                        // 检查是否是引用: $node_id.SLOT
                        if (!val_str.empty() && val_str[0] == '$') {
                            size_t dot_pos = val_str.find('.');
                            if (dot_pos != std::string::npos) {
                                std::string ref_node_id = val_str.substr(1, dot_pos - 1);
                                std::string ref_slot_name = val_str.substr(dot_pos + 1);
                                
                                auto src_it = node_output_slots.find(ref_node_id);
                                if (src_it != node_output_slots.end()) {
                                    auto slot_it = src_it->second.find(ref_slot_name);
                                    if (slot_it != src_it->second.end()) {
                                        Link link;
                                        link.src_node_id = ref_node_id;
                                        link.src_slot = slot_it->second;
                                        link.dst_node_id = node_id;
                                        auto dst_slot_it = input_name_to_slot.find(input_name);
                                        link.dst_slot = (dst_slot_it != input_name_to_slot.end()) ? dst_slot_it->second : -1;
                                        
                                        input_links_[node_id].push_back(link);
                                        output_links_[ref_node_id].push_back(link);
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    
                    // 字面量值
                    auto val = json_to_any(input_value);
                    if (val.has_value()) {
                        input_values_[node_id][input_name] = val;
                    }
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void Workflow::clear() {
    nodes_.clear();
    input_links_.clear();
    output_links_.clear();
    input_values_.clear();
}

} // namespace sdengine
