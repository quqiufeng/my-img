// ============================================================================
// sd-engine/core/workflow.h
// ============================================================================
// 
// 工作流定义和解析
// ============================================================================

#pragma once

#include "node.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <any>

namespace sdengine {

// 工作流类
class Workflow {
public:
    Workflow() = default;
    ~Workflow() = default;
    
    // 从 JSON 加载
    bool load_from_file(const std::string& path);
    bool load_from_string(const std::string& json_str);
    
    // 节点管理
    Node* get_node(const std::string& id);
    const Node* get_node(const std::string& id) const;
    std::vector<Node*> get_all_nodes();
    std::vector<std::string> get_all_node_ids() const;
    
    // 连接关系
    std::vector<Link> get_input_links(const std::string& node_id) const;
    std::vector<Link> get_output_links(const std::string& node_id) const;
    
    // 获取节点的字面量输入值（非连接值）
    std::map<std::string, std::any> get_input_values(const std::string& node_id) const;
    
    // 拓扑排序（执行顺序）
    std::vector<std::string> topological_sort() const;
    
    // 验证工作流有效性
    bool validate(std::string& error_msg) const;
    
    // 清空
    void clear();

private:
    std::map<std::string, std::unique_ptr<Node>> nodes_;
    std::map<std::string, std::vector<Link>> input_links_;   // 节点的输入连接
    std::map<std::string, std::vector<Link>> output_links_;  // 节点的输出连接
    std::map<std::string, std::map<std::string, std::any>> input_values_;  // 节点的字面量输入值
    
    // 解析 ComfyUI JSON 格式
    bool parse_comfyui_json(const std::string& json_str);
};

} // namespace sdengine
