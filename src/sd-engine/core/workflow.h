// ============================================================================
// sd-engine/core/workflow.h
// ============================================================================
/// @file workflow.h
/// @brief 工作流定义和解析
///
/// Workflow 类负责管理节点集合、连接关系，并提供拓扑排序和验证功能。
/// 支持从 ComfyUI 格式的 JSON 文件加载工作流。
// ============================================================================

#pragma once

#include "node.h"
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <any>

namespace sdengine {

/// @brief 工作流类
///
/// 管理一组节点及其连接关系，负责从 JSON 解析、验证工作流合法性，
/// 并生成拓扑排序后的执行顺序。
class Workflow {
public:
    Workflow() = default;
    ~Workflow() = default;

    /// @brief 从 JSON 文件加载工作流
    /// @param path JSON 文件路径
    /// @return 是否加载成功
    bool load_from_file(const std::string& path);

    /// @brief 从 JSON 字符串加载工作流
    /// @param json_str JSON 字符串内容
    /// @return 是否加载成功
    bool load_from_string(const std::string& json_str);

    /// @brief 根据 ID 获取节点（非 const）
    Node* get_node(const std::string& id);

    /// @brief 根据 ID 获取节点（const）
    const Node* get_node(const std::string& id) const;

    /// @brief 获取所有节点指针
    std::vector<Node*> get_all_nodes();

    /// @brief 获取所有节点 ID 列表
    std::vector<std::string> get_all_node_ids() const;

    /// @brief 获取指定节点的所有输入连接
    std::vector<Link> get_input_links(const std::string& node_id) const;

    /// @brief 获取指定节点的所有输出连接
    std::vector<Link> get_output_links(const std::string& node_id) const;

    /// @brief 获取节点的字面量输入值（非连接值）
    std::map<std::string, std::any> get_input_values(const std::string& node_id) const;

    /// @brief 拓扑排序，返回节点的执行顺序
    ///
    /// 使用 Kahn 算法对工作流图进行拓扑排序。若图中存在环，返回空列表。
    std::vector<std::string> topological_sort() const;

    /// @brief 验证工作流有效性
    /// @param error_msg 出错时填充错误信息
    /// @return 是否验证通过
    bool validate(std::string& error_msg) const;

    /// @brief 清空所有节点和连接
    void clear();

private:
    std::map<std::string, std::unique_ptr<Node>> nodes_;                    ///< 节点实例
    std::map<std::string, std::vector<Link>> input_links_;                 ///< 节点的输入连接
    std::map<std::string, std::vector<Link>> output_links_;                ///< 节点的输出连接
    std::map<std::string, std::map<std::string, std::any>> input_values_;  ///< 节点的字面量输入值

    /// @brief 解析 ComfyUI JSON 格式
    bool parse_comfyui_json(const std::string& json_str);
};

} // namespace sdengine
