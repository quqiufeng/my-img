// ============================================================================
// sd-engine/core/executor.h
// ============================================================================
/// @file executor.h
/// @brief DAG 执行器
///
/// DAGExecutor 负责按拓扑排序执行 Workflow 中的节点，支持缓存、进度回调、
/// 错误处理以及多线程并行执行无依赖节点。
// ============================================================================

#pragma once

#include "workflow.h"
#include "cache.h"
#include "node.h"
#include <functional>

namespace sdengine {

/// @brief 进度回调函数类型
/// @param node_id 当前正在执行的节点 ID
/// @param current 当前执行进度（已完成的节点数）
/// @param total   总节点数
using ProgressCallback = std::function<void(const std::string& node_id,
                                                int current,
                                                int total)>;

/// @brief 执行配置参数
struct ExecutionConfig {
    bool use_cache = true;                        ///< 是否启用 ExecutionCache
    bool verbose = false;                         ///< 是否打印详细执行日志
    int max_threads = 0;                          ///< 最大线程数（0 = 硬件并发数，1 = 串行）
    std::map<std::string, std::any> overrides;   ///< 全局参数覆盖
};

/// @brief DAG 执行器
///
/// 按拓扑顺序执行工作流节点，支持以下特性：
/// - 节点执行结果缓存（通过 ExecutionCache）
/// - 进度和错误回调
/// - 多线程并行执行同层无依赖节点
class DAGExecutor {
public:
    /// @brief 构造函数
    /// @param cache 可选的 ExecutionCache 指针
    explicit DAGExecutor(ExecutionCache* cache = nullptr);

    /// @brief 执行完整工作流
    /// @param workflow 要执行的工作流
    /// @param config   执行配置
    /// @return 执行结果错误码
    sd_error_t execute(Workflow* workflow, const ExecutionConfig& config);

    /// @brief 设置进度回调
    void set_progress_callback(ProgressCallback cb);

    /// @brief 设置错误回调
    void set_error_callback(std::function<void(const std::string&)> cb);

private:
    ExecutionCache* cache_;
    ProgressCallback progress_cb_;
    std::function<void(const std::string&)> error_cb_;

    /// @brief 准备节点输入（从上游已执行节点和字面量值中获取）
    sd_error_t prepare_inputs(Workflow* workflow,
                              const std::string& node_id,
                              NodeInputs& inputs,
                              std::map<std::string, NodeOutputs>& computed);

    /// @brief 执行单个节点（含缓存查询和存储）
    sd_error_t execute_node(Node* node,
                            const NodeInputs& inputs,
                            NodeOutputs& outputs,
                            const ExecutionConfig& config);
};

} // namespace sdengine
