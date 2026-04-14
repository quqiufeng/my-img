// ============================================================================
// sd-engine/core/executor.h
// ============================================================================
// 
// DAG 执行器
// ============================================================================

#pragma once

#include "workflow.h"
#include "cache.h"
#include <functional>

namespace sdengine {

// 进度回调
using ProgressCallback = std::function<void(const std::string& node_id, 
                                               int current, 
                                               int total)>;

// 执行配置
struct ExecutionConfig {
    bool use_cache = true;
    bool verbose = false;
    std::map<std::string, std::any> overrides;  // 参数覆盖
};

class DAGExecutor {
public:
    explicit DAGExecutor(ExecutionCache* cache = nullptr);
    
    // 执行完整工作流
    bool execute(Workflow* workflow, const ExecutionConfig& config);
    
    // 设置回调
    void set_progress_callback(ProgressCallback cb);
    void set_error_callback(std::function<void(const std::string&)> cb);

private:
    ExecutionCache* cache_;
    ProgressCallback progress_cb_;
    std::function<void(const std::string&)> error_cb_;
    
    // 准备节点输入（从上游节点获取）
    bool prepare_inputs(Workflow* workflow, 
                        const std::string& node_id,
                        NodeInputs& inputs,
                        std::map<std::string, NodeOutputs>& computed);
    
    // 执行单个节点
    bool execute_node(Node* node, 
                      const NodeInputs& inputs, 
                      NodeOutputs& outputs,
                      const ExecutionConfig& config);
};

} // namespace sdengine
