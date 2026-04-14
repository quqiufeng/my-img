// ============================================================================
// sd-engine/core/executor.cpp
// ============================================================================

#include "executor.h"
#include <cstdio>

namespace sdengine {

DAGExecutor::DAGExecutor(ExecutionCache* cache) : cache_(cache) {}

void DAGExecutor::set_progress_callback(ProgressCallback cb) {
    progress_cb_ = cb;
}

void DAGExecutor::set_error_callback(std::function<void(const std::string&)> cb) {
    error_cb_ = cb;
}

bool DAGExecutor::execute(Workflow* workflow, const ExecutionConfig& config) {
    if (!workflow) {
        if (error_cb_) error_cb_("Workflow is null");
        return false;
    }
    
    std::string error_msg;
    if (!workflow->validate(error_msg)) {
        if (error_cb_) error_cb_(error_msg);
        return false;
    }
    
    auto order = workflow->topological_sort();
    int total = (int)order.size();
    int current = 0;
    
    // 存储已计算节点的输出
    std::map<std::string, NodeOutputs> computed;
    
    for (const auto& node_id : order) {
        current++;
        
        Node* node = workflow->get_node(node_id);
        if (!node) continue;
        
        if (progress_cb_) {
            progress_cb_(node_id, current, total);
        }
        
        if (config.verbose) {
            printf("[Executor] Executing node %s (%s) [%d/%d]\n", 
                   node_id.c_str(), node->get_class_type().c_str(),
                   current, total);
        }
        
        // 准备输入
        NodeInputs inputs;
        if (!prepare_inputs(workflow, node_id, inputs, computed)) {
            if (error_cb_) error_cb_("Failed to prepare inputs for node: " + node_id);
            return false;
        }
        
        // 应用参数覆盖
        for (const auto& [key, value] : config.overrides) {
            inputs[key] = value;
        }
        
        // 执行节点
        NodeOutputs outputs;
        if (!execute_node(node, inputs, outputs, config)) {
            if (error_cb_) error_cb_("Failed to execute node: " + node_id);
            return false;
        }
        
        computed[node_id] = outputs;
    }
    
    if (progress_cb_) {
        progress_cb_("", total, total);
    }
    
    return true;
}

bool DAGExecutor::prepare_inputs(Workflow* workflow,
                                 const std::string& node_id,
                                 NodeInputs& inputs,
                                 std::map<std::string, NodeOutputs>& computed) {
    Node* node = workflow->get_node(node_id);
    if (!node) return false;
    
    // 获取节点的输入定义
    auto input_defs = node->get_inputs();
    
    // 设置默认值
    for (const auto& def : input_defs) {
        if (!def.required && def.default_value.has_value()) {
            inputs[def.name] = def.default_value;
        }
    }
    
    // 从连接获取输入
    auto links = workflow->get_input_links(node_id);
    for (const auto& link : links) {
        auto it = computed.find(link.src_node_id);
        if (it == computed.end()) {
            return false;  // 上游节点未执行
        }
        
        // 获取上游节点的第 src_slot 个输出
        auto outputs = it->second;
        auto output_defs = workflow->get_node(link.src_node_id)->get_outputs();
        
        if (link.src_slot >= 0 && link.src_slot < (int)output_defs.size()) {
            std::string output_name = output_defs[link.src_slot].name;
            auto out_it = outputs.find(output_name);
            if (out_it != outputs.end()) {
                // 需要映射到当前节点的输入端口名称
                // 简化：假设输入端口顺序和连接顺序一致
                if (link.dst_slot >= 0 && link.dst_slot < (int)input_defs.size()) {
                    inputs[input_defs[link.dst_slot].name] = out_it->second;
                }
            }
        }
    }
    
    // 从工作流 JSON 中读取字面量输入值
    auto literal_values = workflow->get_input_values(node_id);
    for (const auto& [name, value] : literal_values) {
        inputs[name] = value;
    }
    
    return true;
}

bool DAGExecutor::execute_node(Node* node,
                               const NodeInputs& inputs,
                               NodeOutputs& outputs,
                               const ExecutionConfig& config) {
    // 检查缓存
    if (config.use_cache && cache_) {
        std::string hash = node->compute_hash(inputs);
        if (cache_->has(node->get_id(), hash)) {
            outputs = cache_->get(node->get_id(), hash);
            return true;
        }
    }

    // 实际执行
    bool success = node->execute(inputs, outputs);

    // 存入缓存
    if (success && config.use_cache && cache_) {
        std::string hash = node->compute_hash(inputs);
        cache_->put(node->get_id(), hash, outputs);
    }
    
    return success;
}

} // namespace sdengine
