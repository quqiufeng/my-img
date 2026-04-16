// ============================================================================
// sd-engine/core/executor.cpp
// ============================================================================

#include "executor.h"
#include "log.h"
#include <algorithm>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

namespace sdengine {

DAGExecutor::DAGExecutor(ExecutionCache* cache) : cache_(cache) {}

void DAGExecutor::set_progress_callback(ProgressCallback cb) {
    progress_cb_ = cb;
}

void DAGExecutor::set_error_callback(std::function<void(const std::string&)> cb) {
    error_cb_ = cb;
}

// 计算每个节点的执行深度（最长前置路径长度）
static std::map<std::string, int> compute_node_depths(const std::vector<std::string>& order, Workflow* workflow) {
    std::map<std::string, int> depths;
    for (const auto& node_id : order) {
        int depth = 0;
        auto links = workflow->get_input_links(node_id);
        for (const auto& link : links) {
            auto it = depths.find(link.src_node_id);
            if (it != depths.end()) {
                depth = std::max(depth, it->second + 1);
            }
        }
        depths[node_id] = depth;
    }
    return depths;
}

sd_error_t DAGExecutor::execute(Workflow* workflow, const ExecutionConfig& config) {
    if (!workflow) {
        if (error_cb_)
            error_cb_("Workflow is null");
        return sd_error_t::ERROR_INVALID_INPUT;
    }

    std::string error_msg;
    if (!workflow->validate(error_msg)) {
        if (error_cb_)
            error_cb_(error_msg);
        return sd_error_t::ERROR_INVALID_INPUT;
    }

    auto order = workflow->topological_sort();
    int total = (int)order.size();

    if (total == 0) {
        return sd_error_t::OK;
    }

    // 存储已计算节点的输出
    std::map<std::string, NodeOutputs> computed;
    std::mutex computed_mutex;

    int max_threads =
        config.max_threads > 0 ? config.max_threads : static_cast<int>(std::thread::hardware_concurrency());
    if (max_threads < 1)
        max_threads = 1;

    // 单线程模式：保持原有串行逻辑
    if (max_threads == 1) {
        int current = 0;
        for (const auto& node_id : order) {
            current++;

            Node* node = workflow->get_node(node_id);
            if (!node)
                continue;

            if (progress_cb_) {
                progress_cb_(node_id, current, total);
            }

            if (config.verbose) {
                LOG_INFO("[Executor] Executing node %s (%s) [%d/%d]", node_id.c_str(), node->get_class_type().c_str(),
                         current, total);
            }

            NodeInputs inputs;
            sd_error_t prep_err = prepare_inputs(workflow, node_id, inputs, computed);
            if (is_error(prep_err)) {
                if (error_cb_)
                    error_cb_("Failed to prepare inputs for node: " + node_id);
                return prep_err;
            }

            for (const auto& [key, value] : config.overrides) {
                inputs[key] = value;
            }

            NodeOutputs outputs;
            sd_error_t exec_err = execute_node(node, inputs, outputs, config);
            if (is_error(exec_err)) {
                if (error_cb_)
                    error_cb_("Failed to execute node: " + node_id);
                return exec_err;
            }

            computed[node_id] = std::move(outputs);
        }

        if (progress_cb_) {
            progress_cb_("", total, total);
        }
        return sd_error_t::OK;
    }

    // 多线程模式：按深度分层并行执行
    auto depths = compute_node_depths(order, workflow);

    // 按深度分组
    int max_depth = 0;
    for (const auto& [_, d] : depths) {
        max_depth = std::max(max_depth, d);
    }

    int current = 0;
    std::mutex progress_mutex;

    for (int depth = 0; depth <= max_depth; depth++) {
        std::vector<std::string> layer_nodes;
        for (const auto& node_id : order) {
            if (depths[node_id] == depth) {
                layer_nodes.push_back(node_id);
            }
        }

        if (layer_nodes.empty())
            continue;

        if (config.verbose) {
            LOG_INFO("[Executor] Layer %d: %zu nodes (max_threads=%d)", depth, layer_nodes.size(), max_threads);
        }

        // 如果该层只有一个节点，直接串行执行
        if (layer_nodes.size() == 1) {
            const auto& node_id = layer_nodes[0];
            Node* node = workflow->get_node(node_id);
            if (node) {
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    current++;
                    if (progress_cb_) {
                        progress_cb_(node_id, current, total);
                    }
                    if (config.verbose) {
                        LOG_INFO("[Executor] Executing node %s (%s) [%d/%d]", node_id.c_str(),
                                 node->get_class_type().c_str(), current, total);
                    }
                }

                NodeInputs inputs;
                {
                    std::lock_guard<std::mutex> lock(computed_mutex);
                    sd_error_t prep_err = prepare_inputs(workflow, node_id, inputs, computed);
                    if (is_error(prep_err)) {
                        if (error_cb_)
                            error_cb_("Failed to prepare inputs for node: " + node_id);
                        return prep_err;
                    }
                }

                for (const auto& [key, value] : config.overrides) {
                    inputs[key] = value;
                }

                NodeOutputs outputs;
                sd_error_t exec_err = execute_node(node, inputs, outputs, config);
                if (is_error(exec_err)) {
                    if (error_cb_)
                        error_cb_("Failed to execute node: " + node_id);
                    return exec_err;
                }

                {
                    std::lock_guard<std::mutex> lock(computed_mutex);
                    computed[node_id] = std::move(outputs);
                }
            }
            continue;
        }

        // 多节点并行执行：先统一准备 inputs，减少锁竞争
        std::vector<std::pair<std::string, NodeInputs>> prepared_inputs;
        prepared_inputs.reserve(layer_nodes.size());

        {
            std::lock_guard<std::mutex> lock(computed_mutex);
            for (const auto& node_id : layer_nodes) {
                NodeInputs inputs;
                sd_error_t prep_err = prepare_inputs(workflow, node_id, inputs, computed);
                if (is_error(prep_err)) {
                    if (error_cb_)
                        error_cb_("Failed to prepare inputs for node: " + node_id);
                    return prep_err;
                }
                for (const auto& [key, value] : config.overrides) {
                    inputs[key] = value;
                }
                prepared_inputs.push_back({node_id, std::move(inputs)});
            }
        }

        std::vector<std::future<std::pair<std::string, sd_error_t>>> futures;

        for (size_t i = 0; i < layer_nodes.size(); i++) {
            const auto& node_id = prepared_inputs[i].first;
            NodeInputs inputs = std::move(prepared_inputs[i].second);

            futures.push_back(
                std::async(std::launch::async,
                           [this, workflow, &config, &computed, &computed_mutex, &current, &progress_mutex, &total,
                            node_id, inputs]() mutable -> std::pair<std::string, sd_error_t> {
                               Node* node = workflow->get_node(node_id);
                               if (!node) {
                                   return {node_id, sd_error_t::ERROR_INVALID_INPUT};
                               }

                               {
                                   std::lock_guard<std::mutex> lock(progress_mutex);
                                   current++;
                                   if (progress_cb_) {
                                       progress_cb_(node_id, current, total);
                                   }
                                   if (config.verbose) {
                                       LOG_INFO("[Executor] Executing node %s (%s) [%d/%d] (thread=%zu)",
                                                node_id.c_str(), node->get_class_type().c_str(), current, total,
                                                std::hash<std::thread::id>{}(std::this_thread::get_id()) % 1000);
                                   }
                               }

                               NodeOutputs outputs;
                               sd_error_t exec_err = execute_node(node, inputs, outputs, config);
                               if (is_error(exec_err)) {
                                   if (error_cb_)
                                       error_cb_("Failed to execute node: " + node_id);
                                   return {node_id, exec_err};
                               }

                               {
                                   std::lock_guard<std::mutex> lock(computed_mutex);
                                   computed[node_id] = std::move(outputs);
                               }

                               return {node_id, sd_error_t::OK};
                           }));
        }

        // 等待该层所有节点完成
        for (auto& fut : futures) {
            auto [node_id, err] = fut.get();
            if (is_error(err)) {
                return err;
            }
        }
    }

    if (progress_cb_) {
        progress_cb_("", total, total);
    }

    return sd_error_t::OK;
}

sd_error_t DAGExecutor::prepare_inputs(Workflow* workflow, const std::string& node_id, NodeInputs& inputs,
                                       std::map<std::string, NodeOutputs>& computed) {
    Node* node = workflow->get_node(node_id);
    if (!node)
        return sd_error_t::ERROR_INVALID_INPUT;

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
            return sd_error_t::ERROR_MISSING_INPUT; // 上游节点未执行
        }

        // 获取上游节点的第 src_slot 个输出
        const auto& outputs = it->second;
        auto output_defs = workflow->get_node(link.src_node_id)->get_outputs();

        if (link.src_slot >= 0 && link.src_slot < (int)output_defs.size()) {
            std::string output_name = output_defs[link.src_slot].name;
            auto out_it = outputs.find(output_name);
            if (out_it != outputs.end()) {
                if (link.dst_slot >= 0 && link.dst_slot < (int)input_defs.size()) {
                    inputs[input_defs[link.dst_slot].name] = out_it->second;
                }
            }
        }

        // 同时传递上游节点的隐藏元数据输出（以下划线开头），
        // 这样 ControlNetApply / IPAdapterApply 等节点可以将附加信息传递给下游
        for (const auto& [out_name, out_value] : outputs) {
            if (!out_name.empty() && out_name[0] == '_') {
                inputs[out_name] = out_value;
            }
        }
    }

    // 从工作流 JSON 中读取字面量输入值
    auto literal_values = workflow->get_input_values(node_id);
    for (const auto& [name, value] : literal_values) {
        inputs[name] = value;
    }

    return sd_error_t::OK;
}

sd_error_t DAGExecutor::execute_node(Node* node, const NodeInputs& inputs, NodeOutputs& outputs,
                                     const ExecutionConfig& config) {
    // 检查缓存
    if (config.use_cache && cache_) {
        std::string hash = node->compute_hash(inputs);
        if (cache_->has(node->get_id(), hash)) {
            outputs = cache_->get(node->get_id(), hash);
            return sd_error_t::OK;
        }
    }

    // 实际执行
    sd_error_t err = node->execute(inputs, outputs);

    // 存入缓存
    if (is_ok(err) && config.use_cache && cache_) {
        std::string hash = node->compute_hash(inputs);
        cache_->put(node->get_id(), hash, outputs);
    }

    return err;
}

} // namespace sdengine
