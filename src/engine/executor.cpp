#include "utils/log.h"
#include "engine/executor.h"
#include <iostream>

namespace myimg {

void Executor::execute(const Workflow& workflow) {
    std::string error_msg;
    if (!workflow.validate(error_msg)) {
        LOG_ERROR("[Executor] Workflow validation failed: %s", error_msg.c_str());
        return;
    }
    
    auto order = workflow.get_execution_order();
    std::cout << "[Executor] Executing " << order.size() << " nodes" << std::endl;
    
    for (size_t i = 0; i < order.size(); i++) {
        const auto& node_id = order[i];
        std::cout << "[Executor] Executing node: " << node_id << std::endl;
        
        if (progress_cb_) {
            progress_cb_(node_id, static_cast<float>(i) / order.size());
        }
    }
}

void Executor::set_progress_callback(ProgressCallback cb) {
    progress_cb_ = cb;
}

} // namespace myimg
