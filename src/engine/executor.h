#pragma once

#include "engine/workflow.h"
#include <functional>
#include <map>

namespace myimg {

class Executor {
public:
    using ProgressCallback = std::function<void(const std::string& node_id, float progress)>;
    
    void execute(const Workflow& workflow);
    void set_progress_callback(ProgressCallback cb);
    
private:
    ProgressCallback progress_cb_;
};

} // namespace myimg
