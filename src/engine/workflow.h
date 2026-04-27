#pragma once

#include "engine/node.h"
#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <string>

namespace myimg {

struct NodeDef {
    std::string id;
    std::string class_type;
    std::map<std::string, std::any> inputs;
};

class Workflow {
public:
    static Workflow from_json(const nlohmann::json& json);
    
    std::vector<std::string> get_execution_order() const;
    const NodeDef& get_node(const std::string& id) const;
    
    bool validate(std::string& error_msg) const;
    
private:
    std::map<std::string, NodeDef> nodes_;
    std::map<std::string, std::vector<std::string>> edges_;
};

} // namespace myimg
