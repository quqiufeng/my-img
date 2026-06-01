#pragma once

#include "cli/cli_options.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>

namespace myimg {

// ComfyUI workflow node
struct WorkflowNode {
    int id;
    std::string type;
    std::map<std::string, nlohmann::json> inputs;
    std::vector<nlohmann::json> outputs;
    std::vector<nlohmann::json> widgets_values;
};

// ComfyUI workflow link
struct WorkflowLink {
    int id;
    int from_node;
    int from_slot;
    int to_node;
    int to_slot;
};

// Workflow parser
class WorkflowParser {
public:
    bool parse(const std::string& json_path);
    bool to_cli_options(CliOptions& opts);
    
    // Get extracted prompt
    std::string get_prompt() const;
    std::string get_negative_prompt() const;
    
    // Check if workflow is valid
    bool is_valid() const;
    
private:
    nlohmann::json workflow_;
    std::map<int, WorkflowNode> nodes_;
    std::vector<WorkflowLink> links_;
    std::string error_msg_;
    
    void parse_nodes();
    void parse_links();
    
    // Helper to find node by type
    WorkflowNode* find_node_by_type(const std::string& type);
    
    // Helper to resolve input from link
    nlohmann::json resolve_input(const WorkflowNode& node, const std::string& input_name);
};

} // namespace myimg
