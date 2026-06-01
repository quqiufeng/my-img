#include "cli/workflow_parser.h"
#include "utils/log.h"
#include <fstream>

namespace myimg {

bool WorkflowParser::parse(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        error_msg_ = "Failed to open workflow file: " + json_path;
        LOG_ERROR("%s", error_msg_.c_str());
        return false;
    }
    
    try {
        file >> workflow_;
    } catch (const std::exception& e) {
        error_msg_ = std::string("Failed to parse JSON: ") + e.what();
        LOG_ERROR("%s", error_msg_.c_str());
        return false;
    }
    
    parse_nodes();
    parse_links();
    
    LOG_INFO("Parsed workflow with %zu nodes and %zu links", nodes_.size(), links_.size());
    return true;
}

void WorkflowParser::parse_nodes() {
    if (!workflow_.contains("nodes")) return;
    
    for (const auto& node_json : workflow_["nodes"]) {
        WorkflowNode node;
        node.id = node_json.value("id", -1);
        node.type = node_json.value("type", "");
        
        // Parse inputs
        if (node_json.contains("inputs")) {
            for (const auto& input : node_json["inputs"]) {
                if (input.contains("name") && input.contains("value")) {
                    node.inputs[input["name"]] = input["value"];
                }
            }
        }
        
        // Parse widgets_values (ComfyUI 特定格式)
        if (node_json.contains("widgets_values")) {
            node.widgets_values = node_json["widgets_values"].get<std::vector<nlohmann::json>>();
        }
        
        nodes_[node.id] = node;
    }
}

void WorkflowParser::parse_links() {
    if (!workflow_.contains("links")) return;
    
    for (const auto& link_json : workflow_["links"]) {
        WorkflowLink link;
        if (link_json.is_array() && link_json.size() >= 5) {
            link.id = link_json[0];
            link.from_node = link_json[1];
            link.from_slot = link_json[2];
            link.to_node = link_json[3];
            link.to_slot = link_json[4];
            links_.push_back(link);
        }
    }
}

WorkflowNode* WorkflowParser::find_node_by_type(const std::string& type) {
    for (auto& [id, node] : nodes_) {
        if (node.type == type) return &node;
    }
    return nullptr;
}

nlohmann::json WorkflowParser::resolve_input(const WorkflowNode& node, const std::string& input_name) {
    // Check if input is directly in inputs
    auto it = node.inputs.find(input_name);
    if (it != node.inputs.end()) {
        return it->second;
    }
    
    // Check if input is connected via link
    for (const auto& link : links_) {
        if (link.to_node == node.id) {
            auto from_it = nodes_.find(link.from_node);
            if (from_it != nodes_.end()) {
                // Try to get output from source node
                // This is simplified - real implementation would need slot mapping
            }
        }
    }
    
    return nlohmann::json();
}

bool WorkflowParser::to_cli_options(CliOptions& opts) {
    if (!is_valid()) {
        LOG_ERROR("Invalid workflow: %s", error_msg_.c_str());
        return false;
    }
    
    // Find KSampler node
    auto* ksampler = find_node_by_type("KSampler");
    if (ksampler) {
        if (ksampler->widgets_values.size() >= 6) {
            opts.seed = ksampler->widgets_values[0].get<int64_t>();
            opts.steps = ksampler->widgets_values[1].get<int>();
            opts.cfg_scale = ksampler->widgets_values[2].get<float>();
            opts.sampling_method = ksampler->widgets_values[3].get<std::string>();
            opts.scheduler = ksampler->widgets_values[4].get<std::string>();
            // widgets_values[5] = denoise (for img2img)
        }
    }
    
    // Find EmptyLatentImage node
    auto* empty_latent = find_node_by_type("EmptyLatentImage");
    if (empty_latent && empty_latent->widgets_values.size() >= 2) {
        opts.width = empty_latent->widgets_values[0].get<int>();
        opts.height = empty_latent->widgets_values[1].get<int>();
    }
    
    // Find CheckpointLoaderSimple node
    auto* checkpoint = find_node_by_type("CheckpointLoaderSimple");
    if (checkpoint && checkpoint->widgets_values.size() >= 1) {
        opts.model = checkpoint->widgets_values[0].get<std::string>();
    }
    
    // Find VAELoader node
    auto* vae = find_node_by_type("VAELoader");
    if (vae && vae->widgets_values.size() >= 1) {
        opts.vae = vae->widgets_values[0].get<std::string>();
    }
    
    // Find CLIPTextEncode nodes (prompt and negative prompt)
    std::vector<WorkflowNode*> clip_text_nodes;
    for (auto& [id, node] : nodes_) {
        if (node.type == "CLIPTextEncode") {
            clip_text_nodes.push_back(&node);
        }
    }
    
    if (!clip_text_nodes.empty()) {
        // First one is usually prompt
        if (clip_text_nodes[0]->widgets_values.size() >= 1) {
            opts.prompt = clip_text_nodes[0]->widgets_values[0].get<std::string>();
        }
        // Second one might be negative prompt (if connected to KSampler's negative input)
        if (clip_text_nodes.size() > 1) {
            if (clip_text_nodes[1]->widgets_values.size() >= 1) {
                opts.negative_prompt = clip_text_nodes[1]->widgets_values[0].get<std::string>();
            }
        }
    }
    
    LOG_INFO("Workflow mapped to CLI options: %dx%d, %s, steps=%d", 
             opts.width, opts.height, opts.sampling_method.c_str(), opts.steps);
    return true;
}

std::string WorkflowParser::get_prompt() const {
    for (const auto& [id, node] : nodes_) {
        if (node.type == "CLIPTextEncode" && !node.widgets_values.empty()) {
            return node.widgets_values[0].get<std::string>();
        }
    }
    return "";
}

std::string WorkflowParser::get_negative_prompt() const {
    bool found_first = false;
    for (const auto& [id, node] : nodes_) {
        if (node.type == "CLIPTextEncode") {
            if (!found_first) {
                found_first = true;
                continue;
            }
            if (!node.widgets_values.empty()) {
                return node.widgets_values[0].get<std::string>();
            }
        }
    }
    return "";
}

bool WorkflowParser::is_valid() const {
    if (nodes_.empty()) return false;
    for (const auto& [id, node] : nodes_) {
        if (node.type == "KSampler") return true;
    }
    return false;
}

} // namespace myimg
