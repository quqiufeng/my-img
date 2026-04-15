// ============================================================================
// sd-engine/core/node.cpp
// ============================================================================

#include "node.h"
#include "sd_ptr.h"
#include <sstream>
#include <iomanip>

namespace sdengine {

std::string Node::compute_hash(const NodeInputs& inputs) const {
    std::stringstream ss;
    ss << get_class_type();

    for (const auto& [name, value] : inputs) {
        ss << "|" << name << "=";

        // 基础类型的哈希
        if (value.type() == typeid(int)) {
            ss << std::any_cast<int>(value);
        } else if (value.type() == typeid(int64_t)) {
            ss << std::any_cast<int64_t>(value);
        } else if (value.type() == typeid(float)) {
            // 避免 -0.0f 和 0.0f 哈希不同
            float f = std::any_cast<float>(value);
            if (f == 0.0f) f = 0.0f;
            ss << std::fixed << std::setprecision(6) << f;
        } else if (value.type() == typeid(double)) {
            double d = std::any_cast<double>(value);
            if (d == 0.0) d = 0.0;
            ss << std::fixed << std::setprecision(6) << d;
        } else if (value.type() == typeid(std::string)) {
            ss << std::any_cast<std::string>(value);
        } else if (value.type() == typeid(bool)) {
            ss << (std::any_cast<bool>(value) ? "1" : "0");
        } else if (value.type() == typeid(LatentPtr)) {
            auto ptr = std::any_cast<LatentPtr>(value);
            ss << (ptr ? "[latent]" : "[latent-null]");
        } else if (value.type() == typeid(ConditioningPtr)) {
            auto ptr = std::any_cast<ConditioningPtr>(value);
            ss << (ptr ? "[cond]" : "[cond-null]");
        } else if (value.type() == typeid(ImagePtr)) {
            auto ptr = std::any_cast<ImagePtr>(value);
            ss << (ptr ? "[image]" : "[image-null]");
        } else if (value.type() == typeid(sd_image_t)) {
            auto img = std::any_cast<sd_image_t>(value);
            ss << "[sdimg:" << img.width << "x" << img.height << "]";
        } else if (value.type() == typeid(UpscalerPtr)) {
            auto ptr = std::any_cast<UpscalerPtr>(value);
            ss << (ptr ? "[upscaler]" : "[upscaler-null]");
        } else {
            ss << "[complex]";
        }
    }

    return ss.str();
}

NodeRegistry& NodeRegistry::instance() {
    static NodeRegistry registry;
    return registry;
}

void NodeRegistry::register_node(const std::string& class_type, NodeCreator creator) {
    creators_[class_type] = creator;
}

std::unique_ptr<Node> NodeRegistry::create(const std::string& class_type) const {
    auto it = creators_.find(class_type);
    if (it != creators_.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> NodeRegistry::get_supported_nodes() const {
    std::vector<std::string> result;
    for (const auto& [type, _] : creators_) {
        result.push_back(type);
    }
    return result;
}

bool NodeRegistry::has_node(const std::string& class_type) const {
    return creators_.find(class_type) != creators_.end();
}

} // namespace sdengine
