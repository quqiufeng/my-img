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
            if (ptr) {
                int w = 0, h = 0, c = 0;
                sd_latent_get_shape(ptr.get(), &w, &h, &c);
                ss << "[latent:" << w << "x" << h << "x" << c << "]";
            } else {
                ss << "[latent-null]";
            }
        } else if (value.type() == typeid(ConditioningPtr)) {
            auto ptr = std::any_cast<ConditioningPtr>(value);
            // conditioning 是 opaque 类型，无法计算内容哈希，使用指针地址区分
            ss << "[cond:" << ptr.get() << "]";
        } else if (value.type() == typeid(ImagePtr)) {
            auto ptr = std::any_cast<ImagePtr>(value);
            if (ptr && ptr->data) {
                // 快速内容哈希：尺寸 + 像素校验和（每 64 像素采样一个，避免大图像遍历过慢）
                size_t pixels = (size_t)ptr->width * ptr->height * ptr->channel;
                uint64_t checksum = 0;
                size_t stride = std::max((size_t)1, pixels / 64);
                for (size_t i = 0; i < pixels; i += stride) {
                    checksum = checksum * 31 + ptr->data[i];
                }
                ss << "[image:" << ptr->width << "x" << ptr->height << "x" << ptr->channel << ":" << checksum << "]";
            } else {
                ss << "[image-null]";
            }
        } else if (value.type() == typeid(sd_image_t)) {
            auto img = std::any_cast<sd_image_t>(value);
            if (img.data) {
                size_t pixels = (size_t)img.width * img.height * img.channel;
                uint64_t checksum = 0;
                size_t stride = std::max((size_t)1, pixels / 64);
                for (size_t i = 0; i < pixels; i += stride) {
                    checksum = checksum * 31 + img.data[i];
                }
                ss << "[sdimg:" << img.width << "x" << img.height << "x" << img.channel << ":" << checksum << "]";
            } else {
                ss << "[sdimg:" << img.width << "x" << img.height << ":null]";
            }
        } else if (value.type() == typeid(UpscalerPtr)) {
            auto ptr = std::any_cast<UpscalerPtr>(value);
            ss << (ptr ? "[upscaler]" : "[upscaler-null]");
        } else if (value.type() == typeid(SDContextPtr)) {
            auto ptr = std::any_cast<SDContextPtr>(value);
            // 模型上下文使用指针地址即可（同一模型实例内容相同）
            ss << "[sdctx:" << ptr.get() << "]";
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
