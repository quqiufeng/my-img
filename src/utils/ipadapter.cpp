#include <filesystem>
#include "utils/ipadapter.h"
#include "utils/log.h"
#include "utils/image_utils.h"

namespace myimg {

IPAdapter::IPAdapter(const IPAdapterConfig& config) 
    : config_(config) {
    if (!config_.model_path.empty() && !config_.clip_vision_path.empty()) {
        load_model(config_.model_path, config_.clip_vision_path);
    }
    if (!config_.image_path.empty()) {
        load_reference_image(config_.image_path);
    }
}

bool IPAdapter::load_model(const std::string& model_path, const std::string& clip_vision_path) {
    LOG_INFO("IPAdapter: loading IPAdapter model from %s", model_path.c_str());
    LOG_INFO("IPAdapter: loading CLIP Vision model from %s", clip_vision_path.c_str());
    
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("IPAdapter: model not found: %s", model_path.c_str());
        return false;
    }
    
    if (!std::filesystem::exists(clip_vision_path)) {
        LOG_ERROR("IPAdapter: CLIP Vision model not found: %s", clip_vision_path.c_str());
        return false;
    }
    
    // TODO: 加载 IPAdapter 和 CLIP Vision 模型
    // 需要 safetensors 加载器和 CLIP Vision 编码器
    LOG_WARN("IPAdapter: model loading not yet implemented");
    LOG_INFO("IPAdapter: to use IPAdapter, please:");
    LOG_INFO("  1. Download IPAdapter model (ip-adapter_sd15.bin or ip-adapter-plus_sd15.safetensors)");
    LOG_INFO("  2. Download CLIP Vision model (CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors)");
    LOG_INFO("  3. Place them in /data/models/image/");
    LOG_INFO("  4. Use --ipadapter-model PATH --clip-vision PATH --ipadapter-image PATH");
    
    model_loaded_ = false;
    clip_vision_loaded_ = false;
    return false;
}

bool IPAdapter::load_reference_image(const std::string& image_path) {
    LOG_INFO("IPAdapter: loading reference image from %s", image_path.c_str());
    
    if (!std::filesystem::exists(image_path)) {
        LOG_ERROR("IPAdapter: image not found: %s", image_path.c_str());
        return false;
    }
    
    try {
        auto image = load_image(image_path);
        image_features_ = extract_image_features(image);
        return image_features_.defined();
    } catch (const std::exception& e) {
        LOG_ERROR("IPAdapter: failed to load reference image: %s", e.what());
        return false;
    }
}

torch::Tensor IPAdapter::extract_image_features(const torch::Tensor& image) { (void)image;
    if (!clip_vision_loaded_) {
        LOG_WARN("IPAdapter: CLIP Vision model not loaded");
        return torch::Tensor();
    }
    
    return clip_vision_encode(image);
}

torch::Tensor IPAdapter::apply_ipadapter(const torch::Tensor& latent, int step, int total_steps) {
    if (!is_loaded() || !image_features_.defined()) {
        return latent;
    }
    
    float progress = static_cast<float>(step) / total_steps;
    
    // 检查是否在注入范围内
    if (progress < config_.start_at || progress > config_.end_at) {
        return latent;
    }
    
    // 计算当前步数的有效权重
    float effective_weight = config_.weight;
    if (progress < config_.start_at + 0.1f) {
        // 渐入
        effective_weight *= (progress - config_.start_at) / 0.1f;
    } else if (progress > config_.end_at - 0.1f) {
        // 渐出
        effective_weight *= (config_.end_at - progress) / 0.1f;
    }
    
    return inject_attention(latent, image_features_ * effective_weight);
}

torch::Tensor IPAdapter::clip_vision_encode(const torch::Tensor& image) { (void)image;
    // TODO: CLIP Vision 编码
    LOG_WARN("IPAdapter: CLIP Vision encoding not yet implemented");
    return torch::Tensor();
}

torch::Tensor IPAdapter::inject_attention(const torch::Tensor& latent, const torch::Tensor& image_features) { (void)image_features;
    // TODO: 注意力注入
    LOG_WARN("IPAdapter: attention injection not yet implemented");
    return latent;
}

} // namespace myimg
