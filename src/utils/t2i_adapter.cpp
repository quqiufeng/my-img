#include <filesystem>
#include "utils/t2i_adapter.h"
#include "utils/log.h"
#include "utils/image_utils.h"

namespace myimg {

T2IAdapter::T2IAdapter(const T2IAdapterConfig& config) 
    : config_(config) {
    if (!config_.model_path.empty()) {
        load_model(config_.model_path);
    }
    if (!config_.condition_image.empty()) {
        load_condition_image(config_.condition_image);
    }
}

bool T2IAdapter::load_model(const std::string& model_path) {
    LOG_INFO("T2IAdapter: loading model from %s", model_path.c_str());
    
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("T2IAdapter: model not found: %s", model_path.c_str());
        return false;
    }
    
    // TODO: 加载 T2I-Adapter 模型
    LOG_WARN("T2IAdapter: model loading not yet implemented");
    LOG_INFO("T2IAdapter: to use T2I-Adapter, please:");
    LOG_INFO("  1. Download T2I-Adapter model (t2iadapter_sketch_sd15v2.pth)");
    LOG_INFO("  2. Place it in /data/models/image/");
    LOG_INFO("  3. Use --t2i-adapter-model PATH --t2i-adapter-image PATH");
    
    model_loaded_ = false;
    return false;
}

bool T2IAdapter::load_condition_image(const std::string& image_path) {
    LOG_INFO("T2IAdapter: loading condition image from %s", image_path.c_str());
    
    if (!std::filesystem::exists(image_path)) {
        LOG_ERROR("T2IAdapter: image not found: %s", image_path.c_str());
        return false;
    }
    
    try {
        auto image = load_image(image_path);
        condition_features_ = extract_features(image);
        return condition_features_.defined();
    } catch (const std::exception& e) {
        LOG_ERROR("T2IAdapter: failed to load condition image: %s", e.what());
        return false;
    }
}

torch::Tensor T2IAdapter::extract_features(const torch::Tensor& condition_image) {
    if (!model_loaded_) {
        LOG_WARN("T2IAdapter: model not loaded");
        return torch::Tensor();
    }
    
    return extract_t2i_features(condition_image);
}

torch::Tensor T2IAdapter::apply(const torch::Tensor& latent, int step, int total_steps) {
    if (!model_loaded_ || !condition_features_.defined()) {
        return latent;
    }
    
    // 检查步数范围
    int effective_end = (config_.end_step < 0) ? total_steps : config_.end_step;
    if (step < config_.start_step || step > effective_end) {
        return latent;
    }
    
    return inject_into_unet(latent, condition_features_ * config_.strength);
}

torch::Tensor T2IAdapter::extract_t2i_features(const torch::Tensor& image) { (void)image;
    // TODO: T2I-Adapter 特征提取
    LOG_WARN("T2IAdapter: feature extraction not yet implemented");
    return torch::Tensor();
}

torch::Tensor T2IAdapter::inject_into_unet(const torch::Tensor& latent, const torch::Tensor& features) { (void)features;
    // TODO: 注入到 UNet
    LOG_WARN("T2IAdapter: UNet injection not yet implemented");
    return latent;
}

} // namespace myimg
