#include <filesystem>
#include "utils/photo_maker.h"
#include "utils/log.h"
#include "utils/image_utils.h"

namespace myimg {

PhotoMaker::PhotoMaker(const PhotoMakerConfig& config) 
    : config_(config) {
    if (!config_.model_path.empty()) {
        load_model(config_.model_path);
    }
    if (!config_.id_images.empty()) {
        load_id_images(config_.id_images);
    }
}

bool PhotoMaker::load_model(const std::string& model_path) {
    LOG_INFO("PhotoMaker: loading model from %s", model_path.c_str());
    
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("PhotoMaker: model not found: %s", model_path.c_str());
        return false;
    }
    
    // TODO: 加载 PhotoMaker 模型
    LOG_WARN("PhotoMaker: model loading not yet implemented");
    LOG_INFO("PhotoMaker: to use PhotoMaker, please:");
    LOG_INFO("  1. Download PhotoMaker model (photomaker-v1.bin)");
    LOG_INFO("  2. Place it in /data/models/image/");
    LOG_INFO("  3. Use --photomaker-model PATH --photomaker-id-images img1.png,img2.png");
    
    model_loaded_ = false;
    return false;
}

bool PhotoMaker::load_id_images(const std::vector<std::string>& image_paths) {
    LOG_INFO("PhotoMaker: loading %zu ID images", image_paths.size());
    
    id_image_tensors_.clear();
    
    for (const auto& path : image_paths) {
        if (!std::filesystem::exists(path)) {
            LOG_WARN("PhotoMaker: ID image not found: %s", path.c_str());
            continue;
        }
        
        try {
            auto image = load_image(path);
            id_image_tensors_.push_back(image);
        } catch (const std::exception& e) {
            LOG_ERROR("PhotoMaker: failed to load ID image %s: %s", path.c_str(), e.what());
        }
    }
    
    if (id_image_tensors_.empty()) {
        LOG_ERROR("PhotoMaker: no valid ID images loaded");
        return false;
    }
    
    // 提取 ID 特征
    auto features = extract_id_features(id_image_tensors_);
    if (features.defined()) {
        id_features_ = features;
        return true;
    }
    
    return false;
}

torch::Tensor PhotoMaker::extract_id_features(const std::vector<torch::Tensor>& id_images) {
    if (!model_loaded_) {
        LOG_WARN("PhotoMaker: model not loaded");
        return torch::Tensor();
    }
    
    std::vector<torch::Tensor> features;
    for (const auto& img : id_images) {
        auto feat = encode_id_image(img);
        if (feat.defined()) {
            features.push_back(feat);
        }
    }
    
    if (features.empty()) {
        return torch::Tensor();
    }
    
    return aggregate_id_features(features);
}

torch::Tensor PhotoMaker::inject_into_text_embedding(const torch::Tensor& text_embedding,
                                                      const torch::Tensor& id_features) {
    if (!id_features.defined()) {
        return text_embedding;
    }
    
    // PhotoMaker 的核心：将 ID 特征注入到文本嵌入中
    // 具体实现取决于模型架构
    // TODO: 实现特征注入
    LOG_WARN("PhotoMaker: text embedding injection not yet implemented");
    return text_embedding;
}

bool PhotoMaker::apply(GenerationParams& params) { (void)params;
    if (!model_loaded_ || !id_features_.defined()) {
        LOG_WARN("PhotoMaker: not ready (model=%d, features=%d)", 
                 model_loaded_, id_features_.defined());
        return false;
    }
    
    // TODO: 修改 GenerationParams 以应用 PhotoMaker
    // 这需要在适配器层中集成
    LOG_WARN("PhotoMaker: apply not yet implemented");
    return false;
}

torch::Tensor PhotoMaker::encode_id_image(const torch::Tensor& image) { (void)image;
    // TODO: 编码 ID 图像
    LOG_WARN("PhotoMaker: ID image encoding not yet implemented");
    return torch::Tensor();
}

torch::Tensor PhotoMaker::aggregate_id_features(const std::vector<torch::Tensor>& features) {
    if (features.empty()) {
        return torch::Tensor();
    }
    
    if (features.size() == 1) {
        return features[0];
    }
    
    // 堆叠特征
    if (config_.use_stack) {
        auto stacked = torch::stack(features, 0); // [N, C]
        return stacked.mean(0); // 平均池化
    } else {
        return features[0]; // 只使用第一张
    }
}

} // namespace myimg
