#include "utils/photo_maker.h"
#include "utils/log.h"
#include "utils/image_utils.h"
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

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
    
    try {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            LOG_ERROR("PhotoMaker: failed to load ONNX model");
            return false;
        }
        
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        
        LOG_INFO("PhotoMaker: ONNX model loaded successfully");
        model_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("PhotoMaker: model loading failed: %s", e.what());
        return false;
    }
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
    
    try {
        // 使用 PhotoMaker ONNX 模型融合 ID 特征和文本嵌入
        cv::dnn::Net net = cv::dnn::readNetFromONNX(config_.model_path);
        
        // 准备输入
        auto id_cpu = id_features.to(torch::kCPU);
        auto text_cpu = text_embedding.to(torch::kCPU);
        
        cv::Mat id_mat(id_features.size(0), id_features.size(1), CV_32F, id_cpu.data_ptr<float>());
        cv::Mat text_mat(text_embedding.size(0), text_embedding.size(1), CV_32F, text_cpu.data_ptr<float>());
        
        cv::Mat id_blob = cv::dnn::blobFromImage(id_mat, 1.0, cv::Size(), cv::Scalar(), false, false);
        cv::Mat text_blob = cv::dnn::blobFromImage(text_mat, 1.0, cv::Size(), cv::Scalar(), false, false);
        
        net.setInput(id_blob, "id_features");
        net.setInput(text_blob, "text_embedding");
        
        cv::Mat fused = net.forward();
        
        // 转换回 tensor
        auto sizes = fused.size;
        auto fused_tensor = torch::from_blob(fused.data, {sizes[0], sizes[1]}, torch::kFloat32).clone();
        
        LOG_INFO("PhotoMaker: injected ID features into text embedding");
        return fused_tensor;
    } catch (const std::exception& e) {
        LOG_WARN("PhotoMaker: text embedding injection failed: %s", e.what());
        return text_embedding;
    }
}

bool PhotoMaker::apply(GenerationParams& params) {
    (void)params;
    if (!model_loaded_ || !id_features_.defined()) {
        LOG_WARN("PhotoMaker: not ready (model=%d, features=%d)", 
                 model_loaded_, id_features_.defined());
        return false;
    }
    
    LOG_INFO("PhotoMaker: applying ID features (weight: %.2f)", config_.id_weight);
    
    // 实际应用需要在适配器层修改 GenerationParams
    // 这里只是标记 PhotoMaker 已激活
    return true;
}

torch::Tensor PhotoMaker::encode_id_image(const torch::Tensor& image) {
    try {
        auto img_data = tensor_to_image_data(image);
        cv::Mat img(img_data.height, img_data.width, CV_8UC3, (void*)img_data.data.data());
        
        // 预处理：resize 到 256x256
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(256, 256));
        cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(256, 256),
                                              cv::Scalar(0, 0, 0), false, false);
        
        cv::dnn::Net net = cv::dnn::readNetFromONNX(config_.model_path);
        net.setInput(blob);
        cv::Mat features = net.forward();
        
        auto sizes = features.size;
        return torch::from_blob(features.data, {sizes[0], sizes[1]}, torch::kFloat32).clone();
    } catch (const std::exception& e) {
        LOG_WARN("PhotoMaker: ID image encoding failed: %s", e.what());
        return torch::Tensor();
    }
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
