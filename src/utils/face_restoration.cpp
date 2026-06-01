#include <filesystem>
#include "utils/face_restoration.h"
#include "utils/log.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/objdetect.hpp>

namespace myimg {

FaceRestoration::FaceRestoration(const FaceRestorationConfig& config) 
    : config_(config) {
    if (!config_.model_path.empty()) {
        load_model(config_.model_path);
    }
}

bool FaceRestoration::load_model(const std::string& model_path) {
    LOG_INFO("FaceRestoration: loading model from %s", model_path.c_str());
    
    if (!std::filesystem::exists(model_path)) {
        LOG_ERROR("FaceRestoration: model file not found: %s", model_path.c_str());
        return false;
    }
    
    // 检查文件扩展名
    if (model_path.find(".onnx") != std::string::npos) {
        // ONNX 模型 (需要 ONNX Runtime 或 OpenCV DNN)
        try {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
            if (net.empty()) {
                LOG_ERROR("FaceRestoration: failed to load ONNX model");
                return false;
            }
            LOG_INFO("FaceRestoration: ONNX model loaded successfully");
            model_loaded_ = true;
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("FaceRestoration: ONNX loading failed: %s", e.what());
            // 回退到基础增强
        }
    }
    
    // PyTorch 模型 (.pth) - 需要模型架构定义
    // 当前作为占位符，使用基础增强
    LOG_WARN("FaceRestoration: PyTorch models require architecture definition");
    LOG_INFO("FaceRestoration: falling back to basic face enhancement");
    
    model_loaded_ = true; // 标记为已加载，使用基础增强模式
    return true;
}

ImageData FaceRestoration::restore_faces(const ImageData& image) {
    if (!model_loaded_) {
        LOG_WARN("FaceRestoration: model not loaded");
        return image;
    }
    
    auto tensor = image_data_to_tensor(image);
    auto restored = restore_faces_tensor(tensor);
    return tensor_to_image_data(restored);
}

torch::Tensor FaceRestoration::restore_faces_tensor(const torch::Tensor& image) {
    if (!model_loaded_) {
        LOG_WARN("FaceRestoration: model not loaded");
        return image;
    }
    
    // 检测人脸
    auto faces = detect_faces(image);
    if (faces.empty()) {
        LOG_INFO("FaceRestoration: no faces detected");
        return image;
    }
    
    LOG_INFO("FaceRestoration: detected %zu faces, applying enhancement", faces.size());
    
    auto result = image.clone();
    
    for (const auto& [x, y, w, h] : faces) {
        // 提取人脸区域
        auto face_crop = result.narrow(1, y, h).narrow(2, x, w);
        
        // 应用基础增强
        auto restored_face = apply_basic_enhancement(face_crop);
        
        // 将修复后的人脸放回原图
        result.narrow(1, y, h).narrow(2, x, w).copy_(restored_face);
    }
    
    return result;
}

std::vector<std::tuple<int, int, int, int>> FaceRestoration::detect_faces(const torch::Tensor& image) {
    std::vector<std::tuple<int, int, int, int>> faces;
    
    try {
        auto img_data = tensor_to_image_data(image);
        cv::Mat img(img_data.height, img_data.width, CV_8UC3, (void*)img_data.data.data());
        
        // 使用 OpenCV DNN 人脸检测 (YuNet)
        cv::dnn::Net yunet = cv::dnn::readNetFromONNX("/data/models/image/yunet_320_320.onnx");
        if (yunet.empty()) {
            LOG_WARN("FaceRestoration: YuNet model not available, using Haar cascade fallback");
            // 回退到 Haar 级联分类器
            cv::CascadeClassifier face_cascade;
            if (face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
                cv::Mat gray;
                cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
                std::vector<cv::Rect> detected;
                face_cascade.detectMultiScale(gray, detected, 1.1, 3, 0, cv::Size(80, 80));
                for (const auto& r : detected) {
                    faces.push_back({r.x, r.y, r.width, r.height});
                }
            }
            return faces;
        }
        
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(320, 320), 
                                              cv::Scalar(0, 0, 0), true, false);
        yunet.setInput(blob);
        cv::Mat detections = yunet.forward();
        
        const float* data = (float*)detections.data;
        for (int i = 0; i < detections.rows; i++) {
            float confidence = data[i * 15 + 14];
            if (confidence < 0.5f) continue;
            
            int x = static_cast<int>(data[i * 15 + 0] * img_data.width / 320.0f);
            int y = static_cast<int>(data[i * 15 + 1] * img_data.height / 320.0f);
            int w = static_cast<int>(data[i * 15 + 2] * img_data.width / 320.0f);
            int h = static_cast<int>(data[i * 15 + 3] * img_data.height / 320.0f);
            
            faces.push_back({x, y, w, h});
        }
    } catch (const std::exception& e) {
        LOG_WARN("FaceRestoration: face detection failed: %s", e.what());
    }
    
    return faces;
}

torch::Tensor FaceRestoration::restore_single_face(const torch::Tensor& face_crop) {
    return apply_basic_enhancement(face_crop);
}

torch::Tensor FaceRestoration::apply_basic_enhancement(const torch::Tensor& face) {
    // 基础增强：双边滤波 + 锐化
    auto img_data = tensor_to_image_data(face);
    cv::Mat img(img_data.height, img_data.width, CV_8UC3, (void*)img_data.data.data());
    
    // 双边滤波：保留边缘的同时平滑皮肤
    cv::Mat smooth;
    cv::bilateralFilter(img, smooth, 9, 75, 75);
    
    // Unsharp Mask 锐化
    cv::Mat blurred;
    cv::GaussianBlur(smooth, blurred, cv::Size(0, 0), 3);
    cv::Mat sharpened;
    cv::addWeighted(smooth, 1.5, blurred, -0.5, 0, sharpened);
    
    // 轻微对比度增强
    cv::Mat enhanced;
    sharpened.convertTo(enhanced, -1, 1.05, -10);
    
    // 限制范围
    cv::Mat result;
    cv::min(enhanced, 255, result);
    cv::max(result, 0, result);
    
    // 转换回 tensor
    auto result_data = img_data;
    std::memcpy(result_data.data.data(), result.data, result.total() * result.elemSize());
    return image_data_to_tensor(result_data);
}

torch::Tensor FaceRestoration::inference_gfpgan(const torch::Tensor& face) {
    LOG_WARN("FaceRestoration: GFPGAN inference requires full model implementation");
    return face;
}

torch::Tensor FaceRestoration::inference_codeformer(const torch::Tensor& face) {
    LOG_WARN("FaceRestoration: CodeFormer inference requires full model implementation");
    return face;
}

} // namespace myimg
