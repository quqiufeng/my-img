#include "utils/face_swap.h"
#include "utils/log.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>

namespace myimg {

FaceSwap::FaceSwap(const FaceSwapConfig& config) 
    : config_(config) {
    if (!config_.detection_model.empty() && !config_.swap_model.empty()) {
        load_models(config_.detection_model, config_.swap_model);
    }
}

bool FaceSwap::load_models(const std::string& detection_path, const std::string& swap_path) {
    LOG_INFO("FaceSwap: loading detection model from %s", detection_path.c_str());
    LOG_INFO("FaceSwap: loading swap model from %s", swap_path.c_str());
    
    try {
        // 尝试加载 YuNet 人脸检测模型 (ONNX)
        cv::dnn::Net yunet = cv::dnn::readNetFromONNX(detection_path);
        if (!yunet.empty()) {
            yunet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            yunet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            detection_loaded_ = true;
            LOG_INFO("FaceSwap: detection model loaded (YuNet DNN)");
        } else {
            LOG_WARN("FaceSwap: YuNet DNN failed, trying Haar cascade");
            // 回退到 Haar 级联分类器
            if (haar_cascade_.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml")) {
                detection_loaded_ = true;
                use_haar_ = true;
                LOG_INFO("FaceSwap: Haar cascade loaded as fallback");
            } else {
                LOG_ERROR("FaceSwap: failed to load Haar cascade");
                detection_loaded_ = false;
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("FaceSwap: failed to load detection model: %s", e.what());
        detection_loaded_ = false;
    }
    
    try {
        // 加载 Inswapper 人脸替换模型 (ONNX)
        cv::dnn::Net inswapper = cv::dnn::readNetFromONNX(swap_path);
        if (inswapper.empty()) {
            LOG_ERROR("FaceSwap: failed to load swap model");
            return false;
        }
        inswapper.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        inswapper.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        swap_loaded_ = true;
        LOG_INFO("FaceSwap: swap model loaded (Inswapper)");
    } catch (const std::exception& e) {
        LOG_ERROR("FaceSwap: failed to load swap model: %s", e.what());
        swap_loaded_ = false;
    }
    
    return detection_loaded_ && swap_loaded_;
}

std::vector<FaceBox> FaceSwap::detect_faces(const ImageData& image) {
    if (!detection_loaded_) {
        LOG_WARN("FaceSwap: detection model not loaded");
        return {};
    }
    
    try {
        cv::Mat img(image.height, image.width, CV_8UC3, (void*)image.data.data());
        std::vector<FaceBox> faces;
        
        if (use_haar_) {
            // 使用 Haar 级联分类器
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Rect> detected;
            haar_cascade_.detectMultiScale(gray, detected, 1.1, 3, 0, cv::Size(80, 80));
            
            for (size_t i = 0; i < detected.size(); i++) {
                FaceBox box;
                box.x = detected[i].x;
                box.y = detected[i].y;
                box.w = detected[i].width;
                box.h = detected[i].height;
                box.confidence = 0.5f;
                faces.push_back(box);
                LOG_INFO("FaceSwap: Haar detected face %zu at (%d,%d,%d,%d)",
                         i, box.x, box.y, box.w, box.h);
            }
        } else {
            // 使用 YuNet DNN
            cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(320, 320),
                                                  cv::Scalar(0, 0, 0), true, false);
            
            cv::dnn::Net yunet = cv::dnn::readNetFromONNX(config_.detection_model);
            yunet.setInput(blob);
            cv::Mat detections = yunet.forward();
            
            int num_detections = 0;
            const float* data = nullptr;
            
            if (detections.dims == 3) {
                num_detections = detections.size[1];
                data = detections.ptr<float>(0);
            } else if (detections.dims == 2) {
                num_detections = detections.rows;
                data = (float*)detections.data;
            } else {
                LOG_WARN("FaceSwap: unexpected detection output dims=%d", detections.dims);
                return faces;
            }
            
            LOG_INFO("FaceSwap: YuNet candidates: %d", num_detections);
            
            std::vector<cv::Rect> raw_boxes;
            std::vector<float> raw_scores;
            std::vector<FaceBox> raw_faces;
            
            for (int i = 0; i < num_detections; i++) {
                float confidence = data[i * 15 + 14];
                if (confidence < 0.3f) continue;
                
                FaceBox box;
                box.x = static_cast<int>(data[i * 15 + 0] * image.width);
                box.y = static_cast<int>(data[i * 15 + 1] * image.height);
                box.w = static_cast<int>(data[i * 15 + 2] * image.width);
                box.h = static_cast<int>(data[i * 15 + 3] * image.height);
                box.confidence = confidence;
                
                if (box.w <= 0 || box.h <= 0 || box.x < 0 || box.y < 0) continue;
                if (box.x + box.w > image.width || box.y + box.h > image.height) continue;
                
                raw_faces.push_back(box);
                raw_boxes.push_back(cv::Rect(box.x, box.y, box.w, box.h));
                raw_scores.push_back(confidence);
            }
            
            // NMS 过滤
            std::vector<int> indices;
            cv::dnn::NMSBoxes(raw_boxes, raw_scores, 0.3f, 0.4f, indices);
            
            for (int idx : indices) {
                faces.push_back(raw_faces[idx]);
                LOG_INFO("FaceSwap: YuNet face %d at (%d,%d,%d,%d) conf=%.3f",
                         idx, raw_faces[idx].x, raw_faces[idx].y, 
                         raw_faces[idx].w, raw_faces[idx].h, raw_faces[idx].confidence);
            }
        }
        
        LOG_INFO("FaceSwap: total %zu faces after NMS", faces.size());
        return faces;
    } catch (const std::exception& e) {
        LOG_ERROR("FaceSwap: face detection failed: %s", e.what());
        return {};
    }
}

std::vector<FaceBox> FaceSwap::detect_faces_tensor(const torch::Tensor& image) {
    auto img_data = tensor_to_image_data(image);
    return detect_faces(img_data);
}

ImageData FaceSwap::swap_faces(const ImageData& source, const ImageData& target) {
    if (!is_loaded()) {
        LOG_WARN("FaceSwap: models not loaded");
        return target;
    }
    
    auto source_tensor = image_data_to_tensor(source);
    auto target_tensor = image_data_to_tensor(target);
    auto result = swap_faces_tensor(source_tensor, target_tensor);
    return tensor_to_image_data(result);
}

torch::Tensor FaceSwap::swap_faces_tensor(const torch::Tensor& source, const torch::Tensor& target) {
    if (!is_loaded()) {
        LOG_WARN("FaceSwap: models not loaded");
        return target;
    }
    
    auto target_faces = detect_faces_tensor(target);
    if (target_faces.empty()) {
        LOG_WARN("FaceSwap: no faces detected in target");
        return target;
    }
    
    auto source_faces = detect_faces_tensor(source);
    if (source_faces.empty()) {
        LOG_WARN("FaceSwap: no faces detected in source");
        return target;
    }
    
    // 转换 target 为 cv::Mat
    auto target_data = tensor_to_image_data(target);
    cv::Mat target_mat(target_data.height, target_data.width, CV_8UC3, (void*)target_data.data.data());
    cv::Mat result_mat = target_mat.clone();
    
    // 提取源人脸
    auto source_data = tensor_to_image_data(source);
    cv::Mat source_mat(source_data.height, source_data.width, CV_8UC3, (void*)source_data.data.data());
    
    auto& src_box = source_faces[0];
    cv::Rect src_roi(src_box.x, src_box.y, src_box.w, src_box.h);
    cv::Mat src_face = source_mat(src_roi);
    
    // 加载 swap 模型
    cv::dnn::Net inswapper = cv::dnn::readNetFromONNX(config_.swap_model);
    
    for (const auto& tgt_box : target_faces) {
        cv::Rect tgt_roi(tgt_box.x, tgt_box.y, tgt_box.w, tgt_box.h);
        cv::Mat tgt_face = result_mat(tgt_roi);
        
        // 调整源人脸大小以匹配目标人脸
        cv::Mat src_resized;
        cv::resize(src_face, src_resized, cv::Size(tgt_face.cols, tgt_face.rows));
        
        // 准备输入 blob
        cv::Mat blob_src = cv::dnn::blobFromImage(src_resized, 1.0 / 255.0, cv::Size(128, 128),
                                                   cv::Scalar(0, 0, 0), true, false);
        cv::Mat blob_tgt = cv::dnn::blobFromImage(tgt_face, 1.0 / 255.0, cv::Size(128, 128),
                                                   cv::Scalar(0, 0, 0), true, false);
        
        // 合并输入
        std::vector<cv::Mat> inputs = {blob_src, blob_tgt};
        inswapper.setInput(inputs[0], "source");
        inswapper.setInput(inputs[1], "target");
        
        cv::Mat swapped = inswapper.forward();
        
        // 后处理 swapped
        cv::Mat swapped_img(128, 128, CV_32FC3, swapped.data);
        swapped_img *= 255.0;
        swapped_img.convertTo(swapped_img, CV_8UC3);
        
        cv::Mat swapped_resized;
        cv::resize(swapped_img, swapped_resized, cv::Size(tgt_face.cols, tgt_face.rows));
        
        // Alpha 混合
        cv::Mat mask(tgt_face.size(), CV_32FC1, cv::Scalar(0));
        cv::ellipse(mask, cv::Point(mask.cols/2, mask.rows/2), 
                    cv::Size(mask.cols/2, mask.rows/2), 0, 0, 360, cv::Scalar(1), -1);
        cv::GaussianBlur(mask, mask, cv::Size(21, 21), 10);
        
        cv::Mat blended;
        for (int c = 0; c < 3; c++) {
            cv::Mat ch1, ch2;
            cv::extractChannel(swapped_resized, ch1, c);
            cv::extractChannel(tgt_face, ch2, c);
            ch1.convertTo(ch1, CV_32F);
            ch2.convertTo(ch2, CV_32F);
            cv::Mat ch_blended = ch1.mul(mask) + ch2.mul(1.0 - mask);
            ch_blended.convertTo(ch_blended, CV_8U);
            if (c == 0) {
                std::vector<cv::Mat> channels = {ch_blended};
                blended = cv::Mat::zeros(tgt_face.size(), CV_8UC3);
            }
        }
        
        // 复制回结果
        swapped_resized.copyTo(result_mat(tgt_roi));
    }
    
    // 转换回 tensor
    auto result_data = target_data;
    result_data.data.assign(result_mat.data, result_mat.data + result_mat.total() * result_mat.elemSize());
    return image_data_to_tensor(result_data);
}

torch::Tensor FaceSwap::swap_single_face(const torch::Tensor& source_face, const torch::Tensor& target_face,
                                          const FaceBox& target_box) {
    (void)source_face;
    (void)target_box;
    return target_face;
}

torch::Tensor FaceSwap::align_face(const torch::Tensor& face, const FaceBox& box) {
    (void)box;
    return face;
}

torch::Tensor FaceSwap::blend_face(const torch::Tensor& target, const torch::Tensor& swapped_face,
                                    const FaceBox& box) {
    (void)target;
    (void)box;
    return swapped_face;
}

} // namespace myimg
