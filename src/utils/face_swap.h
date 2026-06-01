#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include <opencv2/objdetect.hpp>
#include "utils/image_utils.h"

namespace myimg {

// Face Swap: 人脸替换
// 将参考图像中的人脸替换到目标图像中
struct FaceSwapConfig {
    std::string source_image;    // 源人脸图像路径
    std::string target_image;    // 目标图像路径（可选，用于 img2img 模式）
    float face_similarity = 0.5f; // 人脸相似度阈值
    bool preserve_expression = true; // 保留目标图像的表情
    bool preserve_pose = true;   // 保留目标图像的姿态
    std::string detection_model; // 人脸检测模型路径
    std::string swap_model;      // 人脸替换模型路径
};

struct FaceBox {
    int x, y, w, h;
    float confidence;
    std::vector<std::pair<float, float>> landmarks; // 68 点或 5 点 landmarks
};

class FaceSwap {
public:
    FaceSwap() = default;
    explicit FaceSwap(const FaceSwapConfig& config);
    
    // 加载模型
    bool load_models(const std::string& detection_path, const std::string& swap_path);
    
    // 检测人脸
    std::vector<FaceBox> detect_faces(const ImageData& image);
    std::vector<FaceBox> detect_faces_tensor(const torch::Tensor& image);
    
    // 提取人脸特征
    torch::Tensor extract_face_features(const torch::Tensor& face_crop);
    
    // 执行人脸替换
    // source: 源人脸图像
    // target: 目标图像
    ImageData swap_faces(const ImageData& source, const ImageData& target);
    torch::Tensor swap_faces_tensor(const torch::Tensor& source, const torch::Tensor& target);
    
    // 单个人脸替换
    torch::Tensor swap_single_face(const torch::Tensor& source_face, const torch::Tensor& target_face,
                                    const FaceBox& target_box);
    
    bool is_loaded() const { return detection_loaded_ && swap_loaded_; }
    const FaceSwapConfig& config() const { return config_; }
    
private:
    FaceSwapConfig config_;
    bool detection_loaded_ = false;
    bool swap_loaded_ = false;
    bool use_haar_ = false;
    cv::CascadeClassifier haar_cascade_;
    
    // 人脸对齐
    torch::Tensor align_face(const torch::Tensor& face, const FaceBox& box);
    
    // 人脸融合（泊松融合或 alpha 混合）
    torch::Tensor blend_face(const torch::Tensor& target, const torch::Tensor& swapped_face,
                              const FaceBox& box);
};

} // namespace myimg
