#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include "utils/image_utils.h"

namespace myimg {

// Face Restoration: 人脸修复
// 支持 GFPGAN 和 CodeFormer
enum class FaceRestorationModel {
    GFPGAN,
    CodeFormer,
    RestoreFormer,
};

struct FaceRestorationConfig {
    FaceRestorationModel model = FaceRestorationModel::GFPGAN;
    std::string model_path;     // ONNX 模型路径
    float fidelity = 0.5f;      // 保真度 (0.0-1.0)，越高越接近原图
    bool has_background = true; // 是否保留背景
    bool only_center_face = false; // 是否只修复中心人脸
    int upscale = 2;            // 超分倍数
};

class FaceRestoration {
public:
    FaceRestoration() = default;
    explicit FaceRestoration(const FaceRestorationConfig& config);
    
    // 加载模型
    bool load_model(const std::string& model_path);
    
    // 修复图像中的人脸
    ImageData restore_faces(const ImageData& image);
    
    // 修复 torch 张量中的人脸
    torch::Tensor restore_faces_tensor(const torch::Tensor& image);
    
    // 是否已加载模型
    bool is_loaded() const { return model_loaded_; }
    
    // 获取配置
    const FaceRestorationConfig& config() const { return config_; }
    
private:
    FaceRestorationConfig config_;
    bool model_loaded_ = false;
    
    // 人脸检测（使用 OpenCV DNN 或 ONNX）
    std::vector<std::tuple<int, int, int, int>> detect_faces(const torch::Tensor& image);
    
    // 单个人脸修复
    torch::Tensor restore_single_face(const torch::Tensor& face_crop);
    
    // GFPGAN 推理
    torch::Tensor inference_gfpgan(const torch::Tensor& face);
    
    // CodeFormer 推理
    torch::Tensor inference_codeformer(const torch::Tensor& face);
};

} // namespace myimg
