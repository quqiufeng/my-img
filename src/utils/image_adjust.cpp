#include "utils/image_utils.h"
#include <torch/torch.h>
#include <cmath>

namespace myimg {

// 色温调整：使用简单的 RGB 通道比例
torch::Tensor adjust_temperature(const torch::Tensor& image, float temperature) {
    // temperature: -1.0 (冷/蓝) 到 1.0 (暖/黄)
    auto img = image.clone();
    
    if (temperature > 0) {
        // 暖色调：增强 R，减弱 B
        img[0] = img[0] * (1.0f + temperature * 0.1f);
        img[2] = img[2] * (1.0f - temperature * 0.1f);
    } else {
        // 冷色调：减弱 R，增强 B
        img[0] = img[0] * (1.0f + temperature * 0.1f);
        img[2] = img[2] * (1.0f - temperature * 0.1f);
    }
    
    return img.clamp(0, 1);
}

// 亮度调整
torch::Tensor adjust_brightness(const torch::Tensor& image, float brightness) {
    // brightness: -1.0 到 1.0
    return (image + brightness).clamp(0, 1);
}

// 对比度调整
torch::Tensor adjust_contrast(const torch::Tensor& image, float contrast) {
    // contrast: -1.0 到 1.0, 0 = 无变化
    float factor = 1.0f + contrast;
    return ((image - 0.5f) * factor + 0.5f).clamp(0, 1);
}

// 饱和度调整
torch::Tensor adjust_saturation(const torch::Tensor& image, float saturation) {
    // saturation: -1.0 到 1.0
    auto gray = image.mean(0, true); // 灰度
    float factor = 1.0f + saturation;
    return (gray + (image - gray) * factor).clamp(0, 1);
}

// 曝光调整（EV）
torch::Tensor adjust_exposure(const torch::Tensor& image, float ev) {
    // ev: -5.0 到 5.0
    float factor = std::pow(2.0f, ev);
    return (image * factor).clamp(0, 1);
}

// 高光压缩
torch::Tensor adjust_highlights(const torch::Tensor& image, float highlights) {
    // highlights: -100 到 100
    auto img = image.clone();
    float factor = highlights / 100.0f;
    
    // 对高亮区域进行压缩/提升
    auto mask = img > 0.5f;
    img = torch::where(mask, img * (1.0f - factor * 0.3f) + factor * 0.15f, img);
    
    return img.clamp(0, 1);
}

// 阴影提亮
torch::Tensor adjust_shadows(const torch::Tensor& image, float shadows) {
    // shadows: -100 到 100
    auto img = image.clone();
    float factor = shadows / 100.0f;
    
    // 对暗部区域进行提升/压暗
    auto mask = img < 0.5f;
    img = torch::where(mask, img * (1.0f + factor * 0.3f), img);
    
    return img.clamp(0, 1);
}

// 自动优化（一键修图）
torch::Tensor auto_enhance(const torch::Tensor& image) {
    auto img = image.clone();
    
    // 1. 自动曝光
    float mean = img.mean().item<float>();
    if (mean < 0.4f) {
        img = adjust_exposure(img, 0.3f);
    } else if (mean > 0.6f) {
        img = adjust_exposure(img, -0.2f);
    }
    
    // 2. 自动对比度
    img = adjust_contrast(img, 0.1f);
    
    // 3. 自动饱和度
    img = adjust_saturation(img, 0.05f);
    
    // 4. 阴影提亮
    img = adjust_shadows(img, 20.0f);
    
    return img.clamp(0, 1);
}

} // namespace myimg
