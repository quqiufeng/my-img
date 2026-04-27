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

// 自然饱和度：智能保护肤色
torch::Tensor adjust_vibrance(const torch::Tensor& image, float strength) {
    if (strength == 0.0f) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    
    // Calculate saturation for each pixel
    auto max_result = torch::max(img, 0);
    auto max_val = std::get<0>(max_result); // [H, W]
    auto min_result = torch::min(img, 0);
    auto min_val = std::get<0>(min_result); // [H, W]
    auto saturation = (max_val - min_val) / (max_val + 1e-6f); // [H, W]
    
    // Skin tone detection: pixels where R > G > B and R is in mid-range
    auto r = img[0];
    auto g = img[1];
    auto b = img[2];
    auto skin_mask = (r > g) * (g > b) * (r > 0.3f) * (r < 0.85f);
    
    // For skin tones, apply less saturation change
    // For already saturated pixels, apply less saturation change
    auto protection = skin_mask.to(torch::kFloat32) * 0.7f + (saturation > 0.7f).to(torch::kFloat32) * 0.3f;
    protection = protection.unsqueeze(0); // [1, H, W]
    
    // Blend factor: less effect on protected pixels
    auto blend = std::abs(strength) * (1.0f - protection * 0.8f);
    
    // Apply saturation change
    auto gray = img.mean(0, true);
    img = gray + (img - gray) * (1.0f + blend * (strength > 0 ? 1.0f : -1.0f));
    
    return img.clamp(0, 1);
}

// 清晰度/纹理增强：提升中频细节
torch::Tensor enhance_clarity(const torch::Tensor& image, float strength) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    
    // High-pass filter for mid-frequency details
    auto blur_kernel = torch::tensor({{{0.0625f, 0.125f, 0.0625f},
                                       {0.125f, 0.25f, 0.125f},
                                       {0.0625f, 0.125f, 0.0625f}}},
                                      torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    blur_kernel = blur_kernel.unsqueeze(0).unsqueeze(0);
    
    auto blurred = torch::zeros_like(img);
    for (int c = 0; c < img.size(0); ++c) {
        auto ch = img[c].unsqueeze(0).unsqueeze(0);
        auto blurred_ch = torch::conv2d(ch, blur_kernel, {}, 1, 1);
        blurred[c] = blurred_ch.squeeze(0).squeeze(0);
    }
    
    // Detail = original - blurred
    auto detail = img - blurred;
    
    // Edge mask to avoid over-enhancing edges
    auto sobel_x = torch::tensor({{{-1.0f, 0.0f, 1.0f}, {-2.0f, 0.0f, 2.0f}, {-1.0f, 0.0f, 1.0f}}},
                                  torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    auto sobel_y = torch::tensor({{{-1.0f, -2.0f, -1.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}}},
                                  torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0);
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0);
    
    auto gray = img.mean(0, true);
    auto grad_x = torch::conv2d(gray.unsqueeze(0), sobel_x, {}, 1, 1);
    auto grad_y = torch::conv2d(gray.unsqueeze(0), sobel_y, {}, 1, 1);
    auto edge = (grad_x.squeeze() + grad_y.squeeze()).abs();
    auto edge_mask = (edge < edge.mean().item<float>() * 1.5f).to(torch::kFloat32);
    
    // Apply clarity: boost mid-frequency details on non-edge areas
    float blend = std::min(strength * 0.5f, 1.0f);
    auto mask = edge_mask.unsqueeze(0);
    
    return (img + detail * blend * mask).clamp(0, 1);
}

// USM (Unsharp Mask) 锐化
torch::Tensor usm_sharpen(const torch::Tensor& image, float amount, int radius, float threshold) {
    if (amount <= 0.0f || radius <= 0) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    
    // 创建高斯核
    int kernel_size = 2 * radius + 1;
    auto kernel = torch::zeros({1, 1, kernel_size, kernel_size}, torch::TensorOptions().dtype(torch::kFloat32));
    float sigma = radius / 2.0f;
    float sum = 0.0f;
    
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            float dx = x - radius;
            float dy = y - radius;
            float val = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            kernel[0][0][y][x] = val;
            sum += val;
        }
    }
    kernel = kernel / sum;
    kernel = kernel.to(device);
    
    // 对每个通道应用高斯模糊
    auto blurred = torch::zeros_like(img);
    for (int c = 0; c < img.size(0); ++c) {
        auto ch = img[c].unsqueeze(0).unsqueeze(0); // [1, 1, H, W]
        auto blurred_ch = torch::conv2d(ch, kernel, {}, 1, radius);
        blurred[c] = blurred_ch.squeeze(0).squeeze(0);
    }
    
    // 计算细节
    auto detail = img - blurred;
    
    // 应用阈值：只对差异大于阈值的像素进行锐化
    if (threshold > 0.0f) {
        auto mask = detail.abs() > (threshold / 255.0f);
        detail = detail * mask.to(torch::kFloat32);
    }
    
    // 锐化
    auto sharpened = img + detail * amount;
    return sharpened.clamp(0, 1);
}

// 基础降噪（高斯模糊）
torch::Tensor denoise(const torch::Tensor& image, float strength) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    
    // 根据强度确定模糊半径
    int radius = std::max(1, static_cast<int>(strength * 3));
    int kernel_size = 2 * radius + 1;
    
    auto kernel = torch::zeros({1, 1, kernel_size, kernel_size}, torch::TensorOptions().dtype(torch::kFloat32));
    float sigma = radius / 2.0f;
    float sum = 0.0f;
    
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            float dx = x - radius;
            float dy = y - radius;
            float val = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            kernel[0][0][y][x] = val;
            sum += val;
        }
    }
    kernel = kernel / sum;
    kernel = kernel.to(device);
    
    // 对每个通道应用高斯模糊
    auto blurred = torch::zeros_like(img);
    for (int c = 0; c < img.size(0); ++c) {
        auto ch = img[c].unsqueeze(0).unsqueeze(0);
        auto blurred_ch = torch::conv2d(ch, kernel, {}, 1, radius);
        blurred[c] = blurred_ch.squeeze(0).squeeze(0);
    }
    
    // 混合原图和模糊图
    float blend = std::min(strength, 1.0f);
    return (img * (1.0f - blend) + blurred * blend).clamp(0, 1);
}

// 智能降噪：保留边缘
torch::Tensor smart_denoise(const torch::Tensor& image, float strength) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    
    // 计算边缘强度（使用 Sobel 算子）
    auto sobel_x = torch::tensor({{{-1.0f, 0.0f, 1.0f}, {-2.0f, 0.0f, 2.0f}, {-1.0f, 0.0f, 1.0f}}},
                                 torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    auto sobel_y = torch::tensor({{{-1.0f, -2.0f, -1.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}}},
                                 torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0); // [1, 1, 3, 3]
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0);
    
    // 转为灰度计算边缘
    auto gray = img.mean(0, true); // [1, H, W]
    auto grad_x = torch::conv2d(gray.unsqueeze(0), sobel_x, {}, 1, 1);
    auto grad_y = torch::conv2d(gray.unsqueeze(0), sobel_y, {}, 1, 1);
    auto edge = (grad_x.squeeze() + grad_y.squeeze()).abs();
    auto edge_mask = (edge < edge.mean().item<float>() * 2.0f).to(torch::kFloat32); // 非边缘区域
    
    // 轻量高斯模糊
    int radius = 1;
    auto kernel = torch::tensor({{{0.0625f, 0.125f, 0.0625f}, {0.125f, 0.25f, 0.125f}, {0.0625f, 0.125f, 0.0625f}}},
                                 torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    kernel = kernel.unsqueeze(0).unsqueeze(0);
    
    auto blurred = torch::zeros_like(img);
    for (int c = 0; c < img.size(0); ++c) {
        auto ch = img[c].unsqueeze(0).unsqueeze(0);
        auto blurred_ch = torch::conv2d(ch, kernel, {}, 1, 1);
        blurred[c] = blurred_ch.squeeze(0).squeeze(0);
    }
    
    // 只在非边缘区域应用降噪
    float blend = std::min(strength * 0.7f, 1.0f);
    auto mask = edge_mask.unsqueeze(0); // [1, H, W]
    return (img * (1.0f - blend * mask) + blurred * blend * mask).clamp(0, 1);
}

// 美白：提升亮度，降低饱和度，偏冷色调
torch::Tensor whiten(const torch::Tensor& image, float strength) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    float factor = std::min(strength, 1.0f);
    
    // 提升亮度
    img = img * (1.0f + factor * 0.15f) + factor * 0.05f;
    
    // 降低饱和度
    auto gray = img.mean(0, true);
    img = gray + (img - gray) * (1.0f - factor * 0.2f);
    
    // 轻微冷色调（减少黄/红色调）
    img[0] = img[0] * (1.0f - factor * 0.05f);
    img[2] = img[2] * (1.0f + factor * 0.05f);
    
    return img.clamp(0, 1);
}

// 磨皮：高斯模糊 + 边缘保留
torch::Tensor skin_smooth(const torch::Tensor& image, float strength) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    
    // 检测皮肤色调区域（简单的 RGB 范围判断）
    // 皮肤通常在 R > G > B 的范围内
    auto r = img[0];
    auto g = img[1];
    auto b = img[2];
    
    // 简单的皮肤 mask：R > G 且 G > B 且 R 在中等亮度
    auto skin_mask = (r > g) * (g > b) * (r > 0.3f) * (r < 0.85f);
    skin_mask = skin_mask.to(torch::kFloat32);
    
    // 对皮肤区域应用高斯模糊
    int radius = std::max(1, static_cast<int>(strength * 3));
    int kernel_size = 2 * radius + 1;
    auto kernel = torch::zeros({1, 1, kernel_size, kernel_size}, torch::TensorOptions().dtype(torch::kFloat32));
    float sigma = radius / 2.0f;
    float sum = 0.0f;
    
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            float dx = x - radius;
            float dy = y - radius;
            float val = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            kernel[0][0][y][x] = val;
            sum += val;
        }
    }
    kernel = kernel / sum;
    kernel = kernel.to(device);
    
    auto blurred = torch::zeros_like(img);
    for (int c = 0; c < img.size(0); ++c) {
        auto ch = img[c].unsqueeze(0).unsqueeze(0);
        auto blurred_ch = torch::conv2d(ch, kernel, {}, 1, radius);
        blurred[c] = blurred_ch.squeeze(0).squeeze(0);
    }
    
    // 只对皮肤区域应用模糊
    float blend = std::min(strength * 0.6f, 1.0f);
    auto mask = skin_mask.unsqueeze(0); // [1, H, W]
    return (img * (1.0f - blend * mask) + blurred * blend * mask).clamp(0, 1);
}

// RGB 曲线调整
// curves: "input,output;input,output;..." (0-255)
torch::Tensor apply_curves(const torch::Tensor& image, const std::string& curves) {
    if (curves.empty()) return image.clone();
    
    // Parse control points
    std::vector<std::pair<float, float>> points;
    size_t pos = 0;
    while (pos < curves.size()) {
        size_t semi = curves.find(';', pos);
        std::string pair_str = (semi == std::string::npos) ? curves.substr(pos) : curves.substr(pos, semi - pos);
        size_t comma = pair_str.find(',');
        if (comma != std::string::npos) {
            float input_val = std::stof(pair_str.substr(0, comma)) / 255.0f;
            float output_val = std::stof(pair_str.substr(comma + 1)) / 255.0f;
            points.push_back({input_val, output_val});
        }
        if (semi == std::string::npos) break;
        pos = semi + 1;
    }
    
    if (points.size() < 2) return image.clone();
    
    // Sort by input value
    std::sort(points.begin(), points.end());
    
    // Create lookup table (256 entries)
    std::vector<float> lut(256);
    for (int i = 0; i < 256; ++i) {
        float x = i / 255.0f;
        // Find segment
        size_t idx = 0;
        for (size_t j = 0; j < points.size() - 1; ++j) {
            if (x >= points[j].first && x <= points[j + 1].first) {
                idx = j;
                break;
            }
        }
        // Linear interpolation
        float x0 = points[idx].first;
        float y0 = points[idx].second;
        float x1 = points[idx + 1].first;
        float y1 = points[idx + 1].second;
        if (x1 - x0 > 0) {
            float t = (x - x0) / (x1 - x0);
            lut[i] = y0 + t * (y1 - y0);
        } else {
            lut[i] = y0;
        }
    }
    
    // Apply lookup table
    auto img = image.clone();
    auto lut_tensor = torch::tensor(lut, torch::TensorOptions().dtype(torch::kFloat32)).to(img.device());
    
    // Scale to 0-255, round, clamp, lookup, scale back
    auto idx = (img * 255.0f).clamp(0, 255).to(torch::kInt64);
    
    for (int c = 0; c < img.size(0); ++c) {
        img[c] = lut_tensor.index({idx[c]});
    }
    
    return img.clamp(0, 1);
}

// 内置滤镜预设
torch::Tensor apply_preset(const torch::Tensor& image, const std::string& name) {
    auto img = image.clone();
    
    if (name == "bw" || name == "blackwhite") {
        // 黑白
        auto gray = img.mean(0, true).expand_as(img);
        return gray.clamp(0, 1);
    }
    else if (name == "sepia") {
        // 复古 sepia
        auto r = img[0];
        auto g = img[1];
        auto b = img[2];
        img[0] = (r * 0.393f + g * 0.769f + b * 0.189f).clamp(0, 1);
        img[1] = (r * 0.349f + g * 0.686f + b * 0.168f).clamp(0, 1);
        img[2] = (r * 0.272f + g * 0.534f + b * 0.131f).clamp(0, 1);
        return img;
    }
    else if (name == "vintage") {
        // 复古：降低饱和度 + 暖色调 + 轻微对比度提升
        auto gray = img.mean(0, true);
        img = gray + (img - gray) * 0.6f; // 降低饱和度
        img = adjust_temperature(img, 0.3f); // 暖色调
        img = adjust_contrast(img, 0.15f); // 提升对比度
        img = adjust_brightness(img, -0.05f); // 轻微压暗
        return img.clamp(0, 1);
    }
    else if (name == "warm") {
        return adjust_temperature(img, 0.5f).clamp(0, 1);
    }
    else if (name == "cool") {
        return adjust_temperature(img, -0.5f).clamp(0, 1);
    }
    else if (name == "dramatic") {
        // 戏剧性：高对比度 + 压暗 + 提升饱和度
        img = adjust_contrast(img, 0.4f);
        img = adjust_brightness(img, -0.1f);
        img = adjust_saturation(img, 0.2f);
        img = adjust_shadows(img, -30.0f);
        return img.clamp(0, 1);
    }
    else if (name == "japanese" || name == "film") {
        // 日系/胶片：低对比度 + 轻微过曝 + 降低饱和度 + 偏青
        img = adjust_contrast(img, -0.1f);
        img = adjust_brightness(img, 0.1f);
        img = adjust_saturation(img, -0.2f);
        img = adjust_temperature(img, -0.1f); // 轻微冷色
        img = adjust_highlights(img, 20.0f);
        return img.clamp(0, 1);
    }
    else if (name == "cyberpunk") {
        // 赛博朋克：高饱和度 + 洋红/青色偏移
        img = adjust_saturation(img, 0.4f);
        img = adjust_contrast(img, 0.2f);
        // 增强蓝和洋红
        img[0] = img[0] * 1.1f; // 红
        img[2] = img[2] * 1.2f; // 蓝
        return img.clamp(0, 1);
    }
    else if (name == "cinematic") {
        // 电影感：宽色阶 + 轻微去饱和 + 压暗阴影
        img = adjust_contrast(img, 0.15f);
        img = adjust_saturation(img, -0.1f);
        img = adjust_shadows(img, -20.0f);
        img = adjust_highlights(img, 10.0f);
        img = adjust_temperature(img, 0.05f);
        return img.clamp(0, 1);
    }
    
    // 未知预设，返回原图
    return img;
}

// 暗角效果
torch::Tensor vignette(const torch::Tensor& image, float strength, float radius) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    int h = img.size(1);
    int w = img.size(2);
    auto device = img.device();
    
    // 创建径向距离图
    auto y = torch::linspace(-1.0f, 1.0f, h, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto x = torch::linspace(-1.0f, 1.0f, w, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto yy = y.unsqueeze(1).expand({h, w});
    auto xx = x.unsqueeze(0).expand({h, w});
    auto dist = (yy * yy + xx * xx).sqrt(); // 中心到边缘的距离
    
    // 创建 vignette mask
    float r = radius; // 影响半径
    auto mask = (1.0f - (dist / r).clamp(0, 1).pow(2)).clamp(0, 1);
    
    // 应用暗角
    float blend = std::min(strength, 1.0f);
    mask = mask * (1.0f - blend) + blend; // 边缘 darker
    mask = mask.unsqueeze(0); // [1, H, W]
    
    return (img * mask).clamp(0, 1);
}

// 径向滤镜
torch::Tensor radial_filter(const torch::Tensor& image, float cx, float cy, float radius,
                            float exposure_val, float contrast_val, float saturation_val) {
    if (radius <= 0.0f) return image.clone();
    
    auto img = image.clone();
    int h = img.size(1);
    int w = img.size(2);
    auto device = img.device();
    
    // 创建坐标网格
    auto y = torch::linspace(0.0f, 1.0f, h, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto x = torch::linspace(0.0f, 1.0f, w, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto yy = y.unsqueeze(1).expand({h, w});
    auto xx = x.unsqueeze(0).expand({h, w});
    
    // 计算到中心的距离
    auto dx = xx - cx;
    auto dy = yy - cy;
    auto dist = (dx * dx + dy * dy).sqrt();
    
    // 创建径向 mask（圆形区域内部为 1，外部渐变到 0）
    auto mask = (1.0f - ((dist - radius * 0.7f) / (radius * 0.3f)).clamp(0, 1)).clamp(0, 1);
    mask = mask.unsqueeze(0); // [1, H, W]
    
    // 应用曝光调整
    if (exposure_val != 0.0f) {
        auto adjusted = adjust_exposure(img, exposure_val);
        img = img * (1.0f - mask) + adjusted * mask;
    }
    
    // 应用对比度调整
    if (contrast_val != 0.0f) {
        auto adjusted = adjust_contrast(img, contrast_val);
        img = img * (1.0f - mask) + adjusted * mask;
    }
    
    // 应用饱和度调整
    if (saturation_val != 0.0f) {
        auto adjusted = adjust_saturation(img, saturation_val);
        img = img * (1.0f - mask) + adjusted * mask;
    }
    
    return img.clamp(0, 1);
}

// 渐变滤镜
torch::Tensor graduated_filter(const torch::Tensor& image, float angle, float position, float width,
                               float exposure_val, float contrast_val, float saturation_val) {
    if (width <= 0.0f) return image.clone();
    
    auto img = image.clone();
    int h = img.size(1);
    int w = img.size(2);
    auto device = img.device();
    
    // 将角度转换为弧度
    float rad = angle * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);
    
    // 创建坐标网格
    auto y = torch::linspace(0.0f, 1.0f, h, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto x = torch::linspace(0.0f, 1.0f, w, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto yy = y.unsqueeze(1).expand({h, w});
    auto xx = x.unsqueeze(0).expand({h, w});
    
    // 计算沿梯度方向的距离
    // 将坐标投影到梯度方向
    auto proj = xx * cos_a + yy * sin_a;
    
    // 创建渐变 mask
    auto mask = (1.0f - ((proj - position) / width).abs().clamp(0, 1)).clamp(0, 1);
    mask = mask.unsqueeze(0); // [1, H, W]
    
    // 应用曝光调整
    if (exposure_val != 0.0f) {
        auto adjusted = adjust_exposure(img, exposure_val);
        img = img * (1.0f - mask) + adjusted * mask;
    }
    
    // 应用对比度调整
    if (contrast_val != 0.0f) {
        auto adjusted = adjust_contrast(img, contrast_val);
        img = img * (1.0f - mask) + adjusted * mask;
    }
    
    // 应用饱和度调整
    if (saturation_val != 0.0f) {
        auto adjusted = adjust_saturation(img, saturation_val);
        img = img * (1.0f - mask) + adjusted * mask;
    }
    
    return img.clamp(0, 1);
}

// Smart sharpen: edge-aware sharpening
// strength: 0.0-3.0 (sharpening strength)
// radius: 1-5 (blur radius for edge detection)
torch::Tensor smart_sharpen(const torch::Tensor& image, float strength, int radius) {
    if (strength <= 0.0f) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    
    // Sobel edge detection
    auto sobel_x = torch::tensor({{{-1.0f, 0.0f, 1.0f}, {-2.0f, 0.0f, 2.0f}, {-1.0f, 0.0f, 1.0f}}},
                                 torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    auto sobel_y = torch::tensor({{{-1.0f, -2.0f, -1.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 1.0f}}},
                                 torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0); // [1, 1, 3, 3]
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0);
    
    // Convert to grayscale for edge detection
    auto gray = img.mean(0, true); // [1, H, W]
    auto grad_x = torch::conv2d(gray.unsqueeze(0), sobel_x, {}, 1, 1);
    auto grad_y = torch::conv2d(gray.unsqueeze(0), sobel_y, {}, 1, 1);
    auto edge_magnitude = (grad_x.squeeze().pow(2) + grad_y.squeeze().pow(2)).sqrt();
    
    // Normalize edge magnitude to [0, 1]
    auto edge_max = edge_magnitude.max();
    auto edge_mask = (edge_magnitude / (edge_max + 1e-6)).clamp(0, 1);
    
    // Create sharpening kernel (Laplacian-like)
    auto kernel = torch::tensor({{{0.0f, -1.0f, 0.0f}, {-1.0f, 5.0f, -1.0f}, {0.0f, -1.0f, 0.0f}}},
                                 torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    kernel = kernel.unsqueeze(0).unsqueeze(0);
    
    // Apply sharpening to each channel
    auto sharpened = torch::zeros_like(img);
    for (int c = 0; c < img.size(0); ++c) {
        auto ch = img[c].unsqueeze(0).unsqueeze(0);
        auto sharp_ch = torch::conv2d(ch, kernel, {}, 1, 1);
        sharpened[c] = sharp_ch.squeeze(0).squeeze(0);
    }
    
    // Blend based on edge mask: more sharpening on edges, less on smooth areas
    float blend = std::min(strength / 3.0f, 1.0f);
    auto mask = edge_mask.unsqueeze(0); // [1, H, W]
    auto result = img * (1.0f - blend * mask) + sharpened * blend * mask;
    
    return result.clamp(0, 1);
}

} // namespace myimg
