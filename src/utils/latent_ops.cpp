#include "utils/latent_ops.h"
#include "utils/log.h"

namespace myimg {

// Latent Composite: 将多个 latent 按位置合成
torch::Tensor latent_composite(
    const std::vector<LatentCompositeLayer>& layers,
    const torch::Tensor& base_latent,
    const std::vector<int64_t>& target_size
) {
    if (layers.empty()) {
        LOG_WARN("latent_composite: no layers provided");
        return base_latent;
    }

    torch::Tensor result;
    
    // 确定目标尺寸
    std::vector<int64_t> final_size;
    if (!target_size.empty()) {
        final_size = target_size;
    } else if (base_latent.defined()) {
        final_size = base_latent.sizes().vec();
    } else {
        // 从第一层推断尺寸
        auto first_sizes = layers[0].latent.sizes();
        final_size = {first_sizes[0], first_sizes[1], first_sizes[2]};
    }
    
    // 初始化结果
    if (base_latent.defined()) {
        result = base_latent.clone();
    } else {
        result = torch::zeros(final_size, layers[0].latent.options());
    }
    
    // 按顺序叠加每一层
    for (const auto& layer : layers) {
        if (!layer.latent.defined() || layer.latent.numel() == 0) {
            continue;
        }
        
        auto layer_sizes = layer.latent.sizes();
        int64_t c = layer_sizes[0];
        int64_t h = layer_sizes[1];
        int64_t w = layer_sizes[2];
        
        // 计算放置范围
        int64_t x_start = layer.x;
        int64_t y_start = layer.y;
        int64_t x_end = std::min(x_start + w, final_size[2]);
        int64_t y_end = std::min(y_start + h, final_size[1]);
        
        if (x_start >= final_size[2] || y_start >= final_size[1]) {
            LOG_WARN("latent_composite: layer out of bounds, skipping");
            continue;
        }
        
        // 提取需要复制的区域
        int64_t copy_w = x_end - x_start;
        int64_t copy_h = y_end - y_start;
        
        auto src = layer.latent.narrow(1, 0, copy_h).narrow(2, 0, copy_w);
        auto dst = result.narrow(1, y_start, copy_h).narrow(2, x_start, copy_w);
        
        if (layer.feather > 0.0f && copy_w > 2 * layer.feather && copy_h > 2 * layer.feather) {
            // 创建羽化 mask
            auto mask = create_feather_mask(static_cast<int>(copy_w), static_cast<int>(copy_h), layer.feather);
            mask = mask.unsqueeze(0); // [1, H, W]
            
            // 扩展到所有通道
            if (c > 1) {
                mask = mask.expand({c, copy_h, copy_w});
            }
            
            // 混合: dst = dst * (1 - mask) + src * mask
            dst.copy_(dst * (1.0f - mask) + src * mask);
        } else {
            // 直接复制
            dst.copy_(src);
        }
    }
    
    return result;
}

// 带 mask 的 Latent Composite
torch::Tensor latent_composite_with_mask(
    const torch::Tensor& base_latent,
    const torch::Tensor& overlay_latent,
    const torch::Tensor& mask,
    int x,
    int y
) {
    if (!base_latent.defined() || !overlay_latent.defined()) {
        LOG_ERROR("latent_composite_with_mask: invalid input tensors");
        return base_latent;
    }
    
    auto result = base_latent.clone();
    auto base_sizes = base_latent.sizes();
    auto overlay_sizes = overlay_latent.sizes();
    
    int64_t c = overlay_sizes[0];
    int64_t h = overlay_sizes[1];
    int64_t w = overlay_sizes[2];
    
    int64_t x_start = x;
    int64_t y_start = y;
    int64_t x_end = std::min(x_start + w, base_sizes[2]);
    int64_t y_end = std::min(y_start + h, base_sizes[1]);
    
    if (x_start >= base_sizes[2] || y_start >= base_sizes[1]) {
        LOG_WARN("latent_composite_with_mask: overlay out of bounds");
        return result;
    }
    
    int64_t copy_w = x_end - x_start;
    int64_t copy_h = y_end - y_start;
    
    auto src = overlay_latent.narrow(1, 0, copy_h).narrow(2, 0, copy_w);
    auto dst = result.narrow(1, y_start, copy_h).narrow(2, x_start, copy_w);
    
    // 处理 mask
    torch::Tensor processed_mask;
    if (mask.dim() == 2) {
        processed_mask = mask.unsqueeze(0); // [1, H, W]
    } else {
        processed_mask = mask;
    }
    
    // 扩展到所有通道
    if (c > 1) {
        processed_mask = processed_mask.expand({c, copy_h, copy_w});
    }
    
    // 混合
    dst.copy_(dst * (1.0f - processed_mask) + src * processed_mask);
    
    return result;
}

// 创建渐变 mask
torch::Tensor create_feather_mask(int width, int height, float feather) {
    auto mask = torch::ones({height, width}, torch::kFloat32);
    
    if (feather <= 0.0f) {
        return mask;
    }
    
    int feather_px = static_cast<int>(feather);
    
    // 水平渐变
    if (width > 2 * feather_px) {
        auto left = torch::linspace(0.0f, 1.0f, feather_px);
        auto right = torch::linspace(1.0f, 0.0f, feather_px);
        
        mask.narrow(1, 0, feather_px).mul_(left.view({1, -1}));
        mask.narrow(1, width - feather_px, feather_px).mul_(right.view({1, -1}));
    }
    
    // 垂直渐变
    if (height > 2 * feather_px) {
        auto top = torch::linspace(0.0f, 1.0f, feather_px);
        auto bottom = torch::linspace(1.0f, 0.0f, feather_px);
        
        mask.narrow(0, 0, feather_px).mul_(top.view({-1, 1}));
        mask.narrow(0, height - feather_px, feather_px).mul_(bottom.view({-1, 1}));
    }
    
    return mask;
}

// 占位符：图像到 latent（需要 VAE 模型）
torch::Tensor image_to_latent(const torch::Tensor& image) { (void)image;
    LOG_WARN("image_to_latent: not implemented yet, requires VAE model");
    return torch::Tensor();
}

// 占位符：latent 到图像（需要 VAE 模型）
torch::Tensor latent_to_image(const torch::Tensor& latent) { (void)latent;
    LOG_WARN("latent_to_image: not implemented yet, requires VAE model");
    return torch::Tensor();
}

} // namespace myimg
