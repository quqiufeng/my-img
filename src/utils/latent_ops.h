#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace myimg {

// Latent 合成结构
struct LatentCompositeLayer {
    torch::Tensor latent;           // latent 张量 [C, H, W]
    int x = 0;                      // 左上角 x 坐标
    int y = 0;                      // 左上角 y 坐标
    float feather = 0.0f;           // 边缘羽化半径（像素）
};

// Latent Composite: 将多个 latent 按位置合成
// 类似于 ComfyUI 的 LatentComposite 节点
// layers: 多个 latent 层，按顺序叠加（后面的覆盖前面的）
// base_latent: 基础 latent（可选，如果不提供则创建空白 latent）
// target_size: 目标尺寸 {channels, height, width}
torch::Tensor latent_composite(
    const std::vector<LatentCompositeLayer>& layers,
    const torch::Tensor& base_latent = {},
    const std::vector<int64_t>& target_size = {}
);

// 带 mask 的 Latent Composite: 使用 mask 控制融合区域
// mask: 0-1 浮点 mask，形状 [1, H, W] 或 [H, W]
torch::Tensor latent_composite_with_mask(
    const torch::Tensor& base_latent,
    const torch::Tensor& overlay_latent,
    const torch::Tensor& mask,
    int x = 0,
    int y = 0
);

// 创建渐变 mask（用于羽化边缘）
// width, height: mask 尺寸
// feather: 羽化半径（像素）
torch::Tensor create_feather_mask(int width, int height, float feather);

// 将图像转换为 latent（通过 VAE 编码）
// 注意：此功能需要 VAE 模型，当前为占位符
torch::Tensor image_to_latent(const torch::Tensor& image);

// 将 latent 转换为图像（通过 VAE 解码）
// 注意：此功能需要 VAE 模型，当前为占位符
torch::Tensor latent_to_image(const torch::Tensor& latent);

} // namespace myimg
