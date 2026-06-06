#pragma once

#include <string>
#include <vector>
#include <memory>

namespace myimg {

// IPAdapter: 图像提示词
// 通过 CLIP Vision 提取图像特征，注入到扩散模型的注意力中
struct IPAdapterConfig {
    std::string model_path;          // IPAdapter 模型路径 (.onnx)
    std::string clip_vision_path;    // CLIP Vision 模型路径 (.onnx)
    std::string image_path;          // 参考图像路径
    float weight = 1.0f;             // 注入权重 (0.0-1.0)
    float start_at = 0.0f;           // 开始注入的步数比例 (0.0-1.0)
    float end_at = 1.0f;             // 结束注入的步数比例 (0.0-1.0)
    bool faceid = false;             // 是否使用 FaceID 模式
};

class IPAdapter {
public:
    IPAdapter();
    explicit IPAdapter(const IPAdapterConfig& config);
    ~IPAdapter();

    // 禁止拷贝，允许移动
    IPAdapter(const IPAdapter&) = delete;
    IPAdapter& operator=(const IPAdapter&) = delete;
    IPAdapter(IPAdapter&&) noexcept;
    IPAdapter& operator=(IPAdapter&&) noexcept;

    // 加载 ONNX 模型（CLIP Vision + IPAdapter MLP）
    bool load_model(const std::string& model_path, const std::string& clip_vision_path);

    // 加载参考图像，提取特征
    // 内部会运行 CLIP Vision → IPAdapter MLP 完整管线
    bool load_reference_image(const std::string& image_path);

    // 获取计算好的 image tokens（扁平化 float 向量）
    // 形状: [1, 768] — 由 IPAdapter MLP 输出
    // 需要投影到 2560-dim (cap_feat_dim) 后方能拼接到 Z-Image 的 text context
    const std::vector<float>& get_image_tokens() const { return image_tokens_; }

    // 是否已加载
    bool is_loaded() const { return model_loaded_; }

    const IPAdapterConfig& config() const { return config_; }

private:
    IPAdapterConfig config_;
    bool model_loaded_ = false;

    // 缓存 image tokens (IPAdapter MLP output, [1, 768])
    std::vector<float> image_tokens_;

    // ONNX Runtime 实现细节 (PIMPL)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace myimg
