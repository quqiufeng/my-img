// ============================================================================
// sd-engine/adapter/sd_adapter.h
// ============================================================================
/// @file sd_adapter.h
/// @brief stable-diffusion.cpp C API 的 C++ 封装
///
/// 仅使用 upstream 官方 API（stable-diffusion.h），不依赖任何扩展。
/// 提供模型管理、图像生成、ESRGAN 放大的高层接口。
// ============================================================================

#pragma once

#include "core/sd_ptr.h"
#include "stable-diffusion.h"
#include <cstring>
#include <string>
#include <vector>

namespace sdengine {

/// @brief LoRA 信息结构体
struct LoRAInfo {
    std::string path;
    float strength;
};

/// @brief SDAdapter 配置参数
struct SDAdapterConfig {
    std::string diffusion_model_path;  ///< 扩散模型路径（.gguf/.safetensors）
    std::string vae_path;              ///< VAE 模型路径（可选）
    std::string llm_path;              ///< LLM/CLIP 模型路径（可选）
    int n_threads = 4;                 ///< CPU 线程数
    bool use_gpu = true;               ///< 是否使用 GPU
    bool flash_attn = false;           ///< 是否启用 Flash Attention
};

/// @brief 图像生成参数
struct GenerateParams {
    std::string prompt;                ///< 正向提示词
    std::string negative_prompt;       ///< 负向提示词
    int width = 512;                   ///< 图像宽度
    int height = 512;                  ///< 图像高度
    int sample_steps = 20;             ///< 采样步数
    float cfg_scale = 7.5f;            ///< CFG Scale
    enum sample_method_t sample_method = EULER_SAMPLE_METHOD;  ///< 采样方法
    enum scheduler_t scheduler = DISCRETE_SCHEDULER;           ///< 调度器
    int64_t seed = 42;                 ///< 随机种子
    float strength = 1.0f;             ///< img2img 强度（1.0 = txt2img）
    ImagePtr init_image;               ///< 初始图像（img2img 用，nullptr = txt2img）
    ImagePtr mask_image;               ///< 蒙版图像（inpaint 用）
    
    // HiRes Fix 参数
    bool enable_hires = false;         ///< 是否启用 HiRes Fix
    int hires_width = 1024;            ///< HiRes 目标宽度
    int hires_height = 1024;           ///< HiRes 目标高度
    float hires_strength = 0.5f;       ///< HiRes 重绘强度
    int hires_steps = 20;              ///< HiRes 采样步数
    
    // LoRA 参数
    std::vector<LoRAInfo> loras;      ///< LoRA 列表
};

/// @brief ESRGAN 放大参数
struct UpscaleParams {
    std::string model_path;            ///< ESRGAN 模型路径
    uint32_t upscale_factor = 2;       ///< 放大倍数（2 或 4）
};

/// @brief SDAdapter —— stable-diffusion.cpp 的 C++ 封装
class SDAdapter {
  public:
    SDAdapter() = default;
    ~SDAdapter() = default;
    
    // 禁止拷贝，允许移动
    SDAdapter(const SDAdapter&) = delete;
    SDAdapter& operator=(const SDAdapter&) = delete;
    SDAdapter(SDAdapter&&) = default;
    SDAdapter& operator=(SDAdapter&&) = default;
    
    /// @brief 初始化：加载模型
    bool init(const SDAdapterConfig& config);
    
    /// @brief 释放资源
    void release();
    
    /// @brief 生成图像（txt2img / img2img / HiRes Fix）
    ImagePtr generate(const GenerateParams& params);
    
    /// @brief 初始化 ESRGAN 放大器
    bool init_upscaler(const UpscaleParams& params);
    
    /// @brief ESRGAN 放大
    ImagePtr upscale(const sd_image_t* input_image);
    
    /// @brief 检查是否已初始化
    bool is_initialized() const { return sd_ctx_ != nullptr; }
    
    /// @brief 获取底层 sd_ctx（高级用）
    sd_ctx_t* get_sd_ctx() const { return sd_ctx_.get(); }
    
  private:
    SDContextPtr sd_ctx_;              ///< 模型上下文
    UpscalerPtr upscaler_ctx_;         ///< ESRGAN 上下文
    
    /// @brief 应用 LoRA
    void apply_loras(const std::vector<LoRAInfo>& loras);
    
    /// @brief 清除 LoRA
    void clear_loras();
};

} // namespace sdengine
