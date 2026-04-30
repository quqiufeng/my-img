#pragma once

#include <string>

namespace myimg {

// VRAM 估算结果
struct VRAMEstimate {
    int width = 0;
    int height = 0;
    int steps = 0;
    bool hires = false;
    int hires_steps = 0;
    int estimated_mb = 0;
};

// OOM 降级配置
struct OOMFallbackConfig {
    bool reduce_hires_steps = true;      // 减少 HiRes 步数
    bool reduce_resolution = true;        // 降低分辨率
    bool enable_aggressive_tiling = true; // 启用更激进的 tiling
    bool disable_enhancements = true;     // 关闭 FreeU/SAG 等增强
    int max_retries = 3;                  // 最大重试次数
};

class OOMHandler {
public:
    // 估算生成所需的显存（MB）
    // 基于经验公式：基础模型约 5GB + 每百万像素约 1.5GB + HiRes 额外 2-4GB
    static int estimate_vram(int width, int height, int steps, bool hires, int hires_steps);

    // 估算单步采样所需的显存（MB）
    static int estimate_sampling_vram(int width, int height);

    // 应用降级策略，返回是否成功调整
    // retry_count: 当前重试次数（0-based）
    static bool apply_fallback(int& width, int& height, int& steps,
                               bool& hires, int& hires_steps,
                               bool& freeu, bool& sag,
                               const OOMFallbackConfig& config,
                               int retry_count);

    // 获取用户友好的建议
    static std::string get_friendly_suggestion(int vram_required_mb, int vram_available_mb);

    // 检查给定配置是否会 OOM
    static bool will_oom(int width, int height, int steps, bool hires, int hires_steps,
                         int vram_mb);

    // 批量生成时的 OOM 处理策略
    static bool handle_batch_oom(int& width, int& height, int& steps,
                                 bool& hires, int& hires_steps,
                                 bool& freeu, bool& sag,
                                 const OOMFallbackConfig& config);

private:
    // 计算 latent 尺寸
    static int calculate_latent_size(int pixel_size);
};

} // namespace myimg
