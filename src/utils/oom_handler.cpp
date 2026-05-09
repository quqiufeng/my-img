#include "oom_handler.h"
#include "utils/log.h"

#include <sstream>
#include <cmath>
#include <iostream>

namespace myimg {

int OOMHandler::calculate_latent_size(int pixel_size) {
    return (pixel_size + 7) / 8;  // 对齐到 8
}

int OOMHandler::estimate_sampling_vram(int width, int height) {
    int latent_w = calculate_latent_size(width);
    int latent_h = calculate_latent_size(height);

    // 基础显存：latent 张量 + 注意力矩阵 + 中间结果
    // UNet 推理主要显存占用：
    // 1. Latent: 2 * 16 * latent_w * latent_h * 4 bytes (float32) ≈ 128 * latent_w * latent_h
    // 2. Attention: ~4 * latent_w * latent_h * 1024 * 4 bytes ≈ 16K * latent_w * latent_h
    // 3. 中间特征图: ~10 * latent_w * latent_h * 1024 * 4 bytes ≈ 40K * latent_w * latent_h

    // 简化估算（基于 RTX 3080 实测数据）
    int64_t pixels = static_cast<int64_t>(latent_w) * latent_h;

    // 基础开销：~2500 MB（模型权重已加载，使用 GGUF 量化后）
    int base_mb = 2500;

    // 每 latent 像素约 0.10 MB（优化后）
    int per_pixel_mb = static_cast<int>(pixels * 0.10);

    // 注意力计算开销（Flash Attention 已优化）
    int attention_mb = static_cast<int>(std::sqrt(pixels) * 30);

    return base_mb + per_pixel_mb + attention_mb;
}

int OOMHandler::estimate_vram(int width, int height, int steps, bool hires, int hires_steps) {
    int base_mb = estimate_sampling_vram(width, height);

    // 步数影响较小（因为每步复用内存），但 HiRes 需要额外内存
    int steps_overhead = steps * 2;  // 每步约 2MB 临时开销

    int hires_mb = 0;
    if (hires) {
        // HiRes Fix 需要额外显存：
        // 1. 放大后的 latent（2x 或更高）
        // 2. 额外的采样步数
        int hires_width = width * 2;
        int hires_height = height * 2;
        hires_mb = estimate_sampling_vram(hires_width, hires_height) * 0.6;  // HiRes 不需要重新加载模型
        hires_mb += hires_steps * 2;
    }

    // VAE 解码开销（与像素数成正比）
    int vae_mb = (width * height) / 50000;  // 约每 50K 像素 1MB

    return base_mb + steps_overhead + hires_mb + vae_mb;
}

bool OOMHandler::apply_fallback(int& width, int& height, int& steps,
                                 bool& hires, int& hires_steps,
                                 bool& freeu, bool& sag,
                                 const OOMFallbackConfig& config,
                                 int retry_count) {
    if (retry_count >= config.max_retries) {
        LOG_ERROR("[OOM] Max retries exceeded. Giving up.");
        return false;
    }

    std::cout << "[OOM] Applying fallback strategy (attempt " << (retry_count + 1)
              << "/" << config.max_retries << ")" << std::endl;

    bool modified = false;

    // 策略 1：减少 HiRes 步数
    if (config.reduce_hires_steps && hires && hires_steps > 20) {
        int old_hires_steps = hires_steps;
        hires_steps = std::max(20, hires_steps - 15);
        std::cout << "[OOM] Reduced HiRes steps: " << old_hires_steps << " -> " << hires_steps << std::endl;
        modified = true;
    }

    // 策略 2：降低分辨率
    if (config.reduce_resolution && (width > 512 || height > 512)) {
        int old_width = width;
        int old_height = height;
        width = std::max(512, width * 3 / 4);
        height = std::max(512, height * 3 / 4);
        // 对齐到 8 的倍数
        width = (width / 8) * 8;
        height = (height / 8) * 8;
        std::cout << "[OOM] Reduced resolution: " << old_width << "x" << old_height
                  << " -> " << width << "x" << height << std::endl;
        modified = true;
    }

    // 策略 3：关闭增强功能
    if (config.disable_enhancements) {
        if (freeu) {
            freeu = false;
            std::cout << "[OOM] Disabled FreeU" << std::endl;
            modified = true;
        }
        if (sag) {
            sag = false;
            std::cout << "[OOM] Disabled SAG" << std::endl;
            modified = true;
        }
    }

    // 策略 4：减少基础步数（最后手段）
    if (steps > 20) {
        int old_steps = steps;
        steps = std::max(15, steps - 5);
        std::cout << "[OOM] Reduced steps: " << old_steps << " -> " << steps << std::endl;
        modified = true;
    }

    // 策略 5：如果仍可能 OOM，关闭 HiRes
    if (hires && retry_count >= 2) {
        hires = false;
        std::cout << "[OOM] Disabled HiRes Fix" << std::endl;
        modified = true;
    }

    if (!modified) {
        LOG_ERROR("[OOM] No more fallback options available.");
        return false;
    }

    return true;
}

std::string OOMHandler::get_friendly_suggestion(int vram_required_mb, int vram_available_mb) {
    std::stringstream ss;
    int shortage_mb = vram_required_mb - vram_available_mb;

    ss << "显存不足（需要约 " << vram_required_mb << "MB，可用 " << vram_available_mb << "MB，短缺 " << shortage_mb << "MB）\n";
    ss << "\n建议：\n";

    if (shortage_mb > 4000) {
        ss << "  1. 大幅降低分辨率（如 1280x720 -> 640x480）\n";
        ss << "  2. 关闭 HiRes Fix\n";
        ss << "  3. 关闭 FreeU 和 SAG\n";
        ss << "  4. 考虑使用更低分辨率的模型\n";
    } else if (shortage_mb > 2000) {
        ss << "  1. 降低分辨率（如 2560x1440 -> 1920x1080）\n";
        ss << "  2. 减少 HiRes 步数（45 -> 25）\n";
        ss << "  3. 关闭 FreeU 和 SAG\n";
    } else if (shortage_mb > 500) {
        ss << "  1. 减少 HiRes 步数\n";
        ss << "  2. 关闭 FreeU/SAG\n";
        ss << "  3. 启用 VAE Tiling\n";
    } else {
        ss << "  1. 关闭其他占用显存的程序\n";
        ss << "  2. 稍微降低分辨率\n";
    }

    ss << "\n已自动应用降级策略，正在重试...\n";

    return ss.str();
}

bool OOMHandler::will_oom(int width, int height, int steps, bool hires, int hires_steps,
                          int vram_mb) {
    int required = estimate_vram(width, height, steps, hires, hires_steps);
    // 预留 10% 安全余量
    return required > vram_mb * 0.9;
}

bool OOMHandler::handle_batch_oom(int& width, int& height, int& steps,
                                   bool& hires, int& hires_steps,
                                   bool& freeu, bool& sag,
                                   const OOMFallbackConfig& config) {
    static int retry_count = 0;

    bool success = apply_fallback(width, height, steps, hires, hires_steps,
                                  freeu, sag, config, retry_count);

    if (success) {
        retry_count++;
    } else {
        retry_count = 0;  // 重置计数器
    }

    return success;
}

} // namespace myimg
