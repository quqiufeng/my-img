// ============================================================================
// sd-engine/core/memory_monitor.cpp
// ============================================================================

#include "memory_monitor.h"
#include "log.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>

namespace sdengine {

GPUMemoryInfo detect_gpu_memory() {
    GPUMemoryInfo info;
    info.available = false;

    // 尝试 nvidia-smi
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            float total_mb, free_mb, used_mb;
            if (sscanf(buffer, "%f, %f, %f", &total_mb, &free_mb, &used_mb) == 3) {
                info.device_id = 0;
                info.total_bytes = static_cast<int64_t>(total_mb * 1024 * 1024);
                info.free_bytes = static_cast<int64_t>(free_mb * 1024 * 1024);
                info.used_bytes = static_cast<int64_t>(used_mb * 1024 * 1024);
                info.available = true;
                LOG_INFO("[Memory] GPU detected: %.0f MB total, %.0f MB free\n", total_mb, free_mb);
            }
        }
        pclose(pipe);
    }

    // 如果没有 GPU，检测系统内存
    if (!info.available) {
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            int64_t mem_total = 0, mem_available = 0;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal:") == 0) {
                    sscanf(line.c_str(), "MemTotal: %ld", &mem_total);
                    mem_total *= 1024; // kB to bytes
                } else if (line.find("MemAvailable:") == 0) {
                    sscanf(line.c_str(), "MemAvailable: %ld", &mem_available);
                    mem_available *= 1024;
                }
            }
            if (mem_total > 0) {
                info.device_id = -1; // CPU
                info.total_bytes = mem_total;
                info.free_bytes = mem_available;
                info.used_bytes = mem_total - mem_available;
                info.available = true;
                LOG_INFO("[Memory] CPU mode: %.0f MB total, %.0f MB free\n", 
                         mem_total / (1024.0 * 1024.0), mem_available / (1024.0 * 1024.0));
            }
        }
    }

    return info;
}

int64_t estimate_max_pixels(int64_t free_bytes, bool use_gpu, int batch_count) {
    // 基础显存需求估算（非常粗略）：
    // - 模型权重: ~2-8GB (SD1.5 ~2GB, SDXL ~7GB, Flux ~23GB)
    // -  latent: 64x64x4x4 = 64KB per sample... actually much more in practice
    // - 激活值: 与分辨率成正比
    // 经验公式: 每 1GB 显存支持约 512x512 的 1 batch
    // 或者更精确: 1GB ≈ 262k pixels (512*512)

    int64_t reserve_bytes = 1024 * 1024 * 1024; // 预留 1GB 给系统/其他
    if (!use_gpu) {
        reserve_bytes = 512 * 1024 * 1024; // CPU 预留少些
    }

    int64_t usable = free_bytes - reserve_bytes;
    if (usable <= 0) {
        usable = 512 * 1024 * 1024; // 最小 512MB
    }

    // 每像素大约需要 3-5 bytes 的激活内存（保守估计）
    // 加上模型本身，我们按每 1GB 支持 256k pixels
    double gb = usable / (1024.0 * 1024.0 * 1024.0);
    int64_t max_pixels = static_cast<int64_t>(gb * 256000 * batch_count);

    // 限制上下界
    if (max_pixels < 256 * 256) max_pixels = 256 * 256;
    if (max_pixels > 4096 * 4096) max_pixels = 4096 * 4096;

    return max_pixels;
}

std::pair<int, int> auto_adjust_resolution(int requested_width, int requested_height,
                                            int64_t free_bytes, bool use_gpu) {
    int64_t requested_pixels = static_cast<int64_t>(requested_width) * requested_height;
    int64_t max_pixels = estimate_max_pixels(free_bytes, use_gpu);

    if (requested_pixels <= max_pixels) {
        // 显存充足，使用请求的分辨率
        return {requested_width, requested_height};
    }

    // 需要降分辨率，保持宽高比
    double ratio = static_cast<double>(requested_width) / requested_height;
    int new_height = static_cast<int>(std::sqrt(max_pixels / ratio));
    int new_width = static_cast<int>(new_height * ratio);

    // 对齐到 64 的倍数（latent 空间要求），保持比例
    new_width = (new_width / 64) * 64;
    new_height = static_cast<int>(new_width / ratio);
    new_height = (new_height / 64) * 64;

    if (new_width < 256) new_width = 256;
    if (new_height < 256) new_height = 256;

    LOG_WARN("[Memory] Resolution auto-adjusted: %dx%d -> %dx%d (free: %.0f MB, max: %.0f MP)\n",
             requested_width, requested_height, new_width, new_height,
             free_bytes / (1024.0 * 1024.0), max_pixels / 1000000.0);

    return {new_width, new_height};
}

} // namespace sdengine
