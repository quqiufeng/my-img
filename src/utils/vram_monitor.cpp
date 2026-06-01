#include "vram_monitor.h"
#include "log.h"

#include <sstream>
#include <iomanip>
#include <atomic>

#ifdef __linux__
#include <fstream>
#endif

namespace myimg {

static std::atomic<float> g_peak_vram_mb{0.0f};

float VRAMMonitor::get_used_vram_mb() {
#ifdef __linux__
    // 尝试从 nvidia-smi 获取
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            try {
                float used_mb = std::stof(buffer);
                pclose(pipe);
                return used_mb;
            } catch (const std::exception&) {
                // 解析失败，关闭管道继续尝试其他方法
            }
        }
        pclose(pipe);
    }
    
    // 备用：尝试从 /proc 读取 (仅对集成显卡有效)
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("VmallocUsed") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    try {
                        float kb = std::stof(line.substr(pos + 1));
                        return kb / 1024.0f;
                    } catch (const std::exception&) {
                        return 0.0f;
                    }
                }
            }
        }
    }
#endif
    return 0.0f;
}

float VRAMMonitor::get_total_vram_mb() {
#ifdef __linux__
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            try {
                float total_mb = std::stof(buffer);
                pclose(pipe);
                return total_mb;
            } catch (const std::exception&) {
                // 解析失败，关闭管道
            }
        }
        pclose(pipe);
    }
#endif
    return 0.0f;
}

float VRAMMonitor::get_vram_usage_percent() {
    float total = get_total_vram_mb();
    if (total <= 0.0f) return 0.0f;
    float used = get_used_vram_mb();
    return (used / total) * 100.0f;
}

float VRAMMonitor::get_peak_vram_mb() {
    float current = get_used_vram_mb();
    float expected = g_peak_vram_mb.load();
    while (current > expected && !g_peak_vram_mb.compare_exchange_weak(expected, current)) {
        // Retry if another thread updated the value
    }
    return g_peak_vram_mb.load();
}

void VRAMMonitor::reset_peak() {
    g_peak_vram_mb.store(0.0f);
}

std::string VRAMMonitor::format_vram_info() {
    float used = get_used_vram_mb();
    float total = get_total_vram_mb();
    float percent = get_vram_usage_percent();
    float peak = get_peak_vram_mb();
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "VRAM: " << used << "/" << total << " MB (" << percent << "%)";
    if (peak > 0.0f) {
        ss << " [Peak: " << peak << " MB]";
    }
    return ss.str();
}

} // namespace myimg
