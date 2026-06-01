#pragma once

#include <string>

namespace myimg {

// VRAM 监控工具
class VRAMMonitor {
public:
    // 获取当前 VRAM 使用量 (MB)
    static float get_used_vram_mb();
    
    // 获取总 VRAM (MB)
    static float get_total_vram_mb();
    
    // 获取 VRAM 使用率 (%)
    static float get_vram_usage_percent();
    
    // 获取峰值 VRAM 使用量 (MB)
    static float get_peak_vram_mb();
    
    // 重置峰值记录
    static void reset_peak();
    
    // 格式化输出 VRAM 信息
    static std::string format_vram_info();
};

} // namespace myimg
