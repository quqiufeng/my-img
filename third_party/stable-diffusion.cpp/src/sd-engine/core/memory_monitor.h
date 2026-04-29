// ============================================================================
// sd-engine/core/memory_monitor.h
// ============================================================================
// 显存监控与自动优化
// ============================================================================

#pragma once

#include <string>

namespace sdengine {

/// @brief GPU 显存信息
struct GPUMemoryInfo {
    int device_id = -1;
    int64_t total_bytes = 0;
    int64_t free_bytes = 0;
    int64_t used_bytes = 0;
    bool available = false;
};

/// @brief 检测 GPU 显存（优先 NVIDIA， fallback 到系统内存）
GPUMemoryInfo detect_gpu_memory();

/// @brief 根据显存估算最大支持分辨率
/// @param free_bytes 可用显存字节
/// @param use_gpu 是否使用 GPU
/// @param batch_count 批次数量
/// @return 推荐的最大像素数（width * height）
int64_t estimate_max_pixels(int64_t free_bytes, bool use_gpu, int batch_count = 1);

/// @brief 自动选择合适的分辨率
/// @param requested_width 用户请求的宽度
/// @param requested_height 用户请求的高度
/// @param free_bytes 可用显存
/// @param use_gpu 是否使用 GPU
/// @return 调整后的分辨率 (width, height)
std::pair<int, int> auto_adjust_resolution(int requested_width, int requested_height, 
                                            int64_t free_bytes, bool use_gpu);

} // namespace sdengine
