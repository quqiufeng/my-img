// ============================================================================
// sd-engine/core/cache.h
// ============================================================================
/// @file cache.h
/// @brief 节点执行结果缓存
///
/// ExecutionCache 用于缓存节点的执行结果，避免重复计算。
/// 支持基于内存上限的自动垃圾回收（LRU 策略）。
// ============================================================================

#pragma once

#include "node.h"
#include <map>
#include <chrono>
#include <mutex>

namespace sdengine {

/// @brief 缓存条目
struct CacheEntry {
    std::string hash;                                       ///< 输入哈希值
    NodeOutputs outputs;                                    ///< 缓存的输出数据
    std::chrono::steady_clock::time_point last_access;      ///< 最后访问时间
    size_t memory_size = 0;                                 ///< 估算的内存占用（字节）
};

/// @brief 节点执行结果缓存管理器
///
/// 以节点 ID + 输入哈希为键存储输出结果，自动估算内存占用并在超限时
/// 清理最久未访问的条目。
class ExecutionCache {
public:
    /// @brief 构造函数
    /// @param max_size_bytes 缓存最大内存限制（默认 1GB）
    explicit ExecutionCache(size_t max_size_bytes = 1024 * 1024 * 1024);

    /// @brief 检查缓存中是否存在指定条目
    bool has(const std::string& node_id, const std::string& hash) const;

    /// @brief 从缓存中获取条目（同时更新访问时间）
    NodeOutputs get(const std::string& node_id, const std::string& hash);

    /// @brief 将条目存入缓存
    void put(const std::string& node_id, const std::string& hash,
             const NodeOutputs& outputs);

    /// @brief 垃圾回收：清理过期或超量的缓存条目
    void gc();

    /// @brief 清空所有缓存
    void clear();

    /// @brief 获取当前缓存估算内存占用
    size_t get_current_size() const { return current_size_; }

    /// @brief 获取缓存内存上限
    size_t get_max_size() const { return max_size_; }

private:
    mutable std::mutex mutex_;
    std::map<std::string, CacheEntry> cache_;  ///< 缓存存储，键格式为 "node_id::hash"
    size_t current_size_ = 0;
    size_t max_size_;

    std::string make_key(const std::string& node_id, const std::string& hash) const;
    size_t estimate_size(const NodeOutputs& outputs) const;
};

} // namespace sdengine
