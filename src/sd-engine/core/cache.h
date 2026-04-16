// ============================================================================
// sd-engine/core/cache.h
// ============================================================================
/// @file cache.h
/// @brief 节点执行结果缓存
///
/// ExecutionCache 用于缓存节点的执行结果，避免重复计算。
/// 支持基于内存上限的自动垃圾回收（LRU 策略），使用 list + unordered_map
/// 实现 O(1) 的 get/put/lru 淘汰。
// ============================================================================

#pragma once

#include "node.h"
#include <chrono>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sdengine {

/// @brief 缓存条目
struct CacheEntry {
    std::string hash;                                  ///< 输入哈希值
    NodeOutputs outputs;                               ///< 缓存的输出数据
    std::chrono::steady_clock::time_point last_access; ///< 最后访问时间
    size_t memory_size = 0;                            ///< 估算的内存占用（字节）
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

    /// @brief 将条目存入缓存（拷贝版本）
    void put(const std::string& node_id, const std::string& hash, const NodeOutputs& outputs);

    /// @brief 将条目存入缓存（移动版本，优先使用以避免不必要的拷贝）
    void put(const std::string& node_id, const std::string& hash, NodeOutputs&& outputs);

    /// @brief 垃圾回收：清理过期或超量的缓存条目
    void gc();

    /// @brief 清空所有缓存
    void clear();

    /// @brief 获取当前缓存估算内存占用
    size_t get_current_size() const {
        return current_size_;
    }

    /// @brief 获取缓存内存上限
    size_t get_max_size() const {
        return max_size_;
    }

    /// @brief 获取当前缓存条目数量
    size_t size() const;

  private:
    using LRUList = std::list<std::string>;
    using LRUListIter = LRUList::iterator;

    mutable std::mutex mutex_;

    // LRU 链表：队首是最久未访问，队尾是最近访问
    mutable LRUList lru_list_;

    // 哈希表：键 -> {条目, LRU链表迭代器}
    struct CacheItem {
        CacheEntry entry;
        LRUListIter lru_iter;
    };
    std::unordered_map<std::string, CacheItem> cache_;

    size_t current_size_ = 0;
    size_t max_size_;

    std::string make_key(const std::string& node_id, const std::string& hash) const;
    size_t estimate_size(const NodeOutputs& outputs) const;

    void touch(const std::string& key) const;
    void evict_one();
};

} // namespace sdengine
