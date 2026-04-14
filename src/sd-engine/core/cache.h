// ============================================================================
// sd-engine/core/cache.h
// ============================================================================
// 
// 节点执行结果缓存
// ============================================================================

#pragma once

#include "node.h"
#include <map>
#include <chrono>
#include <mutex>

namespace sdengine {

struct CacheEntry {
    std::string hash;
    NodeOutputs outputs;
    std::chrono::steady_clock::time_point last_access;
    size_t memory_size = 0;
};

class ExecutionCache {
public:
    ExecutionCache(size_t max_size_bytes = 1024 * 1024 * 1024);  // 默认 1GB
    
    bool has(const std::string& node_id, const std::string& hash) const;
    NodeOutputs get(const std::string& node_id, const std::string& hash);
    void put(const std::string& node_id, const std::string& hash, 
             const NodeOutputs& outputs);
    
    // 清理过期/超量缓存
    void gc();
    void clear();
    
    size_t get_current_size() const { return current_size_; }
    size_t get_max_size() const { return max_size_; }

private:
    mutable std::mutex mutex_;
    std::map<std::string, CacheEntry> cache_;  // key = node_id + "::" + hash
    size_t current_size_ = 0;
    size_t max_size_;
    
    std::string make_key(const std::string& node_id, const std::string& hash) const;
    size_t estimate_size(const NodeOutputs& outputs) const;
};

} // namespace sdengine
