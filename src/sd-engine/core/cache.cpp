// ============================================================================
// sd-engine/core/cache.cpp
// ============================================================================

#include "cache.h"
#include "sd_ptr.h"
#include <algorithm>

namespace sdengine {

ExecutionCache::ExecutionCache(size_t max_size_bytes) 
    : max_size_(max_size_bytes) {}

bool ExecutionCache::has(const std::string& node_id, const std::string& hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.find(make_key(node_id, hash)) != cache_.end();
}

NodeOutputs ExecutionCache::get(const std::string& node_id, const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(make_key(node_id, hash));
    if (it != cache_.end()) {
        it->second.last_access = std::chrono::steady_clock::now();
        return it->second.outputs;
    }
    return {};
}

void ExecutionCache::put(const std::string& node_id, const std::string& hash,
                         const NodeOutputs& outputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = make_key(node_id, hash);
    size_t entry_size = estimate_size(outputs);
    
    // 如果已存在，先减去旧大小
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        current_size_ -= it->second.memory_size;
    }
    
    // 检查是否需要清理
    while (current_size_ + entry_size > max_size_ && !cache_.empty()) {
        gc();
    }
    
    CacheEntry entry;
    entry.hash = hash;
    entry.outputs = outputs;
    entry.last_access = std::chrono::steady_clock::now();
    entry.memory_size = entry_size;
    
    cache_[key] = entry;
    current_size_ += entry_size;
}

void ExecutionCache::gc() {
    if (cache_.empty()) return;
    
    // 找到最久未访问的条目
    auto oldest = std::min_element(cache_.begin(), cache_.end(),
        [](const auto& a, const auto& b) {
            return a.second.last_access < b.second.last_access;
        });
    
    if (oldest != cache_.end()) {
        current_size_ -= oldest->second.memory_size;
        cache_.erase(oldest);
    }
}

void ExecutionCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    current_size_ = 0;
}

std::string ExecutionCache::make_key(const std::string& node_id, 
                                     const std::string& hash) const {
    return node_id + "::" + hash;
}

size_t ExecutionCache::estimate_size(const NodeOutputs& outputs) const {
    size_t size = 0;
    for (const auto& [_, value] : outputs) {
        size += sizeof(value);
        // 估算智能指针类型的实际内存占用
        if (value.type() == typeid(LatentPtr)) {
            auto ptr = std::any_cast<LatentPtr>(value);
            if (ptr) {
                int w = 0, h = 0, c = 0;
                sd_latent_get_shape(ptr.get(), &w, &h, &c);
                size += (size_t)w * h * c * sizeof(float);
            }
        } else if (value.type() == typeid(ConditioningPtr)) {
            auto ptr = std::any_cast<ConditioningPtr>(value);
            if (ptr) size += 512 * 1024;  // conditioning 约 512KB
        } else if (value.type() == typeid(ImagePtr)) {
            auto ptr = std::any_cast<ImagePtr>(value);
            if (ptr) {
                size += ptr->width * ptr->height * ptr->channel;
            }
        } else if (value.type() == typeid(sd_image_t)) {
            auto img = std::any_cast<sd_image_t>(value);
            size += img.width * img.height * img.channel;
        } else if (value.type() == typeid(UpscalerPtr)) {
            size += 1024 * 1024;  // upscaler model ~1MB
        }
    }
    return std::max(size, (size_t)1024);  // 最小 1KB
}

} // namespace sdengine
