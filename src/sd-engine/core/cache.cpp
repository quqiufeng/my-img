// ============================================================================
// sd-engine/core/cache.cpp
// ============================================================================

#include "cache.h"
#include "sd_ptr.h"
#include <algorithm>

namespace sdengine {

ExecutionCache::ExecutionCache(size_t max_size_bytes) : max_size_(max_size_bytes) {}

bool ExecutionCache::has(const std::string& node_id, const std::string& hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(node_id, hash);
    return cache_.find(key) != cache_.end();
}

NodeOutputs ExecutionCache::get(const std::string& node_id, const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(node_id, hash);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second.entry.last_access = std::chrono::steady_clock::now();
        touch(key);
        return it->second.entry.outputs;
    }
    return {};
}

void ExecutionCache::put(const std::string& node_id, const std::string& hash, const NodeOutputs& outputs) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = make_key(node_id, hash);
    size_t entry_size = estimate_size(outputs);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // 更新已有条目
        current_size_ -= it->second.entry.memory_size;
        it->second.entry.outputs = outputs;
        it->second.entry.last_access = std::chrono::steady_clock::now();
        it->second.entry.memory_size = entry_size;
        current_size_ += entry_size;
        touch(key);
    } else {
        // 新条目
        while (current_size_ + entry_size > max_size_ && !cache_.empty()) {
            evict_one();
        }

        lru_list_.push_back(key);
        CacheEntry entry;
        entry.hash = hash;
        entry.outputs = outputs;
        entry.last_access = std::chrono::steady_clock::now();
        entry.memory_size = entry_size;

        CacheItem item;
        item.entry = std::move(entry);
        item.lru_iter = std::prev(lru_list_.end());
        cache_[key] = std::move(item);
        current_size_ += entry_size;
    }
}

void ExecutionCache::touch(const std::string& key) const {
    auto it = cache_.find(key);
    if (it == cache_.end())
        return;

    // 移动到 LRU 链表尾部（最近使用）
    lru_list_.splice(lru_list_.end(), lru_list_, it->second.lru_iter);
}

void ExecutionCache::evict_one() {
    if (lru_list_.empty())
        return;

    std::string key = lru_list_.front();
    lru_list_.pop_front();

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        current_size_ -= it->second.entry.memory_size;
        cache_.erase(it);
    }
}

void ExecutionCache::gc() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_.empty())
        return;
    evict_one();
}

void ExecutionCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
    current_size_ = 0;
}

size_t ExecutionCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

std::string ExecutionCache::make_key(const std::string& node_id, const std::string& hash) const {
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
            if (ptr)
                size += 512 * 1024; // conditioning 约 512KB
        } else if (value.type() == typeid(ImagePtr)) {
            auto ptr = std::any_cast<ImagePtr>(value);
            if (ptr) {
                size += ptr->width * ptr->height * ptr->channel;
            }
        } else if (value.type() == typeid(sd_image_t)) {
            auto img = std::any_cast<sd_image_t>(value);
            size += img.width * img.height * img.channel;
        } else if (value.type() == typeid(UpscalerPtr)) {
            size += 1024 * 1024; // upscaler model ~1MB
        }
    }
    return std::max(size, (size_t)1024); // 最小 1KB
}

} // namespace sdengine
