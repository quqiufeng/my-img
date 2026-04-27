#include "engine/cache.h"

namespace myimg {

Cache::Cache(size_t max_size) : max_size_(max_size), current_size_(0) {}

bool Cache::has(const std::string& key) const {
    return cache_.find(key) != cache_.end();
}

std::any Cache::get(const std::string& key) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second.first;
    }
    return std::any();
}

void Cache::set(const std::string& key, const std::any& value, size_t size) {
    cache_[key] = {value, size};
    current_size_ += size;
}

void Cache::clear() {
    cache_.clear();
    current_size_ = 0;
}

} // namespace myimg
