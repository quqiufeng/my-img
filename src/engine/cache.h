#pragma once

#include <string>
#include <map>
#include <any>
#include <limits>

namespace myimg {

class Cache {
public:
    Cache(size_t max_size = 100 * 1024 * 1024); // 100MB default
    
    bool has(const std::string& key) const;
    std::any get(const std::string& key);
    void set(const std::string& key, const std::any& value, size_t size);
    void clear();
    
private:
    size_t max_size_;
    size_t current_size_;
    std::map<std::string, std::pair<std::any, size_t>> cache_;
};

} // namespace myimg
