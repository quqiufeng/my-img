/* Symbol Index V3 - C++ Wrapper
 * Modern C++ interface with RAII and iterators
 * 
 * Usage:
 *   SymbolIndexV3 idx("/path/to/index.bin");
 *   
 *   // Single lookup
 *   auto result = idx.find("my_function");
 *   if (result) {
 *       std::cout << result->name << " at " << result->file << ":" << result->line;
 *   }
 *   
 *   // Range-based iteration over all symbols
 *   for (const auto& sym : idx) {
 *       std::cout << sym.name << std::endl;
 *   }
 *   
 *   // Search operations
 *   auto matches = idx.find_prefix("test_");
 *   auto globs = idx.glob("foo*bar");
 *   auto fuzzy = idx.fuzzy("myfuncton", 2);  // max distance 2
 *   auto regex = idx.regex("^test_.*");
 */

#ifndef SYMBOL_INDEX_V3_HPP
#define SYMBOL_INDEX_V3_HPP

#include "symbol_index_v3.h"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iterator>

namespace code_index {

/* Symbol information structure */
struct SymbolInfo {
    std::string name;
    std::string signature;
    std::string file;
    uint32_t line;
    int kind;
    std::string code_snippet;
    std::string context_json;
    
    /* Check if symbol has code snippet */
    bool has_code() const { return !code_snippet.empty(); }
    
    /* Check if symbol has context JSON */
    bool has_context() const { return !context_json.empty(); }
    
    /* Get kind as string */
    std::string kind_name() const { return symbol_kind_name(kind); }
};

/* Forward declaration */
class SymbolIndexV3;

/* Iterator for range-based for loops */
class SymbolIterator {
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = SymbolInfo;
    using difference_type = std::ptrdiff_t;
    using pointer = const SymbolInfo*;
    using reference = const SymbolInfo&;

    SymbolIterator() : idx_(nullptr), current_(0), cached_info_(nullptr) {}
    
    SymbolIterator(const SymbolIndexV3* idx, uint32_t pos) 
        : idx_(idx), current_(pos), cached_info_(nullptr) {
        advance_to_valid();
    }
    
    ~SymbolIterator() = default;
    
    // Copy
    SymbolIterator(const SymbolIterator& other) 
        : idx_(other.idx_), current_(other.current_), cached_info_(nullptr) {}
    SymbolIterator& operator=(const SymbolIterator& other) {
        idx_ = other.idx_;
        current_ = other.current_;
        cached_info_.reset();
        return *this;
    }
    
    // Move
    SymbolIterator(SymbolIterator&& other) noexcept
        : idx_(other.idx_), current_(other.current_), cached_info_(std::move(other.cached_info_)) {}
    SymbolIterator& operator=(SymbolIterator&& other) noexcept {
        idx_ = other.idx_;
        current_ = other.current_;
        cached_info_ = std::move(other.cached_info_);
        return *this;
    }
    
    reference operator*() const {
        if (!cached_info_) {
            const_cast<SymbolIterator*>(this)->cache_current();
        }
        return *cached_info_;
    }
    
    pointer operator->() const {
        if (!cached_info_) {
            const_cast<SymbolIterator*>(this)->cache_current();
        }
        return cached_info_.get();
    }
    
    SymbolIterator& operator++() {
        ++current_;
        cached_info_.reset();
        advance_to_valid();
        return *this;
    }
    
    SymbolIterator operator++(int) {
        SymbolIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    bool operator==(const SymbolIterator& other) const {
        return idx_ == other.idx_ && current_ == other.current_;
    }
    
    bool operator!=(const SymbolIterator& other) const {
        return !(*this == other);
    }

private:
    const SymbolIndexV3* idx_;
    uint32_t current_;
    std::unique_ptr<SymbolInfo> cached_info_;
    
    void advance_to_valid();
    void cache_current();
};

/* RAII wrapper for SymbolIndex */
class SymbolIndexV3 {
public:
    /* Constructor - opens index file
     * Throws std::runtime_error on failure
     */
    explicit SymbolIndexV3(const std::string& path) 
        : handle_(symbol_index_open(path.c_str())) {
        if (!handle_) {
            throw std::runtime_error("Failed to open index: " + path);
        }
    }
    
    /* Destructor - automatically closes index */
    ~SymbolIndexV3() {
        if (handle_) {
            symbol_index_close(handle_);
        }
    }
    
    /* Disable copy */
    SymbolIndexV3(const SymbolIndexV3&) = delete;
    SymbolIndexV3& operator=(const SymbolIndexV3&) = delete;
    
    /* Enable move */
    SymbolIndexV3(SymbolIndexV3&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    SymbolIndexV3& operator=(SymbolIndexV3&& other) noexcept {
        if (this != &other) {
            if (handle_) symbol_index_close(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    /* Get symbol count */
    uint32_t count() const {
        return symbol_index_count(handle_);
    }
    
    /* Get file count */
    uint32_t file_count() const {
        return symbol_index_file_count(handle_);
    }
    
    /* Iterator interface for range-based for loops */
    SymbolIterator begin() const {
        return SymbolIterator(this, 0);
    }
    
    SymbolIterator end() const {
        return SymbolIterator(this, count());
    }
    
    /* Find symbol by exact name */
    std::unique_ptr<SymbolInfo> find(const std::string& name) const {
        SymbolResult* res = symbol_index_find(handle_, name.c_str());
        if (!res) return nullptr;
        
        auto info = std::make_unique<SymbolInfo>();
        fill_info_from_result(res, *info);
        symbol_result_free(res);
        
        return info;
    }
    
    /* Find symbols by prefix */
    std::vector<SymbolInfo> find_prefix(const std::string& prefix) const {
        int count = 0;
        SymbolResult* results = symbol_index_find_prefix(handle_, prefix.c_str(), &count);
        return convert_results(results, count);
    }
    
    /* Find symbols by glob pattern */
    std::vector<SymbolInfo> glob(const std::string& pattern) const {
        int count = 0;
        SymbolResult* results = symbol_index_glob(handle_, pattern.c_str(), &count);
        return convert_results(results, count);
    }
    
    /* Fuzzy search by edit distance */
    std::vector<SymbolInfo> fuzzy(const std::string& name, int max_dist = 2) const {
        int count = 0;
        SymbolResult* results = symbol_index_fuzzy(handle_, name.c_str(), max_dist, &count);
        return convert_results(results, count);
    }
    
    /* Regex search */
    std::vector<SymbolInfo> regex(const std::string& pattern) const {
        int count = 0;
        SymbolResult* results = symbol_index_regex(handle_, pattern.c_str(), &count);
        return convert_results(results, count);
    }
    
    /* Check if index is valid */
    bool valid() const { return handle_ != nullptr; }
    
    /* Access underlying C handle (for advanced use) */
    SymbolIndex* handle() const { return handle_; }

private:
    SymbolIndex* handle_;
    
    friend class SymbolIterator;
    
    void fill_info_from_result(SymbolResult* res, SymbolInfo& info) const {
        const char* name = symbol_result_get_name(res);
        const char* sig = symbol_result_get_signature(res);
        const char* file = symbol_result_get_file(res);
        const char* code = symbol_result_get_code_snippet(res);
        const char* ctx = symbol_result_get_context_json(res);
        
        if (name) info.name = name;
        if (sig) info.signature = sig;
        if (file) info.file = file;
        info.line = symbol_result_get_line(res);
        info.kind = symbol_result_get_kind(res);
        if (code) info.code_snippet = code;
        if (ctx) info.context_json = ctx;
    }
    
    std::vector<SymbolInfo> convert_results(SymbolResult* results, int count) const {
        std::vector<SymbolInfo> vec;
        if (!results || count <= 0) return vec;
        
        vec.reserve(count);
        for (int i = 0; i < count; i++) {
            SymbolInfo info;
            fill_info_from_result(&results[i], info);
            vec.push_back(std::move(info));
        }
        
        symbol_results_free(results, count);
        return vec;
    }
};

/* Iterator implementation */
inline void SymbolIterator::advance_to_valid() {
    // In this implementation, all positions are valid until end
    // since we're iterating by index
}

inline void SymbolIterator::cache_current() {
    if (!idx_ || !idx_->handle_) return;
    
    cached_info_ = std::make_unique<SymbolInfo>();
    SymbolResult* res = symbol_index_get_by_index(idx_->handle_, current_);
    if (res) {
        const char* name = symbol_result_get_name(res);
        const char* sig = symbol_result_get_signature(res);
        const char* file = symbol_result_get_file(res);
        const char* code = symbol_result_get_code_snippet(res);
        const char* ctx = symbol_result_get_context_json(res);
        
        if (name) cached_info_->name = name;
        if (sig) cached_info_->signature = sig;
        if (file) cached_info_->file = file;
        cached_info_->line = symbol_result_get_line(res);
        cached_info_->kind = symbol_result_get_kind(res);
        if (code) cached_info_->code_snippet = code;
        if (ctx) cached_info_->context_json = ctx;
        
        symbol_result_free(res);
    }
}

/* Convenience type alias */
using SymbolIndexPtr = std::unique_ptr<SymbolIndexV3>;

/* Factory function */
inline SymbolIndexPtr make_symbol_index(const std::string& path) {
    return std::make_unique<SymbolIndexV3>(path);
}

} /* namespace code_index */

#endif /* SYMBOL_INDEX_V3_HPP */
