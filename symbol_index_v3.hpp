/* Symbol Index V3 - C++ Wrapper
 * Modern C++ interface with RAII
 * 
 * Usage:
 *   SymbolIndexV3 idx("/path/to/index.bin");
 *   auto result = idx.find("my_function");
 *   if (result) {
 *       std::cout << result->name << " at " << result->file << ":" << result->line;
 *   }
 */

#ifndef SYMBOL_INDEX_V3_HPP
#define SYMBOL_INDEX_V3_HPP

#include "symbol_index_v3.h"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

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
    
    /* Find symbol by exact name */
    std::unique_ptr<SymbolInfo> find(const std::string& name) const {
        SymbolResultPy* res = symbol_index_find_py(handle_, name.c_str());
        if (!res) return nullptr;
        
        auto info = std::make_unique<SymbolInfo>();
        fill_info(res, *info);
        symbol_result_py_free(res);
        
        return info;
    }
    
    /* Find symbols by prefix */
    std::vector<SymbolInfo> find_prefix(const std::string& prefix) const {
        // Note: These functions return SymbolResult* which is opaque
        // Use Python-compatible API or add accessor functions
        // For now, return empty vector - TODO: implement iterator
        (void)prefix;
        return std::vector<SymbolInfo>{};
    }
    
    /* Find symbols by glob pattern */
    std::vector<SymbolInfo> glob(const std::string& pattern) const {
        // Note: These functions return SymbolResult* which is opaque
        // Use Python-compatible API or add accessor functions
        (void)pattern;
        return std::vector<SymbolInfo>{};
    }
    
    /* Fuzzy search */
    std::vector<SymbolInfo> fuzzy(const std::string& name, int max_dist = 2) const {
        // Note: These functions return SymbolResult* which is opaque
        // Use Python-compatible API or add accessor functions
        (void)name;
        (void)max_dist;
        return std::vector<SymbolInfo>{};
    }
    
    /* Regex search */
    std::vector<SymbolInfo> regex(const std::string& pattern) const {
        // Note: These functions return SymbolResult* which is opaque
        // Use Python-compatible API or add accessor functions
        (void)pattern;
        return std::vector<SymbolInfo>{};
    }
    
    /* Check if index is valid */
    bool valid() const { return handle_ != nullptr; }

private:
    SymbolIndex* handle_;
    
    void fill_info(SymbolResultPy* res, SymbolInfo& info) const {
        char* name = symbol_result_name(res);
        char* sig = symbol_result_signature(res);
        char* file = symbol_result_file(res);
        char* code = symbol_result_code_snippet(res);
        char* ctx = symbol_result_context_json(res);
        
        if (name) info.name = name;
        if (sig) info.signature = sig;
        if (file) info.file = file;
        info.line = symbol_result_line(res);
        info.kind = symbol_result_kind(res);
        if (code) info.code_snippet = code;
        if (ctx) info.context_json = ctx;
    }
};

/* Convenience type alias */
using SymbolIndexPtr = std::unique_ptr<SymbolIndexV3>;

/* Factory function */
inline SymbolIndexPtr make_symbol_index(const std::string& path) {
    return std::make_unique<SymbolIndexV3>(path);
}

} /* namespace code_index */

#endif /* SYMBOL_INDEX_V3_HPP */
