/* Test program for Symbol Index V3 C++ wrapper with iterators */
#include "symbol_index_v3.hpp"
#include <iostream>
#include <cstdlib>

void test_single_lookup(code_index::SymbolIndexV3& idx, const char* name) {
    std::cout << "\n=== Single Lookup: " << name << " ===" << std::endl;
    
    auto result = idx.find(name);
    if (result) {
        std::cout << "Found: " << result->name << std::endl;
        std::cout << "  Kind: " << result->kind_name() << " (" << result->kind << ")" << std::endl;
        std::cout << "  File: " << result->file << ":" << result->line << std::endl;
        if (!result->signature.empty()) {
            std::cout << "  Signature: " << result->signature << std::endl;
        }
        if (result->has_code()) {
            std::cout << "  Code: " << result->code_snippet.substr(0, 50) << "..." << std::endl;
        }
    } else {
        std::cout << "Not found" << std::endl;
    }
}

void test_prefix_search(code_index::SymbolIndexV3& idx, const char* prefix) {
    std::cout << "\n=== Prefix Search: " << prefix << " ===" << std::endl;
    
    auto results = idx.find_prefix(prefix);
    std::cout << "Found " << results.size() << " matches" << std::endl;
    
    for (const auto& sym : results) {
        std::cout << "  - " << sym.name << " (" << sym.kind_name() << ")" << std::endl;
    }
}

void test_glob_search(code_index::SymbolIndexV3& idx, const char* pattern) {
    std::cout << "\n=== Glob Search: " << pattern << " ===" << std::endl;
    
    auto results = idx.glob(pattern);
    std::cout << "Found " << results.size() << " matches" << std::endl;
    
    for (const auto& sym : results) {
        std::cout << "  - " << sym.name << std::endl;
    }
}

void test_fuzzy_search(code_index::SymbolIndexV3& idx, const char* name, int max_dist) {
    std::cout << "\n=== Fuzzy Search: " << name << " (max_dist=" << max_dist << ") ===" << std::endl;
    
    auto results = idx.fuzzy(name, max_dist);
    std::cout << "Found " << results.size() << " matches" << std::endl;
    
    for (const auto& sym : results) {
        std::cout << "  - " << sym.name << std::endl;
    }
}

void test_regex_search(code_index::SymbolIndexV3& idx, const char* pattern) {
    std::cout << "\n=== Regex Search: " << pattern << " ===" << std::endl;
    
    auto results = idx.regex(pattern);
    std::cout << "Found " << results.size() << " matches" << std::endl;
    
    for (const auto& sym : results) {
        std::cout << "  - " << sym.name << std::endl;
    }
}

void test_iteration(code_index::SymbolIndexV3& idx) {
    std::cout << "\n=== Iterating All Symbols ===" << std::endl;
    std::cout << "Total symbols: " << idx.count() << std::endl;
    
    int count = 0;
    for (const auto& sym : idx) {
        std::cout << "  " << (++count) << ". " << sym.name 
                  << " (" << sym.kind_name() << ")" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <index.bin> [symbol_name]" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " test_index.bin" << std::endl;
        std::cerr << "  " << argv[0] << " test_index.bin test_func" << std::endl;
        return 1;
    }

    try {
        // Open index using C++ wrapper
        code_index::SymbolIndexV3 idx(argv[1]);
        
        std::cout << "Index loaded successfully!" << std::endl;
        std::cout << "Symbols: " << idx.count() << std::endl;
        std::cout << "Files: " << idx.file_count() << std::endl;
        
        // Test iteration
        test_iteration(idx);
        
        // Test single lookup
        if (argc >= 3) {
            test_single_lookup(idx, argv[2]);
        } else {
            test_single_lookup(idx, "test_func");
        }
        
        // Test prefix search
        test_prefix_search(idx, "test");
        test_prefix_search(idx, "hel");
        
        // Test glob search
        test_glob_search(idx, "*func*");
        test_glob_search(idx, "util?");
        
        // Test fuzzy search
        test_fuzzy_search(idx, "tst_func", 2);
        test_fuzzy_search(idx, "hlp", 1);
        
        // Test regex search
        test_regex_search(idx, "^test.*");
        test_regex_search(idx, "util");
        
        std::cout << "\n=== All tests completed successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
