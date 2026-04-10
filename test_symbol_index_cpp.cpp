/* Test program for Symbol Index V3 C++ wrapper */
#include "symbol_index_v3.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <index.bin> <symbol_name>" << std::endl;
        return 1;
    }

    try {
        // Open index using C++ wrapper
        code_index::SymbolIndexV3 idx(argv[1]);
        
        std::cout << "Index loaded successfully!" << std::endl;
        std::cout << "Symbols: " << idx.count() << std::endl;
        std::cout << "Files: " << idx.file_count() << std::endl;
        std::cout << std::endl;
        
        // Find symbol
        auto result = idx.find(argv[2]);
        
        if (result) {
            std::cout << "Found: " << result->name << std::endl;
            std::cout << "Kind: " << result->kind_name() << " (" << result->kind << ")" << std::endl;
            std::cout << "File: " << result->file << ":" << result->line << std::endl;
            
            if (!result->signature.empty()) {
                std::cout << "Signature: " << result->signature << std::endl;
            }
            
            if (result->has_code()) {
                std::cout << "\nCode snippet (" << result->code_snippet.length() << " bytes):" << std::endl;
                std::string snippet = result->code_snippet;
                if (snippet.length() > 500) {
                    snippet = snippet.substr(0, 500) + "...";
                }
                std::cout << snippet << std::endl;
            }
            
            if (result->has_context()) {
                std::cout << "\nContext JSON (" << result->context_json.length() << " bytes):" << std::endl;
                std::string ctx = result->context_json;
                if (ctx.length() > 500) {
                    ctx = ctx.substr(0, 500) + "...";
                }
                std::cout << ctx << std::endl;
            }
        } else {
            std::cout << "Symbol not found: " << argv[2] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
