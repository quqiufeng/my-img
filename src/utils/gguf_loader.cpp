#include "utils/gguf_loader.h"
#include <gguf.h>
#include <ggml.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace myimg {

std::map<std::string, torch::Tensor> GGUFLoder::load(const std::string& path) {
    std::map<std::string, torch::Tensor> tensors;
    
    std::cout << "[GGUF] Loading: " << path << std::endl;
    
    // Initialize GGUF context
    struct gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    
    struct gguf_context* ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        std::cerr << "[GGUF] Failed to load file: " << path << std::endl;
        return tensors;
    }
    
    // Print metadata
    std::cout << "[GGUF] Version: " << gguf_get_version(ctx) << std::endl;
    std::cout << "[GGUF] Alignment: " << gguf_get_alignment(ctx) << std::endl;
    std::cout << "[GGUF] Number of tensors: " << gguf_get_n_tensors(ctx) << std::endl;
    std::cout << "[GGUF] Number of KV pairs: " << gguf_get_n_kv(ctx) << std::endl;
    
    // Read metadata
    for (int64_t i = 0; i < gguf_get_n_kv(ctx); i++) {
        const char* key = gguf_get_key(ctx, i);
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        
        if (type == GGUF_TYPE_STRING) {
            std::cout << "[GGUF] " << key << " = " << gguf_get_val_str(ctx, i) << std::endl;
        } else if (type == GGUF_TYPE_UINT32) {
            std::cout << "[GGUF] " << key << " = " << gguf_get_val_u32(ctx, i) << std::endl;
        }
    }
    
    // Open file for reading tensor data
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[GGUF] Failed to open file for reading tensor data" << std::endl;
        gguf_free(ctx);
        return tensors;
    }
    
    // Read tensors
    int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);
        size_t size = gguf_get_tensor_size(ctx, i);
        
        // Get number of dimensions
        // Note: gguf_get_tensor_ndims is not available in this version
        // We'll need to read the dimensions from the file directly
        // For now, assume common shapes
        
        // Calculate number of elements
        int64_t n_elements = size * ggml_blck_size(type) / ggml_type_size(type);
        
        // Read raw data
        std::vector<uint8_t> raw_data(size);
        file.seekg(offset);
        file.read(reinterpret_cast<char*>(raw_data.data()), size);
        
        torch::Tensor tensor;
        
        if (type == GGML_TYPE_F32) {
            // FP32: Direct copy, no dequantization needed
            tensor = torch::from_blob(raw_data.data(), {n_elements}, torch::kFloat32);
            tensor = tensor.to(torch::kFloat16).clone(); // Clone to own the memory
        } else if (type == GGML_TYPE_F16) {
            // FP16: Direct copy
            tensor = torch::from_blob(raw_data.data(), {n_elements}, torch::kFloat16);
            tensor = tensor.clone(); // Clone to own the memory
        } else {
            // Quantized types: Dequantize
            const struct ggml_type_traits* traits = ggml_get_type_traits(type);
            if (!traits || !traits->to_float) {
                std::cerr << "[GGUF] No dequantization function for type " << type << " in tensor " << name << std::endl;
                continue;
            }
            
            // Dequantize to float32
            std::vector<float> float_data(n_elements);
            traits->to_float(raw_data.data(), float_data.data(), n_elements);
            
            // Convert to torch::Tensor (FP16)
            tensor = torch::from_blob(float_data.data(), {n_elements}, torch::kFloat32);
            tensor = tensor.to(torch::kFloat16);
        }
        
        tensors[name] = tensor;
        
        std::cout << "[GGUF] Loaded tensor: " << name 
                  << " (type=" << type << ", elements=" << n_elements << ")" << std::endl;
    }
    
    file.close();
    gguf_free(ctx);
    
    std::cout << "[GGUF] Loaded " << tensors.size() << " tensors" << std::endl;
    
    return tensors;
}

} // namespace myimg
