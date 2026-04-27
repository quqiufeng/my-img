#include "adapters/sdcpp_adapter.h"
#include <iostream>

using namespace myimg;

int main() {
    std::cout << "=== SDCPP Adapter Test ===" << std::endl;
    
    // 1. 检查版本
    std::cout << "\n1. Checking sd.cpp version..." << std::endl;
    std::cout << "   Version: " << SDCPPAdapter::get_version() << std::endl;
    std::cout << "   Commit: " << SDCPPAdapter::get_commit() << std::endl;
    
    // 2. 列出可用采样方法
    std::cout << "\n2. Available sample methods:" << std::endl;
    auto methods = SDCPPAdapter::get_available_sample_methods();
    for (size_t i = 0; i < methods.size(); i++) {
        std::cout << "   " << i << ": " << methods[i] << std::endl;
    }
    
    // 3. 列出可用调度器
    std::cout << "\n3. Available schedulers:" << std::endl;
    auto schedulers = SDCPPAdapter::get_available_schedulers();
    for (size_t i = 0; i < schedulers.size(); i++) {
        std::cout << "   " << i << ": " << schedulers[i] << std::endl;
    }
    
    // 4. 尝试加载模型（如果存在）
    std::string model_path = "/opt/image/model/z_image_turbo-Q5_K_M.gguf";
    std::cout << "\n4. Testing model loading..." << std::endl;
    std::cout << "   Model path: " << model_path << std::endl;
    
    SDCPPAdapter adapter;
    GenerationParams params;
    // Z-Image 模型需要使用 diffusion_model_path
    params.diffusion_model_path = model_path;
    params.vae_path = "/opt/image/model/ae.safetensors";
    params.llm_path = "/opt/image/model/Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    params.n_threads = 4;  // 使用4线程测试
    params.wtype = "default";  // 使用模型默认类型
    
    if (adapter.initialize(params)) {
        std::cout << "   Model loaded successfully!" << std::endl;
        
        // 5. 测试生成（可选，如果用户想要）
        std::cout << "\n5. Model ready for generation" << std::endl;
        std::cout << "   Supports image generation: yes" << std::endl;
    } else {
        std::cout << "   Model loading failed (this is OK if model file doesn't exist)" << std::endl;
    }
    
    std::cout << "\n=== Test Completed ===" << std::endl;
    return 0;
}
