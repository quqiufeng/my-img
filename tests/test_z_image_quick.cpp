#include "backend/z_image_model.h"
#include <iostream>
#include <chrono>

using namespace myimg;

int main() {
    std::cout << "=== Quick ZImage Test ===" << std::endl;
    
    // 创建小模型来测试架构
    ZImageParams params;
    params.hidden_size = 3840;
    params.num_layers = 30;
    params.num_heads = 30;
    params.head_dim = 128;
    
    std::cout << "Creating model with:" << std::endl;
    std::cout << "  hidden_size: " << params.hidden_size << std::endl;
    std::cout << "  num_layers: " << params.num_layers << std::endl;
    std::cout << "  num_heads: " << params.num_heads << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        ZImageDiT dit(params);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        
        std::cout << "Model created in " << duration << " seconds" << std::endl;
        
        // 计算参数量
        int64_t total_params = 0;
        for (const auto& p : dit->parameters()) {
            total_params += p.numel();
        }
        std::cout << "Total parameters: " << (total_params / 1e6) << "M" << std::endl;
        
        // 测试CPU上的简单前向传播（小latent）
        std::cout << "\nTesting forward pass on CPU with small latent..." << std::endl;
        auto latent = torch::randn({1, 16, 32, 32});  // 256x256图像
        auto timestep = torch::tensor({500.0f});
        auto context = torch::randn({1, 64, 2560});
        
        start = std::chrono::high_resolution_clock::now();
        auto output = dit->forward(latent, timestep, context);
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Forward pass took: " << duration << " ms" << std::endl;
        std::cout << "Output shape: " << output.sizes() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Test Completed ===" << std::endl;
    return 0;
}
