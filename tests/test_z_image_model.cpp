#include "backend/z_image_model.h"
#include "utils/gguf_loader.h"
#include <iostream>
#include <cassert>

using namespace myimg;

int main() {
    std::cout << "=== ZImage Model Test ===" << std::endl;
    
    // 1. 创建模型
    std::cout << "\n1. Creating ZImage model..." << std::endl;
    ZImageModel model;
    std::cout << "   Model created successfully" << std::endl;
    
    // 2. 检查模型参数
    auto dit = model.get_dit();
    auto params = dit->get_params();
    std::cout << "   hidden_size: " << params.hidden_size << std::endl;
    std::cout << "   num_layers: " << params.num_layers << std::endl;
    std::cout << "   num_heads: " << params.num_heads << std::endl;
    
    // 3. 尝试加载模型（如果存在）
    std::string model_path = "/opt/image/model/z_image_turbo-Q5_K_M.gguf";
    std::cout << "\n2. Loading model from: " << model_path << std::endl;
    bool loaded = model.load(model_path);
    
    if (loaded) {
        std::cout << "   Model loaded successfully!" << std::endl;
        
        // 4. 测试前向传播
        std::cout << "\n3. Testing forward pass..." << std::endl;
        
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        std::cout << "   Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
        
        // 创建测试输入
        // latent: [1, 16, 64, 64] (512x512图像的latent)
        auto latent = torch::randn({1, 16, 64, 64}, device);
        auto timestep = torch::tensor({500.0f}, torch::dtype(torch::kFloat32).device(device));
        auto context = torch::randn({1, 256, 2560}, device);  // 文本embedding
        
        std::cout << "   Input latent shape: " << latent.sizes() << std::endl;
        std::cout << "   Timestep shape: " << timestep.sizes() << std::endl;
        std::cout << "   Context shape: " << context.sizes() << std::endl;
        
        // 前向传播
        auto start = std::chrono::high_resolution_clock::now();
        auto output = model.forward(latent, timestep, context);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "   Output shape: " << output.sizes() << std::endl;
        std::cout << "   Forward pass took: " << duration << " ms" << std::endl;
        
        // 验证输出形状
        assert(output.size(0) == 1);
        assert(output.size(1) == 16);
        assert(output.size(2) == 64);
        assert(output.size(3) == 64);
        
        std::cout << "   Output shape validation: PASSED" << std::endl;
        
        // 5. 测试不同分辨率
        std::cout << "\n4. Testing different resolutions..." << std::endl;
        
        std::vector<std::pair<int, int>> resolutions = {
            {512, 512},   // 64x64 latent
            {1024, 1024}, // 128x128 latent
            {1280, 720},  // 160x90 latent
        };
        
        for (const auto& [w, h] : resolutions) {
            auto test_latent = torch::randn({1, 16, h/8, w/8}, device);
            auto test_context = torch::randn({1, 256, 2560}, device);
            
            auto out = model.forward(test_latent, timestep, test_context);
            
            std::cout << "   " << w << "x" << h << " -> output: " << out.sizes() << std::endl;
            
            assert(out.size(2) == h/8);
            assert(out.size(3) == w/8);
        }
        
        std::cout << "   Resolution tests: PASSED" << std::endl;
        
    } else {
        std::cout << "   Model loading failed (this is OK if model file doesn't exist)" << std::endl;
    }
    
    std::cout << "\n=== All Tests Completed ===" << std::endl;
    return 0;
}
