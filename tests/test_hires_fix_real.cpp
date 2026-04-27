#include "adapters/sdcpp_adapter.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace myimg;

int main() {
    std::cout << "=== Z-Image HiRes Fix Test ===" << std::endl;
    
    // 模型路径
    std::string diffusion_model = "/opt/image/model/z_image_turbo-Q5_K_M.gguf";
    std::string vae_model = "/opt/image/model/ae.safetensors";
    std::string llm_model = "/opt/image/model/Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    
    std::cout << "\n1. Initializing adapter..." << std::endl;
    
    SDCPPAdapter adapter;
    GenerationParams params;
    params.diffusion_model_path = diffusion_model;
    params.vae_path = vae_model;
    params.llm_path = llm_model;
    params.n_threads = -1;
    
    adapter.set_progress_callback([](int step, int steps, float time) {
        float progress = (float)step / steps * 100.0f;
        std::cout << "\r  Progress: " << step << "/" << steps << " (" << (int)progress << "%)" << std::flush;
    });
    
    if (!adapter.initialize(params)) {
        std::cerr << "Failed to initialize adapter!" << std::endl;
        return 1;
    }
    
    std::cout << "\n2. Generating image with HiRes Fix..." << std::endl;
    
    // HiRes Fix 参数
    params.prompt = "masterpiece, best quality, a serene mountain landscape with a crystal clear lake, morning mist, golden sunrise, highly detailed, 8k uhd";
    params.negative_prompt = "blurry, low quality, worst quality, jpeg artifacts";
    
    // 低分辨率生成
    params.width = 1280;
    params.height = 720;
    params.sample_steps = 20;
    params.cfg_scale = 3.2f;
    params.sample_method = SampleMethod::Euler;
    params.scheduler = Scheduler::Discrete;
    params.seed = 12345;
    
    // 启用 HiRes Fix
    params.enable_hires = true;
    params.hires_width = 2560;
    params.hires_height = 1440;
    params.hires_strength = 0.30f;
    params.hires_sample_steps = 30;
    
    std::cout << "   Low-res: " << params.width << "x" << params.height << std::endl;
    std::cout << "   Hi-res: " << params.hires_width << "x" << params.hires_height << std::endl;
    std::cout << "   HiRes strength: " << params.hires_strength << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto images = adapter.generate(params);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "\n\n3. Generation completed in " << duration << " seconds" << std::endl;
    
    if (!images.empty()) {
        std::cout << "   Generated " << images.size() << " image(s)" << std::endl;
        
        std::string output_path = "/tmp/test_hires_output.png";
        std::ofstream file(output_path, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(images[0].data.data()), images[0].data.size());
            file.close();
            std::cout << "   Saved to: " << output_path << std::endl;
            std::cout << "   Final size: " << images[0].width << "x" << images[0].height << std::endl;
        }
    } else {
        std::cerr << "   No images generated!" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== HiRes Fix Test Completed ===" << std::endl;
    return 0;
}
