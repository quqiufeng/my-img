#include "adapters/sdcpp_adapter.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace myimg;

int main() {
    std::cout << "=== Z-Image txt2img Test ===" << std::endl;
    
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
    params.n_threads = -1;  // 使用所有可用线程
    
    // 设置进度回调
    adapter.set_progress_callback([](int step, int steps, float time) {
        float progress = (float)step / steps * 100.0f;
        std::cout << "\r  Progress: " << step << "/" << steps << " (" << (int)progress << "%) - " << time << "s/step" << std::flush;
    });
    
    if (!adapter.initialize(params)) {
        std::cerr << "Failed to initialize adapter!" << std::endl;
        return 1;
    }
    
    std::cout << "\n2. Generating image..." << std::endl;
    
    // 设置生成参数
    params.prompt = "masterpiece, best quality, a beautiful sunset over the ocean, golden hour, peaceful, serene landscape";
    params.negative_prompt = "blurry, low quality, worst quality, jpeg artifacts, noise";
    params.width = 1280;
    params.height = 720;
    params.sample_steps = 20;
    params.cfg_scale = 3.2f;
    params.sample_method = SampleMethod::Euler;
    params.scheduler = Scheduler::Discrete;
    params.seed = 42;  // 固定种子以便复现
    
    auto start = std::chrono::high_resolution_clock::now();
    auto images = adapter.generate(params);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "\n\n3. Generation completed in " << duration << " seconds" << std::endl;
    
    if (!images.empty()) {
        std::cout << "   Generated " << images.size() << " image(s)" << std::endl;
        
        // 保存图像
        std::string output_path = "/tmp/test_output.png";
        std::ofstream file(output_path, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(images[0].data.data()), images[0].data.size());
            file.close();
            std::cout << "   Saved to: " << output_path << std::endl;
            std::cout << "   Size: " << images[0].width << "x" << images[0].height << std::endl;
        }
    } else {
        std::cerr << "   No images generated!" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Test Completed ===" << std::endl;
    return 0;
}
