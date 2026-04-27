#include <gguf_loader.h>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace myimg;

int main(int argc, char** argv) {
    std::cout << "=== GGUF Loader Test ===" << std::endl;
    
    // Test 1: Load Z-Image model
    std::string model_path = "/opt/image/model/z_image_turbo-Q5_K_M.gguf";
    
    std::cout << "\n[Test 1] Loading model: " << model_path << std::endl;
    
    if (!std::ifstream(model_path).good()) {
        std::cerr << "Model file not found: " << model_path << std::endl;
        std::cerr << "Skipping test..." << std::endl;
        return 0;
    }
    
    auto tensors = GGUFLoder::load(model_path);
    
    // Assertions
    assert(!tensors.empty() && "Should load at least one tensor");
    std::cout << "✓ Loaded " << tensors.size() << " tensors" << std::endl;
    
    // Check tensor properties
    for (const auto& [name, tensor] : tensors) {
        assert(tensor.dim() > 0 && "Tensor should have dimensions");
        assert(tensor.dtype() == torch::kFloat16 && "Tensor should be FP16");
        assert(tensor.numel() > 0 && "Tensor should have elements");
    }
    std::cout << "✓ All tensors have valid properties" << std::endl;
    
    // Print first few tensor names and shapes
    std::cout << "\nTensor samples:" << std::endl;
    int count = 0;
    for (const auto& [name, tensor] : tensors) {
        std::cout << "  " << name << ": ";
        for (int i = 0; i < tensor.dim(); i++) {
            std::cout << tensor.size(i) << " ";
        }
        std::cout << "(dtype=" << tensor.dtype() << ")" << std::endl;
        
        if (++count >= 5) {
            std::cout << "  ... (" << tensors.size() - 5 << " more)" << std::endl;
            break;
        }
    }
    
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
