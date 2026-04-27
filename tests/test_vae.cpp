#include <backend/model.h>
#include <utils/image_utils.h>
#include <iostream>
#include <cassert>

using namespace myimg;

int main() {
    std::cout << "=== VAE Test ===" << std::endl;
    
    // Test 1: Create dummy image and encode/decode
    std::cout << "\n[Test 1] Encode/Decode roundtrip" << std::endl;
    
    VAEModel vae;
    
    // Create a dummy image (1, 3, 512, 512)
    auto image = torch::rand({1, 3, 512, 512});
    std::cout << "Original image shape: " << image.sizes() << std::endl;
    
    // Encode
    auto latent = vae.encode(image);
    std::cout << "Latent shape: " << latent.sizes() << std::endl;
    
    // Check dimensions: 512/8 = 64
    assert(latent.size(2) == 64 && "Latent height should be 64");
    assert(latent.size(3) == 64 && "Latent width should be 64");
    std::cout << "✓ Latent dimensions correct (64x64)" << std::endl;
    
    // Decode
    auto decoded = vae.decode(latent);
    std::cout << "Decoded shape: " << decoded.sizes() << std::endl;
    
    assert(decoded.size(2) == 512 && "Decoded height should be 512");
    assert(decoded.size(3) == 512 && "Decoded width should be 512");
    std::cout << "✓ Decoded dimensions correct (512x512)" << std::endl;
    
    // Test 2: Different resolutions
    std::cout << "\n[Test 2] Different resolutions" << std::endl;
    
    std::vector<std::pair<int, int>> resolutions = {
        {512, 512},
        {1024, 1024},
        {1280, 720},
        {2560, 1440}
    };
    
    for (const auto& [w, h] : resolutions) {
        auto img = torch::rand({1, 3, h, w});
        auto lat = vae.encode(img);
        auto dec = vae.decode(lat);
        
        std::cout << "  " << w << "x" << h << " -> latent " << lat.size(3) << "x" << lat.size(2) 
                  << " -> decoded " << dec.size(3) << "x" << dec.size(2) << std::endl;
        
        assert(dec.size(3) == w && dec.size(2) == h);
    }
    std::cout << "✓ All resolutions work correctly" << std::endl;
    
    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
