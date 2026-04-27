#include "utils/lut_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace myimg {

bool LUT3D::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[LUT] Failed to open file: " << path << std::endl;
        return false;
    }
    
    std::string line;
    bool has_size = false;
    int expected_entries = 0;
    int entries_read = 0;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        std::string keyword;
        ss >> keyword;
        
        if (keyword == "TITLE") {
            // Extract title (quoted string)
            size_t start = line.find('"');
            size_t end = line.rfind('"');
            if (start != std::string::npos && end > start) {
                title = line.substr(start + 1, end - start - 1);
            }
        }
        else if (keyword == "LUT_3D_SIZE") {
            ss >> size;
            if (size <= 0) {
                std::cerr << "[LUT] Invalid LUT size: " << size << std::endl;
                return false;
            }
            expected_entries = size * size * size;
            data.reserve(expected_entries * 3);
            has_size = true;
        }
        else if (keyword == "DOMAIN_MIN" || keyword == "DOMAIN_MAX") {
            // Skip domain specification (assume 0-1)
            continue;
        }
        else {
            // Parse RGB values
            if (!has_size) {
                std::cerr << "[LUT] LUT_3D_SIZE not specified before data" << std::endl;
                return false;
            }
            
            float r, g, b;
            std::stringstream val_ss(line);
            val_ss >> r >> g >> b;
            
            if (val_ss.fail()) {
                std::cerr << "[LUT] Failed to parse RGB values: " << line << std::endl;
                continue;
            }
            
            data.push_back(r);
            data.push_back(g);
            data.push_back(b);
            entries_read++;
        }
    }
    
    if (!has_size) {
        std::cerr << "[LUT] No LUT_3D_SIZE found" << std::endl;
        return false;
    }
    
    if (entries_read != expected_entries) {
        std::cerr << "[LUT] Expected " << expected_entries << " entries, got " << entries_read << std::endl;
        return false;
    }
    
    std::cout << "[LUT] Loaded " << title << " (" << size << "x" << size << "x" << size << ")" << std::endl;
    return true;
}

torch::Tensor LUT3D::apply(const torch::Tensor& image) const {
    if (empty()) return image.clone();
    
    auto img = image.clone();
    auto device = img.device();
    int h = img.size(1);
    int w = img.size(2);
    
    // Create LUT tensor [size, size, size, 3]
    auto lut_tensor = torch::from_blob(
        const_cast<float*>(data.data()),
        {size, size, size, 3},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).to(device).clone();
    
    // Scale to LUT coordinates
    float scale = size - 1.0f;
    auto coords = img.permute({1, 2, 0}) * scale; // [H, W, 3]
    
    // Get integer coordinates
    auto coords0 = coords.floor().clamp(0, size - 2).to(torch::kInt64);
    auto coords1 = (coords0 + 1).clamp(0, size - 1);
    
    // Fractional parts
    auto frac = coords - coords0.to(torch::kFloat32);
    auto fx = frac.index({"...", 0}).unsqueeze(-1);
    auto fy = frac.index({"...", 1}).unsqueeze(-1);
    auto fz = frac.index({"...", 2}).unsqueeze(-1);
    
    // Get 8 corner values
    auto v000 = lut_tensor.index({coords0.index({"...", 0}), coords0.index({"...", 1}), coords0.index({"...", 2}), "..."});
    auto v001 = lut_tensor.index({coords0.index({"...", 0}), coords0.index({"...", 1}), coords1.index({"...", 2}), "..."});
    auto v010 = lut_tensor.index({coords0.index({"...", 0}), coords1.index({"...", 1}), coords0.index({"...", 2}), "..."});
    auto v011 = lut_tensor.index({coords0.index({"...", 0}), coords1.index({"...", 1}), coords1.index({"...", 2}), "..."});
    auto v100 = lut_tensor.index({coords1.index({"...", 0}), coords0.index({"...", 1}), coords0.index({"...", 2}), "..."});
    auto v101 = lut_tensor.index({coords1.index({"...", 0}), coords0.index({"...", 1}), coords1.index({"...", 2}), "..."});
    auto v110 = lut_tensor.index({coords1.index({"...", 0}), coords1.index({"...", 1}), coords0.index({"...", 2}), "..."});
    auto v111 = lut_tensor.index({coords1.index({"...", 0}), coords1.index({"...", 1}), coords1.index({"...", 2}), "..."});
    
    // Trilinear interpolation
    auto c00 = v000 * (1 - fz) + v001 * fz;
    auto c01 = v010 * (1 - fz) + v011 * fz;
    auto c10 = v100 * (1 - fz) + v101 * fz;
    auto c11 = v110 * (1 - fz) + v111 * fz;
    
    auto c0 = c00 * (1 - fy) + c01 * fy;
    auto c1 = c10 * (1 - fy) + c11 * fy;
    
    auto result = c0 * (1 - fx) + c1 * fx;
    
    return result.permute({2, 0, 1}).clamp(0, 1);
}

} // namespace myimg
