#include "utils/regional_prompting.h"
#include "utils/log.h"
#include <sstream>

namespace myimg {

void RegionalPromptManager::add_region(const RegionalPrompt& region) {
    regions_.push_back(region);
}

bool RegionalPromptManager::parse(const std::string& regions_str) {
    regions_.clear();
    
    if (regions_str.empty()) {
        return true;
    }
    
    std::stringstream ss(regions_str);
    std::string segment;
    
    while (std::getline(ss, segment, '|')) {
        if (segment.empty()) continue;
        
        RegionalPrompt region;
        
        // 检查是否是预设格式
        if (segment.find("top:") == 0) {
            size_t comma = segment.find(',');
            if (comma == std::string::npos) continue;
            float ratio = std::stof(segment.substr(4, comma - 4));
            region = RegionalPrompt::top(ratio, segment.substr(comma + 1));
        } else if (segment.find("bottom:") == 0) {
            size_t comma = segment.find(',');
            if (comma == std::string::npos) continue;
            float ratio = std::stof(segment.substr(7, comma - 7));
            region = RegionalPrompt::bottom(ratio, segment.substr(comma + 1));
        } else if (segment.find("left:") == 0) {
            size_t comma = segment.find(',');
            if (comma == std::string::npos) continue;
            float ratio = std::stof(segment.substr(5, comma - 5));
            region = RegionalPrompt::left(ratio, segment.substr(comma + 1));
        } else if (segment.find("right:") == 0) {
            size_t comma = segment.find(',');
            if (comma == std::string::npos) continue;
            float ratio = std::stof(segment.substr(6, comma - 6));
            region = RegionalPrompt::right(ratio, segment.substr(comma + 1));
        } else if (segment.find("center:") == 0) {
            size_t comma1 = segment.find(',');
            if (comma1 == std::string::npos) continue;
            std::string dims = segment.substr(7, comma1 - 7);
            size_t x_pos = dims.find('x');
            if (x_pos == std::string::npos) continue;
            float w = std::stof(dims.substr(0, x_pos));
            float h = std::stof(dims.substr(x_pos + 1));
            region = RegionalPrompt::center(w, h, segment.substr(comma1 + 1));
        } else {
            // 解析 "x,y,w,h,prompt" 格式
            std::stringstream seg_ss(segment);
            std::string part;
            std::vector<std::string> parts;
            
            while (std::getline(seg_ss, part, ',')) {
                parts.push_back(part);
            }
            
            if (parts.size() < 5) {
                LOG_WARN("RegionalPrompt: invalid format: %s", segment.c_str());
                continue;
            }
            
            try {
                region.x = std::stof(parts[0]);
                region.y = std::stof(parts[1]);
                region.width = std::stof(parts[2]);
                region.height = std::stof(parts[3]);
                region.prompt = parts[4];
            } catch (...) {
                LOG_WARN("RegionalPrompt: failed to parse: %s", segment.c_str());
                continue;
            }
        }
        
        regions_.push_back(region);
    }
    
    return !regions_.empty();
}

std::vector<torch::Tensor> RegionalPromptManager::generate_masks(int latent_width, int latent_height) const {
    std::vector<torch::Tensor> masks;
    
    for (const auto& region : regions_) {
        // 计算像素坐标
        int x = static_cast<int>(region.x * latent_width);
        int y = static_cast<int>(region.y * latent_height);
        int w = static_cast<int>(region.width * latent_width);
        int h = static_cast<int>(region.height * latent_height);
        
        // 确保在范围内
        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, latent_width - x);
        h = std::min(h, latent_height - y);
        
        // 创建 mask
        auto mask = torch::zeros({1, latent_height, latent_width}, torch::kFloat32);
        
        if (w > 0 && h > 0) {
            mask.narrow(1, y, h).narrow(2, x, w).fill_(region.weight);
            
            // 羽化边缘
            if (region.feather > 0.0f) {
                int feather_px_x = static_cast<int>(region.feather * latent_width);
                int feather_px_y = static_cast<int>(region.feather * latent_height);
                
                if (feather_px_x > 0 && feather_px_y > 0) {
                    // 水平羽化
                    if (w > 2 * feather_px_x) {
                        auto left = torch::linspace(0.0f, 1.0f, feather_px_x);
                        auto right = torch::linspace(1.0f, 0.0f, feather_px_x);
                        mask.narrow(1, y, h).narrow(2, x, feather_px_x).mul_(left.view({1, 1, -1}));
                        mask.narrow(1, y, h).narrow(2, x + w - feather_px_x, feather_px_x).mul_(right.view({1, 1, -1}));
                    }
                    
                    // 垂直羽化
                    if (h > 2 * feather_px_y) {
                        auto top = torch::linspace(0.0f, 1.0f, feather_px_y);
                        auto bottom = torch::linspace(1.0f, 0.0f, feather_px_y);
                        mask.narrow(1, y, feather_px_y).narrow(2, x, w).mul_(top.view({-1, 1, 1}));
                        mask.narrow(1, y + h - feather_px_y, feather_px_y).narrow(2, x, w).mul_(bottom.view({-1, 1, 1}));
                    }
                }
            }
        }
        
        masks.push_back(mask);
    }
    
    return masks;
}

} // namespace myimg
