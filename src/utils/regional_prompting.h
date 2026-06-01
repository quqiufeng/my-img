#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

namespace myimg {

// Regional Prompting: 分区提示词
// 类似于 ComfyUI 的 ConditioningSetArea / RegionalPrompting
struct RegionalPrompt {
    // 区域定义（相对坐标 0.0-1.0）
    float x = 0.0f;             // 区域左上角 x（相对图像宽度）
    float y = 0.0f;             // 区域左上角 y（相对图像高度）
    float width = 1.0f;         // 区域宽度（相对图像宽度）
    float height = 1.0f;        // 区域高度（相对图像高度）
    
    // 提示词
    std::string prompt;         // 此区域的正向提示词
    std::string negative_prompt; // 此区域的负面提示词（可选）
    
    // 权重
    float weight = 1.0f;        // 区域权重（0.0-1.0）
    float feather = 0.1f;       // 边缘羽化（相对尺寸）
    
    // 便捷构造区域
    static RegionalPrompt full(const std::string& prompt) {
        RegionalPrompt r;
        r.prompt = prompt;
        return r;
    }
    
    static RegionalPrompt top(float ratio, const std::string& prompt) {
        RegionalPrompt r;
        r.y = 0.0f;
        r.height = ratio;
        r.prompt = prompt;
        return r;
    }
    
    static RegionalPrompt bottom(float ratio, const std::string& prompt) {
        RegionalPrompt r;
        r.y = 1.0f - ratio;
        r.height = ratio;
        r.prompt = prompt;
        return r;
    }
    
    static RegionalPrompt left(float ratio, const std::string& prompt) {
        RegionalPrompt r;
        r.x = 0.0f;
        r.width = ratio;
        r.prompt = prompt;
        return r;
    }
    
    static RegionalPrompt right(float ratio, const std::string& prompt) {
        RegionalPrompt r;
        r.x = 1.0f - ratio;
        r.width = ratio;
        r.prompt = prompt;
        return r;
    }
    
    static RegionalPrompt center(float w_ratio, float h_ratio, const std::string& prompt) {
        RegionalPrompt r;
        r.x = (1.0f - w_ratio) / 2.0f;
        r.y = (1.0f - h_ratio) / 2.0f;
        r.width = w_ratio;
        r.height = h_ratio;
        r.prompt = prompt;
        return r;
    }
};

class RegionalPromptManager {
public:
    // 添加区域提示词
    void add_region(const RegionalPrompt& region);
    
    // 解析字符串格式
    // 格式: "top:0.3,prompt1|bottom:0.3,prompt2|center:0.5x0.5,prompt3"
    // 或: "0,0,0.5,0.5,prompt1|0.5,0.5,0.5,0.5,prompt2"
    bool parse(const std::string& regions_str);
    
    // 生成区域 mask（用于 latent 空间）
    // 返回 mask 列表，每个 mask 对应一个区域
    std::vector<torch::Tensor> generate_masks(int latent_width, int latent_height) const;
    
    // 获取所有区域
    const std::vector<RegionalPrompt>& regions() const { return regions_; }
    
    // 是否启用了区域提示词
    bool enabled() const { return !regions_.empty(); }
    
    // 清空
    void clear() { regions_.clear(); }
    
private:
    std::vector<RegionalPrompt> regions_;
};

} // namespace myimg
