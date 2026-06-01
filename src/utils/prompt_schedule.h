#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace myimg {

// Prompt Schedule: 按步数调度提示词
// 类似于 ComfyUI 的 PromptSchedule 节点
struct PromptScheduleEntry {
    int start_step = 0;     // 起始步数（包含）
    int end_step = -1;      // 结束步数（-1 表示到最后）
    std::string prompt;     // 此阶段的提示词
    std::string negative_prompt; // 此阶段的负面提示词（可选）
    float cfg_scale = -1.0f;     // 此阶段的 CFG（-1 = 使用全局）
};

class PromptSchedule {
public:
    // 解析 schedule 字符串
    // 格式: "0-10:prompt1|11-20:prompt2|21-:prompt3"
    // 或 JSON 格式
    bool parse(const std::string& schedule_str);
    
    // 添加一个 schedule 条目
    void add_entry(const PromptScheduleEntry& entry);
    
    // 获取指定步数的提示词
    // 如果当前步数不在任何条目中，返回默认提示词
    std::optional<PromptScheduleEntry> get_entry(int current_step, int total_steps) const;
    
    // 检查是否启用了 schedule
    bool enabled() const { return !entries_.empty(); }
    
    // 清空所有条目
    void clear() { entries_.clear(); }
    
    // 获取所有条目
    const std::vector<PromptScheduleEntry>& entries() const { return entries_; }
    
private:
    std::vector<PromptScheduleEntry> entries_;
    
    // 按 start_step 排序
    void sort_entries();
};

} // namespace myimg
