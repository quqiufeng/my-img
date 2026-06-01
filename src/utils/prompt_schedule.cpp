#include "utils/prompt_schedule.h"
#include "utils/log.h"
#include <sstream>
#include <algorithm>

namespace myimg {

bool PromptSchedule::parse(const std::string& schedule_str) {
    entries_.clear();
    
    if (schedule_str.empty()) {
        return true;
    }
    
    // 检查是否是 JSON 格式
    if (schedule_str.front() == '[' || schedule_str.front() == '{') {
        LOG_WARN("PromptSchedule: JSON format not yet supported, use simple format");
        return false;
    }
    
    // 简单格式: "0-10:prompt1|11-20:prompt2|21-:prompt3"
    std::stringstream ss(schedule_str);
    std::string segment;
    
    while (std::getline(ss, segment, '|')) {
        if (segment.empty()) continue;
        
        // 解析 "start-end:prompt" 格式
        size_t colon_pos = segment.find(':');
        if (colon_pos == std::string::npos) {
            LOG_WARN("PromptSchedule: invalid segment format: %s", segment.c_str());
            continue;
        }
        
        std::string range_str = segment.substr(0, colon_pos);
        std::string prompt = segment.substr(colon_pos + 1);
        
        // 解析范围
        size_t dash_pos = range_str.find('-');
        if (dash_pos == std::string::npos) {
            LOG_WARN("PromptSchedule: invalid range format: %s", range_str.c_str());
            continue;
        }
        
        PromptScheduleEntry entry;
        try {
            entry.start_step = std::stoi(range_str.substr(0, dash_pos));
            std::string end_str = range_str.substr(dash_pos + 1);
            if (end_str.empty() || end_str == "") {
                entry.end_step = -1; // 到最后
            } else {
                entry.end_step = std::stoi(end_str);
            }
        } catch (...) {
            LOG_WARN("PromptSchedule: failed to parse range: %s", range_str.c_str());
            continue;
        }
        
        entry.prompt = prompt;
        entries_.push_back(entry);
    }
    
    sort_entries();
    return !entries_.empty();
}

void PromptSchedule::add_entry(const PromptScheduleEntry& entry) {
    entries_.push_back(entry);
    sort_entries();
}

std::optional<PromptScheduleEntry> PromptSchedule::get_entry(int current_step, int total_steps) const {
    for (const auto& entry : entries_) {
        int effective_end = (entry.end_step < 0) ? total_steps : entry.end_step;
        if (current_step >= entry.start_step && current_step <= effective_end) {
            return entry;
        }
    }
    return std::nullopt;
}

void PromptSchedule::sort_entries() {
    std::sort(entries_.begin(), entries_.end(), 
        [](const PromptScheduleEntry& a, const PromptScheduleEntry& b) {
            return a.start_step < b.start_step;
        });
}

} // namespace myimg
