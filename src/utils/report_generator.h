#pragma once

#include "adapters/sdcpp_adapter.h"
#include <nlohmann/json.hpp>
#include <string>
#include <chrono>
#include <vector>

namespace myimg {

// 生成步骤记录
struct GenerationStep {
    int step;
    float time_seconds;
    float vram_mb;
    std::string timestamp;
};

// 生成报告
struct GenerationReport {
    // 输入参数
    std::string prompt;
    std::string negative_prompt;
    int width;
    int height;
    int steps;
    float cfg_scale;
    std::string sampling_method;
    std::string scheduler;
    int64_t seed;
    std::string model;
    
    // 性能数据
    float total_time_seconds;
    float model_load_time_seconds;
    std::vector<GenerationStep> steps_data;
    float peak_vram_mb;
    float avg_step_time;
    
    // 输出
    std::string output_path;
    int output_width;
    int output_height;
    
    // 时间戳
    std::string start_time;
    std::string end_time;
    
    // 转换为 JSON
    nlohmann::json to_json() const;
    
    // 保存到文件
    bool save_to_file(const std::string& path) const;
};

// 报告生成器
class ReportGenerator {
public:
    void start_generation(const GenerationParams& params);
    void record_step(int step, float time_seconds, float vram_mb);
    void end_generation(const std::string& output_path, int output_width, int output_height);
    
    GenerationReport get_report() const;
    
private:
    GenerationReport report_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point model_load_start_;
};

} // namespace myimg
