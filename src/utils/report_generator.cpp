#include "report_generator.h"
#include "vram_monitor.h"
#include "log.h"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace myimg {

nlohmann::json GenerationReport::to_json() const {
    nlohmann::json j;
    
    // 输入参数
    j["input"]["prompt"] = prompt;
    j["input"]["negative_prompt"] = negative_prompt;
    j["input"]["width"] = width;
    j["input"]["height"] = height;
    j["input"]["steps"] = steps;
    j["input"]["cfg_scale"] = cfg_scale;
    j["input"]["sampling_method"] = sampling_method;
    j["input"]["scheduler"] = scheduler;
    j["input"]["seed"] = seed;
    j["input"]["model"] = model;
    
    // 性能数据
    j["performance"]["total_time_seconds"] = total_time_seconds;
    j["performance"]["model_load_time_seconds"] = model_load_time_seconds;
    j["performance"]["peak_vram_mb"] = peak_vram_mb;
    j["performance"]["avg_step_time"] = avg_step_time;
    
    // 步骤数据
    for (const auto& step : steps_data) {
        nlohmann::json step_json;
        step_json["step"] = step.step;
        step_json["time_seconds"] = step.time_seconds;
        step_json["vram_mb"] = step.vram_mb;
        step_json["timestamp"] = step.timestamp;
        j["performance"]["steps"].push_back(step_json);
    }
    
    // 输出
    j["output"]["path"] = output_path;
    j["output"]["width"] = output_width;
    j["output"]["height"] = output_height;
    
    // 时间戳
    j["timestamps"]["start"] = start_time;
    j["timestamps"]["end"] = end_time;
    
    return j;
}

bool GenerationReport::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to write report to %s", path.c_str());
        return false;
    }
    
    file << to_json().dump(2);
    LOG_INFO("Generation report saved to %s", path.c_str());
    return true;
}

void ReportGenerator::start_generation(const GenerationParams& params) {
    report_ = GenerationReport{};
    start_time_ = std::chrono::high_resolution_clock::now();
    
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    report_.start_time = ss.str();
    
    report_.prompt = params.prompt;
    report_.negative_prompt = params.negative_prompt;
    report_.width = params.width;
    report_.height = params.height;
    report_.steps = params.sample_steps;
    report_.cfg_scale = params.cfg_scale;
    report_.sampling_method = std::to_string(static_cast<int>(params.sample_method));
    report_.scheduler = std::to_string(static_cast<int>(params.scheduler));
    report_.seed = params.seed;
    report_.model = params.diffusion_model_path;
    
    VRAMMonitor::reset_peak();
}

void ReportGenerator::record_step(int step, float time_seconds, float vram_mb) {
    GenerationStep step_data;
    step_data.step = step;
    step_data.time_seconds = time_seconds;
    step_data.vram_mb = vram_mb;
    
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%H:%M:%S");
    step_data.timestamp = ss.str();
    
    report_.steps_data.push_back(step_data);
}

void ReportGenerator::end_generation(const std::string& output_path, int output_width, int output_height) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    report_.total_time_seconds = duration / 1000.0f;
    
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    report_.end_time = ss.str();
    
    report_.output_path = output_path;
    report_.output_width = output_width;
    report_.output_height = output_height;
    report_.peak_vram_mb = VRAMMonitor::get_peak_vram_mb();
    
    if (!report_.steps_data.empty()) {
        float total_step_time = 0.0f;
        for (const auto& step : report_.steps_data) {
            total_step_time += step.time_seconds;
        }
        report_.avg_step_time = total_step_time / report_.steps_data.size();
    }
}

GenerationReport ReportGenerator::get_report() const {
    return report_;
}

} // namespace myimg
