#pragma once

#include "cli/cli_options.h"
#include "cli/workflow_parser.h"
#include "adapters/sdcpp_adapter.h"
#include <string>
#include <vector>

namespace myimg {

// 打印帮助信息
void print_usage(const char* argv0);

// 解析命令行参数
bool parse_args(int argc, char** argv, CliOptions& opts);

// 解析 embedding 语法
std::string parse_embedding_syntax(const std::string& prompt, std::vector<std::string>& referenced_embeddings);

// 扩展输出模板
std::string expand_output_template(const std::string& template_str, const std::string& filename, int index);

// 保存预设
bool save_preset(const CliOptions& opts, const std::string& preset_name);

// 加载预设
bool load_preset(CliOptions& opts, const std::string& preset_path);

// 加载 JSON 配置文件
bool load_config_file(CliOptions& opts);

// 保存 JSON 配置文件
bool save_config_file(const CliOptions& opts, const std::string& path);

// 解析采样方法
SampleMethod parse_sampling_method(const std::string& name);

// 解析调度器
Scheduler parse_scheduler(const std::string& name);

} // namespace myimg
