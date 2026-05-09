#pragma once

#include "cli/cli_options.h"
#include "adapters/sdcpp_adapter.h"
#include <string>

namespace myimg {

// 打印帮助信息
void print_usage(const char* argv0);

// 解析命令行参数
bool parse_args(int argc, char** argv, CliOptions& opts);

// 解析采样方法
SampleMethod parse_sampling_method(const std::string& name);

// 解析调度器
Scheduler parse_scheduler(const std::string& name);

} // namespace myimg
