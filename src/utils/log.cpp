#include "log.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace myimg {

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

Logger::Logger() = default;

Logger::~Logger() {
    if (file_.is_open()) {
        file_.close();
    }
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

void Logger::set_output_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.close();
    }
    file_.open(path, std::ios::app);
}

void Logger::enable_colors(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    colors_enabled_ = enabled;
}

bool Logger::is_level_enabled(LogLevel level) const {
    return static_cast<int>(level) >= static_cast<int>(level_);
}

const char* Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Fatal: return "FATAL";
        default:              return "UNKNOWN";
    }
}

const char* Logger::level_to_color(LogLevel level) const {
    if (!colors_enabled_) return "";
    switch (level) {
        case LogLevel::Trace: return "\033[0;37m";  // White
        case LogLevel::Debug: return "\033[0;36m";  // Cyan
        case LogLevel::Info:  return "\033[0;32m";  // Green
        case LogLevel::Warn:  return "\033[1;33m";  // Yellow
        case LogLevel::Error: return "\033[0;31m";  // Red
        case LogLevel::Fatal: return "\033[1;31m";  // Bold Red
        default:              return "";
    }
}

std::string Logger::format_message(LogLevel level, const char* file, int line, const char* func, const char* msg) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setw(3) << std::setfill('0') << ms.count();
    ss << " [" << level_to_string(level) << "]";
    ss << " [" << file << ":" << line << " " << func << "()] ";
    ss << msg;
    return ss.str();
}

void Logger::log(LogLevel level, const char* file, int line, const char* func, const char* fmt, ...) {
    if (!is_level_enabled(level)) {
        return;
    }

    // 格式化消息
    va_list args;
    va_start(args, fmt);
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    auto message = format_message(level, file, line, func, buffer);

    std::lock_guard<std::mutex> lock(mutex_);

    // 输出到控制台（带颜色）
    std::cout << level_to_color(level) << message << "\033[0m" << std::endl;

    // 输出到文件（不带颜色）
    if (file_.is_open()) {
        file_ << message << std::endl;
    }

    // Fatal 级别自动终止
    if (level == LogLevel::Fatal) {
        abort();
    }
}

} // namespace myimg
