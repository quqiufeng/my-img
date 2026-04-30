#pragma once

#include <cstdarg>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace myimg {

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal
};

class Logger {
public:
    static Logger& instance();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void set_level(LogLevel level);
    void set_output_file(const std::string& path);
    void enable_colors(bool enabled);

    void log(LogLevel level, const char* file, int line, const char* func, const char* fmt, ...);

    bool is_level_enabled(LogLevel level) const;

private:
    Logger();
    ~Logger();

    LogLevel level_ = LogLevel::Info;
    bool colors_enabled_ = true;
    std::mutex mutex_;
    std::ofstream file_;

    const char* level_to_string(LogLevel level) const;
    const char* level_to_color(LogLevel level) const;
    std::string format_message(LogLevel level, const char* file, int line, const char* func, const char* msg);
};

} // namespace myimg

// 日志宏
#define LOG_TRACE(fmt, ...) myimg::Logger::instance().log(myimg::LogLevel::Trace, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) myimg::Logger::instance().log(myimg::LogLevel::Debug, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  myimg::Logger::instance().log(myimg::LogLevel::Info,  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  myimg::Logger::instance().log(myimg::LogLevel::Warn,  __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) myimg::Logger::instance().log(myimg::LogLevel::Error, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) myimg::Logger::instance().log(myimg::LogLevel::Fatal, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
