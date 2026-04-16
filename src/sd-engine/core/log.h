// ============================================================================
// sd-engine/core/log.h
// ============================================================================
// 统一日志系统
// ============================================================================

#pragma once

#include <cstdio>
#include <cstdarg>
#include <string>
#include <mutex>

namespace sdengine {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    SILENT = 4
};

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level);
    LogLevel get_level() const;

    void set_file(const std::string& path);
    void close_file();

    void log(LogLevel level, const char* file, int line, const char* fmt, ...);

private:
    Logger() = default;
    ~Logger();

    LogLevel level_ = LogLevel::INFO;
    FILE* file_ = nullptr;
    std::mutex mutex_;

    const char* level_to_string(LogLevel level) const;
    void write(LogLevel level, const char* file, int line, const char* msg);
};

// 宏接口
#define LOG_DEBUG(...) ::sdengine::Logger::instance().log(::sdengine::LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)  ::sdengine::Logger::instance().log(::sdengine::LogLevel::INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  ::sdengine::Logger::instance().log(::sdengine::LogLevel::WARNING,__FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) ::sdengine::Logger::instance().log(::sdengine::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)

} // namespace sdengine
