// ============================================================================
// sd-engine/core/log.cpp
// ============================================================================
// 统一日志系统实现
// ============================================================================

#include "log.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace sdengine {

Logger::~Logger() {
    close_file();
}

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

LogLevel Logger::get_level() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
    return level_;
}

void Logger::set_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
        fclose(file_);
    }
    file_ = fopen(path.c_str(), "a");
}

void Logger::close_file() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
}

const char* Logger::level_to_string(LogLevel level) const {
    switch (level) {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::WARNING:
        return "WARN";
    case LogLevel::ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

void Logger::write(LogLevel level, const char* file, int line, const char* msg) {
    // 提取文件名（不带路径）
    const char* filename = file;
    const char* last_slash = strrchr(file, '/');
    if (last_slash) {
        filename = last_slash + 1;
    } else {
        const char* last_backslash = strrchr(file, '\\');
        if (last_backslash) {
            filename = last_backslash + 1;
        }
    }

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    char buf[256];
    snprintf(buf, sizeof(buf), "[%04d-%02d-%02d %02d:%02d:%02d.%03d] [%5s] [%s:%d] ", tm_now.tm_year + 1900,
             tm_now.tm_mon + 1, tm_now.tm_mday, tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec, (int)ms.count(),
             level_to_string(level), filename, line);

    if (level_ <= level) {
        FILE* out = (level >= LogLevel::ERROR) ? stderr : stdout;
        fprintf(out, "%s%s\n", buf, msg);
        fflush(out);
    }

    if (file_ && level_ <= level) {
        fprintf(file_, "%s%s\n", buf, msg);
        fflush(file_);
    }
}

void Logger::log(LogLevel level, const char* file, int line, const char* fmt, ...) {
    if (level_ > level && !file_) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    int len = vsnprintf(nullptr, 0, fmt, args_copy);
    va_end(args_copy);

    if (len < 0) {
        va_end(args);
        return;
    }

    std::string msg;
    msg.resize(len + 1);
    vsnprintf(msg.data(), msg.size(), fmt, args);
    va_end(args);
    msg.resize(len);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        write(level, file, line, msg.c_str());
    }
}

} // namespace sdengine
