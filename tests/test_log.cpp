#include "utils/log.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace myimg;

int main(int argc, char** argv) {
    std::cout << "=== Logger Test Suite ===" << std::endl;

    Logger& logger = Logger::instance();
    logger.enable_colors(false);

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: Level filtering
    std::cout << "\n[Test 1] Level filtering..." << std::endl;
    {
        logger.set_level(LogLevel::Info);
        bool check1 = logger.is_level_enabled(LogLevel::Info);
        bool check2 = !logger.is_level_enabled(LogLevel::Debug);
        bool check3 = !logger.is_level_enabled(LogLevel::Trace);
        bool check4 = logger.is_level_enabled(LogLevel::Error);

        if (check1 && check2 && check3 && check4) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 2: File output
    std::cout << "\n[Test 2] File output..." << std::endl;
    {
        logger.set_level(LogLevel::Debug);
        std::string test_file = "/tmp/test_log_output.txt";
        std::filesystem::remove(test_file);

        logger.set_output_file(test_file);
        LOG_INFO("Test message 123");
        LOG_WARN("Warning message 456");
        logger.set_output_file("");

        bool file_exists = std::filesystem::exists(test_file);
        std::string content;
        if (file_exists) {
            std::ifstream file(test_file);
            content = std::string((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
        }

        bool has_info = content.find("Test message 123") != std::string::npos;
        bool has_warn = content.find("Warning message 456") != std::string::npos;
        bool has_level_info = content.find("[INFO]") != std::string::npos;
        bool has_level_warn = content.find("[WARN]") != std::string::npos;

        if (file_exists && has_info && has_warn && has_level_info && has_level_warn) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (file_exists=" << file_exists << ", has_info=" << has_info
                      << ", has_warn=" << has_warn << ")" << std::endl;
            tests_failed++;
        }

        std::filesystem::remove(test_file);
    }

    // Test 3: Format message structure
    std::cout << "\n[Test 3] Format message structure..." << std::endl;
    {
        logger.set_level(LogLevel::Info);
        std::string test_file = "/tmp/test_log_format.txt";
        std::filesystem::remove(test_file);

        logger.set_output_file(test_file);
        LOG_INFO("Format test");
        logger.set_output_file("");

        std::ifstream file(test_file);
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        bool has_year = content.find("2026-") != std::string::npos || content.find("2025-") != std::string::npos;
        bool has_level = content.find("[INFO]") != std::string::npos;
        bool has_file = content.find("test_log.cpp") != std::string::npos;
        bool has_msg = content.find("Format test") != std::string::npos;

        if (has_year && has_level && has_file && has_msg) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }

        std::filesystem::remove(test_file);
    }

    // Test 4: Level progression
    std::cout << "\n[Test 4] Level progression..." << std::endl;
    {
        bool check1 = static_cast<int>(LogLevel::Trace) < static_cast<int>(LogLevel::Debug);
        bool check2 = static_cast<int>(LogLevel::Debug) < static_cast<int>(LogLevel::Info);
        bool check3 = static_cast<int>(LogLevel::Info) < static_cast<int>(LogLevel::Warn);
        bool check4 = static_cast<int>(LogLevel::Warn) < static_cast<int>(LogLevel::Error);
        bool check5 = static_cast<int>(LogLevel::Error) < static_cast<int>(LogLevel::Fatal);

        if (check1 && check2 && check3 && check4 && check5) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 5: Multiple log calls
    std::cout << "\n[Test 5] Multiple log calls..." << std::endl;
    {
        logger.set_level(LogLevel::Trace);
        std::string test_file = "/tmp/test_log_multi.txt";
        std::filesystem::remove(test_file);
        logger.set_output_file(test_file);

        for (int i = 0; i < 5; ++i) {
            LOG_INFO("Message %d", i);
        }

        logger.set_output_file("");

        std::ifstream file(test_file);
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        bool all_found = true;
        for (int i = 0; i < 5; ++i) {
            std::string expected = "Message " + std::to_string(i);
            if (content.find(expected) == std::string::npos) {
                all_found = false;
                break;
            }
        }

        if (all_found) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }

        std::filesystem::remove(test_file);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
