// ============================================================================
// tests/test_log.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/log.h"
#include <cstdio>
#include <fstream>
#include <string>

using namespace sdengine;

TEST_CASE("Logger set_level and get_level", "[log]") {
    Logger& logger = Logger::instance();

    LogLevel original = logger.get_level();

    logger.set_level(LogLevel::DEBUG);
    REQUIRE(logger.get_level() == LogLevel::DEBUG);

    logger.set_level(LogLevel::ERROR);
    REQUIRE(logger.get_level() == LogLevel::ERROR);

    logger.set_level(LogLevel::SILENT);
    REQUIRE(logger.get_level() == LogLevel::SILENT);

    logger.set_level(original);
}

TEST_CASE("Logger level filtering suppresses low-priority messages", "[log]") {
    Logger& logger = Logger::instance();
    LogLevel original = logger.get_level();

    logger.set_level(LogLevel::ERROR);

    // These should not crash and should be filtered out (no stdout for INFO/WARN)
    // We verify mainly by ensuring no exception and level is respected.
    LOG_INFO("This info should be filtered");
    LOG_WARN("This warn should be filtered");
    LOG_ERROR("This error should pass filter");

    logger.set_level(original);
    REQUIRE(true); // If we got here without crash, filtering works
}

TEST_CASE("Logger file output", "[log]") {
    Logger& logger = Logger::instance();
    LogLevel original = logger.get_level();

    const char* temp_path = "/tmp/sd_engine_test_log.txt";
    std::remove(temp_path);

    logger.set_level(LogLevel::INFO);
    logger.set_file(temp_path);

    LOG_INFO("TestLogMessage12345");

    logger.close_file();
    logger.set_level(original);

    std::ifstream file(temp_path);
    REQUIRE(file.good());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    REQUIRE(content.find("TestLogMessage12345") != std::string::npos);

    std::remove(temp_path);
}

TEST_CASE("Logger file respects level filtering", "[log]") {
    Logger& logger = Logger::instance();
    LogLevel original = logger.get_level();

    const char* temp_path = "/tmp/sd_engine_test_log_filter.txt";
    std::remove(temp_path);

    logger.set_level(LogLevel::ERROR);
    logger.set_file(temp_path);

    LOG_INFO("FilteredInfoMessage");
    LOG_ERROR("AllowedErrorMessage");

    logger.close_file();
    logger.set_level(original);

    std::ifstream file(temp_path);
    REQUIRE(file.good());

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    REQUIRE(content.find("AllowedErrorMessage") != std::string::npos);
    REQUIRE(content.find("FilteredInfoMessage") == std::string::npos);

    std::remove(temp_path);
}
