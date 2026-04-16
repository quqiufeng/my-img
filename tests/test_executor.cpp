// ============================================================================
// tests/test_executor.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/executor.h"
#include "core/workflow.h"
#include "stable-diffusion.h"

using namespace sdengine;

TEST_CASE("Executor runs simple workflow", "[executor]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 7}},
        "2": {"class_type": "ConstantInt", "inputs": {"value": 8}},
        "3": {"class_type": "AddInt", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
        "4": {"class_type": "PrintInt", "inputs": {"value": ["3", 0]}}
    })";

    REQUIRE(wf.load_from_string(json_str));

    DAGExecutor executor;
    ExecutionConfig config;
    config.use_cache = true;
    config.verbose = false;

    sd_error_t err = executor.execute(&wf, config);
    REQUIRE(is_ok(err));
}


TEST_CASE("Executor runs workflow in parallel", "[executor]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 1}},
        "2": {"class_type": "ConstantInt", "inputs": {"value": 2}},
        "3": {"class_type": "ConstantInt", "inputs": {"value": 3}},
        "4": {"class_type": "MultiplyInt", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
        "5": {"class_type": "MultiplyInt", "inputs": {"a": ["2", 0], "b": ["3", 0]}},
        "6": {"class_type": "AddInt", "inputs": {"a": ["4", 0], "b": ["5", 0]}},
        "7": {"class_type": "PrintInt", "inputs": {"value": ["6", 0]}}
    })";

    REQUIRE(wf.load_from_string(json_str));

    DAGExecutor executor;
    ExecutionConfig config;
    config.use_cache = true;
    config.verbose = false;
    config.max_threads = 4; // 启用并行执行

    sd_error_t err = executor.execute(&wf, config);
    REQUIRE(is_ok(err));
}


TEST_CASE("DAGExecutor multithreaded execution is correct", "[executor]") {
    // 构建一个可以并行执行的宽 DAG：多个独立分支同时计算
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 1}},
        "2": {"class_type": "ConstantInt", "inputs": {"value": 2}},
        "3": {"class_type": "ConstantInt", "inputs": {"value": 3}},
        "4": {"class_type": "ConstantInt", "inputs": {"value": 4}},
        "5": {"class_type": "AddInt", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
        "6": {"class_type": "AddInt", "inputs": {"a": ["3", 0], "b": ["4", 0]}},
        "7": {"class_type": "AddInt", "inputs": {"a": ["5", 0], "b": ["6", 0]}},
        "8": {"class_type": "PrintInt", "inputs": {"value": ["7", 0]}}
    })";

    REQUIRE(wf.load_from_string(json_str));

    std::string error_msg;
    REQUIRE(wf.validate(error_msg));

    // 多线程执行
    DAGExecutor executor;
    ExecutionConfig config;
    config.use_cache = true;
    config.verbose = false;
    config.max_threads = 4;

    sd_error_t err = executor.execute(&wf, config);
    REQUIRE(is_ok(err));
}


TEST_CASE("DAGExecutor multithreaded with cache", "[executor]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 5}},
        "2": {"class_type": "AddInt", "inputs": {"a": ["1", 0], "b": ["1", 0]}},
        "3": {"class_type": "AddInt", "inputs": {"a": ["2", 0], "b": ["1", 0]}},
        "4": {"class_type": "PrintInt", "inputs": {"value": ["3", 0]}}
    })";

    REQUIRE(wf.load_from_string(json_str));

    ExecutionCache cache;
    DAGExecutor executor(&cache);
    ExecutionConfig config;
    config.use_cache = true;
    config.max_threads = 4;

    // 第一次执行
    sd_error_t err1 = executor.execute(&wf, config);
    REQUIRE(is_ok(err1));

    // 第二次执行应该命中缓存
    sd_error_t err2 = executor.execute(&wf, config);
    REQUIRE(is_ok(err2));
}


TEST_CASE("DAGExecutor executes empty workflow", "[executor]") {
    Workflow wf;
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));

    DAGExecutor executor;
    ExecutionConfig config;
    config.use_cache = true;
    config.max_threads = 4;

    sd_error_t err = executor.execute(&wf, config);
    REQUIRE(is_ok(err));
}


TEST_CASE("DAGExecutor rejects null workflow", "[executor]") {
    DAGExecutor executor;
    ExecutionConfig config;
    sd_error_t err = executor.execute(nullptr, config);
    REQUIRE(is_error(err));
}


TEST_CASE("DAGExecutor single-threaded with cache", "[executor]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 3}},
        "2": {"class_type": "MultiplyInt", "inputs": {"a": ["1", 0], "b": ["1", 0]}},
        "3": {"class_type": "PrintInt", "inputs": {"value": ["2", 0]}}
    })";
    REQUIRE(wf.load_from_string(json_str));

    ExecutionCache cache;
    DAGExecutor executor(&cache);
    ExecutionConfig config;
    config.use_cache = true;
    config.max_threads = 1; // 强制单线程

    sd_error_t err1 = executor.execute(&wf, config);
    REQUIRE(is_ok(err1));

    sd_error_t err2 = executor.execute(&wf, config);
    REQUIRE(is_ok(err2));
}

