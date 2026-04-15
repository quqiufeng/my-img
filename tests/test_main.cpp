// ============================================================================
// tests/test_main.cpp
// ============================================================================
// Catch2 测试入口（使用 amalgamated 版本时不需要单独定义 CATCH_CONFIG_MAIN）
// ============================================================================

#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"
#include "core/init.h"

struct GlobalSetup {
    GlobalSetup() {
        sdengine::init_builtin_nodes();
    }
} global_setup;
