// ============================================================================
// tests/test_cache.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/cache.h"

using namespace sdengine;

TEST_CASE("ExecutionCache stores and retrieves", "[cache]") {
    ExecutionCache cache(1024 * 1024);

    NodeOutputs outputs;
    outputs["value"] = 42;

    cache.put("node1", "hash1", outputs);
    REQUIRE(cache.has("node1", "hash1"));

    auto retrieved = cache.get("node1", "hash1");
    REQUIRE(retrieved.count("value"));
    REQUIRE(std::any_cast<int>(retrieved["value"]) == 42);
}


TEST_CASE("ExecutionCache LRU eviction", "[cache]") {
    // 限制很小的缓存，只能容纳约 2 个条目
    ExecutionCache cache(3000);

    NodeOutputs out1, out2, out3;
    out1["v"] = 1;
    out2["v"] = 2;
    out3["v"] = 3;

    cache.put("n1", "h1", out1);
    cache.put("n2", "h2", out2);
    // 访问 n1，使其变为最近使用
    (void)cache.get("n1", "h1");

    cache.put("n3", "h3", out3);

    // n2 应该被淘汰（最久未使用）
    REQUIRE(cache.has("n1", "h1"));
    REQUIRE(!cache.has("n2", "h2"));
    REQUIRE(cache.has("n3", "h3"));
}


TEST_CASE("ExecutionCache clear", "[cache]") {
    ExecutionCache cache;
    NodeOutputs out;
    out["v"] = 1;
    cache.put("n", "h", out);
    REQUIRE(cache.size() == 1);

    cache.clear();
    REQUIRE(cache.size() == 0);
    REQUIRE(!cache.has("n", "h"));
}


