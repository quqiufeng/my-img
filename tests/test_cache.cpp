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

TEST_CASE("ExecutionCache get non-existent key returns empty", "[cache]") {
    ExecutionCache cache;
    auto retrieved = cache.get("missing", "hash");
    REQUIRE(retrieved.empty());
}

TEST_CASE("ExecutionCache overwrite existing key updates value", "[cache]") {
    ExecutionCache cache;

    NodeOutputs out1, out2;
    out1["v"] = 1;
    out2["v"] = 99;

    cache.put("n", "h", out1);
    REQUIRE(std::any_cast<int>(cache.get("n", "h")["v"]) == 1);

    cache.put("n", "h", out2);
    REQUIRE(cache.size() == 1);
    REQUIRE(std::any_cast<int>(cache.get("n", "h")["v"]) == 99);
}

TEST_CASE("ExecutionCache zero max size rejects new entries", "[cache]") {
    ExecutionCache cache(0);

    NodeOutputs out;
    out["v"] = 1;

    cache.put("n", "h", out);
    REQUIRE(cache.size() == 0);
    REQUIRE(!cache.has("n", "h"));
}

TEST_CASE("ExecutionCache gc evicts oldest entry", "[cache]") {
    ExecutionCache cache(10240);

    NodeOutputs out1, out2;
    out1["v"] = 1;
    out2["v"] = 2;

    cache.put("n1", "h1", out1);
    cache.put("n2", "h2", out2);
    REQUIRE(cache.size() == 2);

    cache.gc();
    REQUIRE(cache.size() == 1);
    REQUIRE(!cache.has("n1", "h1"));
    REQUIRE(cache.has("n2", "h2"));
}

TEST_CASE("ExecutionCache memory estimation is at least 1KB per entry", "[cache]") {
    ExecutionCache cache(1024 * 1024);

    NodeOutputs out;
    out["v"] = 1; // small int

    cache.put("n", "h", out);
    REQUIRE(cache.get_current_size() >= 1024);
}

TEST_CASE("ExecutionCache move put works correctly", "[cache]") {
    ExecutionCache cache;

    NodeOutputs out;
    out["v"] = std::string("moved_value");

    cache.put("n", "h", std::move(out));
    REQUIRE(cache.has("n", "h"));
    REQUIRE(std::any_cast<std::string>(cache.get("n", "h")["v"]) == "moved_value");
}
