// ============================================================================
// tests/test_object_pool.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/object_pool.h"

using namespace sdengine;

TEST_CASE("ObjectPool basic acquire and release", "[object_pool]") {
    int created = 0;
    int resetted = 0;
    ObjectPool<int> pool(
        [&created]() {
            ++created;
            return new int(0);
        },
        [&resetted](int* p) {
            ++resetted;
            *p = 0;
        },
        4);

    int* a = pool.acquire();
    REQUIRE(a != nullptr);
    *a = 42;

    int* b = pool.acquire();
    REQUIRE(b != nullptr);
    REQUIRE(b != a);

    pool.release(a);
    REQUIRE(pool.size() == 1);

    int* c = pool.acquire();
    REQUIRE(c == a); // 优先复用池中对象
    REQUIRE(pool.size() == 0);
    REQUIRE(created == 2);
    REQUIRE(resetted == 1);
}


TEST_CASE("ObjectPool respects max size", "[object_pool]") {
    int resetted = 0;
    ObjectPool<int> pool([]() { return new int(0); },
                         [&resetted](int* p) {
                             *p = -1;
                             ++resetted;
                         },
                         2);

    int* a = pool.acquire();
    int* b = pool.acquire();
    int* c = pool.acquire();

    pool.release(a);
    pool.release(b);
    pool.release(c); // 超出 max_size，应直接删除

    REQUIRE(pool.size() == 2);
    REQUIRE(resetted == 3);
}


TEST_CASE("ObjectPool reserve and clear", "[object_pool]") {
    ObjectPool<int> pool([]() { return new int(0); }, nullptr, 8);
    pool.reserve(4);
    REQUIRE(pool.size() == 4);

    pool.clear();
    REQUIRE(pool.size() == 0);
}


TEST_CASE("ObjectPool set_max_size shrinks pool", "[object_pool]") {
    ObjectPool<int> pool([]() { return new int(0); }, nullptr, 8);

    pool.reserve(6);
    REQUIRE(pool.size() == 6);

    pool.set_max_size(3);
    REQUIRE(pool.size() == 3);
}


