// ============================================================================
// sd-engine/core/init.cpp
// ============================================================================

#include "init.h"

namespace sdengine {

// 外部声明的初始化函数
extern void init_test_nodes();
extern void init_core_nodes();

void init_builtin_nodes() {
    init_test_nodes();
    init_core_nodes();
}

} // namespace sdengine
