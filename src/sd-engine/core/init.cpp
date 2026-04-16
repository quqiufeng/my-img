// ============================================================================
// sd-engine/core/init.cpp
// ============================================================================

#include "init.h"

namespace sdengine {

// 外部声明的初始化函数
extern void init_test_nodes();
extern void init_core_nodes();
extern void init_image_io_nodes();
extern void init_image_transform_nodes();
extern void init_image_filter_nodes();
extern void init_image_nodes();
extern void init_loader_nodes();
extern void init_conditioning_nodes();
extern void init_latent_nodes();
extern void init_preprocessor_nodes();
extern void init_face_nodes();

void init_builtin_nodes() {
    init_test_nodes();
    init_core_nodes();
    init_image_io_nodes();
    init_image_transform_nodes();
    init_image_filter_nodes();
    init_image_nodes();
    init_loader_nodes();
    init_conditioning_nodes();
    init_latent_nodes();
    init_preprocessor_nodes();
    init_face_nodes();
}

} // namespace sdengine
