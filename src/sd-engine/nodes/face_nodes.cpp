// ============================================================================
// sd-engine/nodes/face_nodes.cpp
// ============================================================================
// 人脸相关节点入口（薄封装，实际实现分散在子模块中）
// ============================================================================

namespace sdengine {

extern void init_face_detect_nodes();
extern void init_face_restore_nodes();
extern void init_face_swap_nodes();

void init_face_nodes() {
    init_face_detect_nodes();
    init_face_restore_nodes();
    init_face_swap_nodes();
}

} // namespace sdengine
