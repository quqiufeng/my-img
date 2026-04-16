// ============================================================================
// sd-engine/nodes/preprocessor_nodes.cpp
// ============================================================================
// 图像预处理器节点入口（实际实现已拆分到 preprocessor_cpu_nodes.cpp
// 和 preprocessor_onnx_nodes.cpp）
// ============================================================================

namespace sdengine {

extern void init_preprocessor_cpu_nodes();
extern void init_preprocessor_onnx_nodes();

void init_preprocessor_nodes() {
    init_preprocessor_cpu_nodes();
    init_preprocessor_onnx_nodes();
}

} // namespace sdengine
