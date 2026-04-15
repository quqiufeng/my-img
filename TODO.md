# my-img 项目待办事项 / 进度追踪

## 进阶优化 (P3)

### 1. latent 插值 GPU 化
- **问题**: `DeepHighResFix` 中的 `upscale_latent_bilinear_node()` 是 CPU 逐像素插值，大图时可能成为瓶颈。
- **方案**: 评估是否可用 ggml 或 CUDA kernel 加速 latent 上采样。
- **状态**: 待处理

### 2. 人脸模块批量并行处理
- **问题**: 人脸检测/修复/换脸对多个人脸时逐个串行处理。
- **方案**: 多个人脸修复时利用线程池并行裁剪-修复-贴回。
- **状态**: 待处理

---

## 已完成 (Recent)

- [x] 恢复 `build/build_sd_cpp.sh` 编译脚本
- [x] 将 `sd-engine-full.patch` 拆分为 4 个逻辑模块化补丁
- [x] 添加 `LineArtLoader` + `LineArtPreprocessor` 节点
- [x] 人脸检测/修复/换脸 ONNX 模块集成 (Phase 1-3)
- [x] **修复 sd_ctx_t 工作流内释放机制** — 引入 `SDContextPtr` + `UnloadModel` 节点
- [x] **修复缓存哈希可靠性** — `ImagePtr`/`LatentPtr` 加入尺寸+像素校验和
- [x] **LineArt 模型实例缓存** — `LineArtLoaderNode` 缓存 ONNX Session
- [x] **提取公共辅助函数 + 修复对象池归还** — `ImageDeleter` 使用 `release_image()`
- [x] 所有测试通过 (89 assertions, 19 test cases)
