# my-img 代码优化进度追踪

> 创建时间: 2026-04-16  
> 目标: 拆分 `core_nodes.cpp` (3865行)，提取公共辅助函数，优化缓存和Executor性能

---

## 总体目标

1. **模块化 `core_nodes.cpp`**：将46个节点按功能拆分为6个独立文件
2. **提取公共代码**：创建 `nodes/node_utils.h` 和 `node_utils.cpp`
3. **消除重复代码**：ONNX占位符节点使用宏统一生成
4. **性能优化**：Cache大小估算更准确，Executor锁粒度更细

---

## 进度看板

| # | 任务 | 状态 | 说明 |
|---|------|------|------|
| 1 | 创建 `nodes/node_utils.h` + `.cpp` | 待开始 | 提取 CLIPWrapper/LoRAInfo/IPAdapterInfo/ControlNetApplyInfo/RemBGModel/DeepHiresNodeState 和公共函数 |
| 2 | 拆分 `loader_nodes.cpp` | 待开始 | CheckpointLoaderSimple, LoRALoader, LoRAStack, ControlNetLoader, UpscaleModelLoader, IPAdapterLoader, RemBGModelLoader, LineArtLoader, 人脸相关Loader, UnloadModel |
| 3 | 拆分 `conditioning_nodes.cpp` | 待开始 | CLIPTextEncode, CLIPSetLastLayer, CLIPVisionEncode, ConditioningCombine/Concat/Average, ControlNetApply, IPAdapterApply |
| 4 | 拆分 `latent_nodes.cpp` | 待开始 | EmptyLatentImage, VAEEncode, VAEDecode, KSampler, KSamplerAdvanced, DeepHighResFix |
| 5 | 拆分 `image_nodes.cpp` | 待开始 | LoadImage, SaveImage, ImageScale, ImageCrop, ImageBlend, ImageCompositeMasked, ImageInvert, ImageColorAdjust, ImageBlur, ImageGrayscale, ImageThreshold, PreviewImage, LoadImageMask |
| 6 | 拆分 `preprocessor_nodes.cpp` | 待开始 | CannyEdgePreprocessor, LineArtLoader, LineArtPreprocessor |
| 7 | 拆分 `face_nodes.cpp` | 待开始 | FaceDetectModelLoader, FaceDetect, FaceRestoreModelLoader, FaceRestoreWithModel, FaceSwapModelLoader, FaceSwap |
| 8 | 更新 `CMakeLists.txt` | 待开始 | 将新源文件加入 sd-engine 静态库 |
| 9 | 编译验证 | 待开始 | 确保所有目标编译通过 |
| 10 | 优化 `Cache::estimate_size` | 待开始 | LatentPtr 使用实际尺寸计算 |
| 11 | 优化 `Executor` 锁粒度 | 待开始 | 同层节点 prepare_inputs 前置化 |

---

## 变更日志

### 2026-04-16
- **初始化**: 创建本进度文档，开始执行优化计划。
- **提取公共代码**: 创建 `nodes/node_utils.h` + `node_utils.cpp`，集中存放 CLIPWrapper/LoRAInfo/IPAdapterInfo/ControlNetApplyInfo/RemBGModel/DeepHiresNodeState 和公共辅助函数（`convert_rgba_to_rgb`, `create_image_ptr`, `extract_sd_ctx`, `run_sampler_common`, `upscale_latent_bilinear_node`, `deep_hires_node_latent_hook`, `deep_hires_node_guidance_hook`）。
- **拆分 core_nodes.cpp**: 将 3865 行的庞然大物拆分为 6 个模块化文件：
  - `loader_nodes.cpp` (622 行) — CheckpointLoaderSimple, LoRALoader, LoRAStack, ControlNetLoader, UpscaleModelLoader, IPAdapterLoader, RemBGModelLoader, LineArtLoader, 人脸相关 Loader, UnloadModel
  - `conditioning_nodes.cpp` (245 行) — CLIPTextEncode, CLIPSetLastLayer, CLIPVisionEncode, ConditioningCombine/Concat/Average, ControlNetApply, IPAdapterApply
  - `latent_nodes.cpp` (272 行) — EmptyLatentImage, VAEEncode, VAEDecode, KSampler, KSamplerAdvanced, DeepHighResFix
  - `image_nodes.cpp` (694 行) — LoadImage, SaveImage, ImageScale, ImageCrop, ImageBlend, ImageCompositeMasked, ImageInvert, ImageColorAdjust, ImageBlur, ImageGrayscale, ImageThreshold, PreviewImage, LoadImageMask, ImageUpscaleWithModel
  - `preprocessor_nodes.cpp` (198 行) — CannyEdgePreprocessor, ImageRemoveBackground, LineArtLoader, LineArtPreprocessor
  - `face_nodes.cpp` (552 行) — FaceDetectModelLoader, FaceDetect, FaceRestoreModelLoader, FaceRestoreWithModel, FaceSwapModelLoader, FaceSwap
- **消除 ONNX 占位符重复**: 使用 `DEFINE_ONNX_PLACEHOLDER_NODE` 宏统一生成 `!HAS_ONNXRUNTIME` 时的占位节点，减少数百行重复代码。
- **修复链接问题**: 为每个拆分后的 .cpp 文件添加 `init_*_nodes()` 空函数，并在 `core_nodes.cpp` 的 `init_core_nodes()` 中统一调用，确保 `REGISTER_NODE` 全局对象不会被链接器丢弃。
- **更新 CMakeLists.txt**: 将新源文件加入 `sd-engine` 静态库编译列表。
- **编译验证**: 所有目标（sd-workflow, sd-hires, sd-img2img, sd-upscale, sd-engine-tests）编译通过。
- **测试验证**: 全部 19 个 test cases / 89 assertions 通过。
- **Cache 优化**: `ExecutionCache::estimate_size()` 中 `LatentPtr` 不再使用硬编码 1MB，而是通过 `sd_latent_get_shape()` 计算实际尺寸（`w * h * c * sizeof(float)`）。
- **Executor 优化**: 多线程模式下，同层节点的 `prepare_inputs` 从 `std::async` 内部移到外层统一执行，显著减少 `computed_mutex` 锁竞争。

