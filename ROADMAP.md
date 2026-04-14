# sd-engine 开发路线图

本文档记录 sd-engine（C++ 版 ComfyUI）的待完成任务和已知问题，按优先级和难度分层。

---

## 🔴 第一层：基础完善（高优先级，低难度）

### 1.1 修复 Prompt 传递 Bug
- **问题**：`DeepHighResFix` 节点从 `CONDITIONING` 拿不到原始 prompt 字符串
- **解决方案**：
  - 方案 A：修改 `CLIPTextEncode` 同时输出原始 prompt 字符串
  - 方案 B：`DeepHighResFix` 直接接收 prompt 字符串输入
- **状态**：⬜ 待完成

### 1.2 多 LoRA 堆叠支持
- **问题**：`KSampler` 目前只支持单个 LoRA
- **解决方案**：
  - 创建 `LoRAStack` 节点，可以串联多个 LoRA
  - 或 `KSampler` 接收 `lora_stack` 数组
- **状态**：⬜ 待完成

### 1.3 统一错误处理机制
- **问题**：节点失败时只是 `fprintf(stderr)`，没有统一错误码
- **解决方案**：
  - 定义 `sd_error_t` 错误码枚举
  - 节点 `execute()` 返回错误码而非 bool
  - `DAGExecutor` 统一处理并上报错误
- **状态**：⬜ 待完成

---

## 🟠 第二层：内存安全（高优先级，中等难度）

### 2.1 智能指针封装
- **问题**：`sd_latent_t*`、`sd_conditioning_t*`、`sd_image_t*` 裸指针容易泄漏
- **解决方案**：
  ```cpp
  using LatentPtr = std::shared_ptr<sd_latent_t>;
  using ConditioningPtr = std::shared_ptr<sd_conditioning_t>;
  using ImagePtr = std::shared_ptr<sd_image_t>;
  ```
- **状态**：⬜ 待完成

### 2.2 缓存深拷贝
- **问题**：`ExecutionCache` 存的是 `std::any`，指针类型没有深拷贝
- **解决方案**：
  - 为每个类型实现 `clone()` 方法
  - 或使用智能指针的引用计数
- **状态**：⬜ 待完成

### 2.3 资源自动释放
- **问题**：节点执行完后资源没有自动释放
- **解决方案**：
  - 节点析构时自动释放输出资源
  - 或使用 RAII 包装器
- **状态**：⬜ 待完成

---

## 🟡 第三层：功能扩展（中等优先级，中等难度）

### 3.1 ControlNet 支持
- **任务**：
  - [ ] 扩展 C API：`sd_apply_controlnet()`
  - [ ] 实现 `ControlNetLoader` 节点
  - [ ] 实现 `ControlNetApply` 节点
  - [ ] 实现 `CannyEdgePreprocessor` 节点
  - [ ] 实现 `MiDaS-DepthMapPreprocessor` 节点
- **状态**：⬜ 待完成

### 3.2 Upscale 模型支持
- **任务**：
  - [ ] 实现 `UpscaleModelLoader` 节点（ESRGAN）
  - [ ] 实现 `ImageUpscaleWithModel` 节点
- **状态**：⬜ 待完成

### 3.3 Inpaint 支持
- **任务**：
  - [ ] 实现 `INPAINT_LoadInpaintModel` 节点
  - [ ] 实现 `INPAINT_ApplyInpaint` 节点
  - [ ] 支持 mask 输入
- **状态**：⬜ 待完成

---

## 🟢 第四层：工程优化（低优先级，中等难度）

### 4.1 单元测试框架
- **任务**：
  - [ ] 引入 Google Test 或 Catch2
  - [ ] 为每个节点编写单元测试
  - [ ] 自动化 CI/CD（GitHub Actions）
- **状态**：⬜ 待完成

### 4.2 性能优化
- **任务**：
  - [ ] 内存池管理 latent 缓冲区
  - [ ] 并行执行无依赖节点
  - [ ] 性能 profiling 和瓶颈分析
- **状态**：⬜ 待完成

### 4.3 文档完善
- **任务**：
  - [ ] API 文档（Doxygen）
  - [ ] 节点开发教程
  - [ ] 示例工作流库
- **状态**：⬜ 待完成

---

## 🔵 第五层：高级功能（低优先级，高难度）

### 5.1 IPAdapter 支持
- **任务**：
  - [ ] 实现 `IPAdapterLoader` 节点
  - [ ] 实现 `IPAdapterApply` 节点
- **状态**：⬜ 待完成

### 5.2 视频生成支持
- **任务**：
  - [ ] AnimateDiff Loader
  - [ ] 视频采样器
- **状态**：⬜ 待完成

### 5.3 HTTP API 服务
- **任务**：
  - [ ] 实现 `sd-server`
  - [ ] RESTful API 设计
  - [ ] 队列管理和并发控制
- **状态**：⬜ 待完成

---

## 🐛 已知 Bug 列表

| # | Bug 描述 | 严重程度 | 所在文件 | 状态 |
|---|---------|---------|---------|------|
| 1 | `DeepHighResFix` 拿不到 prompt 字符串 | 高 | core_nodes.cpp | ⬜ |
| 2 | 多 LoRA 无法堆叠 | 中 | core_nodes.cpp | ⬜ |
| 3 | 内存泄漏（裸指针传递） | 高 | 多处 | ⬜ |
| 4 | 缓存系统对指针类型不安全 | 中 | cache.cpp | ⬜ |
| 5 | 节点错误信息不统一 | 低 | 多处 | ⬜ |

---

## 📊 当前完成度

### 已实现（✅）
- [x] 核心引擎（Workflow、DAGExecutor、ExecutionCache）
- [x] 17 个基础节点
- [x] 真正的中间 latent/conditioning 传递
- [x] Deep HighRes Fix（节点 + 命令行）
- [x] LoRA 支持（单 LoRA）
- [x] 命令行 ↔ 节点桥接
- [x] 快速模式（txt2img/img2img/process/DeepHires）

### 进行中（🔄）
- [ ] 内存管理优化
- [ ] 多 LoRA 堆叠

### 待开始（⬜）
- [ ] ControlNet
- [ ] Upscale 模型
- [ ] Inpaint
- [ ] 单元测试
- [ ] HTTP API

---

## 🎯 下一冲刺目标

建议按以下顺序推进：

1. **修复 Prompt Bug**（1-2 小时）
2. **实现多 LoRA 堆叠**（2-3 小时）
3. **引入智能指针**（3-4 小时）
4. **添加单元测试框架**（4-6 小时）
5. **实现 ControlNet**（1-2 天）

---

## 🧹 待清理项目

### 清理脚本和代码
- [ ] 删除测试 JSON 文件（test_*.json）
- [ ] 删除旧的 deep_hires_hook.patch（已被 sd-engine-full.patch 替代）
- [ ] 清理 sd-core 目录（逻辑已迁移到 DeepHighResFix 节点）
- [ ] 移除 stable-diffusion.cpp 软链接
- [ ] 添加 .gitignore 规则

---

*最后更新：2025-04-14*
