# sd-engine 开发路线图

本文档记录 sd-engine（C++ 版 ComfyUI）的待完成任务和已知问题，按优先级和难度分层。

---

## 更新说明

**2025-04-15（第二次更新）**：完成第三层所有功能扩展任务，包括 Upscale 节点化、ControlNet 支持、Inpaint 支持、单元测试框架、KSamplerAdvanced。当前节点数已达 25 个。

**2025-04-15（第三次更新）**：完成 IPAdapter 支持（SD1.5 + SDXL），新增 `IPAdapterLoader`、`IPAdapterApply` 节点，以及 `IPAdapterTxt2ImgBuilder` 快速工作流。上游 `stable-diffusion.cpp` 增加 IPAdapter 权重缓存避免重复加载。当前节点数 27 个。

**2025-04-15（第四次更新）**：新增 `ImageBlend`、`ImageCompositeMasked`、`ConditioningCombine`、`ConditioningConcat`、`ConditioningAverage` 节点。引入 ONNX Runtime 可选依赖，新增 `RemBGModelLoader`、`ImageRemoveBackground` 背景抠图节点。当前节点数 34 个。

---

## ✅ 已完成

### 第一层：基础完善 ✅
| 任务 | 说明 | 状态 |
|------|------|------|
| 修复 Prompt 传递 Bug | `CLIPTextEncode` 增加 `text` 输出 | ✅ |
| 多 LoRA 堆叠支持 | 新增 `LoRAStack` 节点 | ✅ |
| 统一错误处理机制 | `sd_error_t` + `execute()` 返回错误码 | ✅ |

### 第二层：内存安全 ✅
| 任务 | 说明 | 状态 |
|------|------|------|
| 智能指针封装 | `LatentPtr` / `ConditioningPtr` / `ImagePtr` / `UpscalerPtr` | ✅ |
| 缓存安全 | `ExecutionCache` 通过 `shared_ptr` 引用计数自动安全 | ✅ |
| 资源自动释放 | 智能指针 deleter 自动释放 | ✅ |

### 第三层：功能扩展 ✅
| 任务 | 说明 | 状态 |
|------|------|------|
| ControlNet 支持 | `ControlNetLoader` + `ControlNetApply` + `CannyEdgePreprocessor` | ✅ |
| Upscale 模型节点化 | `UpscaleModelLoader` + `ImageUpscaleWithModel` | ✅ |
| Inpaint 支持 | `LoadImageMask` + `KSampler` mask 输入 | ✅ |
| 采样器扩展 | `KSamplerAdvanced`（start/end step / add_noise） | ✅ |

### 第四层：工程优化 ✅
| 任务 | 说明 | 状态 |
|------|------|------|
| 单元测试框架 | Catch2 v3 + 8 个测试用例（全部通过） | ✅ |
| 并行执行无依赖节点 | DAGExecutor 按深度分层并行执行 | ✅ |
| 对象池（image） | `sd_image_t` 全局对象池 + `acquire_image()` | ✅ |

---

## 🟢 第四层：工程优化（剩余）

### 4.2 性能优化
- **任务**：
  - [x] 并行执行无依赖节点
  - [x] `sd_image_t` 对象池
  - [ ] 内存池管理 latent / conditioning 缓冲区（不完整类型，需上游支持）
  - [ ] 性能 profiling 和瓶颈分析
- **状态**：🔄 进行中

### 4.3 文档完善
- **任务**：
  - [x] API 文档（Doxygen）
  - [x] 节点开发教程
  - [x] 示例工作流库（txt2img / img2img）
- **状态**：✅ 已完成

---

## 🔵 第五层：高级功能（低优先级，高难度）

### 5.1 IPAdapter 支持
- **任务**：
  - [x] 上游 `stable-diffusion.cpp` 补丁：ImageProjModel + CrossAttention K/V 注入
  - [x] 实现 `IPAdapterLoader` 节点
  - [x] 实现 `IPAdapterApply` 节点
  - [x] `WorkflowBuilder` 增加 IPAdapter 工作流构建方法
  - [x] 上游 `load_ipadapter` 增加路径/参数缓存，避免重复加载
- **状态**：✅ 已完成

### 5.2 视频生成支持
- **任务**：
  - [ ] AnimateDiff Loader
  - - [ ] 视频采样器
- **状态**：⬜ 待完成

### 5.3 HTTP API 服务
- **任务**：
  - [ ] 实现 `sd-server`
  - [ ] RESTful API 设计
  - [ ] 队列管理和并发控制
- **状态**：⬜ 待完成

---

## 🎯 主要对齐节点清单

> 策略：不对齐全部数千个节点，只覆盖**真正高频好用**的核心节点。按优先级分批实现。

### 第一批：基础工作流（已对齐 ✅）
| 节点 | 状态 | 说明 |
|------|------|------|
| `CheckpointLoaderSimple` | ✅ | 模型加载，支持 ControlNet |
| `CLIPTextEncode` | ✅ | 文本编码，输出 CONDITIONING + text |
| `EmptyLatentImage` | ✅ | 创建空 latent |
| `KSampler` | ✅ | 核心采样器，支持 LoRA Stack / ControlNet / Mask |
| `KSamplerAdvanced` | ✅ | 高级采样器，支持 start/end step / add_noise |
| `VAEDecode` | ✅ | latent 解码为图像 |
| `VAEEncode` | ✅ | 图像编码为 latent |
| `LoadImage` | ✅ | 加载图像 |
| `SaveImage` | ✅ | 保存图像 |
| `ImageScale` | ✅ | 图像缩放 |
| `ImageCrop` | ✅ | 图像裁剪 |
| `PreviewImage` | ✅ | 预览信息 |
| `LoRALoader` | ✅ | 单 LoRA 加载 |
| `LoRAStack` | ✅ | 多 LoRA 堆叠 |
| `DeepHighResFix` | ✅ | 原生 Deep HighRes Fix |

### 第二批：图像质量控制（已对齐 ✅）
| 节点 | 状态 | 说明 |
|------|------|------|
| `UpscaleModelLoader` | ✅ | ESRGAN 等放大模型加载 |
| `ImageUpscaleWithModel` | ✅ | 模型放大 |
| `ControlNetLoader` | ✅ | ControlNet 模型加载 |
| `ControlNetApply` | ✅ | 应用 ControlNet |
| `CannyEdgePreprocessor` | ✅ | Canny 边缘检测预处理 |
| `LoadImageMask` | ✅ | 加载 mask |

### 第三批：采样与调度扩展（已对齐 ✅）
| 节点 | 状态 | 说明 |
|------|------|------|
| `KSamplerAdvanced` | ✅ | 高级采样器参数 |
| `CLIPSetLastLayer` | ⬜ | CLIP Skip 控制 |
| `CLIPVisionEncode` | ⬜ | 图像反推编码 |

### 第四批：实用特效与增强
| 节点 | 状态 | 说明 |
|------|------|------|
| `FaceRestoreCFWithModel` / `FaceRestore` | ⬜ | 人脸修复 |
| `ImageBlend` | ✅ | 图像混合（normal/add/multiply/screen） |
| `ImageCompositeMasked` | ✅ | 蒙版合成 |
| `ConditioningCombine` | ✅ | 条件合并（token 维度拼接） |
| `ConditioningAverage` | ✅ | 条件加权平均 |
| `ConditioningConcat` | ✅ | 条件拼接 |
| `RemBGModelLoader` | ✅ | 背景抠图模型加载（ONNX） |
| `ImageRemoveBackground` | ✅ | 背景抠图（输出 RGBA + Mask） |
| `MiDaS-DepthMapPreprocessor` | ⬜ | 深度图预处理 |

### 第五批：高级功能（远期）
| 节点 | 状态 | 说明 |
|------|------|------|
| `IPAdapterLoader` | ✅ | IPAdapter 加载（支持 SD1.5/SDXL，带权重缓存） |
| `IPAdapterApply` | ✅ | IPAdapter 应用 |
| `AnimateDiffLoader` | ⬜ | 视频动画 |

---

## 🐛 已知 Bug 列表

| # | Bug 描述 | 严重程度 | 所在文件 | 状态 |
|---|---------|---------|---------|------|
| 1 | `DeepHighResFix` 拿不到 prompt 字符串 | 高 | core_nodes.cpp | ✅ 已修复 |
| 2 | 多 LoRA 无法堆叠 | 中 | core_nodes.cpp | ✅ 已修复 |
| 3 | 内存泄漏（裸指针传递） | 高 | 多处 | ✅ 已修复 |
| 4 | 缓存系统对指针类型不安全 | 中 | cache.cpp | ✅ 已修复 |
| 5 | 节点错误信息不统一 | 低 | 多处 | ✅ 已修复 |
| 6 | `Workflow::parse_comfyui_json` 中 `dst_slot` 未正确设置 | 中 | workflow.cpp | ✅ 已修复 |
| 7 | `KSampler` 采样失败后 LoRA 状态残留 | 中 | core_nodes.cpp | ✅ 已修复 |

---

## 📊 当前完成度

### 已实现（✅）
- [x] 核心引擎（Workflow、DAGExecutor、ExecutionCache）
- [x] **34 个基础节点**
- [x] 真正的中间 latent/conditioning 传递
- [x] Deep HighRes Fix（节点 + 命令行）
- [x] LoRA 支持（单 LoRA + LoRA Stack 多堆叠）
- [x] 统一错误处理机制（`sd_error_t`）
- [x] 智能指针内存管理
- [x] **ControlNet 支持**（Canny 预处理 + Apply）
- [x] **Upscale 模型节点化**（ESRGAN）
- [x] **Inpaint 支持**（Mask 输入）
- [x] **KSamplerAdvanced**（start/end step / add_noise）
- [x] **单元测试框架**（Catch2 v3，8 个测试用例全部通过）
- [x] 命令行 ↔ 节点桥接
- [x] 快速模式（txt2img/img2img/process/DeepHires/IPAdapter）
- [x] **IPAdapter 支持**（SD1.5/SDXL，含权重缓存优化）
- [x] **Conditioning 操作**（Combine / Concat / Average）
- [x] **图像混合与合成**（ImageBlend / ImageCompositeMasked）
- [x] **背景抠图**（RemBG，ONNX Runtime 可选依赖）

### 进行中（🔄）
- [ ] 性能优化
- [ ] 文档完善

### 待开始（⬜）
- [ ] 视频生成
- [ ] HTTP API

---

## 🎯 下一冲刺目标

建议按以下顺序推进：

1. **更多预处理节点**（MiDaS Depth、OpenPose 等）
2. **HTTP API 服务**（`sd-server`）
3. **视频生成支持**（AnimateDiff）

---

## 🧹 待清理项目

### 清理脚本和代码
- [x] 删除测试 JSON 文件（test_*.json）
- [x] 删除旧的 deep_hires_hook.patch（已被 sd-engine-full.patch 替代）
- [x] 清理 sd-core 目录（逻辑已迁移到 DeepHighResFix 节点）
- [x] 移除 stable-diffusion.cpp 软链接
- [x] 添加 .gitignore 规则
- [x] 清理 CMakeLists.txt 中 sd-core 的残留逻辑

---

*最后更新：2025-04-15（RemBG 背景抠图已完成，节点总数 34）*
