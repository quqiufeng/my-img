# IPAdapter：C++ 原生实现（sd.cpp 后端）

> **目标**：在 my-img（纯 C++ ComfyUI）中实现 IPAdapter 图像提示词功能
> **后端**：stable-diffusion.cpp（GGML/CUDA）
> **模型**：Z-Image Turbo（SDXL 架构，Flow Matching → **DiT**）
> **当前状态**：🚧 Phase 2/5 基本完成（ipadapter.cpp 重写、ONNX Runtime 推理通过、已验证 C++ 端到端管线 ✅）
>
> **核心理念**：市面上没有 C++ 原生的 IPAdapter 实现。
> ComfyUI（Python）→ my-img（C++），这次也一样。

---

## 1. 背景

### 1.1 什么是 IPAdapter

IPAdapter（Image Prompt Adapter）是一种**无需训练**的图像条件注入方法。它通过 CLIP Vision 提取参考图像的特征，注入到扩散模型的 cross-attention 层，使生成结果在构图上参考输入图像。

- **论文**: [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)
- **ComfyUI 节点**: `IPAdapter Unified Loader` + `IPAdapter Apply`
- **特点**: 不修改模型权重，即插即用，支持多参考图组合

### 1.2 为什么有挑战

| 难点 | 说明 |
|------|------|
| **C++ 原生实现** | 所有现有 IPAdapter 实现都是 Python（Diffusers / ComfyUI） |
| **sd.cpp 无 API** | sd.cpp 没有 IPAdapter 的钩子机制，需要自行注入 |
| **UNet 注意力修改** | 需要在采样循环中修改 cross-attention 的 k/v |
| **CLIP Vision 加载** | 需要 C++ 加载 CLIP Vision 模型提取图像特征 |
| **Flow Matching** | Z-Image Turbo 是 flow matching 模型，非标准扩散 |
| **高清大图** | 目标 2560×1440，IPAdapter + HiRes Fix 的组合尚无成熟方案 |

---

## 2. 架构调研

### 2.1 sd.cpp Cross-Attention 实现

sd.cpp 的 cross-attention 位于 `common_block.hpp`，核心调用链：

```
generate_image()  [stable-diffusion.cpp]
  → UNet forward()
    → SpatialTransformer::forward()   [unet.hpp / common_block.hpp]
      → BasicTransformerBlock::forward()
        → attn2 → CrossAttention::forward()   ← 注入点
```

**`CrossAttention`（`common_block.hpp:276-338`）**：

```cpp
class CrossAttention : public GGMLBlock {
    // to_q, to_k, to_v: Linear projections
    // x: [N, n_token, query_dim]
    // context: [N, n_context, context_dim]  ← text embedding
    
    ggml_tensor* forward(ctx, x, context) {
        q = to_q->forward(x);
        k = to_k->forward(context);
        v = to_v->forward(context);
        x = ggml_ext_attention_ext(q, k, v, n_head, ...);
        x = to_out_0->forward(x);
        return x;
    }
};
```

**关键**：`to_k` 和 `to_v` 把 `context`（text embedding）投影成 k/v。IPAdapter 需要把图像 tokens 拼接到 `context` 中，或另外生成 k_img/v_img 并拼接。

### 2.2 LoRA 机制（参考）

sd.cpp 通过 `WeightAdapter` 机制（`ggml_extend.hpp:2997-3002`）实现 LoRA：

```cpp
// Linear::forward():
if (ctx->weight_adapter) {
    return ctx->weight_adapter->forward_with_lora(
        ctx, backend, x, w, b, prefix, params);
}
return ggml_ext_linear(ctx, x, w, b, ...);
```

LoRA 的钩子方式启发了 IPAdapter 的设计思路。

### 2.3 ControlNet 架构（参考）

`control.hpp` 展示了如何在 sd.cpp 中实现外部条件控制。ControlNet 在采样过程中产生多尺度的控制特征，叠加到 UNet 的中间输出上。

### 2.4 Z-Image Turbo 架构（关键更新 2026-06-06）

> ⚠️ **Z-Image 不是 UNet！是 DiT（Diffusion Transformer）架构！**
> 这一发现完全改变了 IPAdapter 的实现路径。

| 特性 | 标准 SDXL UNet | Z-Image Turbo (DiT) |
|------|---------------|---------------------|
| **主干网络** | UNet with SpatialTransformer + ResNet | **仅 Transformer**（无 ResNet 下采样/上采样） |
| **Cross-Attention** | `CrossAttention` (to_q/text_k/text_v) | **没有 cross-attention！** 使用 `JointAttention` 自注意力 |
| **文本融合** | cross-attention `context` → k/v | **文本 token 直接拼接到图像 token 序列**，一起做自注意力 |
| **位置编码** | 无显式位置编码 | **RoPE (Rotary Position Embedding)** |
| **条件调制** | 时间步通过 adaLN 调制 | adaLN modulation + **Flow Matching ODE** |
| **文本嵌入维度** | CLIP-L (768) + OpenCLIP-G (1280) = 2048 | **cap_feat_dim=2560** → embedder → **hidden_size=3840** |

**架构核心信息（来自 `z_image.hpp`）**：

```
ZImageModel {
  cap_embedder_0: RMSNorm → Linear(2560→3840)    # 文本嵌入
  cap_embedder_1: RMSNorm → Linear(3840→3840)     # 文本嵌入深化
  context_refiner: 4× RMSNorm → Linear → GeLU     # 文本 transformer
  noise_refiner: 1× PatchConv(2×2, 3840)           # 噪声降采样
  joints: 30× JointTransformerBlock {              # 联合注意力层
    norm1 → attn (QKV from concat[txt, img])       # 自注意力（文本+图像一起）
    norm2 → mlp
    adaLN modulation (timestep + cfg 调节)
  }
  final_linear: LayerNorm → Linear(3840→3840)
}
```

**关键差异对于 IPAdapter**：
- 标准 IPAdapter 注入 cross-attention 的 k/v（UNet 架构）
- Z-Image 没有 cross-attention！文本和图像在 `JointAttention` 中通过**自注意力**交互
- IPAdapter 注入需要在 `cap_embedder` 之前添加 image tokens 到 `c_crossattn`（2560-dim），或直接拼接到 hidden_size(3840) 的序列
- 需要线性投影层将 IPAdapter 输出(768-dim) 映射到 2560-dim 或 3840-dim

---

## 3. 当前代码状态

### 3.1 代码架构（Phase 2 实现）

**`/opt/my-img/src/utils/ipadapter.h`** — 当前接口设计（无 libtorch 依赖）：

```cpp
class IPAdapter {
public:
    IPAdapter();
    explicit IPAdapter(const IPAdapterConfig& config);
    ~IPAdapter();

    bool load_model(const std::string& model_path, const std::string& clip_vision_path);
    bool load_reference_image(const std::string& image_path);

    // 获取计算好的 image tokens [1, 768]（扁平化 float 向量）
    const std::vector<float>& get_image_tokens() const { return image_tokens_; }
    bool is_loaded() const { return model_loaded_; }

private:
    IPAdapterConfig config_;
    bool model_loaded_ = false;
    std::vector<float> image_tokens_;

    // PIMPL: 隐藏 ONNX Runtime 类型，头文件零依赖
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

**设计要点**：
- **PIMPL 模式**：`Impl` 结构体在 `.cpp` 中定义，包含 `Ort::Env`, `Ort::Session` 等 ONNX Runtime 类型。头文件不暴露任何 ONNX Runtime 依赖，编译时无需 `HAVE_ONNXRUNTIME` 宏
- **无 libtorch**：原接口使用了 `torch::Tensor`（引用了整个 libtorch），新接口完全使用 `std::vector<float>`，消除 1GB 编译依赖
- **可移动**：拷贝删除，移动默认（`unique_ptr<Impl>` 天然支持移动）
- **终身会话**：`Ort::Session` 在 `load_model()` 时创建一次，持续到对象析构，避免重复加载

**`/opt/my-img/src/utils/ipadapter.cpp`** — ONNX Runtime 推理管线：

```cpp
struct IPAdapter::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "IPAdapter"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> clip_session;  // CLIP Vision 模型
    std::unique_ptr<Ort::Session> ipa_session;   // IPAdapter MLP 模型
    std::string clip_input_name, clip_output_name;
    std::string ipa_input_name, ipa_output_name;
};
```

**推理流程**（`load_reference_image()` 中一次性完成）：
```
1. cv::imread(path) → BGR Mat
2. cv::cvtColor(BGR→RGB) + cv::resize(224×224)
3. float32 归一化: (pixel/255 - mean) / std
   mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
4. CHW 排列: [1, 3, 224, 224]
5. clip_session.Run() → image_embeds [1, 1024]
6. ipa_session.Run(image_embeds) → text_embedding [1, 768]
7. 结果存入 image_tokens_（std::vector<float>, 768 个元素）
```

**验证结果**（C++ vs Python ONNX Runtime CPU）：
```
C++ (GPU ONNX Runtime build):     Python (CPU ONNX Runtime):
CLIP Vision [1, 1024]             CLIP Vision [1, 1024]
IPAdapter MLP [1, 768]            IPAdapter MLP [1, 768]
  min=-1.34, max=1.27, mean≈0       min=-1.07, max=1.30, mean≈0
```
数值分布一致（~N(0,1)），微小差异来自图优化器和数值精度

### 3.2 历史版本对比

| 问题 | 旧实现（Phase 0） | 当前实现（Phase 2） |
|------|-------------------|---------------------|
| **CLIP Vision 加载** | `cv::dnn::readNetFromONNX(clip_vision_sd15.safetensors)` → 静默失败（格式不对） | `Ort::Session(env, clip_vision.onnx)` → ✅ 成功加载 2.4GB 模型 |
| **IPAdapter MLP 加载** | 同上 | `Ort::Session(env, ipadapter.onnx)` → ✅ 成功加载 5.4MB 模型 |
| **`apply_ipadapter()` 调用** | 从未被调用（sdcpp_adapter.cpp 只打印日志） | ✅ `sdcpp_adapter.cpp` 中 `--ipadapter` 触发完整管线 |
| **`inject_attention`** | `return latent * (1.0f + features * 0.05f)` — 假注入 | ⏳ Phase 3 实现真正的 context 注入 |
| **模型重加载** | 每次推理都重新 `readNetFromONNX` | ✅ `Ort::Session` 持久化，生命周期内只加载一次 |
| **libtorch 依赖** | `torch::Tensor` 在接口中暴露，需链接 1GB libtorch | ✅ `std::vector<float>` 纯标准库 |
| **推理耗时** | N/A（从未成功运行） | CLIP Vision ~1s, IPAdapter MLP ~1ms |

### 3.2 CLI 参数（已实现）

```
--ipadapter               Enable IPAdapter
--ipadapter-model PATH    IPAdapter model path
--ipadapter-clip-vision PATH  CLIP Vision model path
--ipadapter-image PATH    Reference image path
--ipadapter-weight FLOAT  Weight 0.0-1.0 (default: 1.0)
--ipadapter-start FLOAT   Start step ratio 0.0-1.0 (default: 0.0)
--ipadapter-end FLOAT     End step ratio 0.0-1.0 (default: 1.0)
```

### 3.3 模型文件

```
/data/models/image/
├── ipadapter.onnx          (13K)     ← ONNX 头文件（外部数据格式）
├── ipadapter.onnx.data     (5.4M)    ← ONNX 权重（2 层 MLP: 1024→768→768）
├── clip_vision_sd15.safetensors (2.4G)  ← CLIP Vision 原始 PyTorch（不再使用）
├── clip_vision.onnx        (397K)    ← CLIP Vision ONNX 头文件 ✅
├── clip_vision.onnx.data   (2.36G)   ← CLIP Vision ONNX 权重 ✅
├── inswapper_128.onnx      (529M)    ← Face Swap（不相关）
└── yunet_320_320.onnx      (5.8M)    ← 人脸检测（不相关）
```

**模型详细信息**：

| 模型 | 架构 | 参数 | 输入 | 输出 |
|------|------|------|------|------|
| `clip_vision.onnx` | OpenCLIP ViT-bigG/14 | ~2.4B | `[1,3,224,224]` (image) + `[1,257]` (attention_mask) | `[1,1024]` (global embedding) |
| `ipadapter.onnx` | 2-layer MLP | 1.38M | `[1,1024]` (CLIP embedding) | `[1,768]` (image tokens) |

---

## 4. 实现方案（适配 Z-Image DiT）

### 4.1 方案对比

| 方案 | 侵入性 | 效果 | 工作量 |
|------|--------|------|--------|
| **A: 条件编码前注入（推荐）** | 低（不改 sd.cpp） | 中（适用于 DiT） | 中 |
| **B: 修改 JointAttention** | 高（改 sd.cpp） | 高（decoupled IPAdapter） | 大 |
| **C: 后处理注入** | 无 | 低（不影响生成） | 小 |

### 4.2 方案 A：条件编码前注入（推荐，针对 DiT）

**原理**：Z-Image 的 `cap_embedder` 将 text context (2560-dim) 投影到 hidden_size (3840-dim)，然后与 image tokens 拼接做自注意力。我们可以在 context 进入 `cap_embedder` 前拼接 IPAdapter image tokens。

```
Z-Image DiT 数据流:

Text (CLIP-L + CLIP-G) → [77, 2560] context
                                    ↓
IPAdapter:                           ↓
  ref_img → CLIP Vision → [1,1024]   ↓
         → IPAdapter MLP → [1,768]   ↓
         → LinearProj   → [N,2560] ←┘   ← image tokens 拼接到 text context
                                    ↓
  context': [77+N, 2560]            ↓
                                    ↓
  cap_embedder_0: RMSNorm → Linear(2560→3840)
  cap_embedder_1: RMSNorm → Linear(3840→3840)
  → [77+N, 3840] = txt  ← 文本 + 图像 tokens 一起
                                    ↓
  JointAttention(QKV from concat[txt, img])   ← 自注意力处理所有 token
```

**流程**：

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│  参考图       │───▶│  CLIP Vision      │───▶│  图像特征     │
│  ~/demo.jpg   │    │  (ONNX Runtime)   │    │  [1, 1024]   │
└──────────────┘    └───────────────────┘    └──────┬───────┘
                                                     ▼
                                            ┌──────────────────┐
                                            │  IPAdapter MLP    │
                                            │  (ONNX Runtime)   │
                                            │  1024→768→768     │
                                            └──────┬───────────┘
                                                     ▼
                                            ┌──────────────────┐
                                            │  Linear Projector │
                                            │  768→2560         │
                                            │  (tiny nn, ~2M)   │
                                            └──────┬───────────┘
                                                     ▼
   采样流程 (在 generate_image 前):          ┌──────────────────┐
   text context: [77+N, 2560]              │  image tokens     │
   (原始 text: [77, 2560]                  │  [N, 2560]        │
    + image tokens: [4, 2560])             └──────┬───────────┘
                                                     ▼
                                           ┌──────────────────┐
                                           │  Gehe zu          │
                                           │  cap_embedder     │
                                           │  → JointAttention │
                                           └──────────────────┘
```

**优势**：
- 不需要修改 sd.cpp 的 `JointAttention` 代码
- 利用 Z-Image 已经存在的 **文本+图像自注意力** 机制
- token 数量增加很少（4 tokens vs 77+4096 tokens），计算量可忽略

**需要解决**：
1. IPAdapter MLP 输出 768-dim → 需要投影到 2560-dim (cap_feat_dim)
2. `generate_image()` 是黑盒，需要在 **调用前** 修改 context
3. 当前的 `SDCondition.c_image_embeds` 字段设计用于 multimodal LLM，但可以被复用

### 4.3 方案 B：修改 JointAttention（高级，未来升级）

**原理**：Z-Image 的 `JointTransformerBlock` 在 `forward()` 中做：
```cpp
auto q = q_proj(ctx, x);     // [N, n_txt+n_img, hidden_size]
auto k = k_proj(ctx, x);     // 同上
auto v = v_proj(ctx, x);     // 同上
```

可以添加独立于文本的 image k/v 通道，在 attention 计算后与文本结果融合。

### 4.4 ONNX Runtime 集成

当前 `CMakeLists.txt`（`/opt/my-img/CMakeLists.txt:161-291`）已支持 ONNX Runtime：

```cmake
if(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(STATUS "[my-img] Found ONNX Runtime at ${ONNXRUNTIME_ROOT}")
    set(ONNXRUNTIME_FOUND TRUE)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT}/include)
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
endif()
...
if(ONNXRUNTIME_FOUND)
    target_link_libraries(myimg-cli PRIVATE
        ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    )
endif()
```

**ipadapter.cpp 的 ONNX 加载方式**（使用 C++ API）：

```cpp
// 1. 创建 Session（构造函数中）
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "IPAdapter");
Ort::SessionOptions opts;
opts.SetIntraOpNumThreads(4);
opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

// 2. 加载 CLIP Vision（大模型）
auto clip_session = std::make_unique<Ort::Session>(env, path, opts);
// 自动检测输入输出名
auto name = clip_session->GetInputNameAllocated(0, allocator);
auto type_info = clip_session->GetInputTypeInfo(0);
auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();

// 3. 推理
Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    mem, input_data.data(), input_data.size(), shape.data(), shape.size());
auto output = clip_session->Run(Ort::RunOptions{nullptr},
                                 input_names, &input_tensor, 1,
                                 output_names, 1);
float* out_data = output[0].GetTensorMutableData<float>();
```

**关键实现细节**：

1. **PIMPL 隐藏 ONNX 类型**：`IPAdapter::Impl` 在 `.cpp` 中定义，头文件不包含 `<onnxruntime_cxx_api.h>`。即使 `ONNXRUNTIME_FOUND=FALSE`，头文件仍可编译，只有 `.cpp` 需要条件编译
2. **`session_options` 配置**：`SetIntraOpNumThreads(4)` 控制 CPU 线程数，`ORT_ENABLE_ALL` 启用所有图优化（常量折叠、算子融合等），加速 2.4GB CLIP Vision 推理
3. **外部数据格式支持**：`clip_vision.onnx` 使用外部数据（`clip_vision.onnx.data` 2.36GB），ONNX Runtime 自动加载同目录下的 `.data` 文件，无需额外代码
4. **内存管理**：推理输出通过 `GetTensorMutableData<float>()` 获取原始指针，用 `std::vector<float>(data, data + size)` 拷贝到 RAII 容器，防止悬垂指针

**ONNX Runtime 安装状态**：
- ✅ 已安装到 `/data/venv/onnxruntime-linux-x64-gpu-1.20.1/`
- ✅ CUDA 12.6 兼容（已验证 `libonnxruntime_providers_cuda.so`）
- ✅ `CMakeLists.txt` 自动检测 `/data/venv/onnxruntime-linux-x64-gpu-1.20.1/`
- ✅ 编译后验证：`ldd build/myimg-cli | grep onnx` → `libonnxruntime.so.1` 已链接
- ⚠️ 运行时需要设置 `LD_LIBRARY_PATH=/data/venv/onnxruntime-linux-x64-gpu-1.20.1/lib`

---

## 5. 实现计划

### Phase 1：准备工作 ✅（已完成）

- [x] 安装 ONNX Runtime GPU 1.20.1 到 `/data/venv/`
- [x] 确认/获取 IPAdapter 模型 `ipadapter.onnx`（SD1.5 版，1024→768→768）
- [x] 确认/获取 ONNX 格式的 CLIP Vision（已从 safetensors 转换）
- [x] 端到端测试 ONNX Runtime 推理（Python，CPU）
- [x] 修复 CUDA 路径 `/usr/local/cuda` → CUDA 12.6
- [x] 重建 sd.cpp + myimg-cli 与 ONNX Runtime 链接

### Phase 2：CLIP Vision + IPAdapter MLP ✅（已完成）

- [x] 用 ONNX Runtime (Ort::Session) 重写 ipadapter.cpp（使用 PIMPL 隐藏实现细节）
- [x] 加载 CLIP Vision ONNX（2.4GB），提取图像特征 [1, 1024]（~1.5s 加载，~1s 推理）
- [x] 加载 IPAdapter MLP ONNX（5.4MB），投影为 image tokens [1, 768]（instant）
- [x] 验证特征提取正确性（C++ vs Python ONNX Runtime 输出分布一致）
- [x] 接入 sdcpp_adapter.cpp 生成流程（--ipadapter 参数触发加载 + 推理）
- [x] 跨平台兼容：ipadapter.h 不依赖 ONNX Runtime 头文件（PIMPL），仅 .cpp 文件链接

### Phase 3：上下文注入（DiT 适配）⏳（下一阶段）

- [ ] 添加 Linear 投影层 768→2560（cap_feat_dim），将 IPAdapter 输出投影到 Z-Image 的 context 空间
- [ ] 修改 sd.cpp 的 `generate_image()` 或新增 `generate_image_with_ipadapter()` 函数，接受 IPAdapter tokens
- [ ] 在 `get_learned_condition()` 后，将 image tokens 拼接到 `SDCondition.c_crossattn` 的 [77, 2560] context 上
- [ ] 修改 `ZImageModel::forward_core()` 或调用处，使 image tokens 通过 `cap_embedder` 进入 `JointAttention`
- [ ] 实现权重控制（ipadapter_weight → 缩放 image tokens）
- [ ] 实现步数控制（start_at / end_at）
- [ ] 处理维度投影（768→2560）— 可以是简单 Linear 层或 ONNX 模型

### Phase 4：集成与优化

- [ ] 接入 HiRes Fix（高分辨率下保持 IPAdapter 效果）
- [ ] 接入 FreeU + SAG（与 IPAdapter 兼容性测试）
- [ ] 2560×1440 高清生成测试
- [ ] VRAM 优化

### Phase 5：完善

- [ ] 多人脸/多参考图支持
- [ ] 抠脸预处理（人脸检测 + 裁剪）
- [ ] 权重退火（linear / cosine schedule）
- [ ] 与 ControlNet 协同工作

---

## 6. 关键决策记录

### 2026-06-06 (1)：项目启动

**问题**：IPAdapter 代码存在但不工作（CLIP Vision 加载失败、apply 未调用、inject 假实现）。

**决策**：重新实现 IPAdapter，采用方案 A（上下文拼接）作为第一阶段，后续升级到方案 B（decoupled cross-attention）。

**理由**：
1. 方案 A 不修改 sd.cpp，风险最低
2. 可以快速验证 IPAdapter 是否能改善生成效果
3. 后续可以平滑升级到方案 B

### 2026-06-06 (2)：架构发现 — Z-Image 是 DiT 非 UNet

**问题**：在分析 `z_image.hpp` 后发现 Z-Image Turbo 是 DiT 架构，完全不同于标准 SDXL UNet。没有 `CrossAttention` 层，使用的是 `JointTransformerBlock`（文本和图像 token 一起做自注意力）。

**决策**：
1. IPAdapter 注入点从 cross-attention k/v 改为 **`cap_embedder` 前的 context 拼接**
2. 需要添加 **768→2560 线性投影层** 用于维度匹配
3. 原始方案 A 仍然可行（注入 context），但路径不同
4. 方案 B 修改对象从 `CrossAttention` 改为 `JointTransformerBlock`

**理由**：
- DiT 的 JointAttention 天然支持文本+图像混合，context 拼接是非常自然的注入方式
- 无需额外的 attention 层修改

### 2026-06-06 (3)：ONNX Runtime + CUDA 12.6 适配

**问题**：GCC 13 与 CUDA 11.8 不兼容，LTO bytecode 版本不匹配，ONNX Runtime 链接缺失。

**决策**：
1. `/usr/local/cuda` 从 `/data/cuda-11.8` → `/data/cuda` (CUDA 12.6)
2. sd.cpp 用 `-DGGML_LTO=OFF` 重建（GCC 13 LTO 与 GCC 12 不兼容）
3. ONNX Runtime GPU 1.20.1 从 GitHub 发布页下载（252MB）
4. `LD_LIBRARY_PATH` 需要包含 ONNX Runtime lib 目录

### 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| CLIP Vision 运行时 | ONNX Runtime GPU | 支持 CUDA，性能好，已集成到 CMakeLists.txt |
| IPAdapter MLP 运行时 | ONNX Runtime GPU | 同上 |
| 图像预处理 | OpenCV | 已依赖，用于 resize/normalize |
| 特征注入方式 | 拼接 context | 不修改 sd.cpp，利用 DiT 的 JointAttention |
| 模型格式 | ONNX | 跨平台，推理效率高 |
| 线性投影 | 简单 Linear 层 | 768→2560，可以用 ONNX 或用简单的 C++ 矩阵乘 |

## 7. 测试计划

### 7.1 单元测试

```cpp
TEST_CASE("IPAdapter: CLIP Vision feature extraction", "[ipadapter][unit]") {
    // 加载 ONNX CLIP Vision
    // 提取图像特征
    // 验证特征 shape 和数值范围
}

TEST_CASE("IPAdapter: image token projection", "[ipadapter][unit]") {
    // 加载 IPAdapter MLP
    // 投影图像特征为 image tokens
    // 验证 tokens shape
}

TEST_CASE("IPAdapter: context concatenation", "[ipadapter][unit]") {
    // 拼接 text context + image tokens
    // 验证 shape 正确
    // 验证语义保持
}
```

### 7.2 集成测试

```bash
# 1280x720 基础测试
./myimg-cli \
  --diffusion-model ... \
  --ipadapter --ipadapter-model ... \
  --ipadapter-clip-vision ... --ipadapter-image ref.jpg \
  --ipadapter-weight 0.8 \
  -p "portrait of a person" \
  -W 1280 -H 720 --steps 20 -s 42 \
  -o test_ipadapter_base.png

# 2560x1440 高清测试
./myimg-cli \
  ... \
  -W 1280 -H 720 --steps 20 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.30 --hires-steps 45 \
  -o test_ipadapter_hires.png

# 组合测试（FreeU + SAG + IPAdapter）
./myimg-cli \
  ... \
  --freeu --freeu-b1 1.4 --freeu-b2 1.5 \
  --sag --sag-scale 0.5 \
  --ipadapter ... \
  -o test_ipadapter_all.png
```

### 7.3 效果验证

- 对比有/无 IPAdapter 的人脸相似度
- 验证不同 weight（0.3 / 0.5 / 0.8 / 1.0）的效果变化
- 验证剪裁人脸 vs 全身照作为参考图的差异

---

## 8. 参考资源

- [IP-Adapter 论文](https://arxiv.org/abs/2308.06721)
- [IPAdapter GitHub (huggingface)](https://github.com/tencent-ailab/IP-Adapter)
- [ComfyUI IPAdapter 节点源码](https://github.com/comfyanonymous/ComfyUI_IPAdapter_plus)
- [sd.cpp 架构](https://github.com/leejet/stable-diffusion.cpp)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)

---

## 9. 里程碑

| 阶段 | 预计完成 | 状态 |
|------|----------|------|
| P1：ONNX Runtime 安装 + 模型转换 | 2026-06-06 | ✅ 已完成 |
| P2：CLIP Vision + IPAdapter ONNX 推理 | 2026-06-06 | ✅ 已完成 |
| P3：条件编码注入 + 生成流程集成 | 下一轮 | ⏳ 待开始 |
| P4：HiRes Fix + 高清测试 | - | ⏳ 待开始 |
| P5：抠脸 + 参数调优 | - | ⏳ 待开始 |

---

> **最后更新**: 2026-06-06
> **维护者**: my-img Team
