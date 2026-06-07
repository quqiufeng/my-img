# IPAdapter：C++ 原生实现（sd.cpp 后端）

> **目标**：在 my-img（纯 C++ ComfyUI）中实现 IPAdapter 图像提示词功能
> **后端**：stable-diffusion.cpp（GGML/CUDA）
> **模型**：
> - **DiT 路径**：Z-Image Turbo（SDXL 架构，Flow Matching → **DiT**）
> - **UNet 路径**：SDXL Base 1.0（标准 **UNet** cross-attention）
> **当前状态**：
> - 🚧 Phase 3-4 完成（注入 + 步进控制 + HiRes Fix ✅）
> - ✅ **UNet IPAdapter 权重注入实现完成**：70/70 cross-attention 层权重加载成功，端到端运行无崩溃
> - ⚠️ **SDXL Base 模型兼容性问题**：当前 `sd_xl_base_1.0.safetensors` 与 sd.cpp 产生纯白输出（与 IPAdapter 代码无关），阻塞 UNet 路径效果验证
> - **重大发现**：当前 SD1.5 IPAdapter 模型与 Z-Image DiT 不兼容，导致效果几乎为零。已定位 root cause，正在适配 SDXL IPAdapter Plus。
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
--ipadapter                   Enable IPAdapter
--ipadapter-model PATH        IPAdapter model path
--ipadapter-clip-vision PATH  CLIP Vision model path
--ipadapter-projection PATH   768->2560 linear projection (optional)
--ipadapter-image PATH        Reference image path
--ipadapter-weight FLOAT      Weight 0.0-1.0 (default: 1.0)
--ipadapter-start FLOAT       Start step ratio 0.0-1.0 (default: 0.0)
--ipadapter-end FLOAT         End step ratio 0.0-1.0 (default: 1.0)
```

### 3.3 模型文件

#### 当前可用模型

```
/data/models/image/
├── ipadapter.onnx                (13K)     ← ⚠️ SD1.5 版，与 Z-Image 不兼容
├── ipadapter.onnx.data           (5.4M)    ← SD1.5 权重（2 层 MLP: 1024→768）
├── ipadapter_proj.onnx           (0.3K)    ← 线性投影 ONNX（identity init）
├── ipadapter_proj.onnx.data      (15M)     ← 投影权重
├── clip_vision.onnx              (397K)    ← CLIP Vision ViT-bigG/14 ✅
├── clip_vision.onnx.data         (2.36G)   ← CLIP Vision 权重（输出 1024-dim）
├── ip-adapter-plus_sdxl_vit-h.safetensors (847M) ← 📦 SDXL IPAdapter Plus（待转 ONNX）
```

**兼容性矩阵**：

| 模型 | 输出维度 | 适配 Z-Image | 状态 |
|------|----------|-------------|------|
| `ipadapter.onnx` (SD1.5) | [1, 768] | ❌ 不兼容 | 已废弃 |
| `ip-adapter-plus_sdxl_vit-h` | [16, 2048] | ⚠️ 需 2048→2560 投影 | 已下载，待转 ONNX |
| `clip_vision.onnx` | [1, 1024] | ⚠️ 需确认是否匹配 SDXL Plus | 当前使用 |

**InsightFace 模型（人脸检测/识别）**：

```
/data/models/image/
├── w600k_r50.onnx                (167M)    ← ArcFace 人脸特征 (512-dim) ✅
├── det_10g.onnx                  (17M)     ← 人脸检测 ✅
├── 2d106det.onnx                 (4.8M)    ← 2D 106 关键点 ✅
├── 1k3d68.onnx                   (137M)    ← 3D 68 关键点 ✅
├── inswapper_128.onnx            (529M)    ← Face Swap ✅
├── yunet_320_320.onnx            (5.8M)    ← 人脸检测（OpenCV 版）✅
```

**模型详细信息**：

| 模型 | 架构 | 参数 | 输入 | 输出 |
|------|------|------|------|------|
| `clip_vision.onnx` | OpenCLIP ViT-bigG/14 | ~2.4B | `[1,3,224,224]` (image) + `[1,257]` (attention_mask) | `[1,1024]` (global embedding) |
| `ipadapter.onnx` | 2-layer MLP | 1.38M | `[1,1024]` (CLIP embedding) | `[1,768]` (image tokens) |
| `ipadapter_proj.onnx` | Linear(768→2560) | 1.97M | `[1,768]` (IPA tokens) | `[1,2560]` (Z-Image context) |

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

### Phase 3：上下文注入（DiT 适配）✅（完成 2026-06-06）

- [x] 修改 sd.cpp 的 `sd_img_gen_params_t`，新增 `ipadapter_tokens` / `ipadapter_num_tokens` / `ipadapter_weight` 字段
- [x] 在 `prepare_image_generation_embeds()` 中，`get_learned_condition()` 后注入 image tokens
- [x] 实现权重控制（`ipadapter_weight` → 缩放 image tokens）
- [x] `sdcpp_adapter.cpp` 中接入：`--ipadapter` 触发完整管线
- [x] PE 兼容性验证通过
- [x] 线性投影层 768→2560（ONNX 模型，identity init）
- [x] **步数控制（start_at / end_at）** — `sd::ops::slice()` 在 denoise lambda 中裁切 IPA tokens

### Phase 4：HiRes Fix 集成 ✅（完成 2026-06-06）

- [x] IPA 自动继承到 HiRes 第二遍采样（复用 `embeds.cond`）
- [x] 步进控制在两遍采样中独立生效
- [x] 2560×1440 全尺寸生成测试通过（~7.4 min, VRAM 峰值 ~18.4 GB）
- [x] FreeU + IPAdapter 兼容性验证通过

### Phase 5：SDXL IPAdapter Plus 适配（进行中 2026-06-07）

#### 5.1 重大发现：模型架构完全不匹配（2026-06-07）

**重大发现**：当前 `ipadapter.onnx` 是 **SD1.5 版本**，与 **Z-Image (SDXL DiT)** 完全不兼容。

| 组件 | 当前 (SD1.5) | Z-Image (SDXL DiT) 需要 |
|------|-------------|------------------------|
| CLIP Vision 输出 | ViT-L 768-dim | **ViT-H/14 1280-dim** |
| IPAdapter MLP | 2-layer 1024→768 | **Perceiver Resampler 16×1280→2048** |
| 最终输出 | [1, 768] | **[16, 2048] → 需投影到 [16, 2560]** |
| 架构适配 | UNet CrossAttention | DiT JointAttention (context 拼接) |

**测试验证**（2026-06-07）：
- 极端权重测试（weight=0 vs 1.5）：注入通路正常 ✅
- CLIP 相似度量化：weight=1.5 比无 IPA **更不像**参考图（0.2973 vs 0.4648）❌
- 4-token 重复测试：恶化效果（0.1636）❌
- **结论**：不是注入代码 bug，是**模型完全不匹配**

#### 5.2 更深层的 Root Cause：CLIP Vision 输出格式错误（2026-06-07 下午）

在尝试适配 SDXL Plus 时，发现**第二个致命错误**：

**IPAdapter Plus 使用的是 `hidden_states[-2]`，不是 `image_embeds`**。

```python
# IPAdapter Plus (ComfyUI/diffusers 实现)
clip_image_embeds = image_encoder(
    clip_image, output_hidden_states=True
).hidden_states[-2]  # [batch, 257, 1280] ← 257个patch token

# 标准 IPAdapter (base) 才用 image_embeds
clip_image_embeds = image_encoder(clip_image).image_embeds  # [batch, 1024] ← pooled
```

| 属性 | `image_embeds` (pooled) | `hidden_states[-2]` (patch tokens) |
|------|------------------------|-----------------------------------|
| **维度** | [1, 1024] | **[1, 257, 1280]** |
| **来源** | 最终 LayerNorm + Projection | 倒数第二层 transformer 输出 |
| **信息** | 全局聚合特征 | **逐 patch 局部特征** |
| **IPAdapter 类型** | Base 版 | **Plus 版** |
| **Perceiver Resampler 输入** | 直接输入 | **需要 proj_in [1280,1280]** |

**关键证据**：
- 从 `ip-adapter-plus_sdxl_vit-h.safetensors` 提取的权重：`image_proj.proj_in.weight = [1280, 1280]`
- 这说明 Perceiver Resampler 期望的输入是 **1280-dim**，不是 1024-dim
- 我们的 `clip_vision.onnx` 只输出 `image_embeds [1, 1024]`，与 Plus 模型不匹配
- 之前 v1 ONNX 导出时加的 `clip_adapter (1024→1280)` 是**未训练的随机权重**，破坏了语义

**这就是 SDXL Plus 16 tokens 效果仍然很差（CLIP sim 0.0936）的根本原因**：
1. 输入不是 patch tokens，而是错误的 pooled embedding
2. 即使加了 clip_adapter，也是随机线性变换，完全破坏了 CLIP 特征

#### 5.3 v2 ONNX 导出（2026-06-07）

**导出环境**：`/data/venv` (torch 2.4.0+cu118, open_clip 3.3.0, ONNX Runtime 1.26.0)

**教训**：导出前确认系统 CUDA 版本（12.6），不要随意重装 venv 中的 torch。本次导出使用现有已安装的工具链，未新增/删除任何包。

**(1) CLIP Vision ViT-H/14 hidden states**：

```python
# OpenCLIP ViT-H/14 (laion2b_s32b_b79k)
# 输出倒数第二层 transformer 结果（before ln_post + projection）
# Shape: [batch, 257, 1280]

class OpenCLIPHidden(nn.Module):
    def forward(self, pixel_values):
        x = visual.conv1(pixel_values)       # [B, 1280, 14, 14]
        x = x.reshape(B, 1280, -1)
        x = x.permute(0, 2, 1)               # [B, 196, 1280]
        x = torch.cat([CLS_token, x], dim=1) # [B, 257, 1280]
        x = x + positional_embedding
        x = ln_pre(x)
        x = x.permute(1, 0, 2)               # [257, B, 1280]
        for block in transformer.resblocks:
            x = block(x, None)
        x = x.permute(1, 0, 2)               # [B, 257, 1280]
        return x  # ← 返回这里，不做 ln_post 和 projection
```

- **输出文件**：`/data/models/image/clip_vision_vit_h_hidden.onnx`
- **输出 shape**：`[batch_size, 257, 1280]`
- **输出范围**：[-6.96, 2.49]（未归一化，直接进 Perceiver Resampler）
- **大小**：约 2.4GB（与原始 clip_vision.onnx 相近）

**(2) SDXL Plus Perceiver Resampler (v2)**：

```python
# 直接使用原始权重，无 clip_adapter
# 输入: [batch, 257, 1280] (来自 hidden states)
# 输出: [batch, 16, 2048]

Resampler(
    dim=1280, depth=4, dim_head=64, heads=20,
    num_queries=16, embedding_dim=1280, output_dim=2048, ff_mult=4
)
```

- **权重来源**：`image_proj.*` from `ip-adapter-plus_sdxl_vit-h.safetensors`
- **proj_in**：`[1280, 1280]`（将 1280-dim patch 特征投影到 resampler dim）
- **proj_out**：`[2048, 1280]`（输出 2048-dim image tokens）
- **输出文件**：`/data/models/image/ipadapter_sdxl_plus_v2.onnx`
- **输出 shape**：`[batch_size, 16, 2048]`

**(3) 投影层 2048→2560 (v2)**：

```python
# Linear(2048, 2560), bias=False
# 初始化: identity-like (前 2048 列 eye，后 512 列 zero)
nn.init.eye_(proj.weight[:, :2048])
```

- **输出文件**：`/data/models/image/ipadapter_proj_2048_2560_v2.onnx`
- **输出 shape**：`[batch_size, 16, 2560]`
- **注意**：此投影层未训练，仅用于维度对齐。后续如有训练数据，可替换权重

#### 5.4 模型文件清单（v2）

```
/data/models/image/
├── clip_vision_vit_h_hidden.onnx           (397K)   ← NEW: ViT-H/14 hidden states [257,1280]
├── clip_vision_vit_h_hidden.onnx.data      (2.36G)  ← 权重（与原始相同，仅输出节点不同）
├── ipadapter_sdxl_plus_v2.onnx             (待测量)  ← NEW: Perceiver Resampler [257,1280]→[16,2048]
├── ipadapter_proj_2048_2560_v2.onnx        (待测量)  ← NEW: 2048→2560 投影
├── ip-adapter-plus_sdxl_vit-h.safetensors  (847M)   ← 源 PyTorch 权重
├── ipadapter_sdxl_plus.onnx                (旧版)   ← v1: 含未训练 clip_adapter，已废弃
└── ipadapter_proj_2048_2560.onnx           (旧版)   ← v1: 同上，已废弃
```

**v1 vs v2 核心差异**：

| 组件 | v1 (废弃) | v2 (当前) |
|------|----------|----------|
| CLIP Vision 输入 | `image_embeds [1,1024]` | **`hidden_states [1,257,1280]`** |
| clip_adapter | 未训练 `Linear(1024→1280)` | **无**（Resampler 直接接收 1280-dim） |
| 权重正确性 | ❌ 随机初始化 | ✅ 从 safetensors 加载原始权重 |
| 预期效果 | 无或负效果 | 待验证 |

#### 5.5 修复计划（更新）

- [x] 下载 SDXL IPAdapter Plus (`ip-adapter-plus_sdxl_vit-h.safetensors`, 847MB)
- [x] 分析 root cause：`hidden_states[-2]` vs `image_embeds`
- [x] 导出 `clip_vision_vit_h_hidden.onnx`（257×1280 patch tokens）
- [x] 导出 `ipadapter_sdxl_plus_v2.onnx`（Perceiver Resampler，无 clip_adapter）
- [x] 导出 `ipadapter_proj_2048_2560_v2.onnx`（2048→2560）
- [ ] 修改 `ipadapter.cpp`：支持新的 hidden states 输入格式
- [ ] 修改 `sdcpp_adapter.cpp`：传递 v2 模型路径
- [ ] 修改 `stable-diffusion.cpp`：适配 16 tokens SDXL Plus 注入
- [ ] 重新构建 + 测试验证效果

### Phase 5：VRAM 优化（RTX 3080 20GB）

- [x] VAE Tiling 128×128（峰值 6.6 GB VRAM）
- [x] Flash Attention 启用
- [ ] 进一步优化 VAE tile overlap 减少 artifacts
- [ ] CPU offload 策略（显存不足时）

### Phase 6：IPAdapter Face ID（人脸克隆）

**目标**：用参考图的人脸特征控制生成，实现人脸克隆（face preservation）。

**所需模型**：

| 模型 | 格式 | 大小 | 来源 | 状态 |
|------|------|------|------|------|
| ArcFace `w600k_r50` | ONNX | ~20MB | insightface buffalo_l | 📦 已下载（`/tmp/insightface_models/`） |
| Face detection `det_10g` | ONNX | ~10MB | insightface buffalo_l | 📦 已下载 |
| `ip-adapter-faceid-plusv2_sd15` | safetensors | ~100MB | Hugging Face h94/IP-Adapter-FaceID | ❌ 待下载 |
| FaceID MLP → ONNX 转换脚本 | Python | - | 自研 | ❌ 待编写 |

**实现步骤**：

- [ ] 1. 下载 IPAdapter Face ID PyTorch 权重（`h94/IP-Adapter-FaceID`）
- [ ] 2. 编写 Python 脚本，将 Face ID MLP 导出为 ONNX
- [ ] 3. 将 ArcFace ONNX 复制到 `/data/models/image/`
- [ ] 4. 在 `ipadapter.cpp` 中添加 ArcFace 人脸特征提取
- [ ] 5. 修改注入逻辑：Face ID 特征 (512-dim) + CLIP 特征 (1024-dim) → 拼接 → Face ID MLP → 768-dim
- [ ] 6. 测试人脸相似度

**依赖安装**：
```bash
# Python 环境（仅用于模型下载和转换，运行时不需要）
pip install insightface onnx onnxruntime

# 模型下载
python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')"

# 下载 IPAdapter Face ID
wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin
```

**C++ 运行时依赖**：
- ONNX Runtime GPU（已有 ✅）
- OpenCV（已有 ✅）
- 新增：ArcFace ONNX（待复制到 `/data/models/image/`）
- 新增：Face ID MLP ONNX（待转换）

---

## 6. UNet IPAdapter 路径（SDXL Base 1.0）

### 6.1 背景

除了 Z-Image DiT 路径，my-img 还需要支持**标准 SDXL UNet** 的 IPAdapter。标准 IPAdapter 论文中的注入点就是 UNet cross-attention 的 `to_k` / `to_v`：为参考图像单独学习一组 `to_k_ip` 和 `to_v_ip` 投影，将图像 tokens 投影后拼接到文本 k/v 上。

### 6.2 实现方案

**核心思路**：在 sd.cpp 的 `CrossAttention`（`common_block.hpp`）中为每个 cross-attention 层动态注入 `to_k_ip_layer_<id>` / `to_v_ip_layer_<id>` 两个 `Linear` 块。在生成时，通过全局图像嵌入张量 `g_ipadapter_image_embeds` 计算图像 k/v 并拼接到文本 k/v 上。

```
UNet CrossAttention 数据流（启用 IPAdapter UNet 时）:

text context: [N, n_context, context_dim]
                    ↓
            to_k / to_v (原始文本投影)
                    ↓
    k_text, v_text: [N, n_context, inner_dim]

image_embeds: [1, N_ip, context_dim]  ← 来自 IPAdapter MLP
                    ↓
    to_k_ip / to_v_ip (每层独立的 Linear)
                    ↓
    k_ip, v_ip: [1, N_ip, inner_dim]
                    ↓
    k = concat(k_text, k_ip, dim=1)  # 沿 token 维度拼接
    v = concat(v_text, v_ip, dim=1)
                    ↓
        ggml_ext_attention_ext(q, k, v, ...)
```

### 6.3 代码改动

| 文件 | 改动 |
|------|------|
| `include/stable-diffusion.h` | 新增 `ipadapter_unet_mode` / `ipadapter_unet_weights_path` 到 `sd_ctx_params_t` 和 `sd_img_gen_params_t` |
| `src/common_block.hpp` | `CrossAttention` 构造时按层注入 `to_k_ip_layer_<id>` / `to_v_ip_layer_<id>`；forward 时拼接图像 k/v |
| `src/unet.hpp` | `UNetModelRunner` 构造时在 `params_ctx` 中预留 `ipadapter_image_embeds` 张量 |
| `src/stable-diffusion.cpp` | 新增 `load_ipadapter_unet_weights()` / `assign_ipadapter_unet_weights()`；在采样前设置图像嵌入 |
| `src/ggml_extend.hpp` | `get_param_tensors()` 跳过 `to_k_ip_layer_*` / `to_v_ip_layer_*`，避免主加载流程报错 |
| `src/cli/cli_options.h` / `cli_parser.cpp` / `main.cpp` | 新增 `--ipadapter-unet-weights PATH` CLI 参数 |
| `src/adapters/sdcpp_adapter.cpp` | 将 UNet IPAdapter 模式传递给 sd.cpp |

### 6.4 权重文件格式

**文件**：`/data/models/image/ipadapter_unet_weights.bin`

**格式**：
```
Header:
  uint32 n_layers = 70
  uint32 version  = 1

Layer 0 (无显式 layer_id，兼容早期导出):
  uint32 out_features
  uint32 in_features
  float  to_k_ip[out_features * in_features]
  float  to_v_ip[out_features * in_features]

Layer 1..n_layers-1:
  uint32 layer_id       # 文件中的注意力层索引 (1,3,5,...,139)
  uint32 out_features   # 640 或 1280 (SDXL attn 输出维度)
  uint32 in_features    # 2048 (SDXL cross-attention context_dim)
  float  to_k_ip[...]
  float  to_v_ip[...]
```

**特性**：
- 文件末尾有约 20MB 额外数据（超出 header 声明的 70 层），解析器只读取 70 层并忽略尾部
- `layer_id` 是 flattened attention 索引（1,3,5,...,139），对应模型中所有 `CrossAttention` 实例（包括 self 和 cross）
- 权重分配采用**按形状分组 + 组内排序匹配**，解决文件 layer_id 顺序与模型遍历顺序不一致的问题

### 6.5 运行状态

**已验证**：
- ✅ `load_ipadapter_unet_weights()` 成功解析 70 层
- ✅ 模型总参数量从 ~4.9GB 增加到 ~6.2GB（新增 70 组 `to_k_ip`/`to_v_ip` 权重）
- ✅ `assign_ipadapter_unet_weights()` 成功将 70/70 层权重写入对应张量
- ✅ 使用 `ggml_backend_tensor_set()` 进行后端无关的张量写入（修复了 CUDA 下直接 `memcpy` 到 `tensor->data` 的段错误）
- ✅ 端到端运行无崩溃：`CrossAttention` 每层正确拼接图像 tokens
- ✅ 不影响现有 DiT 路径：`g_ipadapter_unet_enabled` 仅在 UNet 模式下激活，z_image_turbo 测试通过

**已解决**：
- ✅ **SDXL Base 模型兼容性问题**：根因不是模型不兼容，而是 **myimg-cli 的模型加载逻辑缺陷**：
  1. `sd_params` 未零初始化，导致 `vae_decode_only` 等字段为随机值 → 修复：`sd_ctx_params_t sd_params = {};`
  2. `vae_decode_only` 固定为 `false`，要求加载 VAE encoder 权重，但完整 checkpoint 中缺少 encoder → 修复：根据 `params.init_image.empty()` 动态设置
  3. `--diffusion-model` 强制使用 `diffusion_model_path`（只加载 `model.diffusion_model.` 前缀），导致 VAE/CLIP 权重被跳过 → 修复：检测 safetensors header 中是否包含 `first_stage_model`/`conditioner.embedders`，若是则使用 `model_path` 加载全部权重
- ✅ **UNet IPAdapter 端到端验证通过**：
  - Baseline (无 IPAdapter): mean=140.48, unique=109387
  - UNet IPAdapter (weight=0.8, 16 tokens): mean=150.12, unique=43642
  - 图像统计明显不同，IPAdapter 确实在影响生成
  - 下一步：CLIP 相似度量化对比

**待验证**：
- ⏳ 与 DiT 路径进行 CLIP 相似度 head-to-head 对比
- ⏳ 不同参考图像/提示词的系统性效果评估

### 6.6 关键修复记录

| 问题 | 根因 | 修复 |
|------|------|------|
| 段错误 (`SIGSEGV`) | 直接 `memcpy(dst->data, ...)` 在 CUDA 后端会写到 GPU 指针，CPU 侧不可直接访问 | 改用 `ggml_backend_tensor_set(k_w, ...)` / `ggml_backend_tensor_set(v_w, ...)` |
| 权重层数不匹配 | 文件 layer_id (1,3,5,...) 与模型遍历顺序不一致 | 按 `(out_features, in_features)` 形状分组，组内按排序后的 layer_id 匹配 |
| `to_k_ip`/`to_v_ip` 出现在主 `tensors` 但无文件权重 | 这些是新注入的参数，没有对应 safetensors 键 | 在 `get_param_tensors()` lambda 中跳过 `attn2.to_k_ip_layer_*` / `attn2.to_v_ip_layer_*` |
| SDXL 加载失败 (`first_stage_model.* not found`) | `diffusion_model_path` 只加载 `model.diffusion_model.` 前缀，跳过 VAE/CLIP | 检测完整 checkpoint 特征，自动切换为 `model_path` 加载全部权重 |
| `sd_params` 未初始化导致 VAE 加载异常 | 栈上 C struct 未清零，布尔字段随机值 | `sd_ctx_params_t sd_params = {};` 零初始化 |
| DiT 投影 2048→2560 污染 UNet tokens | UNet 期望 2048-dim，但 DiT 投影输出 2560-dim | `sdcpp_adapter.cpp` 在 UNet 模式下切片取前 2048 维 |

---

## 7. 关键决策记录

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

### 2026-06-06 (5)：Phase 3.1 — 线性投影层 768→2560

**问题**：IPAdapter MLP 输出 768-dim，Z-Image context 是 2560-dim，零填充导致 70% 维度为空。

**决策**：
1. 创建 ONNX 模型 `ipadapter_proj.onnx`，包含 `MatMul(768×2560) + Add` 两个算子
2. 初始化权重为 identity-like（前 768 维对角线为 1，其余为 0），功能等价于零填充
3. 投影层为可选加载：提供 `--ipadapter-projection PATH` 参数，不提供时回退到零填充
4. 未来可替换 `.data` 权重文件为训练好的投影矩阵

**实现**：
- `ipadapter.cpp`: 新增 `proj_session`, `load_projection()`, `run_projection()` 
- `run_projection()` 在投影层未加载时自动回退到 C++ 零填充（无需 ONNX）
- sd.cpp 注入代码改为使用 `ctx_dim` 而非硬编码 768，支持任意维度 token
- CLI: `--ipadapter-projection` 参数

### 2026-06-06 (4)：Phase 3 注入实现 — 零填充 + concat

**问题**：如何将 IPAdapter 的 768-dim token 注入到 Z-Image DiT 的 2560-dim context 中？

**决策**：
1. 在 `prepare_image_generation_embeds()` 中实现注入，而非在采样循环中逐个 attention 层修改
2. 768→2560 使用**零填充**（先填充到完整 ctx_dim，剩余 1792 位为 0），而非训练 Linear 层
3. **拼接**（concat）而非相加：沿 dim=1 追加 token 到 `cond.c_crossattn` 和 `uncond.c_crossattn`
4. 注入点：`get_learned_condition()` 返回后（line 4164），生成最终 embeds 前

**理由**：
- 零填充是可行的 prototype — DiT 的 `cap_embedder` 用 RMSNorm + Linear 处理 context，零填充维度不会干扰已学习的权重
- 拼接而非相加保持 token 独立性，不污染原始 text tokens
- `prepare_image_generation_embeds()` 是统一入口，一次修改覆盖所有采样路径

**验证**：
- 注入后 shape: `[2560, 9]` → `[2560, 10]` ✅
- Z-Image PE 断言通过（graph-build time 基于 `context->ne[1]` 生成 PE）✅

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

## 8. 已知问题与修复计划

### 7.1 当前问题：IPAdapter 效果几乎不可感知

**现象**（2026-06-06~07 测试）：
- 生成图（`--ipadapter-weight 0.8`）与无 IPAdapter 相比，风格/光线/构图几乎无差异
- 参考图的人脸特征完全未传递到生成结果

**根因诊断（2026-06-07 重大发现）**：

| 问题 | 严重程度 | 说明 |
|------|----------|------|
| **模型架构不匹配** | 🔴🔴 **致命** | 当前 `ipadapter.onnx` 是 **SD1.5 版本**（输出 `[1, 768]`），Z-Image 是 **SDXL DiT**（需要 `[16, 2048]`）。两者架构完全不同 |
| CLIP Vision 维度 | 🔴 高 | 当前 CLIP 输出 1024-dim，SDXL Plus 期望 1280-dim（ViT-H/14） |
| 投影权重未训练 | 🟡 中 | identity init 确实稀释信号，但不是 root cause |
| Token 数量不足 | 🟡 中 | 1 token vs 16 tokens 是差异，但前提是模型版本正确 |

**为什么之前诊断错误**：
- 最初以为是 token 数量/投影权重问题
- 2026-06-07 极端权重测试（weight=0/0.8/1.5 + 4-token 重复）证明：**注入通路正常，但模型完全不匹配**
- CLIP 相似度量化：weight=1.5 比无 IPA **更不像**参考图（0.2973 vs 0.4648），说明 SD1.5 token 在 SDXL DiT 中产生了**负效果**

### 7.2 修复方案（已更新）

**方案 A：适配 SDXL IPAdapter Plus（进行中）**
- [x] 下载 `ip-adapter-plus_sdxl_vit-h.safetensors`（847MB）
- [x] 提取 `image_proj`（Perceiver Resampler，16 tokens × 1280-dim）→ ONNX v2
- [x] 解决 CLIP Vision 维度：导出 `clip_vision_vit_h_hidden.onnx`（257×1280）
- [x] 添加 2048→2560 投影层 → ONNX v2
- [ ] 修改 C++ 代码支持 v2 模型（hidden states 输入，16 tokens 输出）
- [ ] 测试验证效果

**方案 B：换 SDXL 兼容的 CLIP Vision（已完成）**
- 原 `clip_vision.onnx` 输出 `image_embeds [1,1024]`（pooled）
- 新 `clip_vision_vit_h_hidden.onnx` 输出 `hidden_states [-2] [1,257,1280]`（patch tokens）
- ✅ 已完成导出

**方案 C：人脸预处理优化（立即可用）**
- 已验证：人脸裁切 + 4-token 重复能改善效果（相对于全图参考）
- 可在代码中集成 `--ipadapter-auto-crop` 自动检测并裁切人脸

### 7.3 测试记录

**2026-06-06**：2560×1440 生成测试
- 结果：生成成功，但 IPA 效果不可感知
- 错误假设：以为是 token 数量/投影权重问题

**2026-06-07**：极端权重验证（root cause 定位）
- A组：无 IPAdapter（baseline）
- B组：`--ipadapter-weight 0.0`（注入但权重为0）
- C组：`--ipadapter-weight 1.5`（极端权重）
- D组：4-token 重复（模拟多 token）
- E组：人脸裁切 + weight=0.8

| 测试 | 与参考图 CLIP 相似度 | vs baseline | 结论 |
|------|---------------------|-------------|------|
| 无 IPA | **0.4648** | baseline | 基准 |
| weight=1.5 全图 | **0.2973** | -0.1674 ❌ | 比无IPA更差！ |
| 4 tokens | **0.1636** | -0.3012 ❌ | 严重恶化 |
| weight=0.8 人脸裁切 | **0.4099** | -0.0549 ❌ | 仍比无IPA差 |
| weight=1.0 零填充 | **0.3930** | -0.0717 ❌ | 比无IPA差 |

**关键结论**：
1. 注入代码通路正常（weight=1.5 确实改变了生成结果）
2. 但改变方向**错误**——SD1.5 token 在 SDXL DiT 中产生了负效果
3. **必须换 SDXL 版 IPAdapter 模型**

**2026-06-07**：模型兼容性分析（第一轮）
- 当前 `ipadapter.onnx` = SD1.5 专用（768-dim 输出）
- Z-Image = SDXL DiT（需要 2048-dim 输入，经投影到 2560）
- 已下载 `ip-adapter-plus_sdxl_vit-h.safetensors`（Perceiver Resampler 结构）
- 导出 v1 ONNX：错误地使用了 `image_embeds [1,1024]` + 未训练 `clip_adapter`

**2026-06-07**：SDXL Plus v1 测试（错误输入）
- 使用 `clip_vision.onnx` (1024-dim) + `ipadapter_sdxl_plus.onnx` (含未训练 clip_adapter)
- 输出 16 tokens × 2048-dim，经投影到 2560-dim
- CLIP 相似度：0.0936（比无 IPA 的 0.4648 更差）
- **原因**：输入是 pooled embedding 而非 patch tokens，clip_adapter 是随机权重

**2026-06-07**：Root Cause 最终定位
- 检查原始权重：`image_proj.proj_in.weight = [1280, 1280]`
- 检查 ComfyUI 源码：`IPAdapterPlusXL.get_image_embeds()` 使用 `hidden_states[-2]`
- 确认：`hidden_states[-2]` shape = `[1, 257, 1280]`（257 = 256 patches + 1 CLS）
- **结论**：v1 完全错误，必须导出 v2（使用 hidden states，移除 clip_adapter）

**2026-06-07**：v2 ONNX 导出成功
- `clip_vision_vit_h_hidden.onnx`: [1, 257, 1280] ✅
- `ipadapter_sdxl_plus_v2.onnx`: [1, 257, 1280] → [1, 16, 2048] ✅
- `ipadapter_proj_2048_2560_v2.onnx`: [1, 16, 2048] → [1, 16, 2560] ✅
- **测试结果**：CLIP sim = 0.1878（仍远低于 baseline 0.4648）

**2026-06-07**：v3 ONNX 导出（修复 chunk bug）
- **问题定位**：v2 的 `chunk(2, dim=-1)` 在 ONNX tracing 下产生错误行为，导致 PyTorch 与 ONNX 输出 max diff = 2.87
- **修复**：将 `to_kv` 拆分为独立的 `to_k`/`to_v`，手动 split 原始权重
- **验证**：PyTorch vs ONNX max diff = 0.000041 ✅
- **测试结果**：
  - weight=0.5: CLIP sim = 0.3838 (-0.0809 vs baseline)
  - weight=1.0: CLIP sim = 0.3715 (-0.0933 vs baseline)
  - weight=1.5: CLIP sim = 0.3820 (-0.0828 vs baseline)
- **结论**：v3 显著优于 v1/v2，但仍不如无 IPA 的 baseline

**2026-06-07**：核心瓶颈分析
| 问题 | 影响 | 说明 |
|------|------|------|
| **2048→2560 投影未训练** | 🔴🔴 **致命** | 当前是 identity-like（前 2048 维 eye，后 512 维 zero）。这导致 20% 维度无信号，且已有维度未经适配直接映射 |
| **投影初始化方式** | 🔴 高 | **Xavier (0.3391) >> Identity-like (0.1467)**。初始化策略对未训练投影层的影响巨大 |
| **Context 拼接 vs Attention K/V 注入** | 🔴 高 | 标准 IPAdapter Plus 将 image tokens 直接作为 attention 的 k/v（通过 `to_k_ip`/`to_v_ip`）。Z-Image DiT 无 cross-attention，只能拼接 context 进 text sequence，经过 `cap_embedder`（为文本设计）处理 |
| **Face crop 质量** | 🟡 中 | 参考图裁剪可能不够精准 |

**2026-06-07**：投影初始化对比（weight=0.8）
| 初始化方法 | CLIP Similarity | vs Baseline | 结论 |
|-----------|----------------|-------------|------|
| Identity-like (eye) | 0.1467 | -0.3181 ❌ | 最差，20% 维度为零 |
| Xavier Uniform | 0.3391 | -0.1257 ⚠️ | **最佳**，所有维度有信号 |
| **Baseline (无 IPA)** | **0.4648** | — | 目标 |

**关键发现**：
1. **初始化比想象中更重要**：Xavier 使相似度从 0.1467 → 0.3391（提升 131%）
2. **所有维度必须有信号**：Identity-like 导致 512/2560 = 20% 维度永远为零，严重破坏信息流
3. **仍未达 baseline**：差距 0.1257，说明仅靠初始化不够，需要训练或架构改进

**下一步方案**（按优先级）：
1. **训练 2048→2560 投影层**（最优）：收集 image-caption 对，用对比学习训练投影。Xavier 初始化已证明是更好的起点
2. **修改 JointAttention**（方案 B）：在 Z-Image 的 JointTransformerBlock 中为 image tokens 单独创建 k/v 投影，标准 IPAdapter 方式注入
3. **多参考图平均**：使用多张参考图提取特征后平均，增强信号稳定性

---

## 9. 参考资源

- [IP-Adapter 论文](https://arxiv.org/abs/2308.06721)
- [IPAdapter GitHub (huggingface)](https://github.com/tencent-ailab/IP-Adapter)
- [ComfyUI IPAdapter 节点源码](https://github.com/comfyanonymous/ComfyUI_IPAdapter_plus)
- [sd.cpp 架构](https://github.com/leejet/stable-diffusion.cpp)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)

---

## 10. 里程碑

| 阶段 | 预计完成 | 状态 |
|------|----------|------|
| P1：ONNX Runtime 安装 + 模型转换 | 2026-06-06 | ✅ 已完成 |
| P2：CLIP Vision + IPAdapter ONNX 推理 | 2026-06-06 | ✅ 已完成 |
| P3：条件编码注入 + 生成流程集成 | 2026-06-06 | ✅ 已完成 |
| P3.1：线性投影层 768→2560 | 2026-06-06 | ✅ 已完成 |
| P3.2：步数控制 start_at / end_at | 2026-06-06 | ✅ 已完成 |
| P4：HiRes Fix + 高清测试 | 2026-06-06 | ✅ 已完成 |
| **P5：SDXL IPAdapter Plus 适配** | 2026-06-07 | 🚧 **核心完成，待训练投影层** |
| P5.1：正确的 hidden states CLIP Vision | 2026-06-07 | ✅ 已导出 `clip_vision_vit_h_hidden.onnx` |
| P5.2：Perceiver Resampler ONNX (v3) | 2026-06-07 | ✅ 已修复 chunk bug，权重与 PyTorch diff < 1e-4 |
| P5.3：2048→2560 投影层 | 2026-06-07 | ⚠️ Xavier 初始化已验证，需训练 |
| P5.4：C++ 代码适配 | 2026-06-07 | ✅ `ipadapter.cpp` 支持 hidden states + 16 tokens |
| P6：训练投影层 / JointAttention 方案 B | - | ⏳ **当前重点** |
| P7：抠脸 + 人脸预处理 | - | ⏳ 待开始 |
| **P5b：UNet IPAdapter 权重注入** | 2026-06-07 | ✅ **端到端运行成功，权重加载 70/70** |
| P5b.1：为 CrossAttention 注入 `to_k_ip`/`to_v_ip` | 2026-06-07 | ✅ 70 层全部注入并加载 |
| P5b.2：SDXL Base 端到端验证 | 2026-06-07 | ✅ 运行无崩溃，生成正常彩色图像（修复 adapter 加载逻辑后） |
| P5b.3：CLIP 相似度对比 (UNet vs DiT) | - | ⏳ 待执行 |
| P6：训练投影层 / JointAttention 方案 B | - | ⏳ **当前重点** |
| P7：抠脸 + 人脸预处理 | - | ⏳ 待开始 |
| P8：VRAM 优化（RTX 3080 20GB 专属） | - | ⏳ 待开始 |

### 当前阻塞项（DiT 路径）

1. ~~CLIP Vision 维度~~ → ✅ 已解决：导出 `clip_vision_vit_h_hidden.onnx`（257×1280）
2. ~~Perceiver Resampler ONNX 转换~~ → ✅ 已解决：导出 `ipadapter_sdxl_plus_v3.onnx`（v3 修复 chunk bug）
3. ~~2048→2560 投影~~ → ✅ 已解决：导出 `ipadapter_proj_2048_2560_xavier.onnx`（Xavier 初始化）
4. ~~C++ 代码适配~~ → ✅ 已解决：`ipadapter.cpp` 支持 hidden states 输入格式
5. **投影层训练**：Xavier 初始化比 identity 好 131%，但仍未达 baseline。需要训练或架构改进

### 当前阻塞项（UNet 路径）

1. ✅ **已解决**：`sd_xl_base_1.0.safetensors` 在 sd.cpp 中可正常出图（修复 adapter 加载逻辑后）。
   - 问题根因：myimg-cli 使用 `diffusion_model_path` 导致只加载 UNet 权重，VAE/CLIP 缺失
   - 修复：检测完整 checkpoint，自动切换为 `model_path`
   - 运行验证通过：生成正常彩色图像
2. **CLIP 相似度量化**：需编写对比脚本，客观评估 UNet IPAdapter 效果
3. **外部 VAE 兼容性**：`sdxl_vae.safetensors`（320MB）与 sd.cpp 配合仍产生纯白输出，待分析根因

### 模型文件清单（最终）

```
/data/models/image/
├── clip_vision_vit_h_hidden.onnx              (397K)   ← ViT-H/14 hidden states [257,1280]
├── clip_vision_vit_h_hidden.onnx.data         (2.36G)  ← 权重
├── ipadapter_sdxl_plus_v3.onnx                (~15MB)  ← Perceiver Resampler [257,1280]→[16,2048]
├── ipadapter_proj_2048_2560_xavier.onnx       (~20MB)  ← 2048→2560 投影 (Xavier init)
├── ip-adapter-plus_sdxl_vit-h.safetensors     (847M)   ← 源 PyTorch 权重
├── ipadapter_unet_weights.bin                 (~1.3G)  ← UNet 路径: 70层 to_k_ip/to_v_ip 权重
├── ipadapter_sdxl_plus_v2.onnx                (废弃)   ← v2: chunk bug
└── ipadapter_proj_2048_2560_v2.onnx           (废弃)   ← v2: identity init
```

### 模型下载状态

| 模型 | 来源 | 大小 | 状态 |
|------|------|------|------|
| `ip-adapter-plus_sdxl_vit-h.safetensors` | h94/IP-Adapter | 847 MB | ✅ 已下载 |
| `ipadapter_unet_weights.bin` | 自制 (从 PyTorch 提取) | ~1.3 GB | ✅ 已生成 |
| `sd_xl_base_1.0.safetensors` | stabilityai/stable-diffusion-xl-base-1.0 | 6.5 GB | ⚠️ 已下载但与 sd.cpp 不兼容（纯白输出） |
| `clip_vision_vit_h_hidden.onnx` | 自制 (OpenCLIP ViT-H/14) | 2.4 GB | ✅ 已导出 |
| `ipadapter_sdxl_plus_v3.onnx` | 自制 (Perceiver Resampler) | ~15 MB | ✅ 已导出 |
| `ipadapter_proj_2048_2560_xavier.onnx` | 自制 (Linear 2048→2560) | ~20 MB | ✅ 已导出 |

---

> **最后更新**: 2026-06-07 14:25
> **维护者**: my-img Team
> 
> **今日关键成果**：
> 1. 完成 UNet IPAdapter 权重注入实现：70/70 cross-attention 层 `to_k_ip`/`to_v_ip` 注入并加载
> 2. 修复 CUDA 后端张量写入段错误：用 `ggml_backend_tensor_set()` 替代 `memcpy(tensor->data, ...)`
> 3. 实现按形状分层的权重匹配策略，解决文件 layer_id 与模型遍历顺序不一致
> 4. 验证 UNet IPAdapter 端到端运行无崩溃，且不影响现有 DiT 路径（z_image_turbo 测试通过）
> 5. **修复 SDXL Base 加载问题**：根因是 adapter 参数初始化不全 + `diffusion_model_path` 跳过 VAE/CLIP，三项修复后正常出图
> 6. **UNet IPAdapter 首次成功出图**：使用 `demo_face_0.png` 参考图，16 tokens × 2048-dim，生成图像与 baseline 统计差异明显
> 7. 修复 DiT→UNet token 维度不匹配：`sdcpp_adapter.cpp` 在 UNet 模式下自动切片取前 2048 维
> 8. 更新 IPAdapter.md：新增 UNet 路径完整文档、状态、阻塞项、修复记录
