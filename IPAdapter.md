# IPAdapter：C++ 原生实现（sd.cpp 后端）

> **目标**：在 my-img（纯 C++ ComfyUI）中实现 IPAdapter 图像提示词功能
> **后端**：stable-diffusion.cpp（GGML/CUDA）
> **模型**：Z-Image Turbo（SDXL 架构，Flow Matching → **DiT**）
> **当前状态**：🚧 Phase 3 全部完成（注入 + 步进控制 ✅），Phase 4 HiRes Fix 集成验证通过 ✅。下一步：**IPAdapter Face ID 人脸克隆**
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

#### 基础 IPAdapter（style transfer，当前可用）

```
/data/models/image/
├── ipadapter.onnx                (13K)     ← ONNX 头文件（外部数据格式）
├── ipadapter.onnx.data           (5.4M)    ← ONNX 权重（2 层 MLP: 1024→768→768）
├── ipadapter_proj.onnx           (0.3K)    ← 线性投影 ONNX 头文件 ✅
├── ipadapter_proj.onnx.data      (15M)     ← 投影权重（MatMul 768×2560 + Bias，identity init）
├── clip_vision.onnx              (397K)    ← CLIP Vision ONNX 头文件 ✅
├── clip_vision.onnx.data         (2.36G)   ← CLIP Vision ONNX 权重 ✅
```

#### IPAdapter Face ID（人脸克隆，待集成）

```
/data/models/image/
├── inswapper_128.onnx            (529M)    ← Face Swap 模型（已有 ✅）
├── yunet_320_320.onnx            (5.8M)    ← 人脸检测（已有 ✅）
```

**需要补充的模型**（从 `insightface` Python 包提取或转换）：

| 模型 | 来源 | 用途 | 状态 |
|------|------|------|------|
| `w600k_r50.onnx` | insightface buffalo_l | ArcFace 人脸特征提取 (512-dim) | 📦 已下载到 `/tmp/insightface_models/` |
| `det_10g.onnx` | insightface buffalo_l | 人脸检测 | 📦 已下载到 `/tmp/insightface_models/` |
| `ip-adapter-faceid_sd15.safetensors` | hf.co/h94/IP-Adapter-FaceID | Face ID MLP 权重 | ❌ 待获取 |
| Face ID → ONNX 转换 | Python 脚本 | Face ID MLP ONNX 推理 | ❌ 待转换 |

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

## 7. 已知问题与修复计划

### 7.1 当前问题：IPAdapter 效果几乎不可感知

**现象**（2026-06-06 测试）：
- 生成图（`--ipadapter-weight 0.8`）与无 IPAdapter 相比，风格/光线/构图几乎无差异
- 参考图的人脸特征完全未传递到生成结果

**根因诊断**：

| 问题 | 严重程度 | 说明 |
|------|----------|------|
| **Token 数量不足** | 🔴 高 | 当前 `ipadapter.onnx` 输出 `[1, 768]`，标准 IPAdapter 用 4-16 个 token 才能编码足够图像信息 |
| **投影权重未训练** | 🔴 高 | 768→2560 投影是 identity init，1792/2560 维为 0，信号被严重稀释 |
| **Token 数与 PE 兼容性** | 🟡 中 | Z-Image DiT PE 在 graph-build time 生成，增加 token 数需验证 PE 是否仍正确 |
| **注入代码通路验证** | 🟡 中 | 尚未做 weight=1.5 极端测试，无法 100% 确认注入代码是否真正生效 |

**为什么不是 Face ID 的问题**：
- 基础 IPAdapter 本应能传递**整体风格/光线/构图**，即使不做 Face ID
- 当前效果弱到连风格都无法感知，说明是注入信号本身太弱，不是 Face ID 缺失

### 7.2 修复方案（优先级排序）

**方案 1：换多 token 的 IPAdapter 模型（高优先级）**
- 目标：找输出 `[N, 768]`（N≥4）的 IPAdapter ONNX 模型替代当前 `[1, 768]`
- 来源：ComfyUI IPAdapter Plus 模型、Diffusers IPAdapter  checkpoint
- 验证：token 数增加后，Z-Image PE 断言是否仍通过

**方案 2：训练投影权重（高优先级）**
- 目标：让 768→2560 投影真正学会映射，而非 identity init
- 方法：收集 50-100 张参考图+生成图对，用对比学习或 MSE loss 训练投影矩阵
- 输入：CLIP Vision 输出 [1024] → IPAdapter MLP → [768] → 投影 → [2560]
- 输出：与目标图像的 CLIP 特征对齐
- 替代：直接用 PyTorch 训练，导出 ONNX 替换 `.data` 文件

**方案 3：极端权重测试（立即可做）**
- 命令：`--ipadapter-weight 1.5`（超出正常范围）
- 目的：确认注入代码通路是否真的在生效
- 如果 1.5 仍无效果 → 注入代码有 bug，需排查
- 如果 1.5 有明显效果 → 只是信号弱，需方案 1/2

### 7.3 测试计划

```bash
# 方案 3：极端权重验证（5 分钟）
./myimg-cli \
  --diffusion-model ... \
  --ipadapter --ipadapter-weight 1.5 \
  --ipadapter-image ref.png \
  -p "a photo" -W 640 -H 384 --steps 5 \
  -o test_extreme_weight.png

# 对照组：无 IPAdapter
./myimg-cli \
  --diffusion-model ... \
  -p "a photo" -W 640 -H 384 --steps 5 \
  -o test_no_ipa.png

# 方案 1：找多 token 模型后测试
# 待补充

# 方案 2：训练投影后测试
# 待补充
```

### 7.4 历史测试记录

**2026-06-06**：2560×1440 生成测试通过，但 IPA 效果不可感知
- 模型：`z-image-turbo-Q6_K.gguf`
- 参考图：`~/demo.png` (897×950)
- Prompt: `solo,single woman,half body portrait...`
- 参数：`--ipadapter-weight 0.8`, `--ipadapter-start 0.0`, `--ipadapter-end 0.5`
- 结果：生成图与无 IPAdapter 对照组无明显差异
- 结论：信号太弱，需修复方案 1/2/3

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
| P3：条件编码注入 + 生成流程集成 | 2026-06-06 | ✅ 已完成（零填充投影 + 无步数控制） |
| P3.1：线性投影层 768→2560 | 2026-06-06 | ✅ 已完成 |
| P3.2：步数控制 start_at / end_at | 下一轮 | ⏳ 待开始 |
| P4：HiRes Fix + 高清测试 | - | ⏳ 待开始 |
| P5：抠脸 + 参数调优 | - | ⏳ 待开始 |

---

> **最后更新**: 2026-06-06
> **维护者**: my-img Team
