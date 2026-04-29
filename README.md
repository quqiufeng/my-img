# my-img - 纯 C++ 版 ComfyUI

> 📚 **项目文档索引**
>
> | 文档 | 说明 |
> |------|------|
> | **📖 [design.md](design.md)** | 架构设计、技术细节、实现路线图 |
> | **🔧 [SD_INTEGRATION.md](SD_INTEGRATION.md)** | stable-diffusion.cpp 集成方式、更新维护流程 |
> | **🚀 [HIRES_FIX_LIBTORCH.md](HIRES_FIX_LIBTORCH.md)** | libTorch 版 HiRes Fix 高清出图原理与实现 |
> | **⚡ [SD_UPGRADE_GUIDE.md](SD_UPGRADE_GUIDE.md)** | sd.cpp 升级后适配指南、AI 辅助检查流程 |
> | **📋 [task.md](task.md)** | 开发任务表、功能完整度对照、进度追踪 |
> | **📜 [AGENTS.md](AGENTS.md)** | 项目开发规范、C++ 代码规范、Git 工作流 |
> | **🤖 [claude.md](claude.md)** | Claude AI 使用说明（项目内部） |

---

## 项目简介

**my-img** 是一个纯 C++ 实现的 ComfyUI 替代方案，旨在彻底摆脱 Python 依赖，提供一个轻量、高效、可独立部署的 AI 图像生成工具。

### 设计初衷

在使用 ComfyUI 的过程中，我们发现 Python 环境带来了很多痛点：

1. **环境依赖复杂**：需要 Python 3.10+、PyTorch、transformers、diffusers 等数百个依赖包，环境配置容易出错
2. **启动速度慢**：Python 解释器 + 大量模块导入，启动需要数秒甚至更久
3. **部署困难**：需要 conda/venv 环境，Docker 镜像体积巨大（>10GB）
4. **GIL 限制**：Python 全局解释器锁限制了多线程性能
5. **内存开销大**：Python 对象头开销显著，同样模型占用更多内存

**my-img 的目标**：保留 ComfyUI 的所有生成能力，但用纯 C++ 重写，实现：
- **零 Python 依赖** - 不嵌入 Python 解释器，不依赖任何 Python 包
- **快速启动** - 直接运行二进制文件，毫秒级启动
- **轻量部署** - 单个可执行文件 + 模型文件即可运行
- **高效内存** - C++ 对象紧凑，内存占用显著降低
- **完整功能** - txt2img、img2img、HiRes Fix、LoRA、ControlNet、IPAdapter 等

---

## 核心架构

### 混合架构：sd.cpp + libtorch

my-img 采用混合架构，结合了 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 和 [libtorch](https://pytorch.org/cppdocs/) 两者的优势：

```
用户输入（CLI 参数 / JSON 工作流）
    │
    ▼
┌─────────────────────────────────────┐
│  my-img CLI / Workflow Engine       │  ← C++ 工作流编排
│  - 参数解析                         │
│  - 节点调度                         │
│  - 图像 I/O                         │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│ SDCPPAdapter     │  │ libtorch 扩展    │
│ (适配层)         │  │ (图像处理等)     │
└────────┬─────────┘  └──────────────────┘
         │
    ┌────┴─────────────────────────────────┐
    ▼                                      ▼
┌──────────────────────┐      ┌──────────────────────┐
│ stable-diffusion.cpp │      │ libtorch (PyTorch C++)│
│ (GGML 推理引擎)      │      │ (张量计算库)          │
│                      │      │                      │
│ - 模型加载           │      │ - 张量操作           │
│ - GGUF 量化推理      │      │ - 图像预处理         │
│ - 文本编码           │      │ - 高级功能扩展       │
│ - 采样器             │      │ - 自定义算子         │
│ - VAE 编解码         │      │                      │
└──────────────────────┘      └──────────────────────┘
```

### 为什么这样设计？

#### 1. stable-diffusion.cpp（推理引擎）

**用途**：负责所有 AI 模型的加载和推理

**为什么选择它**：
- ✅ **成熟稳定**：完整支持 Z-Image、Flux、SDXL、SD3 等主流模型
- ✅ **GGUF 原生支持**：直接加载量化模型，无需转换
- ✅ **高效推理**：基于 GGML 框架，支持 CPU/GPU 混合推理
- ✅ **显存友好**：Q4_K/Q5_K 量化显著降低显存占用（RTX 3080 10GB 可跑 2560x1440）
- ✅ **C API**：提供稳定 C 接口，易于 C++ 封装

**在 my-img 中的角色**：
- 扩散模型加载与推理（DiT/UNet）
- 文本编码器（CLIP/Qwen3/T5）
- VAE 编码器/解码器
- 采样器实现（Euler、DPM++、Heun 等 15 种）
- 噪声调度器（Discrete、Karras、AYS 等 11 种）
- LoRA 权重注入
- ControlNet 推理
- ESRGAN 图像放大

#### 2. libtorch（PyTorch C++ 前端）

**用途**：提供张量计算和图像处理能力，用于扩展功能

**为什么选择它**：
- ✅ **完整 PyTorch 生态**：支持自动求导、自定义算子、CUDA 加速
- ✅ **图像处理**：torchvision 风格的图像变换、张量操作
- ✅ **高级功能**：IPAdapter 特征提取、ControlNet 预处理等
- ✅ **与 Python 互通**：可加载 Python 训练的模型权重

**在 my-img 中的角色**：
- 图像预处理（缩放、裁剪、归一化）
- 张量工具函数
- 未来扩展：IPAdapter 图像特征提取、ControlNet 预处理器
- 自定义采样器实现

#### 3. SDCPPAdapter（适配层）

**用途**：隔离 sd.cpp 的 C API，提供类型安全的 C++ 接口

**设计原则**：
- **封装隔离**：所有 sd.cpp C API 调用都通过适配层
- **RAII 资源管理**：自动管理 sd_ctx_t 生命周期
- **类型转换**：C 结构体 ↔ C++ 对象（Image、GenerationParams）
- **版本兼容**：sd.cpp 升级时只需修改适配层实现

```cpp
// 使用示例
myimg::SDCPPAdapter adapter;
myimg::GenerationParams params;
params.diffusion_model_path = "model.gguf";
params.prompt = "a beautiful landscape";
params.width = 1280;
params.height = 720;

adapter.initialize(params);
myimg::Image image = adapter.generate_single(params);
image.save_to_file("output.png");
```

---

## 功能模块

### 已完成功能 ✅

#### 1. txt2img（文本到图像）
- 完整文本编码 → 扩散去噪 → VAE 解码管线
- 支持 Z-Image（GGUF）、Flux、SDXL 等模型
- 支持 Qwen3/CLIP/T5 文本编码器

#### 2. 采样方法（15 种）
- Euler、Euler Ancestral
- DPM++ 2M、DPM++ 2M v2、DPM++ 2S a
- Heun、DPM2
- IPNDM、IPNDM-V
- LCM、DDIM Trailing
- TCD、RES Multistep、RES 2S
- ER-SDE

#### 3. 调度器（11 种）
- Discrete、Karras、Exponential
- AYS、GITS、SGM Uniform
- Simple、Smoothstep
- KL Optimal、LCM、Bong Tangent

#### 4. HiRes Fix（高分辨率修复）
- 两阶段生成：低分辨率基础图 + latent 放大 refine
- 自动计算低分辨率（保持宽高比）
- 支持自定义 hires_strength 和 hires_steps
- 实测：1280×720 → 2560×1440，RTX 3080 10GB 可跑

#### 5. VAE Tiling（显存优化）
- 大分辨率图像分块解码
- 可配置 tile_size 和 overlap
- 解决 2560×1440+ 分辨率 OOM 问题

#### 6. Flash Attention
- 加速注意力计算
- 降低显存占用

#### 7. ESRGAN 放大
- 2×/4× AI 超分辨率
- 支持 tile 模式（省显存）

#### 8. 图像 I/O
- PNG/JPG 加载和保存
- 元数据嵌入（预留）

### 开发中功能 🚧

#### 1. img2img（图像到图像）
- 加载参考图
- 根据 strength 控制保留程度
- VAE encode → 加噪 → 去噪

#### 2. LoRA 集成
- 加载 LoRA 权重（.safetensors）
- 运行时权重注入
- 支持多个 LoRA 叠加

#### 3. Inpainting（局部重绘）
- 加载 mask 图像
- 保留未遮罩区域
- 仅重绘遮罩区域

#### 4. ControlNet
- ControlNet 模型加载
- Canny/Depth/Lineart/OpenPose 预处理
- 精确控制构图

### 计划中功能 📋

#### 1. IPAdapter（图像提示词）
- 参考图像风格/人脸迁移
- CLIP Vision 特征提取

#### 2. Workflow JSON 支持
- 解析 ComfyUI workflow JSON
- 批量生成
- 工作流复用

#### 3. Server 模式
- HTTP API 服务
- 兼容 SD WebUI API
- 队列管理

#### 4. 图像预处理节点
- Canny 边缘检测
- Depth 深度估计
- Lineart 线条提取

---

## HiRes Fix 深度解析

### 1. 为什么需要 HiRes Fix

扩散模型在训练时通常只见过 512×512 ~ 1024×1024 的图像。直接以 2560×1440 等高分辨率生成会导致：

- **多人症**：画面中出现多个重复的人物或物体
- **畸形五官**：面部特征扭曲、五官错位
- **结构崩坏**：肢体比例失调、物体变形
- **细节缺失**：纹理模糊、边缘不清晰

**HiRes Fix 的核心思想**：先在低分辨率（模型熟悉的尺寸）生成正确的构图和结构，然后在 latent 空间放大并 refine，在保留整体结构的同时补充高分辨率细节。

### 2. 工作原理

HiRes Fix 是一个**两阶段生成**流程：

```
阶段 1：基础生成（低分辨率）
  Prompt → Text Encoder → 扩散去噪 → VAE 解码
  输出：1280×720 基础图（构图正确、结构完整）

阶段 2：HiRes Refine（高分辨率）
  基础图 → VAE 编码回 Latent → Latent 空间放大 → 部分加噪 → 扩散去噪 → VAE 解码
  输出：2560×1440 高清图（保留结构 + 补充细节）
```

**Latent 空间放大**（而非像素空间）：
- 像素空间放大（如 bicubic）只是插值，不会增加真实细节
- Latent 空间放大后再次进行扩散去噪，AI 会"想象"出符合 prompt 的高分辨率细节
- 这就是 HiRes Fix 比简单放大更清晰、更真实的原因

### 3. 涉及代码位置

#### 3.1 脚本层：`img1.sh` / `img2.sh`

**分辨率计算逻辑**（`img1.sh:128-150`）：

```bash
# 目标分辨率直接作为 HiRes 目标，基础分辨率固定为一半
# 2560x1440 -> 基础 1280x720 (latent 160x90)
# 1920x1080 -> 基础 1024x576 (latent 128x72)
if [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    LOW_W=1280
    LOW_H=720
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    LOW_W=1024
    LOW_H=576
fi
```

**参数配置**（`img1.sh:88-94`）：
```bash
SAMPLING_METHOD="euler"
SCHEDULER="discrete"
CFG_SCALE="2.8"
STEPS="55"
HIRES_STEPS="25"
HIRES_STRENGTH="0.28"
```

**CLI 调用**（`img1.sh:174-196`）：
```bash
$SD_CLI \
  -W "$LOW_W" -H "$LOW_H" \
  --hires \
  --hires-width "$WIDTH" \
  --hires-height "$HEIGHT" \
  --hires-strength "$HIRES_STRENGTH" \
  --hires-steps "$HIRES_STEPS" \
  --diffusion-fa --vae-tiling
```

#### 3.2 CLI 入口：`src/main.cpp`

**参数解析**（`main.cpp:55-65`）：
```cpp
struct CliOptions {
    bool hires = false;
    int hires_width = 2560;
    int hires_height = 1440;
    float hires_strength = 0.30f;
    int hires_steps = 60;
    std::string hires_upscaler = "latent";
    float hires_scale = 2.0f;
};
```

**参数构建**（`main.cpp:1172-1200`）：
```cpp
params.enable_hires = opts.hires;
if (opts.hires) {
    params.hires_width = opts.hires_width;
    params.hires_height = opts.hires_height;
    params.hires_strength = opts.hires_strength;
    params.hires_sample_steps = opts.hires_steps;
    params.hires_upscaler = myimg::HiresUpscaler::Latent; // 或其他
}
```

#### 3.3 适配器层：`src/adapters/sdcpp_adapter.cpp`

**枚举转换**（`sdcpp_adapter.cpp:104-117`）：
```cpp
static sd_hires_upscaler_t convert_hires_upscaler(HiresUpscaler upscaler) {
    switch (upscaler) {
        case HiresUpscaler::Latent:     return SD_HIRES_UPSCALER_LATENT;
        case HiresUpscaler::Lanczos:    return SD_HIRES_UPSCALER_LANCZOS;
        case HiresUpscaler::Model:      return SD_HIRES_UPSCALER_MODEL;
        // ... 其他 upscaler
    }
}
```

**参数映射**（`sdcpp_adapter.cpp:316-329`）：
```cpp
gen_params.hires.enabled = params.enable_hires;
if (params.enable_hires) {
    gen_params.hires.upscaler = convert_hires_upscaler(params.hires_upscaler);
    gen_params.hires.target_width = params.hires_width;
    gen_params.hires.target_height = params.hires_height;
    gen_params.hires.denoising_strength = params.hires_strength;
    gen_params.hires.steps = params.hires_sample_steps;
}
```

**实际生成**（`sdcpp_adapter.cpp:358-363`）：
```cpp
sd_image_t* sd_images = generate_image(ctx_, &gen_params);
// ↑ sd.cpp 内部完成两阶段生成
```

### 4. 参数调优原理

#### 4.1 `--hires-strength`（去噪强度）

**作用**：控制第二阶段对基础图的修改程度。

| 值 | 效果 | 适用场景 |
|---|---|---|
| 0.15 ~ 0.25 | 轻微 refine，高度保留基础图 | 基础图已经很好，只需轻微增强 |
| **0.28 ~ 0.35** | **平衡，推荐值** | **大多数场景** |
| 0.40 ~ 0.60 | 较强修改，可能改变构图 | 基础图有缺陷，需要大幅修正 |
| > 0.60 | 几乎重新生成，失去 HiRes 意义 | 不建议 |

**原理**：strength 对应 img2img 的 denoising strength。低 strength 只去除少量噪声，保留基础图结构；高 strength 加入更多噪声，生成结果偏离基础图。

**img1.sh 默认值**：`0.28`（低显存下保守值，避免过度修改导致显存激增）

#### 4.2 `--hires-steps`（ refine 步数）

**作用**：第二阶段扩散去噪的采样步数。

| 值 | 效果 | 适用场景 |
|---|---|---|
| 15 ~ 20 | 快速 refine，可能细节不足 | 测试、草稿 |
| **20 ~ 30** | **平衡，推荐值** | **日常出图** |
| 40 ~ 60 | 精细 refine，细节更丰富 | 高质量要求 |
| > 60 | 收益递减，耗时增加 | 极致画质 |

**img1.sh 默认值**：`25`（RTX 3080 10G 的显存和时间平衡）

#### 4.3 `--hires-upscaler`（放大算法）

| 算法 | 原理 | 优点 | 缺点 |
|---|---|---|---|
| `latent` | 默认，latent 空间直接插值 | 速度快，质量尚可 | 最基础 |
| `latent-bicubic` | bicubic 插值后 refine | 边缘更平滑 | 可能过平滑 |
| `lanczos` | Lanczos 重采样 | 保留锐利边缘 | 计算量稍大 |
| `model` | 加载外部超分模型 | 质量最高 | 需要额外模型文件 |

**推荐**：默认 `latent` 即可。如果追求极致锐利边缘，尝试 `lanczos`。

#### 4.4 基础分辨率选择

**核心原则**：基础分辨率必须是 64 的倍数（latent 空间为 8 的倍数，且通常要求 8×8=64）。

| 目标分辨率 | 推荐基础分辨率 | latent 尺寸 | 放大倍数 | 适用显存 |
|---|---|---|---|---|
| 2560×1440 | **1280×720** | 160×90 | 2× | 10GB |
| 2560×1440 | 1920×1080 | 240×135 | 1.33× | 16GB+ |
| 1920×1080 | **1024×576** | 128×72 | 1.875× | 10GB |
| 3840×2160 | 1920×1080 | 240×135 | 2× | 16GB+ |

**img1.sh（10GB 显存）**：基础分辨率必须 ≤ 1280×720，否则第一阶段就 OOM。
**img2.sh（24GB 显存）**：可以用 1920×1080 作为基础，放大倍数更小，latent 插值损失更少，画质更好。

### 5. 显存优化策略

HiRes Fix 两阶段都在显存中进行，必须配合以下优化：

```bash
--diffusion-fa        # Flash Attention：减少注意力计算的显存占用
--vae-tiling          # VAE 分块解码：避免 2560×1440 VAE 解码时 OOM
--vae-tile-size 256x256
--vae-tile-overlap 0.8
```

**VAE Tiling 原理**：VAE 解码器将大图像分成 256×256 的小块分别解码，通过 0.8 的重叠避免块间接缝。

**实测显存占用**（RTX 3080 10GB）：
- 不使用优化：2560×1440 直接 OOM
- 仅 Flash Attention：勉强运行，但可能不稳定
- Flash Attention + VAE Tiling：**稳定运行，峰值显存 ~9.5GB**

### 6. 完整示例

```bash
# 10GB 显存：1280×720 → 2560×1440
./myimg-cli \
  --diffusion-model model.gguf --vae vae.safetensors --llm llm.gguf \
  -p "masterpiece, best quality, portrait" \
  -W 1280 -H 720 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.28 --hires-steps 25 \
  --diffusion-fa --vae-tiling \
  -o output.png

# 24GB 显存：1920×1080 → 2560×1440（画质更好）
./myimg-cli \
  --diffusion-model model.gguf --vae vae.safetensors --llm llm.gguf \
  -p "masterpiece, best quality, portrait" \
  -W 1920 -H 1080 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.30 --hires-steps 30 \
  --diffusion-fa \
  -o output.png
```

### 7. 与 ComfyUI HiRes Fix 的区别

| 维度 | ComfyUI | my-img |
|------|---------|--------|
| **实现层级** | 工作流节点级（显式） | 引擎封装级（隐式） |
| **用户可控性** | 每步可替换/调整 | 通过 CLI 参数配置 |
| **代码复杂度** | 需要理解节点连接 | 只需几个参数 |
| **阶段扩展** | 可在 HiRes 阶段加 ControlNet/IPAdapter | 目前仅支持基础两阶段 |
| ** sampler 选择** | 基础/HiRes 可用不同 sampler | 目前使用相同 sampler |

#### ComfyUI 的实现方式

ComfyUI 的 HiRes Fix 是**显式工作流**，用户可以看到并修改每个步骤：

```
EmptyLatentImage(512x512) → KSampler(20步) → VAEDecode → 基础图
                                      ↓
                              LatentUpscale(2x)
                                      ↓
                              KSampler(20步, denoise=0.3) → VAEDecode → 高清图
```

**关键特点**：
1. **节点可替换**：LatentUpscale 可以换成 ModelUpscale、BicubicUpscale 等
2. **条件可叠加**：HiRes 阶段的 KSampler 可以接入 ControlNet、IPAdapter
3. **Prompt 可切换**：基础阶段和 HiRes 阶段可以使用不同 prompt（Prompt Scheduling）
4. **参数完全独立**：基础步数、HiRes 步数、CFG、Sampler 都可以分别设置

#### my-img 的实现方式

my-img 的 HiRes Fix 是**隐式封装**，由 `stable-diffusion.cpp` 内部完成：

```
用户传入 --hires 参数
        ↓
SDCPPAdapter 将参数映射到 sd_img_gen_params_t.hires
        ↓
sd.cpp 的 generate_image() 内部完成两阶段
        ↓
返回最终高清图
```

**关键特点**：
1. **一键启用**：只需 `--hires --hires-width 2560 --hires-height 1440`
2. **黑盒执行**：用户无法干预中间过程（如替换 upscaler 算法）
3. **参数有限**：只能控制 strength、steps、upscaler 类型，无法做阶段级条件注入
4. **性能优化**：sd.cpp 内部做了显存和计算优化，不需要用户手动管理中间 latent

#### 底层代码差异

**ComfyUI（Python 节点编排）**：
```python
# ComfyUI 的 HiRes Fix 是多个节点的组合
latent = EmptyLatentImage(width=512, height=512)
latent = KSampler(latent, steps=20, ...)
image = VAEDecode(latent)

# HiRes 阶段
latent_hires = LatentUpscale(latent, upscale_method="bicubic", scale=2)
latent_hires = KSampler(latent_hires, steps=20, denoise=0.3, ...)
image_hires = VAEDecode(latent_hires)
```

**my-img（C++ 引擎封装）**：
```cpp
// my-img 的 HiRes Fix 是单个函数调用
sd_img_gen_params_t gen_params;
gen_params.hires.enabled = true;
gen_params.hires.target_width = 2560;
gen_params.hires.target_height = 1440;
gen_params.hires.denoising_strength = 0.28;
gen_params.hires.steps = 25;

// 一次调用完成两阶段
sd_image_t* images = generate_image(ctx, &gen_params);
```

#### 如何选择？

- **用 ComfyUI**：需要精细控制（如 HiRes 阶段加 ControlNet、换不同 sampler、Prompt Scheduling）
- **用 my-img**：追求简单快速出图，不需要干预中间过程，或者部署到生产环境

**my-img 的未来计划**：
当 Workflow JSON 支持完成后，用户将可以像 ComfyUI 一样通过 JSON 定义显式工作流，届时 HiRes Fix 也可以实现阶段级条件注入。

---

## 目录结构

```
my-img/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── design.md                   # 详细设计文档
├── task.md                     # 开发任务表
├── build.sh                    # 一键构建脚本（编译 sd.cpp + my-img）
├── img1.sh                     # RTX 3080 10G 出图脚本
├── img2.sh                     # RTX 4090D 24G 出图脚本
├── src/
│   ├── main.cpp               # CLI 入口
│   ├── adapters/
│   │   ├── sdcpp_adapter.h    # sd.cpp 适配器
│   │   └── sdcpp_adapter.cpp
│   ├── engine/                # 工作流引擎（预留）
│   ├── nodes/                 # 节点实现（预留）
│   ├── backend/               # 推理后端（预留）
│   └── utils/                 # 工具
│       ├── image_utils.h      # 图像 I/O
│       └── gguf_loader.h      # GGUF 加载
├── tests/                      # 测试
│   ├── test_gguf_loader.cpp
│   ├── test_txt2img.cpp
│   ├── test_hires_fix_real.cpp
│   └── ...
└── third_party/               # 第三方依赖
    ├── json/                  # nlohmann/json
    ├── stb/                   # stb_image
    └── ggml/                  # GGML（GGUF 解析）
```

---

## 快速开始

### 环境要求

- **OS**: Linux (Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA 12.0+
- **显存**: 8GB+（推荐 10GB+）
- **编译器**: GCC 11+ 或 Clang 14+
- **CMake**: 3.18+

### 构建

```bash
# 1. 克隆仓库
git clone https://github.com/yourname/my-img.git
cd my-img

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置（自动检测 libtorch）
cmake ..

# 4. 编译
make -j$(nproc)
```

### 运行

```bash
# 基础 txt2img
./myimg-cli \
  --diffusion-model /path/to/z_image_turbo-Q5_K_M.gguf \
  --vae /path/to/ae.safetensors \
  --llm /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "a beautiful landscape" \
  -o output.png

# 带 HiRes Fix 的 2560x1440 人像
./myimg-cli \
  --diffusion-model /path/to/z_image_turbo-Q5_K_M.gguf \
  --vae /path/to/ae.safetensors \
  --llm /path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "portrait of a young woman, soft lighting" \
  -W 1280 -H 720 \
  --hires --hires-width 2560 --hires-height 1440 \
  --hires-strength 0.30 --hires-steps 60 \
  --diffusion-fa --vae-tiling \
  -o portrait_2k.png

# 查看所有参数
./myimg-cli --help
```

### 使用脚本

项目提供三个实用脚本：

#### `build.sh` - 一键构建

自动检测 GPU、编译 sd.cpp 静态库、编译 my-img 主程序：

```bash
# 完整构建（自动检测 CUDA）
./build.sh

# 指定构建类型
BUILD_TYPE=Debug ./build.sh
```

**功能**：
- 自动检测 NVIDIA GPU 并启用 CUDA
- 编译 `third_party/stable-diffusion.cpp` 静态库
- 编译 `myimg-cli` 可执行文件
- 编译测试程序

#### `img1.sh` - RTX 3080 10G 优化出图

针对低显存（10GB）优化的出图脚本：

```bash
# 基础用法
./img1.sh "prompt" ~/output.png 2560 1440

# 带 ESRGAN 2x 放大
./img1.sh "prompt" ~/output.png 2560 1440 --upscale
```

**特点**：
- 低分辨率基础（1280×720）→ HiRes 放大到目标分辨率
- 自动添加质量前缀词和负面提示词
- 启用 VAE Tiling 和 Flash Attention 节省显存
- 针对人像/风景优化采样参数

#### `img2.sh` - RTX 4090D 24G 优化出图

针对高显存（24GB）优化的出图脚本：

```bash
# 基础用法
./img2.sh "prompt" ~/output.png 2560 1440

# 带 ESRGAN 2x 放大
./img2.sh "prompt" ~/output.png 2560 1440 --upscale
```

**特点**：
- 更高基础分辨率（1920×1080）→ 更小的放大倍数，画质更好
- 与 img1.sh 相同的自动化质量优化
- 充分利用 24GB 显存，减少 latent 插值损失

**脚本对比**：

| 脚本 | 目标显存 | 基础分辨率 | 放大倍数 | 适用场景 |
|------|----------|------------|----------|----------|
| `img1.sh` | 10GB | 1280×720 | 2× | RTX 3080 等低显存 |
| `img2.sh` | 24GB | 1920×1080 | 1.33× | RTX 4090 等高显存 |

---

## 技术栈

| 组件 | 库 | 用途 |
|------|-----|------|
| **推理引擎** | [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | 模型加载、GGUF 推理、采样 |
| **张量计算** | [libtorch](https://pytorch.org/cppdocs/) | 图像处理、高级功能扩展 |
| **JSON 解析** | [nlohmann/json](https://github.com/nlohmann/json) | 配置和工作流解析 |
| **图像 I/O** | [stb_image](https://github.com/nothings/stb) | PNG/JPG 加载保存 |
| **构建系统** | CMake 3.18+ | 跨平台构建 |

---

## 与 ComfyUI 的对比

| 特性 | ComfyUI (Python) | my-img (C++) |
|------|------------------|--------------|
| **Python 依赖** | 需要 Python 3.10+ | ❌ 零 Python |
| **启动速度** | 慢（数秒） | ⚡ 毫秒级 |
| **部署体积** | >10GB (Docker) | ~100MB (二进制) |
| **内存占用** | 高（Python 对象） | 低（C++ 紧凑） |
| **模型支持** | Safetensors 为主 | GGUF + Safetensors |
| **功能完整度** | 完整 | 🚧 核心功能已支持 |
| **工作流** | JSON 可视化 | CLI / JSON（开发中） |
| **插件生态** | 丰富 | 计划中 |

---

## 设计哲学

1. **零 Python**：不是 Python-lite，而是彻底零依赖
2. **混合架构**：sd.cpp 负责推理（成熟），libtorch 负责扩展（灵活）
3. **适配层隔离**：sd.cpp 升级不影响上层代码
4. **功能对等**：ComfyUI 能做的，my-img 最终都要能做
5. **性能优先**：C++ 的性能，Python 的便捷（最终目标）

---

## 贡献

欢迎提交 Issue 和 PR！

### 当前最需要帮助的方向
- img2img 实现
- LoRA 权重注入
- ControlNet 支持
- Workflow JSON 解析
- 文档完善

---

## 许可证

MIT License

---

## 致谢

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - 核心推理引擎
- [llama.cpp](https://github.com/ggerganov/llama.cpp) / [ggml](https://github.com/ggerganov/ggml) - GGUF 格式和量化推理
- [PyTorch](https://pytorch.org/) - libtorch C++ 前端
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 灵感来源和工作流设计
