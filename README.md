# my-img

**GGUF 模型下载**: https://huggingface.co/leejet

复刻 ComfyUI 生态理念，但无需麻烦的 Python 依赖。只有**干净的 C++ 二进制程序**和**命令管道**。

> 📖 **架构设计文档**: [sd-engine-design.md](docs/sd-engine-design.md) - C++ 版 ComfyUI 工作流执行引擎详细设计

## 已完成的工具

| 工具 | 功能 | 状态 |
|------|------|------|
| `sd-workflow` | **C++ 版 ComfyUI 工作流引擎**：解析 JSON 工作流，执行 DAG 拓扑排序，支持 txt2img/img2img/图像处理/LoRA | ✅ |
| `sd-hires` | AI 高清修复：ESRGAN 放大 + Deep HighRes Fix 分阶段重绘 | ✅ |
| `sd-img2img` | 图生图重绘（支持普通 img2img 和 Deep HighRes Fix） | ✅ |
| `sd-upscale` | ESRGAN 独立超分放大 | ✅ |

---

## 项目架构

```
my-img/
├── CMakeLists.txt          # 通用编译模板
├── README.md               # 本文档
├── docs/
│   └── sd-engine-design.md # 架构设计文档
├── claude.md               # AI 开发指南
├── apply_patches.sh        # 应用 stable-diffusion.cpp 补丁
├── build/
│   └── build_sd_cpp.sh     # 编译 stable-diffusion.cpp 脚本（自动打补丁）
├── src/                    # 源代码
│   ├── sd-core/            # 核心扩展库（Deep HighRes Fix 原生实现）
│   │   ├── deep_hires.h
│   │   └── deep_hires.cpp
│   ├── sd-engine/          # C++ 版 ComfyUI 工作流引擎
│   │   ├── core/           # 引擎核心（Workflow/DAGExecutor/Cache）
│   │   ├── nodes/          # 节点实现
│   │   └── tools/          # CLI 工具
│   ├── sd-hires/
│   ├── sd-img2img/
│   └── sd-upscale/
├── stable-diffusion.cpp-patched/   # 修改备份（升级后覆盖用）
│   ├── src/stable-diffusion.cpp
│   └── include/stable-diffusion-ext.h
└── bin/                    # 编译后的二进制
```

---

## 编译

### 1. 编译 stable-diffusion.cpp（自动应用补丁）

```bash
cd ~/my-img/build
./build_sd_cpp.sh
```

这个脚本会自动调用 `apply_patches.sh`，将 my-img 所需的修改应用到 `stable-diffusion.cpp` 源码中，然后编译。

### 2. 编译 my-img

```bash
cd ~/my-img
rm -rf build && mkdir build && cd build
cmake .. -DSD_PATH=/home/dministrator/stable-diffusion.cpp
make -j$(nproc)
```

### 3. 安装

```bash
cp sd-workflow sd-hires sd-img2img sd-upscale ../bin/

sudo ln -sf ~/my-img/bin/sd-workflow /usr/local/bin/sd-workflow
sudo ln -sf ~/my-img/bin/sd-hires /usr/local/bin/sd-hires
sudo ln -sf ~/my-img/bin/sd-img2img /usr/local/bin/sd-img2img
sudo ln -sf ~/my-img/bin/sd-upscale /usr/local/bin/sd-upscale
```

---

## sd-workflow：C++ 版 ComfyUI 工作流引擎

### 双模式设计

`sd-workflow` 支持两种使用方式：

1. **JSON 工作流模式**：执行 ComfyUI 风格的 JSON 工作流
2. **快速命令行模式**：无需 JSON，直接通过命令行参数生成并执行

### 快速模式

#### txt2img 快速生成

```bash
sd-workflow --txt2img \
  --model /path/to/model.gguf \
  --prompt "masterpiece, best quality, a cat" \
  --negative "bad quality" \
  --width 512 --height 512 \
  --seed 42 --steps 20 --cfg 7.5 \
  --output mycat

# 保存为 JSON 供后续复用
sd-workflow --txt2img ... --save-json workflow.json
```

#### img2img 快速生成

```bash
sd-workflow --img2img \
  --model /path/to/model.gguf \
  --input input.png \
  --prompt "add details" \
  --denoise 0.75 \
  --seed 42 --steps 20
```

#### 图像处理

```bash
sd-workflow --process \
  --input photo.png \
  --scale-w 1024 --scale-h 1024 \
  --crop-x 256 --crop-y 256 --crop-w 512 --crop-h 512 \
  --output processed
```

#### Deep HighRes Fix（高清修复）

```bash
# txt2img + Deep HighRes Fix（从小分辨率生成高清大图）
sd-workflow --deep-hires \
  --model /path/to/model.gguf \
  --prompt "masterpiece, best quality, extremely detailed" \
  --target-width 1024 \
  --target-height 1024 \
  --seed 42 --steps 30 \
  --output hires_output

# img2img + Deep HighRes Fix（从输入图高清重绘）
sd-workflow --deep-hires \
  --model /path/to/model.gguf \
  --input input_512.png \
  --prompt "masterpiece, best quality" \
  --target-width 1024 \
  --target-height 1024 \
  --denoise 0.45 \
  --seed 42 --steps 30 \
  --output hires_output
```

### JSON 工作流模式

```bash
# 执行工作流
sd-workflow --workflow my_workflow.json --verbose

# 仅验证不执行
sd-workflow --workflow my_workflow.json --dry-run

# 查看支持的节点
sd-workflow --list-nodes
```

### 支持的节点（17个）

| 类别 | 节点 | 说明 |
|------|------|------|
| 加载器 | `CheckpointLoaderSimple` | 加载 SD 模型 |
| 加载器 | `LoRALoader` | 加载 LoRA 权重 |
| 条件编码 | `CLIPTextEncode` | 文本编码为 conditioning |
| Latent | `EmptyLatentImage` | 创建空 latent |
| Latent | `VAEEncode` | 图像编码为 latent（img2img） |
| Latent | `VAEDecode` | latent 解码为图像 |
| 采样 | `KSampler` | 执行扩散采样（支持 LoRA） |
| 采样 | `DeepHighResFix` | **原生 Deep HighRes Fix**，单次采样动态改变分辨率 |
| 图像 | `LoadImage` | 加载图像 |
| 图像 | `SaveImage` | 保存 PNG |
| 图像 | `ImageScale` | 图像缩放（bilinear/nearest/lanczos） |
| 图像 | `ImageCrop` | 图像裁剪 |
| 图像 | `PreviewImage` | 终端预览 |
| 测试 | `ConstantInt` / `AddInt` / `MultiplyInt` / `PrintInt` | 测试节点 |

### 核心特性

- **真正的中间 latent/conditioning 传递**：通过扩展 `stable-diffusion.cpp` C API 实现
- **DAG 拓扑排序执行**：自动计算节点依赖顺序
- **节点结果缓存**：只重新执行变化的节点
- **命令行 ↔ 节点桥接**：`WorkflowBuilder` 支持程序化构建工作流

---

## sd-hires：AI 高清修复

**流程**：ESRGAN 像素级放大 → Deep HighRes Fix 分阶段重绘

```bash
sd-hires \
  --diffusion-model /opt/image/z_image_turbo-Q6_K.gguf \
  --vae /opt/image/ae.safetensors \
  --llm /opt/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --upscale-model /opt/image/RealESRGAN_x2plus.bin \
  --input input.png \
  --output output.png \
  --prompt "masterpiece, best quality, extremely detailed" \
  --scale 2 \
  --steps 30 \
  --deep-hires
```

**参数说明**：
- `--diffusion-model`：扩散模型（.gguf）
- `--vae`：VAE 模型
- `--llm`：LLM / CLIP 模型
- `--upscale-model`：ESRGAN 模型（.bin）
- `--scale`：ESRGAN 放大倍数（2 或 4，默认 2）
- `--strength`：img2img 强度（默认 0.40）
- `--steps`：总采样步数（默认 30）
- `--deep-hires`：启用 Deep HighRes Fix（默认开启）
- `--no-deep-hires`：禁用，只用单次 img2img
- `--target-width/--target-height`：指定输出尺寸
- `--vae-tiling`：大图启用 VAE 分块解码
- `--flash-attn`：启用 Flash Attention
- `--cpu`：强制 CPU 运行

**Deep HighRes Fix 流程**（原生实现，单次采样中动态改变分辨率）：

| 阶段 | 分辨率 | 步数 | 作用 |
|------|--------|------|------|
| Phase 1 | 低分辨率（如 512x512） | 总步数 × 25% | 确定构图 |
| Phase 2 | 中分辨率（如 768x768） | 总步数 × 25% | 过渡细化 |
| Phase 3 | 目标分辨率 | 总步数 × 50% | 细节增强 |

**核心优势**：
- ✅ **单次采样过程**：不是多次调用 `generate_image()`
- ✅ **Latent 空间过渡**：在采样过程中直接对 latent 插值上采样
- ✅ **只 VAE decode 一次**：信息损失最小化

> 实现方式：通过在 `stable-diffusion.cpp` 源码中添加 `sd_latent_hook_t` hook，在 `sample()` 函数的每个采样步骤前调用，动态改变 latent 分辨率。

> 💡 **节点化支持**：Deep HighRes Fix 的核心逻辑也已封装为 `DeepHighResFix` 节点，可以在 `sd-workflow` 的工作流中灵活组合使用（见下文 `--deep-hires` 快速模式）。

---

## sd-img2img：图生图

```bash
# 普通 img2img
sd-img2img \
  --diffusion-model /opt/image/z_image_turbo-Q6_K.gguf \
  --vae /opt/image/ae.safetensors \
  --llm /opt/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --input input.png \
  --output output.png \
  --prompt "masterpiece, best quality" \
  --strength 0.45 \
  --steps 20

# Deep HighRes Fix（从小图生成大图）
sd-img2img \
  --diffusion-model /opt/image/z_image_turbo-Q6_K.gguf \
  --vae /opt/image/ae.safetensors \
  --llm /opt/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --input input_512.png \
  --output output_1024.png \
  --prompt "masterpiece, best quality" \
  --deep-hires \
  --target-width 1024 \
  --target-height 1024 \
  --steps 30
```

---

## sd-upscale：ESRGAN 独立放大

```bash
sd-upscale \
  --model /opt/image/RealESRGAN_x2plus.bin \
  --input input.png \
  --output output.png \
  --scale 2
```

---

## 升级 stable-diffusion.cpp

由于 my-img 修改了 `stable-diffusion.cpp` 源码（添加了 latent hook、分离式 C API、LoRA 支持等），升级后需要重新应用修改：

```bash
cd ~/stable-diffusion.cpp
git pull

cd ~/my-img
./apply_patches.sh

cd ~/my-img/build
./build_sd_cpp.sh

cd ~/my-img/build
cmake .. && make -j$(nproc)
```

修改的备份文件存放在 `~/my-img/stable-diffusion.cpp-patched/` 目录下。

---

## 3080 (10GB) 高清修复方案

### 核心原理

显存隔离策略：不要直接把 2K 大图传给 SD。在 C++ 业务层手动将大图切成 1024x1024 的小块（Tile），分批次送入 GPU。

### 关键技术

1. **分块处理 (Tiled Processing)**
   - 将大图切成 1024x1024 小块
   - 分批次送入 GPU 处理
   - 显存占用约 5GB

2. **线性羽化融合 (Linear Feathering)**
   - 相邻分块必须有 64 像素重叠区
   - 拼接时使用 Alpha 混合消除接缝

3. **尺寸对齐 (64-bit Alignment)**
   - Stable Diffusion 要求宽高必须是 64 的倍数
   - 使用 `(w + 63) & ~63` 强制对齐

4. **VAE 卸载**
   - 开启 `keep_vae_on_cpu = true` 减少显存压力
   - VAE 在 CPU 跑，显存全力供给 U-Net

### 参数建议

| 参数 | 值 | 说明 |
|------|-----|------|
| TILE_SIZE | 1024 | 10GB 显存最稳尺寸 |
| OVERLAP | 64 | 重叠区域，必须≥64 |
| strength | 0.35-0.45 | 保持结构，修复细节 |
| steps | 30 | 足够步数压实细节 |
| cfg_scale | 4.5-7.0 | 根据模型调整 |
| VAE tile_size | 128 | 防爆显存 |

---

## 提示词处理逻辑

**核心原则**：
- 保留原始主题关键词，否则分块可能产生风格不符的内容
- 增加细节增强词，引导 AI 细化纹理

**推荐结构**：
```
正向: (Masterpiece:1.2), (Best quality:1.2), (Highres:1.2), [原始提示词], extremely detailed, sharp focus, intricate textures.
负向: (low quality, worst quality:1.4), blurry, noise, grain, distorted.
```

---

## 模型准备

### Stable Diffusion 模型

下载 `.safetensors` 或 `.ckpt`，转换为 GGUF 格式。

### ESRGAN 超分模型

- https://github.com/xinntao/Real-ESRGAN/releases
- `RealESRGAN_x2plus.bin` - 2x 放大
- `RealESRGAN_x4plus.bin` - 4x 放大

### TAESD 轻量级 VAE（可选）

适合高分辨率或低显存：
```bash
wget https://huggingface.co/leejet/taesd/resolve/main/taesd_encoder.static.gguf -P /opt/image/
wget https://huggingface.co/leejet/taesd/resolve/main/taesd_decoder.static.gguf -P /opt/image/
```

---

## 依赖说明

编译后运行只需系统自带库：
- `libgomp.so.1` - GCC OpenMP
- `libstdc++.so.6` - C++ 标准库
- `libm.so.6` - 数学库
- `libc.so.6` - C 库

---

## 代码索引

本项目提供高性能代码符号索引，用于快速探索 `stable-diffusion.cpp` 等第三方项目源码。

```bash
# 构建索引
python3 code_index.py ~/stable-diffusion.cpp ~/stable-diffusion-cpp.bin

# 搜索符号
python3 code_search.py ~/stable-diffusion-cpp.bin --find generate_image --json
python3 code_search.py ~/stable-diffusion-cpp.bin --search "upscale" --json --limit 10
```

详见 [claude.md](claude.md) 中的 Code Index 章节。

---

## 待复刻的 ComfyUI 生态

### 已完成的独立工具

| # | 项目 | 进度 |
|---|------|------|
| 1 | `sd-workflow` | ✅ 支持 JSON 工作流 + 快速命令行模式 |
| 2 | `sd-hires` | ✅ |
| 3 | `sd-img2img` | ✅ |
| 4 | `sd-upscale` | ✅ |

### 已完成的 sd-engine 节点

| # | 节点类别 | 具体节点 | 进度 |
|---|---------|---------|------|
| 1 | **加载器** | CheckpointLoaderSimple | ✅ |
| 2 | **加载器** | LoRALoader | ✅ |
| 3 | **条件编码** | CLIPTextEncode | ✅ |
| 4 | **Latent** | EmptyLatentImage | ✅ |
| 5 | **Latent** | VAEEncode / VAEDecode | ✅ |
| 6 | **采样** | KSampler（支持 LoRA） | ✅ |
| 7 | **采样** | DeepHighResFix | ✅ |
| 8 | **图像** | LoadImage / SaveImage | ✅ |
| 9 | **图像** | ImageScale / ImageCrop | ✅ |
| 10 | **图像** | PreviewImage | ✅ |
| 11 | **引擎** | sd-workflow CLI + DAG 执行器 + 缓存 | ✅ |
| 12 | **引擎** | WorkflowBuilder（命令行桥接） | ✅ |

### 计划中的节点

| # | 节点类别 | 具体节点 | 进度 |
|---|---------|---------|------|
| 1 | **条件编码** | ConditioningCombine / ConditioningSetArea | ⬜ |
| 2 | **Latent** | LatentUpscale / LatentComposite | ⬜ |
| 3 | **采样** | KSamplerAdvanced / SamplerCustom | ⬜ |
| 4 | **图像** | ImageBlur / ImageSharpen / ImageComposite | ⬜ |
| 5 | **超分** | UpscaleModelLoader / ImageUpscaleWithModel | ⬜ |
| 6 | **修复** | INPAINT_LoadInpaintModel / INPAINT_ApplyInpaint | ⬜ |
| 7 | **ControlNet** | ControlNetLoader / ControlNetApply | ⬜ |
| 8 | **ControlNet** | CannyEdgePreprocessor / MiDaS-DepthMapPreprocessor / OpenPosePreprocessor | ⬜ |
| 9 | **IPAdapter** | IPAdapterLoader / IPAdapterApply | ⬜ |
| 10 | **视频** | AnimateDiff Loader / Sampler | ⬜ |
| 11 | **服务** | sd-server HTTP API | ⬜ |

---

## 参考

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
