# my-img

**GGUF 模型下载**: https://huggingface.co/leejet

复刻 ComfyUI 生态理念，但无需麻烦的 Python 依赖。只有**干净的 C++ 二进制程序**和**命令管道**。

> 📖 **架构设计文档**: [sd-engine-design.md](docs/sd-engine-design.md) - C++ 版 ComfyUI 工作流执行引擎详细设计

## 已完成的工具

| 工具 | 功能 | 状态 |
|------|------|------|
| `sd-hires` | AI 高清修复：ESRGAN 放大 + Deep HighRes Fix 分阶段重绘 | ✅ |
| `sd-img2img` | 图生图重绘（支持普通 img2img 和 Deep HighRes Fix） | ✅ |
| `sd-upscale` | ESRGAN 独立超分放大 | ✅ |

---

## 项目架构

```
my-img/
├── CMakeLists.txt          # 通用编译模板
├── README.md               # 本文档
├── claude.md               # AI 开发指南
├── apply_patches.sh        # 应用 stable-diffusion.cpp 补丁
├── build/
│   └── build_sd_cpp.sh     # 编译 stable-diffusion.cpp 脚本（自动打补丁）
├── src/                    # 源代码
│   ├── sd-core/            # 核心扩展库（Deep HighRes Fix 原生实现）
│   │   ├── deep_hires.h
│   │   └── deep_hires.cpp
│   ├── sd-hires/
│   │   ├── main.cpp
│   │   └── design.md
│   ├── sd-img2img/
│   │   ├── main.cpp
│   │   └── design.md
│   └── sd-upscale/
│       └── main.cpp
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
make -j2
```

### 3. 安装

```bash
cp sd-hires sd-img2img sd-upscale ../bin/

sudo ln -sf ~/my-img/bin/sd-hires /usr/local/bin/sd-hires
sudo ln -sf ~/my-img/bin/sd-img2img /usr/local/bin/sd-img2img
sudo ln -sf ~/my-img/bin/sd-upscale /usr/local/bin/sd-upscale
```

---

## 使用说明

### sd-hires：AI 高清修复

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

---

### sd-img2img：图生图

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

### sd-upscale：ESRGAN 独立放大

```bash
sd-upscale \
  --model /opt/image/RealESRGAN_x2plus.bin \
  --input input.png \
  --output output.png \
  --scale 2
```

---

## 升级 stable-diffusion.cpp

由于 my-img 修改了 `stable-diffusion.cpp` 源码（添加了 latent hook），升级后需要重新应用修改：

```bash
cd ~/stable-diffusion.cpp
git pull

cd ~/my-img
./apply_patches.sh

cd ~/my-img/build
./build_sd_cpp.sh

cd ~/my-img/build
cmake .. && make -j2
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
| 1 | sd-hires | ✅ |
| 2 | sd-img2img | ✅ |
| 3 | sd-upscale | ✅ |

### 计划中的 ComfyUI 节点（C++ 复刻）

| # | 节点类别 | 具体节点 | 进度 |
|---|---------|---------|------|
| 1 | **加载器** | CheckpointLoaderSimple | ⬜ |
| 2 | **加载器** | VAELoader / CLIPLoader | ⬜ |
| 3 | **条件编码** | CLIPTextEncode | ⬜ |
| 4 | **条件编码** | ConditioningCombine / ConditioningSetArea | ⬜ |
| 5 | **Latent** | EmptyLatentImage | ⬜ |
| 6 | **Latent** | VAEEncode / VAEDecode | ⬜ |
| 7 | **Latent** | LatentUpscale / LatentComposite | ⬜ |
| 8 | **采样** | KSampler / KSamplerAdvanced | ⬜ |
| 9 | **采样** | SamplerCustom | ⬜ |
| 10 | **图像** | LoadImage / SaveImage | ⬜ |
| 11 | **图像** | ImageScale / ImageCrop | ⬜ |
| 12 | **图像** | ImageBlur / ImageSharpen | ⬜ |
| 13 | **超分** | UpscaleModelLoader | ⬜ |
| 14 | **超分** | ImageUpscaleWithModel | ⬜ |
| 15 | **修复** | INPAINT_LoadInpaintModel | ⬜ |
| 16 | **修复** | INPAINT_ApplyInpaint | ⬜ |
| 17 | **ControlNet** | ControlNetLoader | ⬜ |
| 18 | **ControlNet** | ControlNetApply / ControlNetApplyAdvanced | ⬜ |
| 19 | **ControlNet** | CannyEdgePreprocessor | ⬜ |
| 20 | **ControlNet** | MiDaS-DepthMapPreprocessor | ⬜ |
| 21 | **ControlNet** | OpenPosePreprocessor | ⬜ |
| 22 | **LoRA** | LoraLoader / LoraLoaderModelOnly | ⬜ |
| 23 | **IPAdapter** | IPAdapterLoader / IPAdapterApply | ⬜ |
| 24 | **视频** | AnimateDiff Loader / Sampler | ⬜ |
| 25 | **视频** | VideoLinearCFGGuidance | ⬜ |
| 26 | **引擎** | sd-workflow (JSON 执行器) | ⬜ |
| 27 | **引擎** | DAG 拓扑排序 + 缓存系统 | ⬜ |

---

## 参考

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
