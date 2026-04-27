# my-img - 纯 C++ 版 ComfyUI

> **📖 详细设计文档**：[design.md](design.md) - 架构设计、技术细节、实现路线图

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

## 目录结构

```
my-img/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── design.md                   # 详细设计文档
├── task.md                     # 开发任务表
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

```bash
# RTX 3080 10G（自动优化）
./img1.sh "prompt here" ~/output.png 2560 1440

# RTX 4090D 24G（更高基础分辨率）
./img2.sh "prompt here" ~/output.png 2560 1440
```

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
