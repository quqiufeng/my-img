# my-img

**GGUF 模型下载**: https://huggingface.co/leejet

复刻 ComfyUI 生态理念，但无需麻烦的 Python 依赖。只有**干净的二进制程序**和**命令管道**。

## 3080 (10GB) 高清修复方案：分块重绘与接缝融合

### 核心原理

显存隔离策略：不要直接把 2K 大图传给 SD。在 C++ 业务层手动将大图切成 1024x1024 的小块（Tile），分批次送入 GPU。

### 关键技术点

1. **分块处理 (Tiled Processing)**
   - 将大图切成 1024x1024 小块
   - 分批次送入 GPU 处理
   - 显存占用约 5GB

2. **线性羽化融合 (Linear Feathering)** - 核心接缝消除算法
   - 相邻分块必须有 64 像素重叠区
   - 拼接时使用 Alpha 混合消除接缝
   
   **原理**：在两个分块重叠的 64 像素区域内，根据距离计算权重：
   ```cpp
   // 权重计算：距离边缘越近，新像素权重越低
   float w_x = (j < overlap) ? (float)j / overlap : 1.0f;
   float w_y = (i < overlap) ? (float)i / overlap : 1.0f;
   float weight = w_x * w_y;  // 综合权重
   
   // 线性插值混合
   canvas[idx] = old_pixel * (1 - weight) + new_pixel * weight;
   ```
   
   **效果**：重叠区域像素平滑过渡，消除"十字架"切割线

3. **尺寸对齐 (64-bit Alignment)**
   - Stable Diffusion 要求宽高必须是 64 的倍数
   - 使用 `(w + 63) & ~63` 强制对齐

4. **VAE 卸载**
   - 开启 `keep_vae_on_cpu = true` 减少显存压力
   - VAE 在 CPU 跑，显存全力供给 U-Net

### 实现代码

```cpp
// 羽化融合函数
static void blend_tile_to_canvas(uint8_t* canvas, int canvas_w, int canvas_h, 
                               uint8_t* tile, int tile_w, int tile_h,
                               int x, int y, int overlap) {
    for (int i = 0; i < tile_h && (y + i) < canvas_h; i++) {
        for (int j = 0; j < tile_w && (x + j) < canvas_w; j++) {
            // 计算权重
            float w_x = (j < overlap && x > 0) ? (float)j / overlap : 1.0f;
            float w_y = (i < overlap && y > 0) ? (float)i / overlap : 1.0f;
            float weight = w_x * w_y;
            
            // 线性插值混合
            int canvas_idx = ((y + i) * canvas_w + (x + j)) * 3;
            int tile_idx = (i * tile_w + j) * 3;
            for (int c = 0; c < 3; c++) {
                canvas[canvas_idx + c] = (uint8_t)(canvas[canvas_idx + c] * (1.0f - weight) + tile[tile_idx + c] * weight);
            }
        }
    }
}

// 分块处理主循环
for (int y = 0; y < h; y += (tile_size - overlap)) {
    for (int x = 0; x < w; x += (tile_size - overlap)) {
        // 1. 裁剪 Tile
        // 2. 推理 generate_image()
        // 3. 羽化拼接 blend_tile_to_canvas()
    }
}
```

### 参数建议

| 参数 | 值 | 说明 |
|------|-----|------|
| TILE_SIZE | 1024 | 10GB 显存最稳尺寸 |
| OVERLAP | 64 | 重叠区域，必须≥64 |
| strength | 0.4 | 保持结构，修复细节 |
| steps | 30 | 足够步数压实细节 |
| cfg_scale | 4.5 | Z-Image/Flux 最优值 |
| tile_size (VAE) | 128 | 防爆显存 |

### 提示词处理逻辑

在分块修复（Tiled Refine）时，提示词处理很关键：

**核心原则**：
- 保留原始主题关键词，否则分块可能产生风格不符的内容
- 增加细节增强词，引导 AI 在每块细化纹理

**推荐的提示词结构**：
```
正向: (Masterpiece:1.2), (Best quality:1.2), (Highres:1.2), [原始提示词], extremely detailed, sharp focus, intricate textures.
负向: (low quality, worst quality:1.4), blurry, noise, grain, distorted.
```

**代码自动增强**：
- 启用 hires_fix 时，自动添加 `masterpiece, ultra-detailed, sharp focus, 8k wallpaper, highly intricate,`

### 为什么 0.4 强度没效果？

1. **显存不足**：VAE 占用 3GB，模型 8GB，10GB 不够
   - 解决：`keep_vae_on_cpu = true`

2. **步数不够**：8 步太少，细节出不来
   - 解决：设为 30 步

3. **Prompt 太短**：AI 不知道要修什么
   - 解决：自动拼接质量词

## 愿景

ComfyUI 以节点式工作流著称，灵活强大，但依赖 Python 生态，安装繁琐。

本项目目标是：
- **纯 C++ 二进制**，零外部依赖（除了系统 libc/gomp）
- **命令管道化**，每个工具独立，通过 Shell 脚本组合工作流
- **开箱即用**，下载二进制 + 模型即可运行

## 项目架构

```
my-img/
├── CMakeLists.txt          # 通用编译模板（自动扫描 src/ 下所有子目录）
├── README.md               # 本文档
├── claude.md               # AI 开发指南
├── bin/                    # 编译后的二进制程序
│   ├── sd-hires            # AI 高清修复 (ESRGAN + Deep HighRes Fix)
│   ├── sd-img2img          # 图生图重绘
│   └── sd-upscale          # ESRGAN 超分放大
├── src/                    # 源代码（每个工具一个目录）
│   ├── sd-hires/
│   │   ├── main.cpp
│   │   └── design.md       # Deep HighRes Fix 设计文档
│   ├── sd-img2img/
│   │   ├── main.cpp
│   │   └── design.md
│   └── sd-upscale/
│       └── main.cpp
├── include/                # API 头文件（从 stable-diffusion.cpp 复制）
│   └── stable-diffusion.h
└── scripts/                # 辅助脚本（如有）
```

### 工具列表

| 工具 | 功能 | 状态 |
|------|------|------|
| sd-hires | AI 高清修复 (ESRGAN 2x/4x + Deep HighRes Fix 分阶段重绘) | ✅ |
| sd-img2img | 图生图重绘（支持普通 img2img 和 Deep HighRes Fix） | ✅ |
| sd-upscale | ESRGAN 超分放大 | ✅ |

## 与 stable-diffusion.cpp 的关系

- **依赖**：使用 `stable-diffusion.cpp` 编译好的静态库 (`libstable-diffusion.a`)
- **引用**：`CMakeLists.txt` 通过 `-DSD_PATH` 指定原项目路径
- **头文件**：从原项目 `include/stable-diffusion.h` 复制到本项目

```
stable-diffusion.cpp (源码)  ──编译──>  libstable-diffusion.a
                                             │
my-img (工具集)  ──编译──>  sd-hires/sd-img2img/sd-upscale  <─────┘
```

## 3080 (10GB) 高清修复方案：分块重绘与接缝融合

### 核心原理

显存隔离策略：不要直接把 2K 大图传给 SD。在 C++ 业务层手动将大图切成 1024x1024 的小块（Tile），分批次送入 GPU。

### 关键技术点

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
   - 开启 `vae_decode_on_cpu` 减少显存压力

### 实现逻辑

```cpp
// 1. 切块循环：按步长(Tile - Overlap)滑动
for (int y = 0; y < H; y += (TILE_SIZE - OVERLAP)) {
    for (int x = 0; x < W; x += (TILE_SIZE - OVERLAP)) {
        
        // 2. 计算当前块尺寸
        int cur_w = std::min(TILE_SIZE, W - x);
        int cur_h = std::min(TILE_SIZE, H - y);
        
        // 3. 尺寸对齐 64
        int render_w = (cur_w + 63) & ~63;
        int render_h = (cur_h + 63) & ~63;
        
        // 4. 抠图 (Crop)
        sd_image_t tile = crop_image(input, x, y, cur_w, cur_h);
        
        // 5. 推理 (Inference)
        sd_image_t* processed = generate_image(ctx, &img_params);
        
        // 6. 羽化拼接回画布
        blend_tile_to_canvas(canvas, processed->data, x, y, ...);
    }
}
```

### 参数建议

| 参数 | 值 | 说明 |
|------|-----|------|
| TILE_SIZE | 1024 | 10GB 显存最稳尺寸 |
| OVERLAP | 64 | 重叠区域 |
| strength | 0.35 | 保持结构，修复细节 |

### 提示词处理逻辑

在分块修复（Tiled Refine）时，提示词处理很关键：

**核心原则**：
- 保留原始主题关键词（如 1girl, forest），否则分块可能产生风格不符的内容
- 增加细节增强词，引导 AI 在每块细化纹理

**推荐的提示词结构**：
```
正向: (Masterpiece:1.2), (Best quality:1.2), (Highres:1.2), [原始提示词], extremely detailed, sharp focus, intricate textures.
负向: (low quality, worst quality:1.4), blurry, noise, grain, distorted.
```

**为什么不需要针对"局部"改提示词**？
- img2img 配合 0.35 的 strength 时，AI 会参考原图像素
- 看到是手就细化手，看到是树叶就细化树叶
- 风险：如果只写 face，AI 可能在背景墙上画出人脸

**代码自动增强**：
- 启用 hires_fix 时，自动添加 `masterpiece, extremely detailed, 8k, high quality,`
- 负向提示词保持简洁，10GB 显存友好
| tile_size (VAE) | 128 | 防爆显存 |

### 1. 编译

```bash
# 进入项目目录
cd ~/my-img

# 创建编译目录
mkdir -p build && cd build

# 配置（指定 stable-diffusion.cpp 路径，使用绝对路径）
cmake .. -DSD_PATH=/home/dministrator/stable-diffusion.cpp

# 编译（自动扫描 src/ 下所有工具）
# 注意：编译时间较长，建议使用 nohup 后台编译
nohup make -j2 > ~/nohup_my-img_compile.log 2>&1 &

# 或前台编译（不推荐，容易超时）
# make -j2

# 查看编译进度
tail -f ~/nohup_my-img_compile.log

# 复制二进制到 bin 目录
cp sd-hires ../bin/
```

### 2. 安装

```bash
# 创建软链接到系统路径
sudo ln -sf ~/my-img/bin/sd-hires /usr/local/bin/sd-hires

# 验证
sd-hires --help
```

## 依赖库编译

my-img 项目依赖 `stable-diffusion.cpp` 的静态库，需要先编译。

### build_sd_cpp.sh

**作用**：编译 my-img 项目需要的依赖库（GPU + Flash Attention 支持）

**位置**：`~/my-shell/build_sd_cpp.sh`

**功能**：
- 克隆/更新 stable-diffusion.cpp 源码
- 编译静态库（libstable-diffusion.a, libggml-cpu.a, libggml-base.a）
- 启用 CUDA GPU 加速
- 启用 Flash Attention 加速
- **保留 build 目录**（之前会删掉，导致 my-img 无法链接）

**编译产物**：
```
stable-diffusion.cpp/build/
├── libstable-diffusion.a   # SD 主库
├── ggml/src/
│   ├── libggml-cpu.a
│   └── libggml-base.a
└── CMakeCache.txt          # 包含 CUDA/Flash Attention 配置
```

**使用方法**：
```bash
~/my-shell/build_sd_cpp.sh
# 或指定路径
~/my-shell/build_sd_cpp.sh /custom/path
```

### img.sh

**作用**：生成图片，作为 my-img 其他工具的**入口文件**

**位置**：`~/my-img/img.sh`

**功能**：
- 使用 Stable Diffusion 生成图片
- 输出 PNG 格式，可直接作为 sd-hires 等工具的输入

**使用方法**：
```bash
# 生成默认图片 (1920x1080)
./img.sh

# 自定义提示词
./img.sh "A cat on the table"

# 指定输出路径和尺寸
./img.sh "A sunset" /opt/sunset.png 2560 1440
```

**典型工作流**：
```bash
# 1. 用 img.sh 生成入口图片
./img.sh "A beautiful landscape" /opt/input.png 1280 720

# 2. 用 sd-hires 超分放大
sd-hires --model models/sd15.gguf --upscale-model models/RealESRGAN_x2plus.bin \
  --input /opt/input.png --output /opt/output.png --prompt "high quality"
```

```bash
# 1. 先编译依赖库（只需一次）
~/my-shell/build_sd_cpp.sh

# 2. 编译 my-img 项目
cd ~/my-img
rm -rf build && mkdir build && cd build
cmake .. -DSD_PATH=~/stable-diffusion.cpp
make
cp sd-hires ../bin/

# 3. 运行
sd-hires --model xxx.gguf --upscale-model xxx.bin --input a.jpg
```

### 3. 使用

```bash
# AI 超分放大 2x
sd-hires \
  --model models/sd15.gguf \
  --upscale-model models/RealESRGAN_x2plus.bin \
  --input input.png \
  --output output.png \
  --prompt "high quality, detailed" \
  --strength 0.45 \
  --steps 20 \
  --seed 42
```

## 编译原理

### CMake 模板

项目使用通用的 CMake 模板，自动扫描 `src/` 下所有子目录：

```cmake
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(src ${SRC_FILES})
    get_filename_component(dir ${src} PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/src/" "" dir ${dir})
    # 目录名即二进制名
    add_executable(${dir} ${src})
    target_link_libraries(${dir} ...)
endforeach()
```

### 链接库

编译需要以下静态库（来自 `stable-diffusion.cpp`）：
- `libstable-diffusion.a` - 主库（SD/ESRGAN 推理）
- `libggml-cpu.a` - GGML 计算库
- `libggml-base.a` - GGML 基础库
- 系统库：`gomp`, `pthread`, `m`

### STB 图像库

使用 header-only 的 `stb_image.h` 处理图片 IO，需要在源文件中定义：
```cpp
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
```

## 开发新工具

### 1. 创建目录和源文件

```bash
mkdir -p src/sd-newtool
touch src/sd-newtool/main.cpp
```

### 2. 编写代码

参考 `src/sd-hires/hires_upscaler.cpp`，使用 `stable-diffusion.h` API。

### 3. 编译

```bash
cd build
cmake .. -DSD_PATH=/path/to/stable-diffusion.cpp
make
# 自动生成 bin/sd-newtool
```

### 4. 安装

```bash
cp sd-newtool ../bin/
sudo ln -sf ~/my-img/bin/sd-newtool /usr/local/bin/sd-newtool
```

## 模型准备

### Stable Diffusion 模型

1. 下载 `.safetensors` 或 `.ckpt` 格式
2. 转换为 GGUF 格式（使用 `stable-diffusion.cpp` 的转换工具）

### ESRGAN 超分模型

下载 `.bin` 格式模型：
- https://github.com/xinntao/Real-ESRGAN/releases

常用模型：
- `RealESRGAN_x2plus.bin` - 2x 放大
- `RealESRGAN_x4plus.bin` - 4x 放大

### TAESD 轻量级 VAE（可选，高显存优化）

TAESD (Tiny AutoEncoder for Stable Diffusion) 是一种极轻量化的 VAE，解码速度快、显存占用极低，适合高分辨率图片处理（2560x1440+）或 10GB 以下显存显卡。

**下载**：
```bash
wget https://huggingface.co/leejet/taesd/resolve/main/taesd_encoder.static.gguf -P /opt/image/
wget https://huggingface.co/leejet/taesd/resolve/main/taesd_decoder.static.gguf -P /opt/image/
```

**使用**：
```bash
./bin/sd-img2img \
  --diffusion-model /opt/image/z_image_turbo-Q6_K.gguf \
  --taesd /opt/image/taesd_encoder.static.gguf \
  --input input.png --output output.png \
  --prompt "high quality" \
  --strength 0.35 --steps 2
```

**说明**：不使用 `--taesd` 时，程序使用模型内置默认 VAE。TAESD 牺牲一些质量换取显存降低。

## 工作流示例

### 生成高清壁纸

```bash
# 1. 生成小图 (1280x720)
sd-cli -m models/z_image.gguf -p "landscape" -o temp.png -W 1280 -H 720

# 2. AI 超分放大 + 重绘 (2560x1440)
sd-hires --model models/sd15.gguf --upscale-model models/RealESRGAN_x2plus.bin \
  -i temp.png -o wallpaper.png -p "high quality, detailed"
```

## 依赖说明

编译后运行只需系统库：
- `libgomp.so.1` - GCC OpenMP
- `libstdc++.so.6` - C++ 标准库
- `libm.so.6` - 数学库
- `libc.so.6` - C 库

这些都是 Linux 系统自带，无需额外安装。

## 代码索引与符号查找 (Code Index)

本项目提供高性能的代码符号索引库，支持 C/C++/LuaJIT 多语言绑定，用于快速查找代码中的函数、类、变量等符号信息。

### 功能特性

- **高性能查询**: 基于内存映射的 V3 索引格式，支持大规模代码库
- **多语言绑定**: 
  - C API - 底层高性能接口
  - C++ Wrapper - 现代 C++ RAII 封装，支持迭代器和范围循环
  - LuaJIT FFI - Lua 高性能绑定
- **多种搜索方式**:
  - 精确匹配 (Exact Match)
  - 前缀搜索 (Prefix Search)
  - Glob 模式匹配 (`*`, `?` 通配符)
  - 模糊搜索 (Fuzzy Search，基于编辑距离)
  - 正则表达式搜索 (Regex)

### 快速开始

```bash
# C++ 示例
#include "symbol_index_v3.hpp"

code_index::SymbolIndexV3 idx("project.bin");

// 范围循环遍历所有符号
for (const auto& sym : idx) {
    std::cout << sym.name << " at " << sym.file << ":" << sym.line << std::endl;
}

// 精确查找
auto result = idx.find("my_function");

// 前缀搜索
auto matches = idx.find_prefix("test_");

// Glob 模式
auto globs = idx.glob("foo*bar");

// 模糊搜索 (编辑距离 <= 2)
auto fuzzy = idx.fuzzy("myfuncton", 2);

// 正则搜索
auto regex = idx.regex("^test_.*");
```

### 详细文档

📖 **[查看完整适配层文档](ADAPTER_LAYER.md)**

文档包含：
- C/C++/LuaJIT API 详细说明
- 编译和使用指南
- 性能优化建议
- 完整示例代码

### 文件说明

| 文件 | 说明 |
|------|------|
| `symbol_index_v3.c` | C 语言核心实现 |
| `symbol_index_v3.h` | C/C++ 头文件 |
| `symbol_index_v3.hpp` | C++ 封装类（支持迭代器） |
| `symbol_index_v3_ffi.lua` | LuaJIT FFI 绑定 |
| `libsymbol_index_v3.so` | 编译后的共享库 |
| `ADAPTER_LAYER.md` | 详细文档 |

## 参考

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

## 待复刻的 ComfyUI 生态项目

按重要性排序，以下是需要复刻的 ComfyUI 常用功能：

| # | 项目名称 | 项目地址 | 主要功能 | 进度 |
|---|----------|----------|----------|------|
| 1 | sd-hires | 本项目 | AI 高清修复 (ESRGAN + Deep HighRes Fix) | ✅ 已完成 |
| 2 | sd-img2img | 本项目 | 图生图重绘（支持 Deep HighRes Fix） | ✅ 已完成 |
| 3 | sd-upscale | 本项目 | ESRGAN 超分放大 | ✅ 已完成 |
| 4 | sd-inpaint | - | 局部重绘/涂抹 | ⬜ 待开发 |
| 5 | sd-canny | - | Canny 边缘检测 + 控制网 | ⬜ 待开发 |
| 6 | sd-depth | - | Depth 控制网 | ⬜ 待开发 |
| 7 | sd-pose | - | OpenPose 姿态检测 | ⬜ 待开发 |
| 8 | sd-scribble | - | 涂鸦/线稿生图 | ⬜ 待开发 |
| 9 | sd-ip2p | - | Inpaint Anything 局部替换 | ⬜ 待开发 |
| 10 | sd-recolor | - | 图像重上色 | ⬜ 待开发 |
| 11 | sd-upscale (Tile) | - | 分块超分（处理超大图） | ⬜ 待开发 |
| 11 | sd-lora | - | LoRA 加载与应用 | ⬜ 待开发 |
| 12 | sd-photomaker | - | PhotoMaker 人脸 ID | ⬜ 待开发 |
| 13 | sd-animate | - | AnimateDiff 动画生成 | ⬜ 待开发 |
| 14 | sd-video | - | 视频生成/视频生视频 | ⬜ 待开发 |
| 15 | sd-segment | - | Segment Anything 语义分割 | ⬜ 待开发 |
| 16 | sd-faceid | - | InsightFace 人脸识别 | ⬜ 待开发 |
| 17 | sd-compose | - | 图像合成/图层叠加 | ⬜ 待开发 |
| 18 | sd-crop | - | 智能裁剪/扩图 | ⬜ 待开发 |
| 19 | sd-face-restore | - | 人脸修复/增强 | ⬜ 待开发 |
| 20 | sd-background-remove | - | 背景抠图 | ⬜ 待开发 |

### 说明

- ✅ 已完成：已有可用的二进制工具
- ⬜ 待开发：等待实现的功能

每个工具都将是独立的 C++ 二进制程序，通过命令行参数调用，可通过 Shell 脚本组合工作流。
