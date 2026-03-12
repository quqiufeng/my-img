# my-img

复刻 ComfyUI 生态理念，但无需麻烦的 Python 依赖。只有**干净的二进制程序**和**命令管道**。

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
├── bin/                    # 编译后的二进制程序
│   └── sd-hires            # AI 超分放大工具
├── src/                    # 源代码（每个工具一个目录）
│   └── sd-hires/
│       └── hires_upscaler.cpp
├── include/                # API 头文件（从 stable-diffusion.cpp 复制）
│   └── stable-diffusion.h
└── scripts/                # 辅助脚本（如有）
```

### 工具列表

| 工具 | 功能 |
|------|------|
| sd-hires | AI 超分放大 (ESRGAN 2x/3x/4x + img2img 重绘) |

## 与 stable-diffusion.cpp 的关系

- **依赖**：使用 `stable-diffusion.cpp` 编译好的静态库 (`libstable-diffusion.a`)
- **引用**：`CMakeLists.txt` 通过 `-DSD_PATH` 指定原项目路径
- **头文件**：从原项目 `include/stable-diffusion.h` 复制到本项目

```
stable-diffusion.cpp (源码)  ──编译──>  libstable-diffusion.a
                                            │
my-img (工具集)  ──编译──>  sd-hires  <─────┘
```

## 快速开始

### 1. 编译

```bash
# 进入项目目录
cd ~/my-img

# 创建编译目录
mkdir -p build && cd build

# 配置（指定 stable-diffusion.cpp 路径）
cmake .. -DSD_PATH=/path/to/stable-diffusion.cpp

# 编译（自动扫描 src/ 下所有工具）
make

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

## 参考

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

## 待复刻的 ComfyUI 生态项目

按重要性排序，以下是需要复刻的 ComfyUI 常用功能：

| # | 项目名称 | 项目地址 | 主要功能 | 进度 |
|---|----------|----------|----------|------|
| 1 | sd-hires (ESRGAN + img2img) | 本项目 | AI 超分放大 + 重绘 | ✅ 已完成 |
| 2 | sd-img2img | - | 图生图重绘 | ⬜ 待开发 |
| 3 | sd-inpaint | - | 局部重绘/涂抹 | ⬜ 待开发 |
| 4 | sd-canny | - | Canny 边缘检测 + 控制网 | ⬜ 待开发 |
| 5 | sd-depth | - | Depth 控制网 | ⬜ 待开发 |
| 6 | sd-pose | - | OpenPose 姿态检测 | ⬜ 待开发 |
| 7 | sd-scribble | - | 涂鸦/线稿生图 | ⬜ 待开发 |
| 8 | sd-ip2p | - | Inpaint Anything 局部替换 | ⬜ 待开发 |
| 9 | sd-recolor | - | 图像重上色 | ⬜ 待开发 |
| 10 | sd-upscale (Tile) | - | 分块超分（处理超大图） | ⬜ 待开发 |
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
