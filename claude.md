# Claude.md

## 项目概述

本项目是 **stable-diffusion.cpp** 的工具集，基于其静态库开发独立的 C++ 图片处理工具。

**远景**：复刻 ComfyUI 生态理念，但无需麻烦的 Python 依赖。只有**干净的二进制程序**和**命令管道**。

**功能**：
- AI 超分放大 (ESRGAN 2x/3x/4x + img2img 重绘)
- 纯 C++ 实现，零外部依赖
- 每个工具独立，通过 Shell 组合工作流

---

## 🔍 重要原则：使用 Code Index 研究第三方项目

**所有第三方依赖项目，必须先建立代码索引，然后通过 code_search.py 搜索功能探索项目。禁止 AI 自主独立研究项目，避免占用宝贵的上下文。**

### 原因

1. **上下文限制**：AI 上下文有限（通常 128K-200K tokens），大型项目代码会占满上下文，导致推理能力下降
2. **效率问题**：让 AI 自己遍历文件，速度慢且容易遗漏关键信息
3. **精准查询**：Code Index 提供毫秒级精准查询，只加载需要的代码片段
4. **可追溯性**：通过索引查询，可以精确定位代码位置，方便后续引用

### 工作流程

#### 1. 为第三方项目建立索引

```bash
# 使用 code_index.py 构建索引
python3 ~/my-img/code_index.py <project_path> <output.bin>

# 示例：为 stable-diffusion.cpp 建立索引
python3 ~/my-img/code_index.py \
  /home/dministrator/stable-diffusion.cpp \
  /home/dministrator/stable-diffusion-cpp.bin
```

**注意事项**：
- 索引文件（.bin）通常保存在项目同级目录或 `~` 目录
- 首次构建需要 1-5 分钟（取决于项目大小）
- 自动过滤超过 5MB 的文件（避免解析超大词汇表导致正则回溯卡死）
- 已修复问题：进度显示、大文件跳过、缓存加载后文件句柄保持打开

#### 2. 使用 code_search.py 查询代码

```bash
# 查看索引统计
python3 ~/my-img/code_search.py <index.bin> --stats

# 精确查找函数
python3 ~/my-img/code_search.py <index.bin> --find <function_name> --json

# 模糊搜索关键字
python3 ~/my-img/code_search.py <index.bin> --search <keyword> --json --limit 10

# 前缀匹配
python3 ~/my-img/code_search.py <index.bin> --prefix <prefix> --json --limit 10

# 正则搜索
python3 ~/my-img/code_search.py <index.bin> --regex "^ggml_.*" --json --limit 10

# 查看文件中的所有函数
python3 ~/my-img/code_search.py <index.bin> --file <filename> --json --limit 10

# 示例：查找 stable-diffusion.cpp 中的 generate_image 函数
python3 ~/my-img/code_search.py \
  /home/dministrator/stable-diffusion-cpp.bin \
  --find generate_image --json
```

**关键参数说明**：
- `--stats`：显示索引统计信息（符号数、文件数等）
- `--find <name>`：精确查找符号名称（O(1)，< 1ms）
- `--search <keyword>`：模糊搜索（不区分大小写）
- `--prefix <prefix>`：前缀匹配搜索
- `--regex <pattern>`：正则表达式搜索
- `--file <filename>`：获取指定文件中的所有函数
- `--json`：输出 JSON 格式，方便 AI 解析
- `--limit <n>`：限制返回结果数量（默认 20）

#### 3. AI 研究项目时的标准流程

**错误做法** ❌：
```
User: 帮我研究 stable-diffusion.cpp 的 img2img 实现
AI:  （自己遍历文件，读几千行代码，占满上下文）
```

**正确做法** ✅：
```
User: 帮我研究 stable-diffusion.cpp 的 img2img 实现

AI:  首先，我需要查询代码索引来获取相关信息：

Step 1: 检查索引是否存在
$ python3 ~/my-img/code_search.py /home/dministrator/stable-diffusion-cpp.bin --stats

Step 2: 搜索 img2img 相关函数
$ python3 ~/my-img/code_search.py /home/dministrator/stable-diffusion-cpp.bin --search img2img --json --limit 10

Step 3: 精确查找核心函数
$ python3 ~/my-img/code_search.py /home/dministrator/stable-diffusion-cpp.bin --find generate_image --json

Step 4: 根据结果，继续深入查询相关函数
... （每次只加载几十到几百 tokens 的代码片段）
```

#### 4. 实际搜索示例

**示例 1：查找 upscale 相关实现**
```bash
$ python3 ~/my-img/code_search.py ~/stable-diffusion-cpp.bin --search upscale --json --limit 5
# 返回：free_upscaler_ctx, upscale, upscale_tensor, ggml_compute_forward_upscale 等
```

**示例 2：查找 VAE tiling 实现**
```bash
$ python3 ~/my-img/code_search.py ~/stable-diffusion-cpp.bin --search tiling --json --limit 5
# 返回：sd_tiling_calc_tiles, make_vae_tiling_json, vae_tiling_params 等
```

**示例 3：查找 Flash Attention 相关代码**
```bash
$ python3 ~/my-img/code_search.py ~/stable-diffusion-cpp.bin --search flash_attn --json --limit 5
# 返回：ggml_compute_forward_flash_attn_back, flash_attn_ext_f16_load_tile 等
```

**示例 4：按文件查找所有符号**
```bash
$ python3 ~/my-img/code_search.py ~/stable-diffusion-cpp.bin --file stable-diffusion.cpp --json --limit 5
# 返回该文件中所有函数、类、变量定义
```

### 索引管理

#### 索引文件命名规范

```
<project_name>.bin

# 示例
stable-diffusion-cpp.bin       # stable-diffusion.cpp 索引
koboldcpp.bin                  # koboldcpp 索引
llama-cpp.bin                  # llama.cpp 索引
```

#### 索引文件位置

```bash
# 推荐位置（优先级从高到低）
1. 家目录：~/stable-diffusion-cpp.bin
2. 项目同级目录：/path/to/project/stable-diffusion-cpp.bin
```

#### 索引更新

```bash
# 当项目代码更新后，重新构建索引
python3 ~/my-img/code_index.py <project_path> <output.bin>

# 建议：每次代码更新后都重新构建索引
# 旧缓存文件（.bin.cache 和 .bin.idx）会自动失效并重建
```

### Code Search 输出格式说明

**JSON 输出字段**：

```json
{
  "name": "generate_image",
  "kind": "function",
  "signature": "sd_image_t* generate_image(...)",
  "location": {
    "file": "stable-diffusion.cpp",
    "line": 3587,
    "column": 0
  },
  "code_snippet": "...",
  "context": {
    "includes": ["ggml_extend.hpp", "model.h"],
    "macros": {},
    "typedefs": {}
  }
}
```

**字段说明**：
- `name`：符号名称
- `kind`：类型（function/class/struct/enum/variable/macro）
- `signature`：函数签名或类型定义
- `location`：文件位置和行列号
- `code_snippet`：代码片段（前 30 行）
- `context`：文件上下文（头文件、宏定义、类型定义）

### 性能指标

- **索引构建**：10万符号 ≈ 130 秒（stable-diffusion.cpp 共 99,887 符号）
- **精确查询**：< 1ms
- **前缀匹配**：< 1ms
- **模糊搜索**：< 100ms
- **正则搜索**：< 500ms
- **内存占用**：索引文件 237MB → 运行时 < 50MB

### 禁止行为

❌ **严禁以下做法**：
1. 让 AI 直接阅读整个项目的源代码
2. 让 AI 遍历文件系统查找代码
3. 在上下文中加载超过 1000 行的代码
4. 不使用索引直接询问项目实现细节

✅ **必须遵守**：
1. 所有第三方项目必须先建立索引
2. 通过 code_search.py 精准查询所需代码
3. 每次只加载必要的代码片段
4. 保持 AI 上下文干净，留给推理使用

---

## ⚡ 重要原则：默认启用 GPU + Flash Attention

**所有新项目必须默认启用 GPU 加速和 Flash Attention，除非硬件/环境确实不支持。**

### 原因

1. **性能提升巨大**：GPU + Flash Attention 可带来 2-10x 加速
2. **用户体验**：开箱即用高性能，无需用户手动配置
3. **未来趋势**：AI 推理必须用 GPU

### 实现要求

| 功能 | 默认值 | 备选 | 说明 |
|------|--------|------|------|
| GPU | **启用** | `--cpu` 关闭 | 优先使用 GPU |
| Flash Attention | **启用** | `--no-flash-attn` 关闭 | GPU 模式下默认开启 |

### 代码模板

```cpp
// 参数定义（默认启用）
bool use_gpu = true;           // 默认 GPU
bool use_flash_attn = true;   // 默认启用 Flash Attention

// 命令行解析
if (strcmp(argv[i], "--cpu") == 0) {
    use_gpu = false;
} else if (strcmp(argv[i], "--no-flash-attn") == 0) {
    use_flash_attn = false;
}

// SD 上下文参数
ctx_params.offload_params_to_cpu = !use_gpu;
ctx_params.keep_vae_on_cpu = !use_gpu;
ctx_params.keep_clip_on_cpu = !use_gpu;
ctx_params.flash_attn = use_gpu && use_flash_attn;
ctx_params.diffusion_flash_attn = use_gpu && use_flash_attn;
```

### 编译要求

编译 `stable-diffusion.cpp` 时必须启用：

**注意**：编译时间较长（10-30分钟），建议使用 `nohup` 后台编译：

```bash
# 使用 nohup 后台编译（推荐）
cd ~/stable-diffusion.cpp
rm -rf build && mkdir build && cd build
cmake .. -DSD_CUDA=ON -DSD_FLASH_ATTN=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc

nohup make -j2 > ~/nohup_sd_compile.log 2>&1 &

# 查看编译进度
tail -f ~/nohup_sd_compile.log

# 等待编译完成（约83%后需要较长时间）
# 检查是否成功
ls ~/stable-diffusion.cpp/build/*.a
```

**编译脚本方式**：
```bash
# 直接使用项目提供的编译脚本
~/my-img/build_sd_cpp.sh
```

---

## 资源目录规范

**所有与本项目相关的资源文件都放在 `/opt/image/` 目录**

AI 需要先搜索这个目录来查找：
- 大模型文件 (`.gguf`, `.bin`, `.safetensors`)
- 脚本文件 (`.sh`)
- 图片素材 (`.png`, `.jpg`, `.webp`)
- 其他资源文件

```bash
/opt/image/
├── models/          # 模型文件
│   ├── sd15/
│   ├── sdxl/
│   └── esrgan/
├── scripts/         # 脚本文件
├── inputs/          # 输入图片
└── outputs/         # 输出图片
```

---

## stable-diffusion.cpp API 详解

### 1. 头文件引入

```cpp
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stable-diffusion.h"
#include "stb_image.h"
#include "stb_image_write.h"
```

### 2. 常用结构体

#### sd_image_t - 图片结构
```cpp
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;
```

#### sd_ctx_params_t - 模型加载参数
```cpp
typedef struct {
    const char* model_path;           // 主模型路径 (GGUF)
    const char* vae_path;             // VAE 模型路径
    const char* diffusion_model_path; // 扩散模型路径
    const char* clip_l_path;          // CLIP 模型路径
    const char* t5xxl_path;           // T5XXL 模型路径
    const char* llm_path;             // LLM 模型路径
    int n_threads;                    // CPU 线程数
    enum sd_type_t wtype;             // 模型量化类型
    enum rng_type_t rng_type;         // 随机数类型
    bool flash_attn;                  // Flash Attention
    bool offload_params_to_cpu;       // 卸载到 CPU
    bool vae_decode_only;            // 仅 VAE 解码
} sd_ctx_params_t;
```

#### sd_img_gen_params_t - 图像生成参数
```cpp
typedef struct {
    const char* prompt;               // 正向提示词
    const char* negative_prompt;     // 负向提示词
    sd_image_t init_image;            // 初始化图片 (img2img 用)
    sd_image_t mask_image;            // 蒙版图片 (inpaint 用)
    int width;                        // 输出宽度
    int height;                       // 输出高度
    float strength;                   // 重绘强度 (0.0-1.0)
    int64_t seed;                     // 随机种子
    int sample_steps;                 // 采样步数
    sd_sample_params_t sample_params; // 采样参数
    sd_tiling_params_t vae_tiling_params; // VAE tiling
    sd_lora_t* loras;                 // LoRA 列表
    uint32_t lora_count;              // LoRA 数量
} sd_img_gen_params_t;
```

#### sd_sample_params_t - 采样参数
```cpp
typedef struct {
    enum sample_method_t sample_method; // 采样方法
    enum scheduler_t scheduler;        // 调度器
    int sample_steps;                  // 步数
    float eta;                         // DDIM eta
    float flow_shift;                  // Flow shift
} sd_sample_params_t;
```

#### sd_tiling_params_t - VAE Tiling 参数（防止大图爆显存）
```cpp
typedef struct {
    bool enabled;          // 是否启用
    int tile_size_x;       // 分块宽度
    int tile_size_y;      // 分块高度
    float target_overlap; // 重叠像素
} sd_tiling_params_t;
```

### 3. 常用枚举

#### 采样方法 (sample_method_t)
| 枚举 | 说明 |
|------|------|
| `EULER_SAMPLE_METHOD` | Euler |
| `EULER_A_SAMPLE_METHOD` | Euler a (常用) |
| `HEUN_SAMPLE_METHOD` | Heun |
| `DPM2_SAMPLE_METHOD` | DPM2 |
| `DPMPP2M_SAMPLE_METHOD` | DPM++ 2M |
| `LCM_SAMPLE_METHOD` | LCM |

#### 调度器 (scheduler_t)
| 枚举 | 说明 |
|------|------|
| `DISCRETE_SCHEDULER` | 默认 |
| `KARRAS_SCHEDULER` | Karras (推荐，图像更清晰) |
| `EXPONENTIAL_SCHEDULER` | 指数 |
| `LCM_SCHEDULER` | LCM 专用 |

#### 模型类型 (sd_type_t) - 量化精度
| 枚举 | 说明 |
|------|------|
| `SD_TYPE_F32` | 全精度 |
| `SD_TYPE_F16` | 半精度 |
| `SD_TYPE_Q8_0` | 8位量化 |
| `SD_TYPE_Q6_K` | 6位量化 |
| `SD_TYPE_Q4_0` | 4位量化 |

#### 随机数类型 (rng_type_t)
| 枚举 | 说明 |
|------|------|
| `STD_DEFAULT_RNG` | 默认 |
| `CUDA_RNG` | CUDA GPU |
| `CPU_RNG` | CPU |

### 4. 核心 API 函数

#### 上下文管理
```c
// 初始化上下文参数
void sd_ctx_params_init(sd_ctx_params_t* params);

// 创建推理上下文
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params);

// 释放上下文
void free_sd_ctx(sd_ctx_t* ctx);
```

#### 图像生成
```c
// 初始化生成参数
void sd_img_gen_params_init(sd_img_gen_params_t* params);

// 文生图 / 图生图
sd_image_t* generate_image(sd_ctx_t* ctx, const sd_img_gen_params_t* params);
```

#### 采样参数
```c
// 初始化采样参数
void sd_sample_params_init(sd_sample_params_t* params);

// 获取默认采样方法
enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* ctx);

// 获取默认调度器
enum scheduler_t sd_get_default_scheduler(sd_ctx_t* ctx, enum sample_method_t method);
```

#### ESRGAN 超分
```c
// 创建超分上下文
upscaler_ctx_t* new_upscaler_ctx(
    const char* esrgan_path,    // ESRGAN 模型路径 (.bin)
    bool offload_params_to_cpu,
    bool direct,
    int n_threads,
    int tile_size
);

// 释放超分上下文
void free_upscaler_ctx(upscaler_ctx_t* ctx);

// 执行超分
sd_image_t upscale(
    upscaler_ctx_t* ctx,
    sd_image_t input_image,
    uint32_t upscale_factor   // 2, 3, 4
);
```

#### 图片预处理
```c
// Canny 边缘检测
sd_image_t preprocess_canny(
    sd_image_t image,
    float high_threshold,
    float low_threshold,
    float weak,
    float strong,
    bool inverse
);
```

#### 模型转换
```c
// 转换模型为 GGUF 格式
bool convert(
    const char* input_path,
    const char* vae_path,
    const char* output_path,
    enum sd_type_t output_type,
    const char* tensor_type_rules,
    bool convert_name
);
```

#### 工具函数
```c
// 获取 CPU 核心数
int32_t sd_get_num_physical_cores();

// 获取系统信息
const char* sd_get_system_info();

// 版本信息
const char* sd_version(void);
const char* sd_commit(void);
```

### 5. 典型使用模式

#### 模式一：文生图 (txt2img)
```cpp
sd_ctx_params_t ctx_params;
sd_ctx_params_init(&ctx_params);
ctx_params.model_path = "model.gguf";
ctx_params.wtype = SD_TYPE_Q8_0;
ctx_params.n_threads = 4;

sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);

sd_img_gen_params_t img_params;
sd_img_gen_params_init(&img_params);
img_params.prompt = "a cat";
img_params.negative_prompt = "blurry";
img_params.width = 512;
img_params.height = 512;
img_params.seed = 42;
img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
img_params.sample_params.scheduler = KARRAS_SCHEDULER;
img_params.sample_params.sample_steps = 20;

sd_image_t result = generate_image(sd_ctx, &img_params);

free_sd_ctx(sd_ctx);
```

#### 模式二：图生图 (img2img)
```cpp
// 先加载图片
int w, h, c;
uint8_t* data = stbi_load("input.png", &w, &h, &c, 4);
sd_image_t input_image = { (uint32_t)w, (uint32_t)h, 4, data };

// 设置 img2img 参数
img_params.init_image = input_image;  // 关键：传入初始图片
img_params.strength = 0.45;            // 重绘强度
```

#### 模式三：ESRGAN 超分
```cpp
// 创建超分器
upscaler_ctx_t* upscaler = new_upscaler_ctx(
    "RealESRGAN_x2plus.bin",
    false, false, 4, 128
);

// 执行 2x 超分
sd_image_t upscaled = upscale(upscaler, input_image, 2);

// 释放
free_upscaler_ctx(upscaler);
```

#### 模式四：VAE Tiling（防止大图爆显存）
```cpp
img_params.vae_tiling_params.enabled = true;
img_params.vae_tiling_params.tile_size_x = 512;
img_params.vae_tiling_params.tile_size_y = 512;
img_params.vae_tiling_params.target_overlap = 32;
```

---

## 依赖引入

### CMake 配置

本项目通过 `-DSD_PATH` 指定 stable-diffusion.cpp 路径，自动引入：

```cmake
cmake_minimum_required(VERSION 3.12)
project(my-img)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT SD_PATH)
    message(FATAL_ERROR "SD_PATH not set")
endif()

# 头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${SD_PATH}/thirdparty)
include_directories(${SD_PATH}/ggml/include)
include_directories(${SD_PATH}/ggml/src)

# STB 图片库
add_definitions(-DSTB_IMAGE_IMPLEMENTATION)
add_definitions(-DSTB_IMAGE_WRITE_IMPLEMENTATION)

# 链接库
link_directories(${SD_PATH}/build)

# 编译
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
foreach(src ${SRC_FILES})
    get_filename_component(name ${src} NAME_WE)
    get_filename_component(dir ${src} PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/src/" "" dir ${dir})
    string(REPLACE "/" "-" dir ${dir})
    set(final_name ${dir})
    
    add_executable(${final_name} ${src})
    
    # ============================================================
    # 默认启用 GPU + Flash Attention（除非确实无法集成）
    # ============================================================
    target_link_libraries(${final_name}
        ${SD_PATH}/build/libstable-diffusion.a 
        ${SD_PATH}/build/ggml/src/libggml-cpu.a
        ${SD_PATH}/build/ggml/src/libggml-base.a
        gomp
        pthread
        m
        # CUDA 库（GPU 支持必需）
        cuda
        cudart
    )
    
    # CUDA 编译选项
    if(EXISTS ${SD_PATH}/build/CMakeCache.txt)
        file(STRINGS ${SD_PATH}/build/CMakeCache.txt CUDA_ENABLE PARENT_SCOPE)
        if(CUDA_ENABLE MATCHES "CUDA_ENABLE:BOOL=ON")
            add_definitions(-DSD_CUDA)
        endif()
    endif()
endforeach()
```

### 依赖的库

| 库 | 说明 |
|---|---|
| `libstable-diffusion.a` | 主库（SD/ESRGAN 推理） |
| `libggml-cpu.a` | GGML CPU 计算 |
| `libggml-base.a` | GGML 基础 |
| `gomp` | OpenMP（多线程） |
| `pthread` | POSIX 线程 |
| `m` | 数学库 |
| `cuda` | CUDA 驱动 |
| `cudart` | CUDA 运行时 |

---

## GPU + Flash Attention 加速（默认启用）

### 重要：所有新项目必须默认启用

**除非硬件/环境确实不支持，否则必须默认启用 GPU + Flash Attention。**

### 1. 编译 stable-diffusion.cpp（前置条件）

```bash
cd stable-diffusion.cpp
cmake . -DSD_CUDA=ON -DSD_FLASH_ATTN=ON -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 2. 代码中启用 GPU + Flash Attention

```cpp
// ============================================================
// 默认配置（GPU + Flash Attention 启用）
// ============================================================
bool use_gpu = true;           // 默认: GPU
bool use_flash_attn = true;    // 默认: 启用 Flash Attention

// ============================================================
// 命令行参数（允许用户覆盖默认值）
// ============================================================
for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--cpu") == 0) {
        use_gpu = false;
    } else if (strcmp(argv[i], "--no-flash-attn") == 0) {
        use_flash_attn = false;
    }
}

// ============================================================
// ESRGAN 超分 - 使用 GPU
// ============================================================
upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(
    upscale_model_path,
    !use_gpu,    // offload_to_cpu: false=GPU, true=CPU
    false,       // use_vulkan
    4,           // n_threads
    128          // tile_size
);

// ============================================================
// SD 模型 - 完整 GPU + Flash Attention 配置
// ============================================================
sd_ctx_params_t ctx_params;
sd_ctx_params_init(&ctx_params);
ctx_params.model_path = model_path;
ctx_params.wtype = SD_TYPE_Q8_0;
ctx_params.n_threads = 4;

// GPU 配置（重要！）
ctx_params.offload_params_to_cpu = !use_gpu;      // 不卸载到 CPU = 使用 GPU
ctx_params.keep_vae_on_cpu = !use_gpu;            // VAE 在 GPU 上
ctx_params.keep_clip_on_cpu = !use_gpu;           // CLIP 在 GPU 上

// Flash Attention（重要！）
ctx_params.flash_attn = use_gpu && use_flash_attn;           // CLIP/VAE Flash Attn
ctx_params.diffusion_flash_attn = use_gpu && use_flash_attn; // 扩散模型 Flash Attn
```

### 3. 运行时指定

```bash
# 默认: GPU + Flash Attention（高性能）
sd-hires --model xxx.gguf --upscale-model xxx.bin --input a.jpg

# 关闭 Flash Attention（兼容性问题时使用）
sd-hires --model xxx.gguf --upscale-model xxx.bin --input a.jpg --no-flash-attn

# 强制 CPU 模式（无 GPU 时自动回退，但建议显式指定）
sd-hires --model xxx.gguf --upscale-model xxx.bin --input a.jpg --cpu

# 调试模式（查看实际使用哪种后端）
sd-hires --model xxx.gguf --upscale-model xxx.bin --input a.jpg --debug
```

### 4. 验证 GPU 使用

```bash
# 查看 GPU 状态
nvidia-smi

# 监控 GPU 使用
watch -n 0.5 nvidia-smi
```

### 5. 常见问题

| 问题 | 解决方案 |
|------|----------|
| 编译报错找不到 `cuda` | 确保 CUDA 已安装，或使用 CPU 模式 |
| 运行时无 GPU 输出 | 检查 `--debug` 确认实际使用模式 |
| Flash Attention 报错 | 使用 `--no-flash-attn` 关闭 |

---

## 远景与功能

本项目旨在打造一个轻量级的 AI 图片处理工具生态：
- 无 Python 环境依赖，开箱即用
- 每个工具独立二进制，通过命令行管道组合
- 参考 ComfyUI 的节点式思维，但用 Shell 脚本实现工作流

---

## 代码风格（借鉴 stable-diffusion.cpp）

- 使用 C++17 标准
- 保持代码简洁，直接调用 API
- 不添加不必要的注释
- 使用 `stb_image.h` 处理图片 IO

### 内存管理原则

- 始终配对：`new_sd_ctx` ↔ `free_sd_ctx`
- 始终配对：`new_upscaler_ctx` ↔ `free_upscaler_ctx`
- 图片使用 `stbi_image_free()` 释放
- 始终检查返回值是否为 NULL

---

## ⚠️ 避免踩坑的最佳实践（stable-diffusion.cpp）

**以下是通过调试 sd-hires 总结出的核心坑点，新建项目时必须注意：**

### 核心问题一：mask_image 必须创建全白图片

**问题**：stable-diffusion.cpp 的 `sd_image_to_ggml_tensor` 函数不检查 data 是否为 NULL，直接处理 mask_image。

**症状**：
```
GGML_ASSERT(image.width == tensor->ne[0]) failed
```

**解决方案**：创建全白 mask（255）
```cpp
// ❌ 错误：设置为 NULL 会崩溃
img_params.mask_image.data = NULL;

// ✅ 正确：创建全白 mask
size_t mask_size = width * height;
uint8_t* mask_data = (uint8_t*)malloc(mask_size);
memset(mask_data, 255, mask_size);  // 全白
img_params.mask_image.data = mask_data;
img_params.mask_image.width = width;
img_params.mask_image.height = height;
img_params.mask_image.channel = 1;
```

---

### 核心问题二：ESRGAN 内存管理

**问题**：`upscale()` 返回的图片数据在 `free_upscaler_ctx()` 后变为悬空指针。

**症状**：ESRGAN 放大后，在 img2img 阶段崩溃。

**解决方案**：在释放 upscaler_ctx 前复制数据
```cpp
// ❌ 错误：直接使用返回的指针，free_upscaler_ctx 后变为悬空
sd_image_t esrgan_result = upscale(upscaler_ctx, input_image, 2);
free_upscaler_ctx(upscaler_ctx);
// esrgan_result.data 现在是悬空指针！

// ✅ 正确：先复制数据再释放
sd_image_t esrgan_result = upscale(upscaler_ctx, input_image, 2);
size_t copy_size = esrgan_result.width * esrgan_result.height * esrgan_result.channel;
upscaled_image.data = (uint8_t*)malloc(copy_size);
memcpy(upscaled_image.data, esrgan_result.data, copy_size);
upscaled_image.width = esrgan_result.width;
upscaled_image.height = esrgan_result.height;
upscaled_image.channel = esrgan_result.channel;
free(esrgan_result.data);           // 先释放原始数据
free_upscaler_ctx(upscaler_ctx);   // 再释放 ctx
```

---

### 核心问题三：img2img 必须设置的关键参数

| 参数 | 必须设置的值 | 说明 |
|------|-------------|------|
| `ctx_params.vae_decode_only` | `false` | img2img 必需，否则只解码不编码 |
| `ctx_params.free_params_immediately` | `false` | 保持模型在内存中 |
| `ctx_params.flash_attn` | `true` | GPU 模式下启用 |
| `img_params.init_image.data` | 有效指针 | 不能是 NULL |
| `img_params.mask_image.data` | 全白数据 | 不能是 NULL |
| `img_params.width` | 图片宽度 | 必须与 init_image 匹配 |
| `img_params.height` | 图片高度 | 必须与 init_image 匹配 |

---

### 核心问题四：大图必须启用 VAE Tiling

**场景**：ESRGAN 放大后图片变大（如 2560x1440），VAE 解码大图会爆显存。

**解决方案**：
```cpp
// ctx_params 和 img_params 都需要启用
ctx_params.vae_tiling_params.enabled = true;
ctx_params.vae_tiling_params.tile_size_x = 512;
ctx_params.vae_tiling_params.tile_size_y = 512;
ctx_params.vae_tiling_params.target_overlap = 32;

img_params.vae_tiling_params.enabled = true;
img_params.vae_tiling_params.tile_size_x = 512;
img_params.vae_tiling_params.tile_size_y = 512;
img_params.vae_tiling_params.target_overlap = 32;
```

---

### 新建项目的正确姿势

1. **直接参考官方 CLI 源码** - 不要自己从头写参数，从 `examples/cli/main.cpp` 复制改
2. **对比命令** - 先用 CLI 跑通，再用代码实现
3. **添加调试输出** - 打印关键参数值确认正确

```cpp
// 调试输出示例
printf("[DEBUG] init_image: %dx%d, channels: %d\n", 
       img_params.init_image.width, img_params.init_image.height, img_params.init_image.channel);
printf("[DEBUG] img_params: width=%d, height=%d\n", 
       img_params.width, img_params.height);
```

---

## 目录结构

```
my-img/
├── CMakeLists.txt          # 通用编译模板
├── README.md               # 项目说明
├── claude.md               # 本文件（AI 操作指南）
├── bin/                    # 编译后的二进制程序
├── src/                    # 源代码（每个工具一个目录）
│   └── sd-hires/
│       └── hires_upscaler.cpp
└── include/                # API 头文件
    └── stable-diffusion.h
```

---

## 新建工具项目

本项目的 CMakeLists.txt 是**通用模板**，会自动扫描 `src/` 下所有子目录，生成对应的二进制。

### 原理

```cmake
file(GLOB_RECURSE SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

foreach(src ${SRC_FILES})
    get_filename_component(dir ${src} PATH)
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/src/" "" dir ${dir})
    # 目录名即二进制名
    add_executable(${dir} ${src})
endforeach()
```

### 步骤

1. **创建目录和源文件**
```bash
mkdir -p src/sd-newtool
touch src/sd-newtool/main.cpp
```

2. **编写代码** - 参考上方 API 文档

3. **编译**（自动检测到新目录并生成二进制）
```bash
cd ~/my-img
rm -rf build && mkdir build && cd build
cmake .. -DSD_PATH=/home/dministrator/stable-diffusion.cpp
make -j2  # 使用 -j2 避免内存不足
```

4. **安装**
```bash
cp sd-newtool ../bin/
sudo ln -sf ~/my-img/bin/sd-newtool /usr/local/bin/sd-newtool
```

### 示例：新增一个工具

```bash
# 1. 创建目录
mkdir -p src/sd-img2img

# 2. 编写代码（参考 src/sd-hires/）
vim src/sd-img2img/main.cpp

# 3. 编译（自动生成 sd-img2tool 二进制）
cd build && make

# 4. 安装
cp sd-img2img ../bin/
sudo ln -sf ~/my-img/bin/<工具名> /usr/local/bin/<工具名>
```

---

## 常用命令

```bash
# 完整编译流程
cd ~/my-img
rm -rf build && mkdir build && cd build
cmake .. -DSD_PATH=/home/dministrator/stable-diffusion.cpp
make -j4

# 安装
cp sd-hires ../bin/
sudo ln -sf ~/my-img/bin/sd-hires /usr/local/bin/sd-hires

# 验证
sd-hires --help
```

---

## 最佳开发实践：调试 stable-diffusion.cpp 代码流程

当需要实现某个功能但不知道如何调用时，按以下步骤调试：

### 1. 找到官方 CLI 实现

```bash
# 查看 sd-cli 命令行参数，找到对应的 mode
~/stable-diffusion.cpp/bin/sd-cli --help | grep -i img2img
```

### 2. 追踪代码执行流程

找到 CLI 中处理该功能的代码位置：

```bash
# 例如查找 img2img 相关代码
grep -rn "img2img\|init_image" ~/stable-diffusion.cpp/examples/cli/main.cpp

# 找到关键的参数设置和函数调用
grep -n "generate_image\|sd_img_gen_params" ~/stable-diffusion.cpp/examples/cli/main.cpp
```

### 3. 分析关键函数调用链

**示例：img2img 实现流程**

1. **加载图片**：`load_sd_image_from_file()`
   ```cpp
   sd_image_t init_image = load_sd_image_from_file(ctx, cli_params.image_path);
   ```

2. **设置尺寸**：用图片尺寸设置 width/height（关键！）
   ```cpp
   gen_params.set_width_and_height_if_unset(init_image.width, init_image.height);
   ```

3. **设置生成参数**：
   ```cpp
   sd_img_gen_params_t img_gen_params = {
       ...
       init_image,                    // 必须是加载的图片
       gen_params.get_resolved_width(),   // 用图片尺寸
       gen_params.get_resolved_height(),
       ...
   };
   ```

4. **调用生成函数**：
   ```cpp
   results = generate_image(sd_ctx, &img_gen_params);
   ```

### 4. 关键注意事项

- **顺序很重要**：必须先加载图片，再用图片尺寸设置参数
- **检查库内部实现**：
  ```bash
  # 查看 sd_img_gen_params_init 默认值
  grep -n "sd_img_gen_params_init" ~/stable-diffusion.cpp/src/stable-diffusion.cpp
  ```
- **可用工具**：
  - `strace` - 跟踪系统调用
  - `gdb` - 调试崩溃
  - `valgrind` - 内存检查

### 5. 调试案例

**问题**：img2img 崩溃 `GGML_ASSERT(image.width == tensor->ne[0]) failed`

**调试过程**：
1. 检查 CLI 代码，找到 IMG_GEN 模式处理
2. 发现 CLI 使用 `set_width_and_height_if_unset(init_image.width, init_image.height)`
3. 对比自己代码，发现直接设置了 width/height
4. **结论**：库内部可能依赖某些初始化顺序或默认值

**实际命令对比**：

#### ✅ CLI 成功运行的命令
```bash
~/stable-diffusion.cpp/bin/sd-cli \
  --diffusion-model /opt/image/z_image_turbo-Q6_K.gguf \
  --vae /opt/image/ae.safetensors \
  --llm /opt/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "masterpiece" \
  --steps 5 \
  --cfg-scale 2.0 \
  --diffusion-fa \
  --scheduler karras \
  --vae-tiling \
  -i /mnt/e/app/input.png \
  --strength 0.35 \
  -o /mnt/e/app/test_cli.png
```

#### ❌ sd-hires 失败的命令
```bash
~/my-img/bin/sd-hires \
  --model /opt/image/z_image_turbo-Q6_K.gguf \
  --vae /opt/image/ae.safetensors \
  --llm /opt/image/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --input /mnt/e/app/input.png \
  --output /mnt/e/app/output_sd_hires.png \
  --prompt "masterpiece" \
  --steps 5 \
  --strength 0.35
```

**关键差异排查方向**：
1. CLI 使用 `load_sd_image_from_file()` 加载图片（强制3通道）
2. CLI 在加载图片后才设置 width/height（用图片尺寸）
3. CLI 使用了 `--vae-tiling` 参数
4. CLI 的 ctx_params 和 img_params 设置顺序和默认值可能不同

**下一步**：对比 CLI 源码中的 `sd_ctx_params_init` 默认值与实际传入值的差异

---

## 项目状态

### ✅ 已完成

1. **编译系统**
   - CMakeLists.txt 配置完成，支持 GPU + Flash Attention
   - 依赖库链接完整（CUDA, cuBLAS, OpenMP 等）
   - 自动扫描 `src/` 下所有子目录生成二进制

2. **sd-hires 工具**
   - ESRGAN 超分放大（2x/4x）
   - Deep HighRes Fix 分阶段重绘
   - 支持命令行参数：`--diffusion-model`, `--vae`, `--llm`, `--upscale-model`, `--prompt`, `--strength`, `--steps`, `--scale`, `--deep-hires`, `--vae-tiling`, `--flash-attn`, `--cpu`

3. **sd-img2img 工具**
   - 普通 img2img 重绘
   - Deep HighRes Fix 多阶段生成
   - 支持目标尺寸调整

4. **sd-upscale 工具**
   - ESRGAN 独立超分放大

5. **文档**
   - README.md 完整使用说明
   - claude.md 开发指南
   - `src/sd-img2img/design.md` Deep HighRes Fix 设计文档

### ❌ 待完成 / 遇到的问题

1. **Deep HighRes Fix 效果验证**
   - 当前实现为多次调用版（非原生 latent 空间过渡）
   - 需要实际测试验证分阶段生成的效果
   - 可能需要调整各阶段的 strength 和步数分配

### 📋 下一步

1. 运行实际测试，验证 sd-hires 和 sd-img2img 的生成效果
2. 根据测试结果优化 Deep HighRes Fix 参数
3. 继续开发 sd-inpaint 等工具

