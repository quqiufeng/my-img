# stable-diffusion.cpp 项目架构详解

## 项目概述

**stable-diffusion.cpp** 是基于 [ggml](https://github.com/ggml-org/ggml) 开发的纯 C++ AI 图像/视频生成库，理念与 [llama.cpp](https://github.com/ggml-org/llama.cpp) 一致。

---

## 目录结构

```
stable-diffusion.cpp/
├── include/                 # 公共头文件
│   └── stable-diffusion.h   # 主 API 头文件
├── src/                     # 核心实现
│   ├── stable-diffusion.cpp # 主入口，模型加载和推理
│   ├── upscaler.cpp         # ESRGAN 超分实现
│   ├── model.cpp            # 模型加载器
│   ├── util.cpp             # 工具函数
│   ├── vae.hpp              # VAE 编解码
│   ├── clip.hpp             # CLIP 文本编码
│   ├── unet.hpp             # UNet 去噪网络
│   ├── diffusion_model.hpp  # 扩散模型
│   ├── denoiser.hpp         # 去噪器
│   ├── lora.hpp             # LoRA 支持
│   ├── control.hpp          # ControlNet
│   ├── ggml_extend.hpp     # GGML 扩展
│   ├── gguf_reader.hpp      # GGUF 格式读取
│   ├── preprocess*.hpp      # 图像预处理
│   ├── rng.hpp              # 随机数生成器
│   └── *.hpp                # 各模型架构
│       ├── sd15/            # SD 1.5
│       ├── sdxl/            # SDXL
│       ├── flux.hpp         # FLUX
│       ├── wan.hpp          # Wan2.1
│       ├── z_image.hpp      # Z-Image
│       ├── qwen_image.hpp   # Qwen-Image
│       └── ...
├── examples/                # 示例程序
│   ├── cli/                 # 命令行工具 (sd-cli)
│   │   └── main.cpp
│   ├── server/              # API 服务 (sd-server)
│   └── common/              # 公共代码
│       └── common.hpp       # 工具函数
├── ggml/                    # GGML 子模块
├── thirdparty/              # 第三方依赖
└── docs/                    # 文档
```

---

## 支持的模型

| 类别 | 模型 | 说明 |
|------|------|------|
| **图像** | SD1.x, SD2.x | 传统 Stable Diffusion |
| | SDXL | SDXL 1.0 |
| | SD-Turbo, SDXL-Turbo | 加速模型 |
| | SD3/SD3.5 | 最新 SD3 |
| | FLUX.1-dev/schnell | FLUX 系列 |
| | FLUX.2-dev/klein | FLUX.2 系列 |
| | Wan2.1/Wan2.2 | 视频模型 |
| | Z-Image | 高效图像模型 |
| | Qwen-Image | 阿里 Qwen 图像 |
| | Anima | Anima 图像 |
| | Chroma | Chroma 图像 |
| **编辑** | FLUX.1-Kontext-dev | 上下文编辑 |
| | Qwen Image Edit | 阿里图像编辑 |
| **视频** | Wan2.1 Vace | 视频生成 |

---

## 核心模块

### 1. 模型加载 (`model.cpp`, `model.h`)

负责加载各种格式的模型文件：
- `.ckpt` / `.pth` (PyTorch)
- `.safetensors` (SafeTensors)
- `.gguf` (GGUF 格式)

```cpp
// 核心 API
ModelLoader model_loader;
model_loader.init_from_file(file_path);           // 通用加载
model_loader.init_from_gguf_file(file_path);      // GGUF 专用
```

### 2. 文本编码 (`clip.hpp`, `t5.hpp`, `llm.hpp`)

| 模块 | 功能 |
|------|------|
| `clip.hpp` | CLIP 文本/图像编码 (SD, SDXL) |
| `t5.hpp` | T5XXL 编码器 (SDXL, FLUX) |
| `llm.hpp` | LLM 编码器 (FLUX, Qwen) |

### 3. 扩散模型 (`diffusion_model.hpp`, `unet.hpp`)

核心去噪网络实现：
- `UNetModel` - 传统 UNet (SD1.x, SDXL)
- `DiffusionModel` - 统一扩散接口

### 4. VAE 编解码 (`vae.hpp`, `tae.hpp`)

| 模块 | 功能 |
|------|------|
| `vae.hpp` | 标准 VAE 编解码 |
| `tae.hpp` | TAESD 快速解码器 |

### 5. 超分 (`upscaler.cpp`, `esrgan.hpp`)

ESRGAN 超分实现，支持 GGUF 格式模型。

### 6. 预处理 (`preprocessing.hpp`)

图像预处理功能：
- Canny 边缘检测
- Depth 深度图
- Normal 法线图

---

## API 详解

### 核心数据类型

```c
// ========== 上下文参数 ==========
typedef struct {
    // 模型路径
    const char* model_path;           // 主模型 (GGUF)
    const char* diffusion_model_path; // 扩散模型
    const char* vae_path;             // VAE 模型
    const char* clip_l_path;           // CLIP L
    const char* clip_g_path;           // CLIP G
    const char* t5xxl_path;            // T5XXL
    const char* llm_path;              // LLM (FLUX)
    
    // 推理控制
    bool vae_decode_only;              // 仅解码 (img2img 必需!)
    bool free_params_immediately;      // 立即释放模型
    int n_threads;                     // CPU 线程
    bool offload_params_to_cpu;        // 卸载到 CPU
    bool keep_vae_on_cpu;              // VAE 留 CPU
    bool keep_clip_on_cpu;             // CLIP 留 CPU
    
    // 加速
    bool flash_attn;                   // Flash Attention
    bool diffusion_flash_attn;         // 扩散模型 Flash Attn
    
    // VAE Tiling (大图防爆显存)
    sd_tiling_params_t vae_tiling_params;
    
    // 高级
    enum sd_type_t wtype;             // 量化类型
    enum rng_type_t rng_type;          // 随机数类型
} sd_ctx_params_t;

// ========== 图像生成参数 ==========
typedef struct {
    // 提示词
    const char* prompt;
    const char* negative_prompt;
    
    // img2img
    sd_image_t init_image;             // 初始图片
    float strength;                    // 重绘强度
    
    // inpaint
    sd_image_t mask_image;             // 蒙版
    
    // 输出
    int width;
    int height;
    
    // 采样
    int64_t seed;
    int sample_steps;
    sd_sample_params_t sample_params;
    
    // VAE Tiling
    sd_tiling_params_t vae_tiling_params;
} sd_img_gen_params_t;

// ========== 采样参数 ==========
typedef struct {
    enum scheduler_t scheduler;        // 调度器
    enum sample_method_t sample_method; // 采样方法
    int sample_steps;
    float eta;
    float flow_shift;
} sd_sample_params_t;

// ========== VAE Tiling 参数 ==========
typedef struct {
    bool enabled;
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
} sd_tiling_params_t;

// ========== 图片结构 ==========
typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;
```

### 核心 API 函数

```c
// ========== 上下文管理 ==========
void sd_ctx_params_init(sd_ctx_params_t* params);
sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params);
void free_sd_ctx(sd_ctx_t* ctx);

// ========== 生成参数 ==========
void sd_img_gen_params_init(sd_img_gen_params_t* params);

// ========== 采样 ==========
void sd_sample_params_init(sd_sample_params_t* params);
enum sample_method_t sd_get_default_sample_method(const sd_ctx_t* ctx);
enum scheduler_t sd_get_default_scheduler(const sd_ctx_t* ctx, enum sample_method_t method);

// ========== 生成 ==========
sd_image_t* generate_image(sd_ctx_t* ctx, const sd_img_gen_params_t* params);

// ========== 超分 ==========
upscaler_ctx_t* new_upscaler_ctx(
    const char* esrgan_path,   // ESRGAN 模型路径
    bool offload_params_to_cpu,
    bool direct,
    int n_threads,
    int tile_size
);
sd_image_t upscale(upscaler_ctx_t* ctx, sd_image_t input, uint32_t factor);
void free_upscaler_ctx(upscaler_ctx_t* ctx);

// ========== 预处理 ==========
bool preprocess_canny(sd_image_t image, ...);

// ========== 模型转换 ==========
bool convert(const char* input_path, const char* vae_path, ...);

// ========== 工具 ==========
int32_t sd_get_num_physical_cores();
const char* sd_get_system_info();
const char* sd_version(void);
```

### 枚举值

```c
// 采样方法
EULER_SAMPLE_METHOD
EULER_A_SAMPLE_METHOD    // 常用
HEUN_SAMPLE_METHOD
DPM2_SAMPLE_METHOD
DPMPP2M_SAMPLE_METHOD    // 推荐
LCM_SAMPLE_METHOD

// 调度器
DISCRETE_SCHEDULER
KARRAS_SCHEDULER         // 推荐，图像更清晰
EXPONENTIAL_SCHEDULER
LCM_SCHEDULER

// 随机数
STD_DEFAULT_RNG
CUDA_RNG                 // 推荐 GPU
CPU_RNG

// 量化类型
SD_TYPE_F32              // 全精度
SD_TYPE_F16              // 半精度
SD_TYPE_Q8_0            // 8位量化 (推荐)
SD_TYPE_Q6_K            // 6位量化
SD_TYPE_Q4_0            // 4位量化
```

---

## 示例程序

### sd-cli (命令行工具)

路径: `examples/cli/main.cpp`

功能：
- 文生图 (txt2img)
- 图生图 (img2img)
- inpaint/outpaint
- 超分 (ESRGAN)
- LoRA
- ControlNet

```bash
# 文生图
./bin/sd-cli -m model.gguf -p "a cat"

# img2img
./bin/sd-cli -m model.gguf -i input.png --strength 0.45 -p "a cat"

# ESRGAN 超分
./bin/sd-cli -m model.gguf --upscale-model esrgan.gguf --upscale-factor 2 -i input.png
```

### sd-server (API 服务)

路径: `examples/server/`

REST API 服务，支持 HTTP 接口调用。

---

## 最佳实践

### 1. img2img 正确初始化

```cpp
// ❌ 错误：默认值会崩溃
sd_ctx_params_t ctx_params;
sd_ctx_params_init(&ctx_params);

// ✅ 正确：必须设置的关键参数
sd_ctx_params_t ctx_params;
sd_ctx_params_init(&ctx_params);
ctx_params.vae_decode_only = false;           // img2img 必需
ctx_params.free_params_immediately = false;
ctx_params.flash_attn = true;                  // GPU 加速
```

### 2. mask_image 不能为 NULL

```cpp
// ❌ 错误
img_params.mask_image.data = NULL;  // 崩溃!

// ✅ 正确：创建全白 mask
size_t mask_size = width * height;
uint8_t* mask = (uint8_t*)malloc(mask_size);
memset(mask, 255, mask_size);
img_params.mask_image.data = mask;
```

### 3. ESRGAN 内存管理

```cpp
// ❌ 错误：释放后指针悬空
sd_image_t result = upscale(ctx, img, 2);
free_upscaler_ctx(ctx);  // result.data 变为悬空!

// ✅ 正确：先复制再释放
sd_image_t result = upscale(ctx, img, 2);
sd_image_t copy = {
    .width = result.width,
    .height = result.height,
    .channel = result.channel,
    .data = malloc(result.width * result.height * result.channel)
};
memcpy(copy.data, result.data, ...);
free(result.data);
free_upscaler_ctx(ctx);
```

### 4. 大图启用 VAE Tiling

```cpp
// 当输出尺寸 > 1024 时启用
img_params.vae_tiling_params.enabled = true;
img_params.vae_tiling_params.tile_size_x = 512;
img_params.vae_tiling_params.tile_size_y = 512;
img_params.vae_tiling_params.target_overlap = 32;
```

---

## 后端支持

| 后端 | 编译选项 | 说明 |
|------|---------|------|
| CPU | 默认 | AVX/AVX2/AVX512 支持 |
| CUDA | `-DSD_CUDA=ON` | NVIDIA GPU |
| Vulkan | `-DSD_VULKAN=ON` | 通用 GPU |
| Metal | `-DSD_METAL=ON` | Apple GPU |
| OpenCL | `-DSD_OPENCL=ON` | 通用 GPU |
| SYCL | `-DSD_SYCL=ON` | Intel GPU |

### Flash Attention

```bash
cmake .. -DSD_CUDA=ON -DSD_FLASH_ATTN=ON
```

---

## 编译指南

```bash
# 1. 克隆
git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git
cd stable-diffusion.cpp

# 2. 配置 (CUDA + Flash Attention)
mkdir build && cd build
cmake .. -DSD_CUDA=ON -DSD_FLASH_ATTN=ON -DCMAKE_BUILD_TYPE=Release

# 3. 编译
make -j$(nproc)

# 4. 输出
ls bin/
# sd-cli    # 命令行工具
# sd-server # API 服务
```

---

## 参考资料

- [官方文档](./docs/)
- [CLI 使用指南](./examples/cli/README.md)
- [模型支持](./docs/)
- [性能优化](./docs/performance.md)
