# Claude.md

## 项目概述

**my-img** 是基于 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 的 C++ 图片生成工具集。

**远景**：复刻 ComfyUI 生态理念，但无需 Python 依赖。只有**干净的二进制程序**和**命令管道**。

当前核心组件：
- `sd-workflow` — C++ 版 ComfyUI 工作流引擎（46+ 节点，支持 DAG 执行、缓存、多线程）
- `sd-hires` — AI 高清修复（ESRGAN + Deep HighRes Fix）
- `sd-img2img` — 图生图重绘
- `sd-upscale` — ESRGAN 独立超分

---

## 快速参考

### 编译

#### 第一步：编译 stable-diffusion.cpp（强制使用 build_sd.sh）

> **⚠️ 强制要求**：必须使用 `build_sd.sh` 脚本编译 stable-diffusion.cpp。该脚本会自动应用 my-img 所需的补丁（C API 扩展、Deep HighRes Fix hooks 等），然后执行编译。手动运行 `cmake` 和 `make` 会导致补丁未应用，链接时缺少符号。

```bash
cd ~/my-img/build
./build_sd.sh [选项]
```

**build_sd.sh 选项：**
- `--cuda` / `--no-cuda`：启用/禁用 CUDA（默认启用）
- `--flash-attn` / `--no-flash`：启用/禁用 Flash Attention（默认启用，需 CUDA）
- `--clean`：清理 build 目录后重新编译
- `--jobs N`：并行编译线程数（默认 `$(nproc)`）
- `--help`：显示帮助信息

**示例：**
```bash
./build_sd.sh                    # 默认：CUDA + Flash Attention
./build_sd.sh --no-cuda          # 仅 CPU
./build_sd.sh --clean --jobs 8   # 清理后使用 8 线程编译
```

#### 第二步：编译 my-img

```bash
cd ~/my-img/build
cmake .. -DSD_PATH=/home/dministrator/stable-diffusion.cpp
make -j$(nproc)
```

### 运行测试

```bash
cd ~/my-img/build
./sd-engine-tests
```

### 代码索引（研究第三方项目用）

```bash
# 建立索引
python3 ~/my-img/code_index.py \
  /home/dministrator/stable-diffusion.cpp \
  /home/dministrator/stable-diffusion-cpp.bin

# 搜索符号
python3 ~/my-img/code_search.py ~/stable-diffusion-cpp.bin --find generate_image --json
```

---

## 核心 API 速查

### 上下文管理

```cpp
sd_ctx_params_t ctx_params;
sd_ctx_params_init(&ctx_params);
ctx_params.diffusion_model_path = "model.gguf";
ctx_params.n_threads = 4;
ctx_params.flash_attn = true;
ctx_params.offload_params_to_cpu = false;

sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
free_sd_ctx(sd_ctx);
```

### 图像生成

```cpp
sd_img_gen_params_t img_params;
sd_img_gen_params_init(&img_params);
img_params.prompt = "a cat";
img_params.width = 512;
img_params.height = 512;
img_params.seed = 42;
img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
img_params.sample_params.scheduler = KARRAS_SCHEDULER;
img_params.sample_params.sample_steps = 20;

sd_image_t* result = generate_image(sd_ctx, &img_params);
```

### ESRGAN 超分

```cpp
upscaler_ctx_t* upscaler = new_upscaler_ctx("model.bin", false, false, 4, 128);
sd_image_t upscaled = upscale(upscaler, input_image, 2);
free_upscaler_ctx(upscaler);
```

---

## 关键坑点

### 0. 禁止手动编译 stable-diffusion.cpp（必须使用 build_sd.sh）

```bash
# ❌ 错误：手动编译会导致补丁未应用，链接失败
mkdir -p ~/stable-diffusion.cpp/build && cd ~/stable-diffusion.cpp/build
cmake .. -DSD_CUDA=ON
make -j$(nproc)

# ✅ 正确：使用 build_sd.sh 自动应用补丁并编译
cd ~/my-img/build
./build_sd.sh
```

**后果**：手动编译后，my-img 链接时会报错 `undefined reference to 'sd_sampler_run'`、`undefined reference to 'sd_conditioning_concat'` 等，因为补丁中的 C API 扩展未注入到 stable-diffusion.cpp 源码中。

### 1. mask_image 不能为 NULL

```cpp
// ❌ 错误
img_params.mask_image.data = NULL;

// ✅ 正确：创建全白 mask
size_t mask_size = width * height;
uint8_t* mask_data = (uint8_t*)malloc(mask_size);
memset(mask_data, 255, mask_size);
img_params.mask_image = {(uint32_t)width, (uint32_t)height, 1, mask_data};
```

### 2. ESRGAN 返回的数据在 free_upscaler_ctx 后悬空

```cpp
sd_image_t esrgan_result = upscale(upscaler_ctx, input_image, 2);
size_t copy_size = esrgan_result.width * esrgan_result.height * esrgan_result.channel;
sd_image_t upscaled_image;
upscaled_image.data = (uint8_t*)malloc(copy_size);
memcpy(upscaled_image.data, esrgan_result.data, copy_size);
upscaled_image.width = esrgan_result.width;
upscaled_image.height = esrgan_result.height;
upscaled_image.channel = esrgan_result.channel;
free(esrgan_result.data);
free_upscaler_ctx(upscaler_ctx);
```

### 3. 大图必须启用 VAE Tiling

```cpp
ctx_params.vae_tiling_params.enabled = true;
ctx_params.vae_tiling_params.tile_size_x = 512;
ctx_params.vae_tiling_params.tile_size_y = 512;
ctx_params.vae_tiling_params.target_overlap = 32;
```

---

## 开发规范

- **C++17** 标准
- **默认启用 GPU + Flash Attention**
- **编译 stable-diffusion.cpp 必须使用 `build_sd.sh`**（自动应用补丁，禁止手动 cmake/make）
- 新增节点继承 `sdengine::Node`，使用 `REGISTER_NODE` 宏注册
- 中间图像缓冲区优先使用 `std::vector<uint8_t>`
- 使用 `LOG_INFO()` / `LOG_ERROR()` 输出日志（禁止裸 `printf`）
- 运行 `clang-format` 保持代码风格一致
- 内存分配使用 `try-catch (std::bad_alloc)`，禁止 `new` 后检查 `nullptr`

---

## 项目结构

```
my-img/
├── CMakeLists.txt
├── README.md
├── .clang-format
├── src/
│   ├── sd-engine/      # 工作流引擎核心
│   │   ├── core/       # Workflow / DAGExecutor / Cache
│   │   ├── nodes/      # 节点实现（已拆分为 6 个模块）
│   │   ├── face/       # 人脸 ONNX 模块
│   │   └── tools/      # CLI 工具入口
│   ├── sd-hires/
│   ├── sd-img2img/
│   └── sd-upscale/
├── tests/              # Catch2 测试
└── docs/
    ├── sd-engine-design.md
    ├── face-onnx-design.md
    └── tutorials/
```

---

## 调试技巧

```bash
# 查看 CLI 实现作为参考
grep -rn "generate_image\|sd_img_gen_params" \
  ~/stable-diffusion.cpp/examples/cli/main.cpp

# 监控 GPU
watch -n 0.5 nvidia-smi

# 内存检查
valgrind --leak-check=full ./sd-engine-tests
```
