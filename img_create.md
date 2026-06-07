# img2.sh / img3.sh 出图逻辑与代码调用路径

> **本文档记录 RTX 4090D 24G 专用出图脚本的完整执行流程和代码调用路径。**
>
> **文件位置**: `/opt/my-img/img2.sh`（Z-Image DiT）、`/opt/my-img/img3.sh`（SDXL Base UNet）  
> **目标设备**: NVIDIA GeForce RTX 4090 D (24GB VRAM)  
> **目标分辨率**: 2560×1440 / 3840×2160

---

## 1. 整体架构

```
img2.sh (Shell 脚本)
    │
    ├── 参数解析与预处理
    │   ├── 模型路径验证
    │   ├── 分辨率计算（4090D 优化）
    │   ├── Prompt 增强（质量前缀 + 负面提示词）
    │   └── CLI 命令构建
    │
    ▼
myimg-cli (C++ 可执行文件)
    │
    ├── src/main.cpp (CLI 入口)
    │   ├── 参数解析 (cli_parser.h)
    │   ├── GenerationParams 构建
    │   └── 模式分发
    │
    ├── SDCPPAdapter::initialize()
    │   └── sd.cpp 模型加载
    │
    ├── SDCPPAdapter::generate_single()
    │   └── sd.cpp 图像生成
    │
    └── 后处理管道
        ├── ESRGAN 放大（可选）
        ├── 摄影后期调整 (photo_adjustment.cpp)
        └── 图像保存 (Image::save_to_file)
```

---

## 2. Shell 脚本层 (img2.sh)

### 2.1 参数解析

```bash
# 用法
./img2.sh <prompt> <output_path> <width> <height> [--upscale]

# 示例
./img2.sh "half body portrait..." "~/portrait.png" 2560 1440
```

**参数处理逻辑** (`img2.sh:54-65`):

```bash
# 扫描 --upscale 标志
for arg in "$@"; do
    if [ "$arg" = "--upscale" ]; then
        UPSCALE_FLAG=1
    else
        ARGS+=("$arg")
    fi
done

# 提取位置参数
PROMPT="${ARGS[0]}"
OUTPUT_FILE="${ARGS[1]}"
WIDTH="${ARGS[2]:-1280}"
HEIGHT="${ARGS[3]:-720}"
```

### 2.2 模型路径配置

```bash
MODEL_DIR="${MODEL_DIR:-/opt/image/model}"
SD_CLI="${SD_CLI:-/opt/my-img/build/myimg-cli}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/z_image_turbo-Q8_0.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/ae.safetensors}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
UPSCALE_MODEL="${UPSCALE_MODEL:-$MODEL_DIR/2x_ESRGAN.gguf}"
```

### 2.3 预检流程

```bash
# 1. 检查 myimg-cli 可执行文件
if [ ! -f "$SD_CLI" ]; then exit 1; fi
if [ ! -x "$SD_CLI" ]; then exit 1; fi

# 2. 检查模型文件
for model in "$DIFFUSION_MODEL" "$VAE_MODEL" "$LLM_MODEL"; do
    if [ ! -f "$model" ]; then exit 1; fi
done

# 3. 检查 upscale 模型（如果启用）
if [ "$UPSCALE_FLAG" -eq 1 ]; then
    if [ ! -f "$UPSCALE_MODEL" ]; then exit 1; fi
fi
```

### 2.4 Prompt 增强

**自动添加质量前缀** (`img2.sh:110-113`):

```bash
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, complete face, full head, professional portrait"

if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi
```

**SAG 参数** (`img2.sh:357-358`):

```bash
--sag                      # 启用 Self-Attention Guidance
--sag-scale 1.0            # SAG 强度
```

> 注：SAG 在 20GB 显存模式下默认启用，可通过环境变量控制。

**默认负面提示词** (`img2.sh:233`):

```bash
NEGATIVE_PROMPT="blurry, low quality, worst quality, jpeg artifacts, noise, grain, soft focus, out of focus, hazy, unclear, bad anatomy, deformed, border artifacts, edge distortion, tiling artifacts, edge artifacts, frame distortion, warped edges, stretched proportions, asymmetrical face, off-center, cropped, out of frame, partial face, cut off, incomplete head, cropped head, watermark, text, logo, signature, cropped shoulders, embedding:EasyNegative, embedding:bad-hands-5"
```

---

## 3. 4090D 分辨率优化策略

### 3.1 核心原则

RTX 4090D 24GB 显存允许使用**更高的基础分辨率**，从而减少 latent 放大倍数，显著提升画质：

| 目标分辨率 | 基础分辨率 | 放大倍数 | latent 尺寸变化 | 适用显存 |
|-----------|-----------|---------|----------------|---------|
| 3840×2160 | 2560×1440 | **1.5x** | 320×180 → 480×270 | 24GB |
| 2560×1440 | 1920×1080 | **1.33x** | 240×135 → 320×180 | 20GB+ |
| 1920×1080 | 1536×864  | **1.25x** | 192×108 → 240×135 | 20GB+ |
| 1280×720  | 1024×576  | **1.25x** | 128×72 → 160×90  | 20GB+ |

> 对比 RTX 3080 10GB：基础分辨率仅 1280×720，放大 2x，latent 插值损失大得多。
> 4090D 24GB 可使用更高基础分辨率（如 2048×1152），通过环境变量覆盖。

### 3.2 分辨率计算代码

```bash
if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    # 4K: 2560x1440 基础 → 1.5x 放大
    LOW_W=2560
    LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    # 2K: 1920x1080 基础 → 1.33x 放大（20G显存安全方案）
    LOW_W=1920
    LOW_H=1080
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    # 1080p: 1536x864 基础 → 1.25x 放大
    LOW_W=1536
    LOW_H=864
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    # 720p: 1024x576 基础 → 1.25x 放大
    LOW_W=1024
    LOW_H=576
else
    # 通用计算：使用目标分辨率的 80% 作为基础
    LOW_LATENT_W=$((TARGET_LATENT_W * 4 / 5))
    LOW_LATENT_H=$((TARGET_LATENT_H * 4 / 5))
    
    # 对齐到 8 的倍数
    LOW_LATENT_W=$(((LOW_LATENT_W + 7) / 8 * 8))
    LOW_LATENT_H=$(((LOW_LATENT_H + 7) / 8 * 8))
    
    LOW_W=$((LOW_LATENT_W * 8))
    LOW_H=$((LOW_LATENT_H * 8))
fi

# 保持比例的最小限制：只在单边小于512时按比例放大
if [ "$LOW_W" -lt 512 ] || [ "$LOW_H" -lt 512 ]; then
    TARGET_RATIO=$(echo "scale=6; $WIDTH / $HEIGHT" | bc)
    if [ "$LOW_W" -lt "$LOW_H" ]; then
        LOW_W=512
        LOW_H=$(echo "scale=0; $LOW_W / $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_H" -lt 512 ]; then LOW_H=512; fi
    else
        LOW_H=512
        LOW_W=$(echo "scale=0; $LOW_H * $TARGET_RATIO / 8 * 8" | bc)
        if [ "$LOW_W" -lt 512 ]; then LOW_W=512; fi
    fi
fi
```

---

## 4. CLI 参数构建

### 4.1 默认参数

```bash
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-3.2}"
STEPS="${STEPS:-25}"
HIRES_STEPS="${HIRES_STEPS:-55}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.30}"
```

### 4.2 完整命令构建

```bash
SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --vae "$VAE_MODEL"
  --llm "$LLM_MODEL"
  -p "$PROMPT"
  -n "$NEGATIVE_PROMPT"
  --cfg-scale "$CFG_SCALE"
  --sampling-method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa              # Flash Attention
  --vae-tiling                # VAE Tiling
  --vae-tile-size 128x128     # 20GB 安全模式（可覆盖为 256x256）
  --vae-tile-overlap 0.3
  --freeu                     # FreeU 增强
  --sag                       # Self-Attention Guidance
  --sag-scale 1.0
  --auto-enhance              # 自动增强
  --clarity 0.6               # 清晰度
  --sharpen 1.5               # USM 锐化
  --sharpen-radius 2
  --smart-sharpen 1.2         # 智能锐化
  --smart-sharpen-radius 2
  --edge-sharpen 1.0          # 边缘锐化
  --edge-sharpen-radius 2
  --edge-sharpen-threshold 0.3
  --embd-dir "$MODEL_DIR/embeddings"  # Textual Inversion
  -W "$LOW_W" -H "$LOW_H"     # 基础分辨率
  --steps "$STEPS"
  --hires                     # 启用 HiRes Fix
  --hires-width "$WIDTH"      # 目标宽度
  --hires-height "$HEIGHT"    # 目标高度
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  -s "$SEED"
  -o "$OUTPUT_PATH"
)

# 可选 ESRGAN 放大
if [ "$UPSCALE_FLAG" -eq 1 ]; then
    SD_CMD+=(--upscale-model "$UPSCALE_MODEL")
    SD_CMD+=(--upscale-repeats 1)
    SD_CMD+=(--upscale-tile-size 1440)
fi
```

---

## 5. C++ 代码调用路径

### 5.1 CLI 入口 (src/main.cpp)

```cpp
// 1. 参数解析
CliOptions opts;
if (!parse_args(argc, argv, opts)) {
    print_usage(argv[0]);
    return 1;
}

// 2. 构建 GenerationParams
myimg::GenerationParams params;
params.diffusion_model_path = opts.diffusion_model;  // z_image_turbo-Q8_0.gguf
params.vae_path = opts.vae;                          // ae.safetensors
params.llm_path = opts.llm;                          // Qwen3-4B-Instruct-2507-Q4_K_M.gguf
params.prompt = processed_prompt;
params.negative_prompt = processed_neg_prompt;
params.width = opts.width;                           // 2048 (基础)
params.height = opts.height;                         // 1152 (基础)
params.sample_steps = opts.steps;                    // 25
params.cfg_scale = opts.cfg_scale;                   // 3.2
params.sample_method = parse_sampling_method("euler");
params.scheduler = parse_scheduler("discrete");
params.seed = opts.seed;
params.batch_count = 1;

// 3. 启用增强功能
params.freeu_enabled = true;                         // --freeu
params.sag_enabled = true;                           // --sag
params.flash_attn = true;                            // --diffusion-fa
params.vae_tiling = true;                            // --vae-tiling
params.vae_tile_size_x = 256;
params.vae_tile_size_y = 256;
params.vae_tile_overlap = 0.8;

// 4. HiRes Fix 参数
params.enable_hires = true;
params.hires_width = 2560;
params.hires_height = 1440;
params.hires_strength = 0.30;
params.hires_sample_steps = 55;
params.hires_upscaler = myimg::HiresUpscaler::Latent;
params.hires_scale = 2.0f;

// 5. Embeddings 目录
params.embedding_dir = "/data/models/image/embeddings";
```

### 5.2 适配器初始化 (src/adapters/sdcpp_adapter.cpp)

```cpp
bool SDCPPAdapter::load_model(const GenerationParams& params) {
    // 设置日志回调
    sd_set_log_callback(sd_log_callback, nullptr);
    
    // 构建 sd_ctx_params_t
    sd_ctx_params_t sd_params;
    sd_ctx_params_init(&sd_params);
    
    sd_params.diffusion_model_path = params.diffusion_model_path.c_str();
    sd_params.vae_path = params.vae_path.c_str();
    sd_params.llm_path = params.llm_path.c_str();
    sd_params.n_threads = 64;                          // 使用 64 线程
    sd_params.flash_attn = true;                       // Flash Attention
    sd_params.diffusion_flash_attn = true;
    
    // 扫描 embeddings
    if (!params.embedding_dir.empty()) {
        for (const auto& entry : fs::directory_iterator(params.embedding_dir)) {
            if (ext == ".pt" || ext == ".safetensors" || ext == ".bin") {
                embeddings.push_back({name.c_str(), path.c_str()});
            }
        }
        sd_params.embeddings = embeddings.data();
        sd_params.embedding_count = embeddings.size();
    }
    
    // 创建 sd.cpp 上下文
    ctx_ = new_sd_ctx(&sd_params);
    
    // 设置进度回调
    sd_set_progress_callback(progress_callback_wrapper, this);
    
    return ctx_ != nullptr;
}
```

### 5.3 sd.cpp 模型加载流程

```
new_sd_ctx(&sd_params)
    │
    ├── stable-diffusion.cpp:234
    │   └── loading diffusion model from 'z_image_turbo-Q8_0.gguf'
    │       └── model.cpp:216 (gguf format)
    │           └── model.cpp:265 (init)
    │               └── 453 tensors, 6891.51 MB VRAM
    │
    ├── stable-diffusion.cpp:281
    │   └── loading llm from 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
    │       └── model.cpp:216 (gguf format)
    │           └── 398 tensors, 3555.38 MB VRAM
    │
    ├── stable-diffusion.cpp:295
    │   └── loading vae from 'ae.safetensors'
    │       └── model.cpp:219 (safetensors format)
    │           └── 244 tensors, 160.00 MB VRAM
    │
    ├── stable-diffusion.cpp:320
    │   └── Version: Z-Image
    │
    └── stable-diffusion.cpp:862
        └── total params memory size = 10606.89MB (VRAM)
            text_encoders:  3555.38 MB
            diffusion_model: 6891.51 MB
            vae:             160.00 MB
```

### 5.4 图像生成流程

```cpp
Image SDCPPAdapter::generate_single(const GenerationParams& params) {
    // 构建 sd_img_gen_params_t
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);
    
    gen_params.prompt = params.prompt.c_str();
    gen_params.negative_prompt = params.negative_prompt.c_str();
    gen_params.width = params.width;                   // 2048
    gen_params.height = params.height;                 // 1152
    gen_params.seed = params.seed;
    gen_params.strength = params.strength;
    
    // 采样参数
    gen_params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    gen_params.sample_params.scheduler = DISCRETE_SCHEDULER;
    gen_params.sample_params.sample_steps = 25;
    gen_params.sample_params.guidance.txt_cfg = 3.2;
    
    // FreeU 参数
    gen_params.freeu.enabled = true;
    gen_params.freeu.b1 = 1.3;
    gen_params.freeu.b2 = 1.4;
    gen_params.freeu.s1 = 0.9;
    gen_params.freeu.s2 = 0.2;
    
    // SAG 参数
    gen_params.sag.enabled = true;
    gen_params.sag.scale = 1.0;
    
    // VAE Tiling
    gen_params.vae_tiling_params.enabled = true;
    gen_params.vae_tiling_params.tile_size_x = 256;
    gen_params.vae_tiling_params.tile_size_y = 256;
    gen_params.vae_tiling_params.target_overlap = 0.8;
    
    // HiRes Fix
    gen_params.hires.enabled = true;
    gen_params.hires.target_width = 2560;
    gen_params.hires.target_height = 1440;
    gen_params.hires.denoising_strength = 0.30;
    gen_params.hires.steps = 55;
    gen_params.hires.upscaler = SD_HIRES_UPSCALER_LATENT;
    gen_params.hires.scale = 2.0f;
    
    // 调用 sd.cpp 生成
    sd_image_t* sd_images = generate_image(ctx_, &gen_params);
    
    // 转换为 Image 对象
    return sd_image_to_image(sd_images[0]);
}
```

---

## 6. sd.cpp 内部生成流程

### 6.1 文本编码

```
[SD DEBUG] conditioner.hpp:1699
    └── 解析 prompt 为 token 序列
        
[SD DEBUG] bpe_tokenizer.cpp:183
    └── split prompt to tokens:
        ["master", "piece", ",", "Ġbest", "Ġquality", ...]
        
[SD DEBUG] ggml_extend.hpp:1880
    └── qwen3 compute buffer size: 8.56 MB(VRAM)
    
[SD DEBUG] conditioner.hpp:1953
    └── computing condition graph completed, taking 296 ms
```

### 6.2 第一阶段采样（基础生成）

```
[SD INFO] stable-diffusion.cpp:3377
    └── generate_image 2048x1152
    
[SD INFO] denoiser.hpp:499
    └── get_sigmas with discrete scheduler
    
[SD INFO] stable-diffusion.cpp:2824
    └── sampling using Euler method
    
[SD DEBUG] ggml_extend.hpp:1880
    └── z_image compute buffer size: 1553.65 MB(VRAM)
    
进度条: 25/25 steps, ~3.00s/it
    
[SD INFO] stable-diffusion.cpp:3459
    └── sampling completed, taking 75.17s
```

### 6.3 HiRes Fix（高分辨率修复）

```
[SD INFO] stable-diffusion.cpp:3482
    └── hires fix: upscaling to 2560x1440
    
[SD INFO] denoiser.hpp:499
    └── get_sigmas with discrete scheduler
    
[SD INFO] stable-diffusion.cpp:3519
    └── hires fix: 183 steps, denoising_strength=0.30, sigma_sched_size=56
    
[SD INFO] stable-diffusion.cpp:3294
    └── hires Latent upscale 256x144 -> 320x180
    
[SD DEBUG] ggml_extend.hpp:1880
    └── z_image compute buffer size: 3007.88 MB(VRAM)
    
进度条: 55/55 steps, ~5.97s/it
    
[SD INFO] stable-diffusion.cpp:3580
    └── hires sampling 1/1 completed, taking 328.12s
    
[SD INFO] stable-diffusion.cpp:3601
    └── hires fix completed, taking 328.42s
```

### 6.4 VAE 解码

```
[SD INFO] stable-diffusion.cpp:3202
    └── decoding 1 latents
    
[SD DEBUG] vae.hpp:177
    └── VAE Tile size: 256x180
    
[SD DEBUG] ggml_extend.hpp:951
    └── num tiles : 2, 1
    └── optimal overlap : 0.750000, 0.000000
    └── processing 2 tiles
    
[SD DEBUG] ggml_extend.hpp:1880
    └── vae compute buffer size: 18722.81 MB(VRAM)
    
进度条: 2/2 tiles
    
[SD DEBUG] vae.hpp:207
    └── computing vae decode graph completed, taking 2.90s
    
[SD INFO] stable-diffusion.cpp:3222
    └── decode_first_stage completed, taking 2.90s
```

---

## 7. 后处理流程

### 7.1 摄影后期调整 (src/pipeline/photo_adjustment.cpp)

```cpp
ImageData apply_photo_adjustments(ImageData img_data, const CliOptions& opts) {
    auto tensor = myimg::image_data_to_tensor(img_data);
    
    // 自动增强
    if (opts.auto_enhance) {
        tensor = myimg::auto_enhance(tensor);
    }
    
    // USM 锐化
    if (opts.sharpen_amount > 0.0f) {
        tensor = myimg::usm_sharpen(tensor, opts.sharpen_amount, 
                                     opts.sharpen_radius, opts.sharpen_threshold);
    }
    
    // 清晰度增强
    if (opts.clarity > 0.0f) {
        tensor = myimg::enhance_clarity(tensor, opts.clarity);
    }
    
    // 转换为 ImageData
    img_data = myimg::tensor_to_image_data(tensor);
    return img_data;
}
```

**img2.sh 启用的后处理参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `--auto-enhance` | true | 自动对比度/色彩优化 |
| `--clarity 0.6` | 0.6 | 中间调纹理增强 |
| `--sharpen 1.5` | 1.5 | USM 锐化强度 |
| `--sharpen-radius 2` | 2 | USM 锐化半径 |
| `--smart-sharpen 1.2` | 1.2 | 智能锐化强度 |
| `--smart-sharpen-radius 2` | 2 | 智能锐化半径 |
| `--edge-sharpen 1.0` | 1.0 | 边缘锐化强度 |
| `--edge-sharpen-radius 2` | 2 | 边缘锐化半径 |
| `--edge-sharpen-threshold 0.3` | 0.3 | 边缘阈值 |

### 7.2 图像保存

```cpp
bool Image::save_to_file(const std::string& path) const {
    // 自动检测格式
    if (ext == "bmp") {
        stbi_write_bmp(path.c_str(), width, height, channels, data.data());
    } else if (ext == "jpg" || ext == "jpeg") {
        save_jpeg_internal(path, width, height, channels, data, jpeg_quality);
    } else if (ext == "webp") {
        save_webp_internal(path, width, height, channels, data, jpeg_quality);
    } else {
        // 默认 PNG
        stbi_write_png(path.c_str(), width, height, channels, 
                        data.data(), width * channels);
    }
}
```

---

## 8. SDXL UNet 路径支持（新增）

> 本节记录 SDXL Base 1.0（UNet 架构）在 `myimg-cli` 中的增强功能支持状态。  
> 完整使用示例请参考 `/opt/my-img/img3.sh`。

### 8.1 模型加载差异

与 Z-Image（DiT）使用 `--diffusion-model` + `--vae` + `--llm` 不同，SDXL Base 是**完整 checkpoint**（包含 UNet + VAE + CLIP-L + CLIP-G）：

```bash
myimg-cli \
  --diffusion-model sd_xl_base_1.0.safetensors \
  --clip-l clip_l.safetensors \
  --clip-g clip_g.safetensors \
  -p "..." -o output.png
```

**关键实现细节**：
- `sdcpp_adapter.cpp` 会检测 safetensors header 中的 `first_stage_model` / `conditioner.embedders` 关键字
- 若检测到完整 checkpoint，自动使用 `model_path` 而非 `diffusion_model_path` 加载，避免 VAE/CLIP 权重被跳过
- `sd_params` 必须零初始化，否则 `vae_decode_only` 等布尔字段为随机值

### 8.2 增强功能支持矩阵

| 功能 | Z-Image (DiT) | SDXL Base (UNet) | 说明 |
|------|---------------|------------------|------|
| **FreeU** | ❌ 不适用（无 UNet skip） | ✅ 已集成 | 修改 `UnetModelBlock` 输出块 skip connection |
| **SAG** | ✅ | ✅ | 采样循环中融合，架构无关 |
| **HiRes Fix** | ✅ | ✅ | Latent upscale + refine，已验证 1536×1024 |
| **VAE Tiling** | ✅ | ✅ | 解码峰值显存可控 |
| **VAE Tile Cap** | ✅ | ✅ | 输出 tile >1024 自动缩放，防止 OOM |
| **IPAdapter UNet** | ❌ | ✅ | 70 层 `to_k_ip`/`to_v_ip` 注入 |

### 8.3 VAE 显存优化

SDXL UNet 的 VAE decode 同样会面临大分辨率 OOM。实现两层保护：

1. **CLI 层**：`--vae-tiling --vae-tile-size 128x128 --vae-tile-overlap 0.5`
2. **后端保护**：`vae.hpp:decode()` 中自动 cap 输出 tile size 到 1024

```cpp
// vae.hpp
const int max_output_tile = 1024;
int output_tile_x = tile_size_x * scale_factor;  // latent->pixel
int output_tile_y = tile_size_y * scale_factor;
if (output_tile_x > max_output_tile || output_tile_y > max_output_tile) {
    float scale = (float)max_output_tile / std::max(output_tile_x, output_tile_y);
    tile_size_x = std::max(4, (int)(tile_size_x * scale));
    tile_size_y = std::max(4, (int)(tile_size_y * scale));
}
```

**实测数据**（RTX 3080 20GB）：

| 测试 | 基础分辨率 | HiRes 目标 | VAE Tile | VAE 峰值 VRAM | 结果 |
|------|-----------|-----------|----------|--------------|------|
| SDXL txt2img | 512×512 | 无 | 128×128 | ~1.9GB | ✅ 正常 |
| SDXL + HiRes | 512×512 | 1024×1024 | 128×128 | ~1.9GB | ✅ 正常 |
| SDXL + HiRes | 768×512 | 1536×1024 | 128×85 | ~5.1GB | ✅ 正常 |

### 8.4 FreeU 在 UNet 中的实现

FreeU 作用于 UNet 输出块与输入块之间的 skip connection：

```cpp
// unet.hpp:UnetModelBlock::forward()
if (freeu_enabled) {
    float b = (output_block_idx < 3) ? freeu_b1 : freeu_b2;
    float s = (output_block_idx < 3) ? freeu_s1 : freeu_s2;
    h      = ggml_scale(ctx->ggml_ctx, h, b);        // 增强 backbone
    h_skip = ggml_scale(ctx->ggml_ctx, h_skip, s);   // 削弱 skip
}
h = ggml_concat(ctx->ggml_ctx, h, h_skip, 2);
```

**CLI 参数**（SDXL UNet 已支持）：

```bash
--freeu                  # 启用 FreeU
--freeu-b1 1.3           # 默认 1.3
--freeu-b2 1.4           # 默认 1.4
--freeu-s1 0.9           # 默认 0.9（新增）
--freeu-s2 0.2           # 默认 0.2（新增）
--sag                    # 启用 SAG
--sag-scale 1.0          # 默认 1.0
```

---

## 9. 性能时间线

### 8.1 模型加载阶段

| 步骤 | 时间 | 说明 |
|------|------|------|
| 加载 diffusion model | ~3s | z_image_turbo-Q8_0.gguf, 6891 MB |
| 加载 LLM | ~2s | Qwen3-4B, 3555 MB |
| 加载 VAE | ~1s | ae.safetensors, 160 MB |
| 模型加载总计 | **~9s** | 10606.89 MB VRAM |

### 8.2 生成阶段

| 阶段 | 时间 | 分辨率 | VRAM |
|------|------|--------|------|
| 文本编码 | 0.32s | - | 8.56 MB |
| 第一阶段采样 (25 steps) | **75.17s** | 1920×1080 | ~1300 MB |
| HiRes latent 放大 | 0.30s | 240×135 → 320×180 | - |
| 第二阶段采样 (55 steps) | **328.12s** | 2560×1440 | ~2500 MB |
| VAE 解码 (8 tiles, 128x128) | **3.4s** | 2560×1440 | ~6657 MB |
| 生成总计 | **~407s** | - | - |

### 8.3 后处理阶段

| 步骤 | 时间 | 说明 |
|------|------|------|
| 自动增强 | ~0.1s | 对比度/色彩优化 |
| USM 锐化 | ~0.1s | 细节增强 |
| 清晰度增强 | ~0.1s | 纹理增强 |
| 图像保存 | ~0.5s | PNG 压缩写入 |
| 后处理总计 | **~0.8s** | - |

**总耗时**: ~**6.8 分钟** (模型加载 9s + 生成 407s + 后处理 0.8s)

---

## 10. VRAM 使用分析

### 9.1 模型常驻内存

```
Component          Size (MB)    Percentage
────────────────────────────────────────────
Diffusion Model    6891.51      65.0%
Text Encoder       3555.38      33.5%
VAE                 160.00       1.5%
────────────────────────────────────────────
Total             10606.89     100.0%
```

### 9.2 生成过程 VRAM 峰值

```
Phase                    VRAM (MB)    Total (MB)
─────────────────────────────────────────────────
Model Loading            8980.00       8980.00
Phase 1 Sampling         1300.00      10280.00
HiRes Sampling           2500.00      11480.00
VAE Decoding (128x128)   6657.00      18137.00  ← 峰值
─────────────────────────────────────────────────
```

> 注：使用 128×128 VAE tiling 时峰值约 18GB，RTX 4090D 24GB 安全余量 5.8GB。
> 若使用 256×256 tiling，峰值约 21GB，24GB 显卡仍可安全运行。

---

## 11. 关键代码文件索引

| 文件 | 作用 |
|------|------|
| `/opt/my-img/img2.sh` | Shell 脚本入口，参数解析和 CLI 构建 |
| `/opt/my-img/src/main.cpp` | C++ CLI 入口，参数分发 |
| `/opt/my-img/src/cli/cli_parser.cpp` | CLI 参数解析实现 |
| `/opt/my-img/src/cli/cli_options.h` | CLI 参数结构体定义 |
| `/opt/my-img/src/adapters/sdcpp_adapter.h` | sd.cpp 适配器头文件 |
| `/opt/my-img/src/adapters/sdcpp_adapter.cpp` | sd.cpp 适配器实现 |
| `/opt/my-img/src/pipeline/photo_adjustment.h` | 后处理管道头文件 |
| `/opt/my-img/src/pipeline/photo_adjustment.cpp` | 后处理管道实现 |
| `/opt/my-img/src/utils/image_utils.h` | 图像工具函数 |
| `/opt/my-img/src/utils/image_adjust.h` | 图像调整算法 |
| `/opt/stable-diffusion.cpp/stable-diffusion.h` | sd.cpp C API 头文件 |
| `/opt/stable-diffusion.cpp/stable-diffusion.cpp` | sd.cpp 核心实现 |

---

## 12. 与 img1.sh 的对比

| 维度 | img1.sh (RTX 3080 10GB) | img2.sh (RTX 4090D 24GB) |
|------|------------------------|--------------------------|
| **基础分辨率** | 1280×720 | 1920×1080 |
| **放大倍数** | 2x | 1.33x |
| **HiRes Steps** | 45 | 55 |
| **显存策略** | VAE Tiling + Flash Attention + CPU offload | Flash Attention + VAE Tiling |
| **SAG** | 关闭 | 启用 |
| **画质** | 良好（latent 损失较大） | 优秀（latent 损失极小） |
| **生成时间** | ~11 分钟 | ~6.8 分钟 |
| **VRAM 峰值** | ~9.5 GB | ~18.1 GB (128x128 tiling) |

---

**文档生成时间**: 2026-06-01  
**对应代码版本**: my-img main (commit fdd7a44)  
**sd.cpp 版本**: leejet/stable-diffusion.cpp (upstream be65ac7 + FreeU/SAG patch)
