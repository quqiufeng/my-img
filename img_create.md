# img3.sh / img2.sh 出图逻辑与代码调用路径

> **本文档记录 RTX 4090D 24G 专用出图脚本的完整执行流程和代码调用路径。**
>
> **主要脚本**: `/opt/my-img/img3.sh`（SDXL Base 1.0 UNet，推荐）  
> **维护脚本**: `/opt/my-img/img2.sh`（Z-Image DiT，仅基础 txt2img）  
> **目标设备**: NVIDIA GeForce RTX 4090 D (24GB VRAM) / RTX 3080 20GB  
> **目标分辨率**: 2560×1440 / 3840×2160

---

## 1. 整体架构

```
img3.sh (SDXL Base 1.0, 推荐)
img2.sh (Z-Image DiT, 维护模式)
    │
    ├── 参数解析与预处理
    │   ├── 模型路径验证
    │   ├── 分辨率计算（SDXL / Z-Image 分别优化）
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
    │   └── sd.cpp 模型加载 (SDXL checkpoint / Z-Image GGUF)
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

## 2. Shell 脚本层

### 2.1 img3.sh（SDXL Base 1.0，推荐）

```bash
# 用法
./img3.sh <prompt> <output_path> <width> <height> [--upscale] [--ipadapter]

# 示例
./img3.sh "professional portrait of a young woman, soft studio lighting" "~/portrait.png" 2560 1440

# 启用 IPAdapter UNet
./img3.sh "portrait in style of reference" "~/out.png" 2560 1440 \
  --ipadapter \
  --ipadapter-unet-weights /data/models/image/ipadapter_xl_unet_weights.safetensors \
  --ipadapter-model /data/models/image/ip-adapter-plus_sdxl_vit-h.safetensors \
  --ipadapter-clip-vision /data/models/image/clip_vision_h.safetensors \
  --ipadapter-image ~/reference.png
```

### 2.2 img2.sh（Z-Image DiT，维护模式）

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

### 2.3 模型路径配置

**img3.sh (SDXL Base 1.0)**:

```bash
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
SD_CLI="${SD_CLI:-/opt/my-img/build/myimg-cli}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/sd_xl_base_1.0.safetensors}"
CLIP_L_MODEL="${CLIP_L_MODEL:-$MODEL_DIR/clip_l.safetensors}"
CLIP_G_MODEL="${CLIP_G_MODEL:-$MODEL_DIR/clip_g.safetensors}"
VAE_MODEL="${VAE_MODEL:-}"                          # SDXL Base 内置 VAE，通常留空
UPSCALE_MODEL="${UPSCALE_MODEL:-$MODEL_DIR/2x_ESRGAN.gguf}"
```

**img2.sh (Z-Image DiT)**:

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

**自动添加质量前缀**:

img3.sh (SDXL):
```bash
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, professional photography"

if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi
```

img2.sh (Z-Image):
```bash
QUALITY_PREFIX="masterpiece, best quality, ultra-detailed, sharp focus, 8k uhd, photorealistic, highly detailed, crisp, clear, centered composition, complete face, full head, professional portrait"

if [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi
```

**SAG 参数**:

```bash
--sag                      # 启用 Self-Attention Guidance
--sag-scale 1.0            # SAG 强度
```

> 注：SAG 在 20GB 显存模式下默认启用，可通过环境变量控制。

**默认负面提示词**:

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

**img3.sh (SDXL Base 1.0)**:

```bash
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-7.0}"
STEPS="${STEPS:-25}"
HIRES_STEPS="${HIRES_STEPS:-45}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.30}"

# FreeU: SDXL 保守值（ComfyUI 默认值 1.3/1.4 会导致伪影）
FREEU_B1="${FREEU_B1:-1.05}"
FREEU_B2="${FREEU_B2:-1.1}"
FREEU_S1="${FREEU_S1:-0.95}"
FREEU_S2="${FREEU_S2:-0.8}"
```

**img2.sh (Z-Image DiT)**:

```bash
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-discrete}"
CFG_SCALE="${CFG_SCALE:-3.2}"
STEPS="${STEPS:-20}"
HIRES_STEPS="${HIRES_STEPS:-45}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.35}"
```

### 4.2 完整命令构建 (img3.sh)

```bash
SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --clip-l "$CLIP_L_MODEL"
  --clip-g "$CLIP_G_MODEL"
  -p "$PROMPT"
  -n "$NEGATIVE_PROMPT"
  --cfg-scale "$CFG_SCALE"
  --sampling-method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa              # Flash Attention
  --vae-tiling                # VAE Tiling
  --vae-tile-size 128x128     # 20GB 安全模式
  --vae-tile-overlap 0.5
  --freeu                     # FreeU 增强
  --freeu-b1 1.05             # SDXL 保守值
  --freeu-b2 1.1
  --freeu-s1 0.95
  --freeu-s2 0.8
  --sag                       # Self-Attention Guidance
  --sag-scale 1.0
  --clarity 0.4               # 清晰度
  --sharpen 0.8               # USM 锐化
  --sharpen-radius 2
  --smart-sharpen 0.5         # 智能锐化
  --smart-sharpen-radius 2
  --edge-sharpen 1.5          # 边缘锐化
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

# 可选：外部 VAE
if [ -n "$VAE_MODEL" ]; then
    SD_CMD+=(--vae "$VAE_MODEL")
fi

# 可选：IPAdapter UNet
if [ "$IPADAPTER_FLAG" -eq 1 ]; then
    SD_CMD+=(
        --ipadapter
        --ipadapter-unet-weights "$IPADAPTER_UNET_WEIGHTS"
        --ipadapter-model "$IPADAPTER_MODEL"
        --ipadapter-clip-vision "$IPADAPTER_CLIP_VISION"
        --ipadapter-image "$IPADAPTER_IMAGE"
        --ipadapter-weight "$IPADAPTER_WEIGHT"
    )
fi

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

// 2. 构建 GenerationParams (SDXL Base 1.0)
myimg::GenerationParams params;
params.diffusion_model_path = opts.diffusion_model;  // sd_xl_base_1.0.safetensors
params.clip_l_path = opts.clip_l;                    // clip_l.safetensors
params.clip_g_path = opts.clip_g;                    // clip_g.safetensors
params.prompt = processed_prompt;
params.negative_prompt = processed_neg_prompt;
params.width = opts.width;                           // 1280 (基础)
params.height = opts.height;                         // 720 (基础)
params.sample_steps = opts.steps;                    // 25
params.cfg_scale = opts.cfg_scale;                   // 7.0
params.sample_method = parse_sampling_method("euler");
params.scheduler = parse_scheduler("discrete");
params.seed = opts.seed;
params.batch_count = 1;

// 3. 启用增强功能
params.freeu_enabled = true;                         // --freeu
params.freeu_b1 = 1.05f;                             // SDXL 保守值
params.freeu_b2 = 1.1f;
params.freeu_s1 = 0.95f;
params.freeu_s2 = 0.8f;
params.sag_enabled = true;                           // --sag
params.flash_attn = true;                            // --diffusion-fa
params.vae_tiling = true;                            // --vae-tiling
params.vae_tile_size_x = 128;
params.vae_tile_size_y = 128;
params.vae_tile_overlap = 0.5;

// 4. HiRes Fix 参数
params.enable_hires = true;
params.hires_width = 2560;
params.hires_height = 1440;
params.hires_strength = 0.30;
params.hires_sample_steps = 45;
params.hires_upscaler = myimg::HiresUpscaler::Latent;
params.hires_scale = 2.0f;

// 5. Embeddings 目录
params.embedding_dir = "/data/models/image/embeddings";
```

### 5.2 适配器初始化 (src/adapters/sdcpp_adapter.cpp)

**SDXL Base 1.0 加载**:

```cpp
bool SDCPPAdapter::load_model(const GenerationParams& params) {
    sd_set_log_callback(sd_log_callback, nullptr);
    
    sd_ctx_params_t sd_params;
    sd_ctx_params_init(&sd_params);  // 必须零初始化
    
    // SDXL: 完整 checkpoint，自动检测 header 中的 first_stage_model
    sd_params.model_path = params.diffusion_model_path.c_str();  // sd_xl_base_1.0.safetensors
    sd_params.clip_l_path = params.clip_l_path.c_str();          // clip_l.safetensors
    sd_params.clip_g_path = params.clip_g_path.c_str();          // clip_g.safetensors
    // VAE 内置，vae_path 留空
    sd_params.n_threads = 64;
    sd_params.flash_attn = true;
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
    
    ctx_ = new_sd_ctx(&sd_params);
    sd_set_progress_callback(progress_callback_wrapper, this);
    return ctx_ != nullptr;
}
```

**Z-Image DiT 加载**（维护模式）:

```cpp
// Z-Image 使用 diffusion_model_path + vae_path + llm_path
sd_params.diffusion_model_path = params.diffusion_model_path.c_str();  // z_image_turbo-Q8_0.gguf
sd_params.vae_path = params.vae_path.c_str();                          // ae.safetensors
sd_params.llm_path = params.llm_path.c_str();                          // Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

### 5.3 sd.cpp 模型加载流程

**SDXL Base 1.0**：

```
new_sd_ctx(&sd_params)
    │
    ├── stable-diffusion.cpp:234
    │   └── loading model from 'sd_xl_base_1.0.safetensors'
    │       └── model.cpp:219 (safetensors format)
    │           └── UNet + VAE + CLIP-L + CLIP-G
    │               └── ~6600 tensors, ~6500 MB VRAM
    │
    ├── stable-diffusion.cpp:281
    │   └── loading clip_l from 'clip_l.safetensors'
    │       └── ~500 tensors, ~500 MB VRAM
    │
    ├── stable-diffusion.cpp:295
    │   └── loading clip_g from 'clip_g.safetensors'
    │       └── ~800 tensors, ~1400 MB VRAM
    │
    ├── stable-diffusion.cpp:320
    │   └── Version: SDXL
    │
    └── stable-diffusion.cpp:862
        └── total params memory size = ~8400MB (VRAM)
```

**Z-Image DiT**（维护模式）：

```
new_sd_ctx(&sd_params)
    │
    ├── stable-diffusion.cpp:234
    │   └── loading diffusion model from 'z_image_turbo-Q8_0.gguf'
    │       └── 453 tensors, 6891.51 MB VRAM
    │
    ├── stable-diffusion.cpp:281
    │   └── loading llm from 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
    │       └── 398 tensors, 3555.38 MB VRAM
    │
    ├── stable-diffusion.cpp:295
    │   └── loading vae from 'ae.safetensors'
    │       └── 244 tensors, 160.00 MB VRAM
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
    
    // FreeU 参数 (SDXL 保守值，避免伪影)
    gen_params.freeu.enabled = true;
    gen_params.freeu.b1 = 1.05;
    gen_params.freeu.b2 = 1.1;
    gen_params.freeu.s1 = 0.95;
    gen_params.freeu.s2 = 0.8;
    
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

**img3.sh 启用的后处理参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `--clarity 0.4` | 0.4 | 中间调纹理增强 |
| `--sharpen 0.8` | 0.8 | USM 锐化强度 |
| `--sharpen-radius 2` | 2 | USM 锐化半径 |
| `--smart-sharpen 0.5` | 0.5 | 智能锐化强度 |
| `--smart-sharpen-radius 2` | 2 | 智能锐化半径 |
| `--edge-sharpen 1.5` | 1.5 | 边缘锐化强度 |
| `--edge-sharpen-radius 2` | 2 | 边缘锐化半径 |
| `--edge-sharpen-threshold 0.3` | 0.3 | 边缘阈值 |

**img2.sh 启用的后处理参数**（Z-Image，维护模式）:

| 参数 | 值 | 说明 |
|------|-----|------|
| `--clarity 0.6` | 0.6 | 中间调纹理增强 |
| `--sharpen 1.5` | 1.5 | USM 锐化强度 |
| `--smart-sharpen 1.2` | 1.2 | 智能锐化强度 |
| `--edge-sharpen 1.0` | 1.0 | 边缘锐化强度 |

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

FreeU 作用于 UNet 输出块与输入块之间的 skip connection。实现采用**半通道激活缩放**（匹配 ComfyUI `nodes_freelunch.py`）：

```cpp
// unet.hpp:UnetModelBlock::forward()
if (freeu_enabled) {
    float b = 1.0f, s = 1.0f;
    int64_t ch = h->ne[2];  // channel dim
    if (ch == model_channels * 4) {
        b = freeu_b1; s = freeu_s1;
    } else if (ch == model_channels * 2) {
        b = freeu_b2; s = freeu_s2;
    }
    if (b != 1.0f && ch > 1) {
        int64_t half_c = ch / 2;
        if (!ggml_is_contiguous(h)) {
            h = ggml_cont(ctx->ggml_ctx, h);
        }
        // 仅缩放 backbone 的前半通道
        auto h_first = ggml_view_4d(ctx->ggml_ctx, h,
                                    h->ne[0], h->ne[1], half_c, h->ne[3],
                                    h->nb[1], h->nb[2], h->nb[3], 0);
        h_first = ggml_scale_inplace(ctx->ggml_ctx, h_first, b);
    }
    if (s != 1.0f) {
        h_skip = ggml_ext_scale(ctx->ggml_ctx, h_skip, s);  // 削弱 skip
    }
}
h = ggml_concat(ctx->ggml_ctx, h, h_skip, 2);
```

**关键修正**：早期实现使用全通道 `ggml_scale` 且采用 ComfyUI 默认值（b1=1.3, b2=1.4），导致 SDXL UNet 严重伪影（气泡/纹理 corruption）。原因是：
1. SDXL GroupNorm + SiLU 对全通道缩放敏感，统计量偏移导致 corruption
2. 简化空间缩放（无 Fourier 滤波）比原始论文更强

**SDXL 安全参数**：

```bash
--freeu                  # 启用 FreeU
--freeu-b1 1.05          # SDXL 保守值（默认 1.3 会导致伪影）
--freeu-b2 1.1
--freeu-s1 0.95
--freeu-s2 0.8
--sag                    # 启用 SAG
--sag-scale 1.0          # 默认 1.0
```

---

## 9. 性能时间线

### 9.1 SDXL Base 1.0 (img3.sh) 预估

| 步骤 | 时间 | 说明 |
|------|------|------|
| 加载 SDXL checkpoint | ~5s | sd_xl_base_1.0.safetensors, ~6500 MB |
| 加载 CLIP-L | ~1s | clip_l.safetensors, ~500 MB |
| 加载 CLIP-G | ~2s | clip_g.safetensors, ~1400 MB |
| 模型加载总计 | **~8s** | ~8400 MB VRAM |

| 阶段 | 时间 | 分辨率 | VRAM |
|------|------|--------|------|
| 文本编码 (CLIP-L+G) | ~0.5s | - | ~50 MB |
| 第一阶段采样 (25 steps) | **~90s** | 1280×720 | ~1500 MB |
| HiRes latent 放大 | 0.30s | 160×90 → 320×180 | - |
| 第二阶段采样 (45 steps) | **~280s** | 2560×1440 | ~2800 MB |
| VAE 解码 (tiling) | **~4s** | 2560×1440 | ~6-8 GB |
| 生成总计 | **~375s** | - | - |

**总耗时**: ~**6.5 分钟** (模型加载 8s + 生成 375s + 后处理 0.8s)

### 9.2 Z-Image DiT (img2.sh) 实测

| 步骤 | 时间 | 说明 |
|------|------|------|
| 加载 diffusion model | ~3s | z_image_turbo-Q8_0.gguf, 6891 MB |
| 加载 LLM | ~2s | Qwen3-4B, 3555 MB |
| 加载 VAE | ~1s | ae.safetensors, 160 MB |
| 模型加载总计 | **~9s** | 10606.89 MB VRAM |

| 阶段 | 时间 | 分辨率 | VRAM |
|------|------|--------|------|
| 文本编码 | 0.32s | - | 8.56 MB |
| 第一阶段采样 (25 steps) | **75.17s** | 1920×1080 | ~1300 MB |
| HiRes latent 放大 | 0.30s | 240×135 → 320×180 | - |
| 第二阶段采样 (55 steps) | **328.12s** | 2560×1440 | ~2500 MB |
| VAE 解码 (8 tiles, 128x128) | **3.4s** | 2560×1440 | ~6657 MB |
| 生成总计 | **~407s** | - | - |

### 9.3 后处理阶段

| 步骤 | 时间 | 说明 |
|------|------|------|
| USM 锐化 | ~0.1s | 细节增强 |
| 清晰度增强 | ~0.1s | 纹理增强 |
| 图像保存 | ~0.5s | PNG 压缩写入 |
| 后处理总计 | **~0.8s** | - |

---

## 10. VRAM 使用分析

### 10.1 SDXL Base 1.0 模型常驻内存

```
Component          Size (MB)    Percentage
────────────────────────────────────────────
UNet + VAE         6500.00      77.4%
CLIP-L              500.00       6.0%
CLIP-G             1400.00      16.6%
────────────────────────────────────────────
Total              8400.00     100.0%
```

### 10.2 Z-Image DiT 模型常驻内存（维护模式）

```
Component          Size (MB)    Percentage
────────────────────────────────────────────
Diffusion Model    6891.51      65.0%
Text Encoder       3555.38      33.5%
VAE                 160.00       1.5%
────────────────────────────────────────────
Total             10606.89     100.0%
```

### 10.3 生成过程 VRAM 峰值

**SDXL Base 1.0 (img3.sh)**：

```
Phase                    VRAM (MB)    Total (MB)
─────────────────────────────────────────────────
Model Loading            8400.00       8400.00
Phase 1 Sampling         1500.00       9900.00
HiRes Sampling           2800.00      11200.00
VAE Decoding (128x128)   8000.00      16400.00  ← 峰值
─────────────────────────────────────────────────
```

> 注：SDXL 使用 128×128 VAE tiling 时峰值约 16GB，RTX 3080 20GB 安全余量 4GB。

**Z-Image DiT (img2.sh)**：

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

---

## 14. 关键代码文件索引

| 文件 | 作用 |
|------|------|
| `/opt/my-img/img3.sh` | SDXL 主脚本，参数解析和 CLI 构建（推荐） |
| `/opt/my-img/img2.sh` | Z-Image 维护脚本 |
| `/opt/my-img/src/main.cpp` | C++ CLI 入口，参数分发 |
| `/opt/my-img/src/cli/cli_parser.cpp` | CLI 参数解析实现 |
| `/opt/my-img/src/cli/cli_options.h` | CLI 参数结构体定义 |
| `/opt/my-img/src/adapters/sdcpp_adapter.h` | sd.cpp 适配器头文件 |
| `/opt/my-img/src/adapters/sdcpp_adapter.cpp` | sd.cpp 适配器实现 |
| `/opt/my-img/src/pipeline/photo_adjustment.h` | 后处理管道头文件 |
| `/opt/my-img/src/pipeline/photo_adjustment.cpp` | 后处理管道实现 |
| `/opt/my-img/src/utils/ipadapter.cpp` | IPAdapter UNet 实现（SDXL Plus） |
| `/opt/my-img/src/utils/ipadapter.h` | IPAdapter 头文件 |
| `/opt/my-img/src/utils/image_utils.h` | 图像工具函数 |
| `/opt/my-img/src/utils/image_adjust.h` | 图像调整算法 |
| `/opt/stable-diffusion.cpp/stable-diffusion.h` | sd.cpp C API 头文件 |
| `/opt/stable-diffusion.cpp/stable-diffusion.cpp` | sd.cpp 核心实现 |
| `/opt/stable-diffusion.cpp/src/unet.hpp` | UNet 前向传播（FreeU 半通道缩放） |

---

## 12. 脚本对比

| 维度 | img3.sh (SDXL, 推荐) | img2.sh (Z-Image, 维护) |
|------|----------------------|------------------------|
| **模型架构** | SDXL Base 1.0 UNet | Z-Image DiT |
| **文本编码器** | CLIP-L + CLIP-G | Qwen3-4B LLM |
| **基础分辨率** | 1280×720 | 1920×1080 |
| **放大倍数** | 2x | 1.33x |
| **CFG Scale** | 7.0 | 3.2 |
| **HiRes Steps** | 45 | 45 |
| **显存策略** | Flash Attention + VAE Tiling | Flash Attention + VAE Tiling |
| **FreeU** | ✅ 保守值 (1.05/1.1) | ✅ 激进值 (1.4/1.5, DiT 路径) |
| **SAG** | ✅ | ✅ |
| **IPAdapter** | ✅ UNet cross-attention | ❌ 已移除 |
| **画质** | 优秀（细节丰富） | 良好 |
| **生成时间** | ~6.5 分钟 | ~6.8 分钟 |
| **VRAM 峰值** | ~16.4 GB | ~18.1 GB |

## 13. 与 img1.sh 的对比

| 维度 | img1.sh (RTX 3080 10GB) | img2.sh (RTX 4090D 24GB) | img3.sh (RTX 3080/4090) |
|------|------------------------|--------------------------|------------------------|
| **基础分辨率** | 1280×720 | 1920×1080 | 1280×720 |
| **放大倍数** | 2x | 1.33x | 2x |
| **HiRes Steps** | 45 | 55 | 45 |
| **显存策略** | VAE Tiling + Flash Attention + CPU offload | Flash Attention + VAE Tiling | Flash Attention + VAE Tiling |
| **SAG** | 关闭 | 启用 | 启用 |
| **画质** | 良好（latent 损失较大） | 优秀（latent 损失极小） | 优秀（SDXL 细节） |
| **生成时间** | ~11 分钟 | ~6.8 分钟 | ~6.5 分钟 |
| **VRAM 峰值** | ~9.5 GB | ~18.1 GB | ~16.4 GB |

---

**文档更新时间**: 2026-06-08  
**对应代码版本**: my-img main (SDXL primary, Z-Image maintenance)  
**sd.cpp 版本**: leejet/stable-diffusion.cpp (upstream + FreeU half-channel + SAG + IPAdapter UNet patch)
