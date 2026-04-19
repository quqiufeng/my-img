# HiRes Fix 实现文档

## 备份时间
2025-04-20

## 修改的文件

### 核心修改（stable-diffusion.cpp 源码）
1. `examples/common/common.cpp` - 添加 CLI 参数解析（--hires-fix 等）
2. `examples/common/common.h` - 添加 hires 参数字段
3. `include/stable-diffusion.h` - 添加 API 参数结构
4. `src/stable-diffusion.cpp` - **核心修改**：
   - **Hook 系统**：`sd_set_latent_hook`, `sd_set_guidance_hook`（my-img 基础）
   - **Node API**：`sd_encode_prompt`, `sd_sampler_run`, `sd_create_empty_latent` 等（ComfyUI 风格）
   - **HiRes Fix**：两阶段采样实现
5. `src/vae.hpp` - VAE tiling 相关修改

### 新增文件
6. `include/stable-diffusion-ext.h` - **扩展头文件**（Node API 和 Hook 声明）
   - 这个文件是 my-img 项目的核心扩展
   - 声明了所有 Node API 函数和 Hook 接口
   - 需要复制到 stable-diffusion.cpp/include/ 目录

## Patch 文件
- `hires-fix.patch` - **完整补丁**（包含所有修改：Hook + Node API + HiRes Fix）
- `stable-diffusion.cpp.patch` - **核心补丁**（Hook 系统 + Node API + HiRes Fix 实现）
- `stable-diffusion.h.patch` - API 参数结构补丁
- `stable-diffusion-ext.h` - **扩展头文件**（Node API 和 Hook 声明，需复制到 include/）
- `common.cpp.patch` - CLI 参数解析补丁
- `common.h.patch` - 参数结构补丁
- `vae.hpp.patch` - VAE tiling 补丁

## 使用方法

### 应用完整补丁（推荐）
```bash
cd /path/to/stable-diffusion.cpp

# 1. 先复制扩展头文件（重要！）
cp /path/to/hires-fix-backup/stable-diffusion-ext.h include/

# 2. 应用完整补丁
git apply /path/to/hires-fix-backup/hires-fix.patch

# 3. 编译
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 手动应用（如果完整补丁失败）
```bash
cd /path/to/stable-diffusion.cpp

# 1. 复制扩展头文件
cp /path/to/hires-fix-backup/stable-diffusion-ext.h include/

# 2. 逐个应用补丁
git apply /path/to/hires-fix-backup/stable-diffusion.cpp.patch
git apply /path/to/hires-fix-backup/stable-diffusion.h.patch
git apply /path/to/hires-fix-backup/common.cpp.patch
git apply /path/to/hires-fix-backup/common.h.patch
git apply /path/to/hires-fix-backup/vae.hpp.patch
```

---

## 架构说明

### my-img 与 stable-diffusion.cpp 的关系

```
stable-diffusion.cpp (上游)
    │
    ├── 基础功能：txt2img, img2img, VAE, UNet
    │
    ├── my-img 扩展（本 Patch）
    │   ├── Hook 系统（基础）
    │   │   ├── sd_set_latent_hook()     - 拦截采样过程中的 latent
    │   │   └── sd_set_guidance_hook()   - 动态修改 cfg_scale
    │   │
    │   ├── Node API（ComfyUI 风格）
    │   │   ├── sd_encode_prompt()       - 文本编码
    │   │   ├── sd_create_empty_latent() - 创建空 latent
    │   │   ├── sd_sampler_run()         - 独立采样
    │   │   └── sd_decode_latent()       - 解码 latent
    │   │
    │   └── HiRes Fix（基于 Hook + Node API 实现）
    │       ├── 第一阶段：小图完整采样
    │       ├── Latent 空间放大
    │       └── 第二阶段：添加噪声 + 重新采样
    │
    └── sd-engine / sd-hires（基于 my-img 扩展开发）
        ├── 使用 Hook 系统实现自定义采样逻辑
        └── 使用 Node API 构建工作流
```

### 为什么需要 Hook 系统

**问题**：stable-diffusion.cpp 的 `generate_image()` 是一个黑盒函数，无法干预采样过程。

**解决方案**：
- **Latent Hook**：在每次采样步骤前拦截 latent，允许修改（如插值放大）
- **Guidance Hook**：动态调整 cfg_scale，实现渐进式引导

**应用场景**：
- HiRes Fix：在特定步骤放大 latent
- 动态分辨率：根据步骤调整图像尺寸
- 渐进式渲染：逐步增加细节

---

## HiRes Fix 原理详解

### 什么是 HiRes Fix

HiRes Fix 是一种在 latent 空间实现图像超分辨率的技术，用于解决直接生成高分辨率图片时的显存不足和细节丢失问题。

**核心思想**：
- 先在小分辨率下生成图片（细节丰富，显存占用低）
- 在 latent 空间放大图片（不经过 VAE encode/decode）
- 添加噪声后重新采样，让 AI "重塑"细节

### 为什么需要 HiRes Fix

1. **显存限制**：直接生成 2560x1440 需要大量显存，容易 OOM
2. **栈溢出**：VAE encode 大图片时计算缓冲区过大（848MB），导致栈溢出
3. **细节丢失**：直接生成高分辨率时，模型难以在全局和局部细节间平衡

### 正确 vs 错误的实现

#### ❌ 错误实现（初始版本）
```cpp
// 错误：在随机噪声上放大，没有先生成小图
latents.init_latent = interpolate(random_noise, target_size);
// 然后直接采样 - 结果：没有利用小图的信息
```

#### ❌ 错误实现（第二版）
```cpp
// 错误：虽然生成了小图，但噪声添加逻辑有误
for (int i = 0; i < steps; ++i) {
    float sigma = sigmas[i];
    // 添加噪声到 x_t...
}
// BUG：这里覆盖了所有噪声！
x_t = ggml_scale(ctx, x_t, 1.0f / 0.18215f);
```

#### ✅ 正确实现（最终版）
```cpp
// 第一阶段：生成小图（完整采样）
sd::Tensor<float> x_0 = sample(..., full_sigmas);

// Latent 空间放大
sd::Tensor<float> upscaled_x0 = interpolate(x_0, target_size, Nearest);

// 第二阶段：重新采样（部分步数）
// 1. 截断 sigmas（基于 hires_strength）
std::vector<float> hires_sigmas(new_sigmas.begin() + start_step, new_sigmas.end());

// 2. 生成新噪声
sd::Tensor<float> hires_noise = randn_like(upscaled_x0);

// 3. 采样器自动用 sigmas[0] 添加噪声并去噪
sd::Tensor<float> result = sample(..., upscaled_x0, hires_noise, hires_sigmas);
```

### 关键原理

#### 1. 两阶段采样
- **阶段 1**：小尺寸完整采样（15-20步）→ 得到 clean latent
- **阶段 2**：大尺寸部分采样（strength * 总步数）→ 添加噪声并去噪

#### 2. Sigma 截断
```cpp
// 根据 hires_strength 计算起始步数
// strength=0.5：从中间开始，保留 50% 步数
// strength=0.3：从 70% 处开始，保留 30% 步数
int start_step = (1.0f - hires_strength) * (sigmas.size() - 1);
hires_sigmas = sigmas[start_step ... end];
```

#### 3. 噪声添加机制
采样器通过 `denoiser->noise_scaling(sigmas[0], noise, init_latent)` 自动添加正确数量的噪声：
- `sigmas[0]` 较大 → 添加更多噪声 → AI 有更多"创作空间"
- `sigmas[0]` 较小 → 添加较少噪声 → 保留更多原图细节

#### 4. 为什么不用 VAE Encode
- VAE 可能处于 `decode_only` 模式（无法 encode）
- VAE encode 大图片容易栈溢出
- Latent 插值更快速、内存效率更高

---

## 代码实现详解

### 1. 参数定义

#### `include/stable-diffusion.h`
```cpp
typedef struct {
    // ... 原有字段 ...
    
    // HiRes Fix 参数
    bool hires_fix;           // 是否启用 HiRes Fix
    int hires_width;          // 目标宽度
    int hires_height;         // 目标高度
    float hires_strength;     // 去噪强度 (0.0-1.0)
} sd_img_gen_params_t;
```

#### `examples/common/common.h`
```cpp
typedef struct {
    // ... 原有字段 ...
    
    // HiRes Fix
    bool hires_fix = false;
    int hires_width = 0;
    int hires_height = 0;
    float hires_strength = 0.5f;
} sd_params_t;
```

#### `examples/common/common.cpp`
```cpp
// CLI 参数解析
if (arg == "--hires-fix") {
    params.hires_fix = true;
}
if (arg == "--hires-width") {
    params.hires_width = std::stoi(argv[++i]);
}
if (arg == "--hires-height") {
    params.hires_height = std::stoi(argv[++i]);
}
if (arg == "--hires-strength") {
    params.hires_strength = std::stof(argv[++i]);
}
```

### 2. 核心实现

#### `src/stable-diffusion.cpp` - generate_image 函数

**第一阶段：生成小图**
```cpp
// 使用原始尺寸生成（如 640x360）
sd::Tensor<float> noise = sd::randn_like<float>(latents.init_latent, sd_ctx->sd->rng);

sd::Tensor<float> x_0 = sd_ctx->sd->sample(
    sd_ctx->sd->diffusion_model,
    true,                    // is_image_generation
    latents.init_latent,     // 随机噪声
    std::move(noise),
    embeds.cond,
    embeds.uncond,
    // ... 其他参数 ...
    plan.sigmas,             // 完整 sigmas（15-20步）
    // ...
);
```

**第二阶段：HiRes Fix 处理**
```cpp
if (sd_img_gen_params->hires_fix && 
    sd_img_gen_params->hires_width > 0 && 
    sd_img_gen_params->hires_height > 0) {
    
    // 1. 计算目标 latent 尺寸
    int vae_scale_factor = sd_ctx->sd->get_vae_scale_factor();
    int target_w = sd_img_gen_params->hires_width / vae_scale_factor;
    int target_h = sd_img_gen_params->hires_height / vae_scale_factor;
    
    // 2. 放大第一阶段的 x_0（clean latent）
    auto current_shape = x_0.shape();
    std::vector<int64_t> target_shape = current_shape;
    target_shape[0] = target_w;
    target_shape[1] = target_h;
    
    sd::Tensor<float> upscaled_x0 = sd::ops::interpolate(
        x_0, target_shape, sd::ops::InterpolateMode::Nearest);
    
    // 3. 放大辅助 latents
    sd::Tensor<float> upscaled_concat_latent;
    if (!latents.concat_latent.empty()) {
        upscaled_concat_latent = sd::ops::interpolate(
            latents.concat_latent, target_shape, sd::ops::InterpolateMode::Nearest);
    }
    
    // 4. 重新计算 sigmas（基于新尺寸）
    float hires_strength = sd_img_gen_params->hires_strength > 0 
                           ? sd_img_gen_params->hires_strength : 0.5f;
    
    scheduler_t scheduler = resolve_scheduler(sd_ctx, ...);
    std::vector<float> new_sigmas = sd_ctx->sd->denoiser->get_sigmas(
        plan.total_steps,
        sd_ctx->sd->get_image_seq_len(hires_height, hires_width),
        scheduler,
        sd_ctx->sd->version);
    
    // 5. 截断 sigmas
    std::vector<float> hires_sigmas;
    if (hires_strength < 1.0f && new_sigmas.size() > 1) {
        int start_step = static_cast<int>(
            (1.0f - hires_strength) * (new_sigmas.size() - 1));
        hires_sigmas = std::vector<float>(
            new_sigmas.begin() + start_step, new_sigmas.end());
    }
    
    // 6. 更新 conditioning
    SDCondition hires_cond = embeds.cond;
    if (!upscaled_concat_latent.empty()) {
        hires_cond.c_concat = upscaled_concat_latent;
    }
    
    // 7. 第二阶段采样
    sd::Tensor<float> hires_noise = sd::randn_like<float>(
        upscaled_x0, sd_ctx->sd->sampler_rng);
    
    sd::Tensor<float> hires_x0 = sd_ctx->sd->sample(
        sd_ctx->sd->diffusion_model,
        true,
        upscaled_x0,             // 放大的 clean latent
        std::move(hires_noise),  // 新噪声
        hires_cond,
        // ...
        hires_sigmas,            // 截断后的 sigmas
        // ...
    );
    
    x_0 = std::move(hires_x0);
}
```

### 3. 关键函数说明

#### `sd::ops::interpolate()`
- 在 latent 空间进行插值放大
- 使用 `Nearest` 模式保留 latent 特征
- 避免 VAE encode/decode 的开销和限制

#### `denoiser->get_sigmas()`
- 生成 sigma 时间步序列
- 基于图像尺寸计算（某些 scheduler 与尺寸相关）
- 返回从大到小的 sigma 值

#### `sample()` 函数
- 如果提供 `noise`，使用 `noise_scaling(sigmas[0], noise, init_latent)` 添加噪声
- 从 `sigmas[0]` 开始，迭代到 `sigmas[end]`
- 每步去噪后返回 clean latent

---

## 性能对比

| 方法 | 时间 | 显存 | 效果 |
|------|------|------|------|
| 直接生成 640x360 | ~10s | ~8GB | 基础质量 |
| 直接生成 1280x720 | ~35s | ~9GB | 可能细节丢失 |
| 直接生成 2560x1440 | 栈溢出 | - | 无法生成 |
| HiRes Fix 640→1280 | ~45s | ~9GB | 细节丰富 |
| HiRes Fix 640→2560 | ~60s | ~9GB | 高清大图 |

**时间构成**：
- 第一阶段（小图）：~10s
- 第二阶段（大图）：~35s
- VAE Decode（大图）：~2s

---

## 使用示例

### 基础用法
```bash
./sd-cli \
  -m model.gguf \
  --vae vae.safetensors \
  -p "beautiful landscape" \
  --width 640 --height 360 \
  --hires-fix \
  --hires-width 1280 \
  --hires-height 720 \
  --hires-strength 0.5 \
  -o output.png
```

### 生成 2K 壁纸
```bash
./sd-cli \
  -m model.gguf \
  --vae vae.safetensors \
  -p "Swiss Alps, majestic mountain peaks, crystal clear lake" \
  --width 640 --height 360 \
  --hires-fix \
  --hires-width 2560 \
  --hires-height 1440 \
  --hires-strength 0.5 \
  --steps 20 \
  -o wallpaper_2k.png
```

### 参数调优
- **`--hires-strength 0.3`**：保留更多原图结构，细节变化小
- **`--hires-strength 0.5`**：平衡，推荐默认值
- **`--hires-strength 0.7`**：更大变化，可能产生更多细节但也可能偏离原图

---

## 注意事项

1. **VAE Tiling**：生成大图时建议开启 `--vae-tiling` 避免显存不足
2. **基础尺寸**：建议基础尺寸不小于 512x512，否则小图质量太差
3. **Strength 范围**：0.0-1.0，推荐 0.3-0.7
4. **Scheduler 影响**：不同 scheduler 的 sigma 曲线不同，效果可能有差异
5. **模型兼容**：适用于 SDXL、Z-Image 等基于 latent diffusion 的模型

---

## 验证修改是否生效

### 方法1：检查二进制文件中的字符串
```bash
# 检查是否包含 "Second phase" 字符串（新代码的标志）
grep -c "Second phase" /home/dministrator/stable-diffusion.cpp/build/bin/sd-cli

# 预期输出：3（表示包含3处"Second phase"字符串）
# 如果输出 0，说明二进制还是旧版本，需要重新编译
```

### 方法2：运行测试并查看日志
```bash
# 运行一次 HiRes Fix 生成
cd /home/dministrator/my-shell/3080

MODEL_DIR="/opt/image/model"
OUTPUT_DIR="$HOME/generated_images"
mkdir -p "$OUTPUT_DIR"

/home/dministrator/stable-diffusion.cpp/build/bin/sd-cli \
  --diffusion-model "$MODEL_DIR/z_image_turbo-Q5_K_M.gguf" \
  --vae "$MODEL_DIR/ae.safetensors" \
  --llm "$MODEL_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  -p "A beautiful mountain landscape" \
  --cfg-scale 1.01 \
  --sampling-method euler \
  --diffusion-fa \
  --vae-tiling \
  --vae-tile-size 64x64 \
  -W 640 -H 360 \
  --hires-fix \
  --hires-width 1280 \
  --hires-height 720 \
  --hires-strength 0.5 \
  --steps 15 \
  -s 42 \
  -o "$OUTPUT_DIR/test_hires.png" 2>&1 | tee /tmp/hires_test.log

# 检查日志中是否包含关键信息
echo "=== 检查关键日志 ==="
grep "HiRes Fix" /tmp/hires_test.log
grep "Second phase" /tmp/hires_test.log
grep "sampling completed" /tmp/hires_test.log

# 预期看到：
# [HiRes Fix] Will generate at 640x360 then upscale to 1280x720
# [HiRes Fix] Processing: 640x360 -> 1280x720
# [HiRes Fix] Starting second phase sampling...
# [HiRes Fix] Second phase: latent 160x90, strength=0.50
# [HiRes Fix] Second phase sampling completed, taking X.XXs
# sampling completed, taking X.XXs（出现两次：第一次小图，第二次大图）
```

### 方法3：对比时间
```bash
# HiRes Fix 应该比直接生成大图慢（因为采样了两次）
# 640x360 基础生成：约 10 秒
# 1280x720 HiRes Fix：约 45-50 秒（第一次 10s + 第二次 35s）
# 1280x720 直接生成：约 35 秒

# 如果 HiRes Fix 时间和直接生成时间差不多（都是 ~35s），
# 说明只运行了一次采样，修改未生效
```

### 方法4：检查图片细节
```bash
# 对比三张图片
# 1. 基础小图（640x360）
# 2. HiRes Fix 结果（1280x720）
# 3. 直接生成大图（1280x720）

# HiRes Fix 结果应该：
# - 比直接生成大图细节更丰富
# - 比小图放大更清晰
# - 文件大小通常比直接生成略大

ls -lh ~/generated_images/test_hires.png
```

### 如果修改未生效

#### 重新编译
```bash
cd /home/dministrator/stable-diffusion.cpp/build

# 清理旧编译产物
rm -rf *

# 重新配置和编译
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# 验证新二进制
grep -c "Second phase" /home/dministrator/stable-diffusion.cpp/build/bin/sd-cli
# 应该输出 3
```

#### 检查源码是否正确修改
```bash
# 检查源码中是否包含新代码
grep -n "Second phase" /home/dministrator/stable-diffusion.cpp/src/stable-diffusion.cpp

# 应该输出3行，大约在 3332、3362、3366 行
```

---

## 参考实现

- **ComfyUI**：使用 `LatentUpscale` + `KSampler` 节点组合
- **AUTOMATIC1111**：`--hires-fix` 参数（使用 VAE encode，有显存限制）
- **本实现**：纯 latent 空间操作，无需 VAE encode，更省显存
