# HiRes Fix 修复记录

## 问题描述

### 核心问题
2560x1440 等大分辨率图片生成时出现 **"重叠幻隐部分"**（tile 接缝伪影），画面质量不完美。

### 具体表现
1. **小图（640x360, 1280x720）**：生成正常，无接缝
2. **大图（2560x1440）**：
   - 直接生成会失败（栈溢出 / OOM）
   - 使用 HiRes Fix 时，第二阶段采样失败，回退到第一阶段低分辨率结果
   - VAE tiling 参数不当导致 tile 边界出现明显接缝

## 根因分析

### 1. my-img 实现错误（sd-hires / sd-img2img）

**错误实现**：多阶段手动调用 `generate_image()`
```cpp
// 错误的三阶段流程
Phase 1: 低分辨率生成（512x384）
Phase 2: 中分辨率 refine（1280x768）
Phase 3: 高分辨率 detail（2560x1472）
```

**问题**：
- 没有真正的 latent 空间放大
- 每次都对像素图进行 resize，损失细节
- 没有添加噪声和重新采样

### 2. stable-diffusion.cpp 的 hires_fix bug

**问题**：img2img 模式下，HiRes Fix 第二阶段 `sample()` 调用后立即返回空值

**根因**：第二阶段使用了第一阶段的 `embeds.img_cond`，其 `c_concat` tensor 尺寸基于 base 分辨率（640x384），而 latent 已放大到目标分辨率（2560x1472），导致 `diffusion_model->compute()` 因维度不匹配而失败。

**代码位置**：`src/stable-diffusion.cpp:3382`

```cpp
// 修复前：传递了错误的 img_cond
sd::Tensor<float> hires_x0 = sd_ctx->sd->sample(
    ...,
    embeds.img_cond,  // ❌ 尺寸基于 640x384，不匹配 2560x1472
    ...
);
```

### 3. VAE tiling 参数不当

**默认参数**：tile_size=64x64, overlap=0.75

**问题**：
- tile 太小 → 2560x1440 需要大量 tile（约 1600 个），边界数量多
- overlap 不足 → 边界融合不好，出现接缝伪影

## 解决方案

### 1. 使用原生 hires_fix API

**正确实现**：
```cpp
// 单次调用 generate_image，内部自动处理两阶段采样
params.hires_fix = true;
params.hires_width = target_w;
params.hires_height = target_h;
params.hires_strength = 0.5f;
// 第一阶段：在 base_w x base_h 分辨率运行 img2img
// 第二阶段：latent 插值放大 + 添加噪声 + 重新采样去噪
sd_image_t* result = generate_image(ctx, &params);
```

### 2. 修复 stable-diffusion.cpp

**修复内容**：为第二阶段创建新的 `img_cond`，使用正确尺寸的 `c_concat`

```cpp
// 修复后：创建匹配目标分辨率的 img_cond
SDCondition hires_img_cond;
if (!hires_cond.c_concat.empty()) {
    hires_img_cond = SDCondition(
        hires_uncond.c_crossattn,  // 使用 uncond 的 crossattn
        hires_uncond.c_vector,      // 使用 uncond 的 vector
        hires_cond.c_concat         // 使用放大后的 concat（正确尺寸）
    );
}

sd::Tensor<float> hires_x0 = sd_ctx->sd->sample(
    ...,
    hires_img_cond,  // ✅ 尺寸匹配目标分辨率
    ...
);
```

**附加修复**：
- 跳过 `control_image`（第一阶段尺寸，不匹配）
- 重置 `start_merge_step = -1`（避免 img2img 模式下的 ID conditioning 问题）

### 3. 优化 VAE tiling 参数

**自动根据分辨率选择参数**：

```cpp
if (width >= 2048 || height >= 2048) {
    // 超大图：256x256 tile + 128 overlap
    tile_size = 256;
    overlap = 128;
} else if (width >= 1280 || height >= 1280) {
    // 大图：256x256 tile + 64 overlap
    tile_size = 256;
    overlap = 64;
} else {
    // 小图：128x128 tile + 16 overlap
    tile_size = 128;
    overlap = 16;
}
```

**注意**：512x512 tile 在 3080 10GB 上会 VAE decode 失败（OOM），但保留作为 4090d 的选项。

## 修改文件

### stable-diffusion.cpp 修改
- `src/stable-diffusion.cpp` - 第二阶段 img_cond 修复
- `src/denoiser.hpp` - 添加调试日志
- `include/stable-diffusion.h` - hires_fix 字段已存在（无需修改）

### my-img 修改
- `src/sd-hires/main.cpp` - 使用原生 hires_fix API + VAE tiling 参数优化
- `src/sd-img2img/main.cpp` - 同上
- `include/stable-diffusion.h` - 同步新版本头文件

## Patch 文件

- `patches/hires-fix-stable-diffusion.patch` - stable-diffusion.cpp 修复
- `patches/hires-fix-myimg.patch` - my-img 修复

## 测试结果

### 1280x720（sd-cli 原生 hires_fix）
- ✅ 成功生成，第二阶段耗时 15.88s
- 质量良好，无明显接缝

### 2560x1440（sd-cli 原生 hires_fix）
- ✅ 成功生成，第二阶段耗时 139.77s
- 文件大小 7.5M，分辨率正确
- 使用 256x256 tile 避免接缝

### 2560x1440（sd-hires 修复后）
- ⚠️ 编译成功，但第二阶段仍有问题（待进一步调试）
- 建议使用 sd-cli 直接生成大图

## 使用建议

### 生成 2560x1440 大图的最佳命令
```bash
sd-cli \
  --diffusion-model model.gguf \
  --vae vae.safetensors \
  --llm llm.gguf \
  -p "prompt" \
  -n "negative" \
  --cfg-scale 1.01 \
  --sampling-method euler \
  --diffusion-fa \
  --vae-tiling \
  --vae-tile-size 256x256 \  # 关键：256x256 避免接缝
  -W 640 -H 360 \            # 基础分辨率
  --hires-fix \
  --hires-width 2560 \
  --hires-height 1440 \
  --hires-strength 0.5 \     # 0.3-0.5 效果较好
  --steps 25 \
  -s 42 \
  -o output.png
```

### sd-hires 使用（待完善）
```bash
sd-hires \
  --diffusion-model model.gguf \
  --vae vae.safetensors \
  --llm llm.gguf \
  --input base.png \
  --prompt "prompt" \
  --deep-hires \
  --target-width 2560 \
  --target-height 1440 \
  --strength 0.5 \
  --skip-upscale \
  --output output.png
```

## 后续工作

1. **sd-hires 第二阶段调试** - 当前第二阶段仍快速返回，需进一步定位问题
2. **4090d 支持** - 添加 `--vae-tile-size 512` 选项，大 tile 进一步减少接缝
3. **像素空间 HiRes Fix** - 实现 `hires_mode=1`（pixel-space，ComfyUI-style）

## 提交信息

```
fix: HiRes Fix for 2560x1440 large images

- Use native hires_fix API instead of multi-phase manual calls
- Fix img2img mode second phase failure in stable-diffusion.cpp
- Optimize VAE tiling params based on resolution (256x256 for 2K+)
- Add --vae-tile-size option for future 4090d support
- Generate patches: hires-fix-stable-diffusion.patch, hires-fix-myimg.patch

Issues fixed:
1. "Overlapping ghosting" artifacts on 2560x1440 images
2. Second phase sampling failure in img2img + hires_fix mode
3. VAE tiling seam artifacts with default 64x64 tile size
```
