# Deep HighRes Fix 设计文档

## 1. 背景与动机

### 1.1 传统 HighRes Fix 的问题

传统的高清修复流程：
1. 生成小图 (512x512)
2. 放大到目标尺寸 (1024x1024)
3. 用 img2img 重绘

**问题**：
- 构图在放大后容易变形
- img2img 重绘时 AI "忘记" 了原图的构图
- 多次 VAE encode/decode 导致信息损失

### 1.2 Kohya-ss 的洞察

日本开发者 Kohya-ss 发现：
- **注意力权重在高分辨率下会偏移**：像素过多时，self-attention 无法有效关联全局信息
- **分段 denoising 更有效**：不同阶段用不同策略
- **潜空间过渡**：在 latent 空间平滑过渡分辨率，而非像素空间

## 2. Deep HighRes Fix 核心原理

### 2.1 核心思想

**在单次采样过程中逐步放大，而非采样前后放大**

```
传统方法:  [小图生成] -> [放大] -> [img2img重绘]
               ↑           ↑           ↑
            512x512    1024x1024   1024x1024
            (VAE decode)  (resize)   (VAE encode+decode)

Deep HRF:  [低分辨率采样] -> [过渡] -> [高分辨率采样]
               ↑              ↑           ↑
            512x512       768x768    1024x1024
            (前25%步)    (中25%步)   (后50%步)
            
关键区别：只 VAE decode 一次（最后），中间在 latent 空间插值过渡
```

### 2.2 技术细节

#### 2.2.1 分段采样策略

**Phase 1: 构图确定阶段 (低分辨率)**
- 分辨率: 512x512 或 768x768
- 步数: 总步数的 20-30%
- 作用: 让 AI 在低分辨率下确定整体构图
- 优势: 低分辨率下注意力机制工作正常，构图稳定

**Phase 2: 过渡阶段 (latent 空间插值)**
- 方法: 在采样循环中，将当前 latent 插值上采样到更高分辨率
- 原理: VAE 的 latent 空间是连续的，插值后仍保持语义一致性
- 实现: 通过 `sd_latent_hook_t` 在 `sample()` 函数内部动态修改 latent

**Phase 3: 细节增强阶段 (高分辨率)**
- 分辨率: 目标尺寸
- 步数: 总步数的 50-70%
- 作用: 在高分辨率下添加细节，同时保持 Phase 1 确定的构图
- 优势: 基于已确定的构图添加细节，不会乱画

#### 2.2.2 原生实现 vs 近似实现

| 特性 | 多次调用版（旧） | 原生 hook 版（当前） |
|------|----------------|---------------------|
| 采样过程 | 3 次独立 `generate_image()` | 1 次 `generate_image()` |
| VAE 编解码 | 3 次 | 1 次（仅 decode） |
| 上采样位置 | 像素空间（resize 后） | latent 空间（采样中） |
| 信息损失 | 大 | 小 |
| 实现复杂度 | 低 | 中（需改 upstream） |

## 3. 原生实现架构

### 3.1 核心机制：Latent Hook

我们在 `stable-diffusion.cpp` 的 `sample()` 函数中添加了一个 hook：

```cpp
// stable-diffusion.cpp/src/stable-diffusion.cpp

typedef sd::Tensor<float> (*sd_latent_hook_t)(
    sd::Tensor<float>& latent,
    int step,
    int total_steps,
    void* user_data);

// 在 sample() 的采样循环中：
auto denoise_with_hook = [&](const sd::Tensor<float>& x, float sigma, int step) {
    sd::Tensor<float> mutable_x = x;
    if (g_sd_latent_hook != nullptr) {
        mutable_x = g_sd_latent_hook(mutable_x, step, steps, user_data);
    }
    return denoise(mutable_x, sigma, step);
};
```

### 3.2 整体流程

```cpp
sd_image_t* generate_image_deep_hires(
    sd_ctx_t* ctx,
    const sd_deep_hires_params_t* params
) {
    // 1. 计算各阶段参数
    phase1_steps = total_steps / 4;      // 如 8 步
    phase2_steps = total_steps / 4;      // 如 7 步
    phase3_steps = total_steps * 2 / 4;  // 如 15 步
    
    phase1_w = 512, phase1_h = 512;
    phase2_w = 768, phase2_h = 768;
    target_w = 1024, target_h = 1024;
    
    // 2. 注册 latent hook
    DeepHiresState state = {
        .phase1_steps = phase1_steps,
        .phase2_steps = phase2_steps,
        .phase1_w = phase1_w, .phase1_h = phase1_h,
        .phase2_w = phase2_w, .phase2_h = phase2_h,
        .target_w = target_w, .target_h = target_h,
    };
    sd_set_latent_hook(deep_hires_latent_hook, &state);
    
    // 3. 单次生成调用（从低分辨率开始）
    sd_img_gen_params_t gen_params;
    gen_params.width = phase1_w;
    gen_params.height = phase1_h;
    gen_params.strength = 1.0f;  // txt2img
    // ... 其他参数 ...
    
    sd_image_t* result = generate_image(ctx, &gen_params);
    
    // 4. 清除 hook
    sd_clear_latent_hook();
    
    return result;
}
```

### 3.3 Hook 回调实现

```cpp
sd::Tensor<float> deep_hires_latent_hook(
    sd::Tensor<float>& latent,
    int step,
    int total_steps,
    void* user_data) {
    
    DeepHiresState* state = (DeepHiresState*)user_data;
    
    // Phase 1 -> Phase 2 过渡
    if (!state->phase1_done && step > state->phase1_steps) {
        state->phase1_done = true;
        return upscale_latent_nearest(
            latent, state->phase2_w, state->phase2_h, state->latent_channel);
    }
    
    // Phase 2 -> Phase 3 过渡
    if (!state->phase2_done && step > (state->phase1_steps + state->phase2_steps)) {
        state->phase2_done = true;
        return upscale_latent_nearest(
            latent, state->target_w, state->target_h, state->latent_channel);
    }
    
    return latent;
}
```

### 3.4 Latent 插值

当前使用 nearest neighbor 插值：

```cpp
sd::Tensor<float> upscale_latent_nearest(
    const sd::Tensor<float>& latent,
    int target_w, int target_h, int channels) {
    
    int current_w = latent.shape()[0];
    int current_h = latent.shape()[1];
    
    sd::Tensor<float> result({target_w, target_h, channels, 1});
    
    for (int y = 0; y < target_h; y++) {
        int src_y = y * current_h / target_h;
        for (int x = 0; x < target_w; x++) {
            int src_x = x * current_w / target_w;
            for (int c = 0; c < channels; c++) {
                result.data()[((y * target_w + x) * channels + c)] =
                    latent.data()[((src_y * current_w + src_x) * channels + c)];
            }
        }
    }
    
    return result;
}
```

> **注意**：当前使用 nearest neighbor 是因为 `sd::Tensor::interpolate()` 只支持 nearest-like 模式。如果测试效果不够平滑，需要实现双线性插值。

## 4. 参数设计

### 4.1 推荐参数组合

**场景 1: 512 -> 1024 (2x 放大)**
```
Phase 1:
  - resolution: 512x512
  - steps: 8 (共30步的27%)
  - cfg_scale: 7.5

Phase 2:
  - interpolate to 768x768 latent
  - steps: 7

Phase 3:
  - interpolate to 1024x1024 latent
  - steps: 15
  - cfg_scale: 7.0
```

**场景 2: 768 -> 1536 (2x 放大)**
```
Phase 1:
  - resolution: 768x768
  - steps: 10 (共40步的25%)
  - cfg_scale: 8.0

Phase 2:
  - interpolate to 1152x1152 latent
  - steps: 10

Phase 3:
  - interpolate to 1536x1536 latent
  - steps: 20
  - cfg_scale: 6.5
```

### 4.2 自动计算逻辑

```cpp
phase1_w = std::min(512, target_w / 2);  // 64 对齐
phase1_h = std::min(512, target_h / 2);
phase1_steps = std::max(6, total_steps / 4);

phase2_w = target_w * 3 / 4;  // 64 对齐
phase2_h = target_h * 3 / 4;
phase2_steps = std::max(4, total_steps - phase1_steps - phase3_steps);

phase3_steps = std::max(8, total_steps * 3 / 4);
```

## 5. 与现有工具的集成

### 5.1 sd-hires

```cpp
// 1. ESRGAN 放大（可选）
sd_image_t upscaled = esrgan_upscale(input_image, scale);

// 2. Deep HighRes Fix 重绘
sd_deep_hires_params_t params;
params.target_width = upscaled.width;
params.target_height = upscaled.height;
params.prompt = prompt;
params.total_steps = 30;

sd_image_t* result = generate_image_deep_hires(ctx, &params);
```

### 5.2 sd-img2img

```cpp
sd_deep_hires_params_t params;
params.target_width = target_width;
params.target_height = target_height;
params.prompt = prompt;
params.total_steps = steps;

sd_image_t* result = generate_image_deep_hires(ctx, &params);
```

## 6. 已知限制与优化方向

### 6.1 当前限制

1. **Latent 插值质量**：nearest neighbor 可能产生块状伪影
2. **不支持 init_image**：当前 `generate_image_deep_hires()` 从随机噪声（txt2img）开始，不支持传入已有图片作为 `init_image`
3. **Latent channel 硬编码**：假设为 4（SD1.5），SDXL/Flux 可能需要调整

### 6.2 优化方向

1. **实现双线性 latent 插值**
2. **支持 init_image 输入**（从已有图片 encode 到 latent 再开始分段采样）
3. **自动检测 latent channel**（根据模型类型）
4. **动态 cfg_scale 调度**（不同阶段使用不同 cfg_scale）

## 7. 参考

- Kohya-ss 的 sd-scripts: https://github.com/kohya-ss/sd-scripts
- Stable Diffusion 潜空间分析: https://arxiv.org/abs/2112.10752
- VAE 插值特性研究: https://arxiv.org/abs/2203.13164
