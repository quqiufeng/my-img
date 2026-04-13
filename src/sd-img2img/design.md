# Deep HighRes Fix 设计文档

## 1. 背景与动机

### 1.1 传统 HighRes Fix 的问题

传统的高清修复流程：
1. 生成小图 (512x512)
2. 放大到目标尺寸 (1024x1024)
3. 用 img2img 重绘

**问题**：
- 构图在放大后容易变形（如人脸拉长、物体比例失调）
- img2img 重绘时 AI "忘记" 了原图的构图
- 高分辨率下注意力机制偏移，导致细节混乱

### 1.2 Kohya-ss 的洞察

日本开发者 Kohya-ss 发现：
- **注意力权重在高分辨率下会偏移**：像素过多时，self-attention 无法有效关联全局信息
- **分段 denoising 更有效**：不同阶段用不同策略，而不是全程统一处理
- **潜空间过渡**：在 latent 空间平滑过渡分辨率，而非像素空间

## 2. Deep HighRes Fix 核心原理

### 2.1 核心思想

**在采样过程中逐步放大，而非采样前后放大**

```
传统方法:  [小图生成] -> [放大] -> [img2img重绘]
              ↑           ↑           ↑
           512x512    1024x1024   1024x1024

Deep HRF:  [低分辨率采样] -> [过渡] -> [高分辨率采样]
              ↑              ↑           ↑
           512x512       768x768    1024x1024
           (前30%步)    (中间步)   (后70%步)
```

### 2.2 技术细节

#### 2.2.1 分段采样策略

**Phase 1: 构图确定阶段 (低分辨率)**
- 分辨率: 512x512 或 768x768
- 步数: 总步数的 20-30%
- 作用: 让 AI 在低分辨率下确定整体构图、物体位置、大致形状
- 优势: 低分辨率下注意力机制工作正常，构图稳定

**Phase 2: 过渡阶段 (分辨率插值)**
- 方法: 在 latent 空间进行双线性插值上采样
- 原理: VAE 的 latent 空间是连续的，插值后仍保持语义一致性
- 公式: `latent_high = interpolate(latent_low, scale_factor, mode='bilinear')`

**Phase 3: 细节增强阶段 (高分辨率)**
- 分辨率: 目标尺寸 (1024x1024 或更高)
- 步数: 总步数的 70-80%
- 作用: 在高分辨率下添加细节，同时保持 Phase 1 确定的构图
- 优势: 基于已确定的构图添加细节，不会乱画

#### 2.2.2 注意力权重修正

**问题**: 在高分辨率下，self-attention 的计算复杂度是 O(n²)，像素过多时注意力权重会分散。

**Kohya-ss 的解决方案**:
- **Attention Slicing**: 将高分辨率特征图分块计算 attention，再融合
- **局部-全局注意力**: 先计算局部 attention，再计算全局 attention，加权融合

在我们的实现中，使用简化的策略：
- 在 Phase 1 充分确定全局构图
- 在 Phase 3 主要关注局部细节
- 通过 ControlNet 或 reference image 保持构图一致性

#### 2.2.3 动态 Denoise 调度

**传统方法**: 全程使用固定的 strength（如 0.45）

**Deep HRF 方法**: 不同阶段使用不同的 effective denoise strength

```
Phase 1 (低分辨率): 较高的 noise level，让 AI 自由确定构图
                   effective_strength = 0.6 ~ 0.8

Phase 2 (过渡):     保持 noise level，平滑过渡
                   effective_strength = 0.5 ~ 0.6

Phase 3 (高分辨率): 较低的 noise level，主要添加细节，保持结构
                   effective_strength = 0.3 ~ 0.45
```

实现方式：通过调整 `sigmas` 调度器参数，而非直接改 strength。

## 3. 实现架构

### 3.1 整体流程

```cpp
sd_image_t* generate_image_deep_hires(
    sd_ctx_t* ctx,
    const deep_hires_params_t* params
) {
    // Phase 1: 低分辨率采样 (构图阶段)
    latent_low = sample_phase1(
        ctx,
        prompt,
        width_low,      // 如 512
        height_low,     // 如 512
        steps_phase1,   // 如 8步 (总共30步的27%)
        seed
    );
    
    // Phase 2: Latent 空间上采样 (过渡)
    latent_high = interpolate_latent(
        latent_low,
        width_high,     // 如 1024
        height_high     // 如 1024
    );
    
    // Phase 3: 高分辨率采样 (细节阶段)
    // 使用 Phase 2 的 latent 作为 init_latent
    // 使用较低的 strength 继续采样
    result = sample_phase3(
        ctx,
        prompt,
        latent_high,    // 作为 init_latent
        width_high,
        height_high,
        steps_phase3,   // 如 22步
        strength_low    // 如 0.4
    );
    
    return result;
}
```

### 3.2 关键组件

#### 3.2.1 Phase 1: 低分辨率采样

```cpp
struct Phase1Params {
    int width;              // 512 或 768
    int height;             // 512 或 768
    int steps;              // 总步数的 20-30%
    float cfg_scale;        // 7.0 ~ 8.0 (正常值)
    sample_method_t method; // EULER_A 或 DPM++
    scheduler_t scheduler;  // KARRAS
};

sd::Tensor<float> sample_phase1(
    sd_ctx_t* ctx,
    const Phase1Params& params,
    const SDCondition& cond,
    const SDCondition& uncond
) {
    // 1. 准备低分辨率 latent (随机噪声或 txt2img)
    auto latent = prepare_latent(params.width, params.height);
    
    // 2. 完整采样流程
    auto sigmas = compute_sigmas(params.steps);
    
    for (int step = 0; step < params.steps; step++) {
        latent = denoise_step(ctx, latent, cond, uncond, sigmas[step]);
    }
    
    return latent;
}
```

#### 3.2.2 Phase 2: Latent 插值

```cpp
sd::Tensor<float> interpolate_latent(
    const sd::Tensor<float>& latent_low,
    int target_width,
    int target_height
) {
    // latent 的形状: [batch, channels, height/8, width/8]
    // 例如 512x512 的图，latent 是 [1, 4, 64, 64]
    // 目标 1024x1024，latent 应该是 [1, 4, 128, 128]
    
    int target_h = target_height / 8;  // VAE 下采样倍数
    int target_w = target_width / 8;
    
    // 使用双线性插值
    return sd::ops::interpolate(
        latent_low,
        {target_h, target_w},
        /*mode=*/"bilinear",
        /*align_corners=*/false
    );
}
```

**为什么 latent 插值有效？**

VAE 的 latent 空间具有以下特性：
1. **连续性**: 相近的 latent 解码后得到相近的图像
2. **线性**: latent 空间的线性插值对应图像空间的语义插值
3. **压缩**: 8x 压缩比，插值计算成本低

#### 3.2.3 Phase 3: 高分辨率采样

```cpp
struct Phase3Params {
    int width;              // 目标宽度 (如 1024)
    int height;             // 目标高度 (如 1024)
    int steps;              // 总步数的 70-80%
    float strength;         // 0.3 ~ 0.45 (较低，保持结构)
    float cfg_scale;        // 可略低于 Phase 1 (如 6.0 ~ 7.0)
};

sd_image_t* sample_phase3(
    sd_ctx_t* ctx,
    const Phase3Params& params,
    sd::Tensor<float>& init_latent,  // 来自 Phase 2
    const SDCondition& cond,
    const SDCondition& uncond
) {
    // 1. 添加噪声到 init_latent (根据 strength)
    float noise_scale = params.strength;  // 简化处理
    auto noisy_latent = add_noise(init_latent, noise_scale);
    
    // 2. 计算剩余步数的 sigmas
    int total_steps = estimate_total_steps(params.strength);
    auto sigmas = compute_sigmas(total_steps);
    
    // 只取前 params.steps 个 sigma
    std::vector<float> phase3_sigmas(
        sigmas.begin(),
        sigmas.begin() + params.steps + 1
    );
    
    // 3. 继续采样
    auto latent = noisy_latent;
    for (int step = 0; step < params.steps; step++) {
        latent = denoise_step(ctx, latent, cond, uncond, phase3_sigmas[step]);
    }
    
    // 4. VAE 解码
    return vae_decode(ctx, latent);
}
```

### 3.3 参数设计

#### 3.3.1 推荐参数组合

**场景 1: 512 -> 1024 (2x 放大)**
```
Phase 1:
  - resolution: 512x512
  - steps: 8 (共30步的27%)
  - cfg_scale: 7.5
  - sampler: EULER_A

Phase 2:
  - interpolate to 1024x1024 latent

Phase 3:
  - resolution: 1024x1024
  - steps: 22
  - strength: 0.40
  - cfg_scale: 7.0
```

**场景 2: 768 -> 1536 (2x 放大，更高质量)**
```
Phase 1:
  - resolution: 768x768
  - steps: 10 (共40步的25%)
  - cfg_scale: 8.0

Phase 2:
  - interpolate to 1536x1536 latent

Phase 3:
  - resolution: 1536x1536
  - steps: 30
  - strength: 0.35
  - cfg_scale: 6.5
```

**场景 3: 1024 -> 2048 (4x 总放大，分阶段)**
```
Phase 1:
  - resolution: 512x512
  - steps: 6

Phase 2:
  - interpolate to 1024x1024

Phase 3a:
  - resolution: 1024x1024
  - steps: 12
  - strength: 0.50

Phase 4:
  - interpolate to 2048x2048

Phase 3b:
  - resolution: 2048x2048
  - steps: 20
  - strength: 0.35
```

#### 3.3.2 参数自动计算

```cpp
struct DeepHiresConfig {
    int target_width;
    int target_height;
    int total_steps;
    float base_cfg_scale;
    
    // 自动计算 Phase 1 参数
    Phase1Params get_phase1() const {
        Phase1Params p;
        // 低分辨率取目标的一半，但不超过 768
        p.width = std::min(768, target_width / 2);
        p.height = std::min(768, target_height / 2);
        // 64 对齐
        p.width = (p.width + 63) & ~63;
        p.height = (p.height + 63) & ~63;
        
        // Phase 1 占 25-30% 步数
        p.steps = std::max(6, total_steps / 4);
        p.cfg_scale = base_cfg_scale;
        return p;
    }
    
    // 自动计算 Phase 3 参数
    Phase3Params get_phase3() const {
        Phase3Params p;
        p.width = target_width;
        p.height = target_height;
        // Phase 3 占剩余步数
        p.steps = total_steps - get_phase1().steps;
        // strength 根据放大倍数调整
        float scale = std::max(
            (float)target_width / get_phase1().width,
            (float)target_height / get_phase1().height
        );
        p.strength = 0.5f - (scale - 1.0f) * 0.1f;  // 2x->0.4, 4x->0.3
        p.strength = std::clamp(p.strength, 0.25f, 0.5f);
        p.cfg_scale = base_cfg_scale * 0.9f;  // 略低于 Phase 1
        return p;
    }
};
```

## 4. 与现有工具的集成

### 4.1 作为 sd-img2img 的增强模式

```cpp
// sd-img2img/main.cpp

int main(int argc, char** argv) {
    // ... 解析参数 ...
    
    if (use_deep_hires) {
        // 使用 Deep HighRes Fix 流程
        deep_hires_params_t params;
        params.target_width = output_width;
        params.target_height = output_height;
        params.total_steps = steps;
        params.prompt = prompt;
        params.init_image = load_image(input_path);  // 可选：从已有图片开始
        
        result = generate_image_deep_hires(ctx, &params);
    } else {
        // 使用普通 img2img
        result = generate_image(ctx, &img_params);
    }
    
    // ... 保存结果 ...
}
```

### 4.2 作为 sd-hires 的核心算法

```cpp
// sd-hires/main.cpp

int main(int argc, char** argv) {
    // 1. 可选：ESRGAN 放大（如果需要像素级放大）
    if (use_esrgan) {
        upscaled = esrgan_upscale(input_image, scale);
    } else {
        upscaled = input_image;
    }
    
    // 2. Deep HighRes Fix 重绘
    deep_hires_params_t params;
    params.target_width = upscaled.width;
    params.target_height = upscaled.height;
    params.init_image = upscaled;  // 从放大后的图开始
    params.prompt = prompt;
    params.total_steps = 30;
    
    result = generate_image_deep_hires(ctx, &params);
    
    // 3. 保存
    save_image(result, output_path);
}
```

## 5. 优化与注意事项

### 5.1 显存优化

**问题**: Phase 3 的高分辨率采样需要大量显存

**解决方案**:
1. **VAE Tiling**: 在 Phase 3 的 VAE decode 阶段启用 tiling
2. **Gradient Checkpointing**: 如果支持，启用以节省显存
3. **分块 Phase 3**: 如果显存仍不足，将 Phase 3 也分块处理

```cpp
// 显存检查与自动降级
if (estimate_vram_needed(target_width, target_height) > available_vram) {
    // 启用 VAE tiling
    params.vae_tiling = true;
    params.tile_size = 512;
    
    // 或者进一步降低 Phase 1 分辨率
    if (available_vram < 8_GB) {
        phase1_width = 512;  // 强制 512
        phase1_height = 512;
    }
}
```

### 5.2 质量控制

**潜在问题**:
1. **Phase 2 插值导致的模糊**: 如果 Phase 1 分辨率过低，插值后细节丢失
2. **Phase 3 过度重绘**: strength 过高导致 Phase 1 的构图被破坏
3. **颜色偏移**: 不同分辨率下 VAE 的解码可能略有差异

**缓解措施**:
1. **最低 Phase 1 分辨率**: 不低于 512x512
2. **渐进式放大**: 超过 2x 时，分多个 Phase 逐步放大
3. **颜色校正**: 在 Phase 3 开始时记录颜色分布，结束时校正

### 5.3 与 LoRA/ControlNet 的兼容性

**LoRA**:
- 在 Phase 1 和 Phase 3 都应用相同的 LoRA
- 注意：某些 LoRA 可能对分辨率敏感，需要测试

**ControlNet**:
- 在 Phase 1 使用低分辨率 control image
- 在 Phase 3 使用高分辨率 control image（需要预先放大）
- 或者：在 Phase 3 使用 Phase 1 的结果作为 reference

## 6. 实现计划

### Phase 1: 基础框架 (MVP)
- [ ] 实现 `interpolate_latent` 函数
- [ ] 实现 `sample_phase1` 基础版本
- [ ] 实现 `sample_phase3` 基础版本
- [ ] 集成到 sd-img2img，支持 `--deep-hires` 参数

### Phase 2: 参数优化
- [ ] 自动参数计算
- [ ] 针对不同模型的预设（SD1.5, SDXL, Flux）
- [ ] 显存自适应

### Phase 3: 高级特性
- [ ] 多阶段放大（支持 4x, 8x）
- [ ] 与 LoRA/ControlNet 集成
- [ ] 注意力权重修正（进阶）

### Phase 4: 性能优化
- [ ] CUDA kernel 优化 latent 插值
- [ ] 异步 VAE decode
- [ ] 内存池管理

## 7. 参考

- Kohya-ss 的 sd-scripts: https://github.com/kohya-ss/sd-scripts
- Stable Diffusion 潜空间分析: https://arxiv.org/abs/2112.10752
- VAE 插值特性研究: https://arxiv.org/abs/2203.13164
- Attention Slicing: https://arxiv.org/abs/2211.16912
