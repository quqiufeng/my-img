# IMG2IMG 实现详解

## 核心 API

### 生成参数结构体
```cpp
typedef struct {
    const char* prompt;           // 正向提示词
    const char* negative_prompt; // 负向提示词
    int clip_skip;               // CLIP 跳过层数
    
    sd_image_t init_image;       // 输入图片（img2img 必需）
    sd_image_t mask_image;        // 蒙版（inpaint 用）
    
    int width;                   // 输出宽度
    int height;                  // 输出高度
    float strength;               // 重绘强度 (0-1)
    int64_t seed;                // 随机种子
    int batch_count;             // 生成数量
    
    sd_sample_params_t sample_params; // 采样参数
    sd_tiling_params_t vae_tiling_params; // VAE tiling
} sd_img_gen_params_t;
```

### 调用生成
```cpp
sd_image_t* generate_image(
    sd_ctx_t* sd_ctx, 
    const sd_img_gen_params_t* params
);
```

---

## 内部实现 (stable-diffusion.cpp)

### generate_image() 函数流程

```
1. 准备 ggml context
   ggml_init(params)  // 1G 内存

2. 对齐尺寸 (width/height 必须是 8 的倍数)
   width = align_up(width, spatial_multiple)
   height = align_up(height, spatial_multiple)

3. 检查是否有 init_image
   if (params->init_image.data) {
       // IMG2IMG 模式
   } else {
       // TXT2IMG 模式
   }
```

---

## IMG2IMG 详细流程

### 1. 计算编码步数
```cpp
// strength 决定从哪个时间步开始采样
// strength=1.0: 从随机噪声开始 (等于 txt2img)
// strength=0.0: 完全使用原图 (不做任何改变)
t_enc = sample_steps * strength
sigmas = sigmas[sample_steps - t_enc - 1 : end]
```

### 2. 创建 Tensor
```cpp
// 图片 tensor (RGB, F32)
init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, 3, 1)

// 蒙版 tensor (灰度, F32)  
mask_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, 1, 1)

// 转换图片数据到 tensor
sd_image_to_ggml_tensor(mask_image, mask_img);
sd_image_to_ggml_tensor(init_image, init_img);
```

### 3. VAE 编码 (像素空间 -> 潜在空间)
```cpp
// 关键步骤：将图片编码到 latent 空间
init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
// init_latent 形状: [b, c, h/8, w/8] (SD1.x) 或 [b, c, h/8, w/8] (SDXL)
```

### 4. Inpaint 处理 (如果有 mask)
```cpp
if (sd_version_is_inpaint(sd_ctx->sd->version)) {
    // 将 mask 应用到 latent
    // 生成 masked_latent 和 concat_latent
}
```

### 5. 采样循环
```cpp
// 在 latent 空间进行去噪
result_latent = sampling_loop(
    init_latent,    // 起始 latent (带噪声)
    prompt,         // 条件
    negative_prompt,
    sigmas,         // 由 strength 决定的噪声调度
    ...
);
```

### 6. VAE 解码 (潜在空间 -> 像素空间)
```cpp
// 最后一步：将结果 latent 解码回像素图片
result_image = sd_ctx->sd->decode_first_stage(work_ctx, result_latent);
```

---

## 关键点

### strength 参数
- `strength = 1.0`: 完全重绘，等同于 txt2img
- `strength = 0.0`: 不改变原图
- `strength = 0.45`: 推荐值，保持 55% 原图特征

### mask_image 参数
- 全黑 (0): 使用 init_image 全部内容
- 全白 (255): 不使用 init_image (等价于 txt2img)
- 灰色: 部分使用

### VAE Tiling
当图片尺寸 > 1024 时，需要启用 tiling 避免显存不足：
```cpp
params.vae_tiling_params.enabled = true;
params.vae_tiling_params.tile_size_x = 512;
params.vae_tiling_params.tile_size_y = 512;
params.vae_tiling_params.target_overlap = 64;
```

---

## 组合：先 ESRGAN 放大后 IMG2IMG

### 完整流程
```cpp
// ========== 步骤 1: ESRGAN 放大 ==========
upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(
    esrgan_path,    // 模型路径
    offload_to_cpu, // 是否卸载到 CPU
    direct,
    n_threads,
    tile_size
);

sd_image_t upscaled = upscale(upscaler_ctx, input_image, 2);
free_upscaler_ctx(upscaler_ctx);

// ========== 步骤 2: IMG2IMG ==========
sd_ctx_params_t ctx_params;
sd_ctx_params_init(&ctx_params);
ctx_params.vae_decode_only = false;
ctx_params.free_params_immediately = false;
// ... 其他参数

sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);

// 准备 img2img 参数
sd_img_gen_params_t img_params;
memset(&img_params, 0, sizeof(img_params));

img_params.prompt = prompt;
img_params.negative_prompt = negative_prompt;
img_params.init_image = upscaled;      // 使用放大后的图片
img_params.width = upscaled.width;
img_params.height = upscaled.height;
img_params.strength = 0.45;
img_params.seed = -1;

// mask: 全黑 = 使用 init_image
img_params.mask_image = {
    .width = upscaled.width,
    .height = upscaled.height,
    .channel = 1,
    .data = malloc(upscaled.width * upscaled.height)
};
memset(img_params.mask_image.data, 0, upscaled.width * upscaled.height);

// 采样参数
img_params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
img_params.sample_params.sample_steps = 20;
img_params.sample_params.scheduler = KARRAS_SCHEDULER;
img_params.sample_params.guidance.txt_cfg = 2.0f;

// VAE tiling (大图必需)
img_params.vae_tiling_params.enabled = true;
img_params.vae_tiling_params.tile_size_x = 512;
img_params.vae_tiling_params.tile_size_y = 512;
img_params.vae_tiling_params.target_overlap = 64;

// 生成
sd_image_t* result = generate_image(sd_ctx, &img_params);

// 清理
free_sd_ctx(sd_ctx);
free(upscaled.data);
free(img_params.mask_image.data);

// result 就是最终图片
```

---

## 常见问题

### 1. 输出图片太小/异常
- 检查 mask_image 是否有效
- 检查 strength 值
- 检查 VAE tiling 设置

### 2. 显存不足
- 启用 VAE tiling
- 减小 tile_size
- 使用 CPU offload

### 3. 崩溃
- 确保 mask_image 有效（非 NULL）
- 确保 width/height 与 init_image 匹配
- 释放所有中间内存
