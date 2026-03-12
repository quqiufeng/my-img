# ESRGAN 放大实现详解

## 核心 API

### 1. 创建放大器上下文
```cpp
upscaler_ctx_t* new_upscaler_ctx(
    const char* esrgan_path,   // ESRGAN 模型路径 (.gguf)
    bool offload_params_to_cpu,
    bool direct,
    int n_threads,
    int tile_size
);
```

### 2. 执行放大
```cpp
sd_image_t upscale(
    upscaler_ctx_t* upscaler_ctx,  // 放大器上下文
    sd_image_t input_image,        // 输入图片
    uint32_t upscale_factor        // 放大倍数 (由模型决定)
);
```

### 3. 释放上下文
```cpp
void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);
```

---

## 内部实现 (upscaler.cpp)

### upscale() 函数流程：

```
1. 计算输出尺寸
   output_width = input_image.width * esrgan_upscaler->scale
   output_height = input_image.height * esrgan_upscaler->scale

2. 创建 ggml context
   ggml_init(params)  // 1G 内存

3. 将输入图片转为 tensor
   input_image_tensor = ggml_new_tensor_4d(..., GGML_TYPE_F32, w, h, 3, 1)
   sd_image_to_ggml_tensor(input_image, input_image_tensor)

4. 创建输出 tensor
   upscaled = ggml_new_tensor_4d(..., GGML_TYPE_F32, output_w, output_h, 3, 1)

5. 执行 tiling 处理
   sd_tiling(input_image_tensor, upscaled, scale, tile_size, overlap, callback)

6. tensor 转回图片
   upscaled_data = ggml_tensor_to_sd_image(upscaled)

7. 返回结果
   sd_image_t = { output_width, output_height, 3, upscaled_data }
```

---

## 关键转换函数

### sd_image_to_ggml_tensor()
```cpp
// 将 sd_image_t (RGB) 转为 ggml_tensor (F32)
// 位置: ggml_extend.hpp
sd_image_to_ggml_tensor(sd_image_t image, ggml_tensor* tensor);
```

### ggml_tensor_to_sd_image()
```cpp
// 将 ggml_tensor 转回 sd_image_t
// 位置: ggml_extend.hpp
uint8_t* ggml_tensor_to_sd_image(ggml_tensor* tensor);
```

---

## SD img2img 实现

### generate_image() 函数中的 img2img 处理

```cpp
// stable-diffusion.cpp:3601
if (sd_img_gen_params->init_image.data) {
    // IMG2IMG 模式
    
    // 1. 计算编码步数
    t_enc = sample_steps * strength
    
    // 2. 创建 tensor
    ggml_tensor* init_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, 3, 1);
    ggml_tensor* mask_img = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, w, h, 1, 1);
    
    // 3. 转换图片和 mask
    sd_image_to_ggml_tensor(sd_img_gen_params->mask_image, mask_img);
    sd_image_to_ggml_tensor(sd_img_gen_params->init_image, init_img);
    
    // 4. VAE encode 到 latent 空间
    init_latent = sd_ctx->sd->encode_first_stage(work_ctx, init_img);
    
    // 5. 在 latent 空间根据 strength 采样
    // 6. VAE decode 回像素空间
}
```

---

## 组合流程（先放大后 img2img）

要实现先 ESRGAN 放大，再 img2img 优化，需要：

```cpp
// 步骤 1: ESRGAN 放大
upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(esrgan_path, ...);
sd_image_t upscaled = upscale(upscaler_ctx, input_image, factor);
free_upscaler_ctx(upscaler_ctx);  // 释放放大器

// 步骤 2: img2img
sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
sd_img_gen_params_t params;
params.init_image = upscaled;      // 使用放大后的图片
params.mask_image = ...;           // 需要有效 mask
params.strength = 0.45;
sd_image_t* result = generate_image(sd_ctx, &params);
free_sd_ctx(sd_ctx);

// 释放内存
free(upscaled.data);
```

---

## 注意事项

1. **内存管理**：ESRGAN 放大和 SD 推理需要分别创建独立的 ggml context
2. **图片数据**：放大后的图片数据需要保存或重新加载给下一步使用
3. **Mask**：img2img 需要有效的 mask_image（全黑=使用原图，全白=不使用）
4. **Tile Size**：ESRGAN 默认 tile_size=128，可调整优化质量和速度
