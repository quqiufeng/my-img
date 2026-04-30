# 原版正常出图逻辑代码

> 来源：`/opt/stable-diffusion.cpp-quqiufeng/`（quqiufeng 修改版，出图正常）

## 1. 调用链

```
main() [examples/cli/main.cpp:499]
  └── parse_args() [examples/cli/main.cpp:270]
        └── gen_params.resolve_and_validate() [examples/common/common.cpp]
              └── 解析 --hires, --hires-width, --hires-height 等参数
  └── gen_params.to_sd_img_gen_params_t() [examples/common/common.cpp:1928]
        └── 填充 sd_img_gen_params_t 结构体（包含 enable_hires, hires_width, hires_height, hires_strength, hires_sample_steps）
  └── generate_image(sd_ctx.get(), &img_gen_params) [examples/cli/main.cpp:723]
        └── sd.cpp 内部完成两阶段生成
```

## 2. 关键代码

### 2.1 参数结构体定义

文件：`examples/common/common.h:146`

```cpp
struct SDGenerationParams {
    // ... 其他字段 ...
    
    bool enable_hires      = false;
    int hires_width        = 1024;
    int hires_height       = 1024;
    float hires_strength   = 0.4f;
    int hires_sample_steps = 0;
    
    // ...
    
    sd_img_gen_params_t to_sd_img_gen_params_t();
};
```

### 2.2 参数转换函数

文件：`examples/common/common.cpp:1928`

```cpp
sd_img_gen_params_t SDGenerationParams::to_sd_img_gen_params_t() {
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);  // 初始化所有字段为默认值
    
    // ... 设置其他参数 ...
    
    params.enable_hires       = enable_hires;
    params.hires_width        = hires_width;
    params.hires_height       = hires_height;
    params.hires_strength     = hires_strength;
    params.hires_sample_steps = hires_sample_steps;
    
    return params;
}
```

### 2.3 主程序调用

文件：`examples/cli/main.cpp:719-723`

```cpp
if (cli_params.mode == IMG_GEN) {
    sd_img_gen_params_t img_gen_params = gen_params.to_sd_img_gen_params_t();
    
    num_results = gen_params.batch_count;
    results.adopt(generate_image(sd_ctx.get(), &img_gen_params), num_results);
}
```

### 2.4 CLI 参数解析

文件：`examples/common/common.cpp`（在 `get_options()` 中）

```cpp
// 简化的参数解析逻辑
// --hires              → enable_hires = true
// --hires-width 2560   → hires_width = 2560
// --hires-height 1440  → hires_height = 1440
// --hires-strength 0.3 → hires_strength = 0.3f
// --hires-steps 60     → hires_sample_steps = 60
```

## 3. API 特点

**quqiufeng 版本使用的是旧版 API**：

```cpp
// sd_img_gen_params_t 中的 HiRes 参数是扁平化的
typedef struct {
    // ... 其他字段 ...
    
    bool enable_hires;
    int hires_width;
    int hires_height;
    float hires_strength;
    int hires_sample_steps;
    
    // ...
} sd_img_gen_params_t;
```

**当前版本（/opt/stable-diffusion.cpp）使用的是新版 API**：

```cpp
// sd_img_gen_params_t 中的 HiRes 参数是嵌套结构体
typedef struct {
    // ... 其他字段 ...
    
    sd_hires_params_t hires;  // 嵌套结构体
    
    // ...
} sd_img_gen_params_t;

// hires 参数在 sd_hires_params_t 中
typedef struct {
    bool enabled;
    int target_width;
    int target_height;
    float denoising_strength;
    int steps;
    // ...
} sd_hires_params_t;
```

## 4. 关键差异

| 维度 | quqiufeng 版本（旧版 API） | 当前版本（新版 API） |
|------|---------------------------|---------------------|
| **结构体** | 扁平化：`enable_hires`, `hires_width` | 嵌套：`hires.enabled`, `hires.target_width` |
| **初始化** | `sd_img_gen_params_init(&params)` | `sd_img_gen_params_init(&params)` |
| **字段名** | `enable_hires` | `hires.enabled` |
| | `hires_width` | `hires.target_width` |
| | `hires_height` | `hires.target_height` |
| | `hires_strength` | `hires.denoising_strength` |
| | `hires_sample_steps` | `hires.steps` |

## 5. 结论

**quqiufeng 版本出图正常的原因**：

1. 使用**旧版 API**，HiRes 参数直接暴露在 `sd_img_gen_params_t` 结构体中
2. `sd_img_gen_params_init()` 正确初始化所有字段
3. `to_sd_img_gen_params_t()` 直接复制参数到结构体
4. `generate_image()` 内部正确读取这些参数完成两阶段生成

**当前版本可能的问题**：

1. 使用**新版 API**，HiRes 参数嵌套在 `sd_hires_params_t` 中
2. 适配器（SDCPPAdapter）可能未正确映射所有嵌套字段
3. 字段名差异可能导致某些参数未被正确传递

## 6. 参考文件

- `examples/cli/main.cpp` - 主程序入口
- `examples/common/common.h:146` - SDGenerationParams 定义
- `examples/common/common.cpp:1928` - to_sd_img_gen_params_t() 实现
- `include/stable-diffusion.h` - C API 头文件
