# SD Upgrade Adaptation Guide

> **sd.cpp 升级后适配指南** - AI 辅助检查和修复流程

---

## 核心原则：最小侵入

经过 libTorch HiRes Fix 实验的教训，我们确立了**最小侵入 sd.cpp 源码**的升级策略：

### 当前对 sd.cpp 的改动（共 21 行）

```diff
# 1. 恢复了 8 个 static 关键字（原生代码本来就有）
-int64_t resolve_seed(int64_t seed) {
+static int64_t resolve_seed(int64_t seed) {

-enum sample_method_t resolve_sample_method(...) {
+static enum sample_method_t resolve_sample_method(...) {

# ... 其他 6 个函数类似

# 2. 删除了末尾的扩展 API include（已废弃）
-#include "stable-diffusion-ext.cpp"
+
```

**结论**：sd.cpp 源码**几乎零改动**，升级时直接覆盖新版即可。

### 升级三步走

```bash
# Step 1: 覆盖新版 sd.cpp（无需担心冲突）
cd /opt/stable-diffusion.cpp
git pull origin master

# Step 2: 检查公开 API 变化
vimdiff include/stable-diffusion.h /tmp/sd.h.old

# Step 3: 在适配器中调整参数映射
# 只需修改 src/adapters/sdcpp_adapter.cpp
```

**不需要做的**：
- ❌ 不需要修改 sd.cpp 内部实现
- ❌ 不需要维护扩展 API
- ❌ 不需要处理符号导出/链接问题

---

## AI 检查法

当 sd.cpp 升级后，**不要手动逐个字段对比**。将以下信息提供给 AI，让 AI 自动检查适配性：

### AI 检查 Prompt 模板

```
stable-diffusion.cpp 已升级，请检查适配器代码是否需要更新。

1. 读取 sd.cpp 最新头文件：/opt/stable-diffusion.cpp/include/stable-diffusion.h
2. 读取适配器实现：src/adapters/sdcpp_adapter.cpp
3. 读取适配器头文件：src/adapters/sdcpp_adapter.h
4. 读取 CLI 入口：src/main.cpp

检查清单：
- sd_ctx_params_t 所有字段是否都在 adapter 的 load_model() 中设置？
- sd_img_gen_params_t 所有字段是否都在 adapter 的 generate() 中设置？
- sd_sample_params_t 的 guidance 子字段（特别是 txt_cfg）是否映射？
- 所有枚举转换函数是否覆盖 sd.cpp 最新的枚举值？
- main.cpp 中所有 CLI 参数是否传递到 GenerationParams？

请列出所有缺失/新增的字段和需要修复的代码。
```

---

## 升级后标准操作流程（SOP）

### Step 1: 更新 sd.cpp

```bash
cd /opt/stable-diffusion.cpp
git pull origin master
git submodule update --init --recursive
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j$(nproc) stable-diffusion
cd /path/to/your/my-img
```

### Step 2: 使用 AI 检查适配性

将以下文件内容提供给 AI：
- `/opt/stable-diffusion.cpp/include/stable-diffusion.h`
- `src/adapters/sdcpp_adapter.cpp`
- `src/adapters/sdcpp_adapter.h`
- `src/main.cpp`

**重点检查以下结构体**：

#### A. sd_ctx_params_t（模型加载参数）

必须检查的字段：
```c
// 模型路径（所有 path 字段）
model_path, clip_l_path, clip_g_path, clip_vision_path, t5xxl_path,
llm_path, llm_vision_path, diffusion_model_path, high_noise_diffusion_model_path,
vae_path, taesd_path, control_net_path, photo_maker_path

// 配置参数
vae_decode_only, free_params_immediately, n_threads, wtype,
rng_type, sampler_rng_type, prediction, lora_apply_mode,
offload_params_to_cpu, enable_mmap,
keep_clip_on_cpu, keep_control_net_on_cpu, keep_vae_on_cpu,
flash_attn, diffusion_flash_attn,  // 两者都要设置！
tae_preview_only, diffusion_conv_direct, vae_conv_direct,
circular_x, circular_y, force_sdxl_vae_conv_scale,
chroma_use_dit_mask, chroma_use_t5_mask, chroma_t5_mask_pad,
qwen_image_zero_cond_t
```

#### B. sd_img_gen_params_t（生成参数）

必须检查的字段：
```c
// 基础参数
prompt, negative_prompt, clip_skip, width, height,
strength, seed, batch_count

// 图像输入
init_image, mask_image, control_image, control_strength

// 参考图像（新增可能）
ref_images, ref_images_count, auto_resize_ref_image, increase_ref_index

// 采样参数（重点检查 guidance）
sample_params.sample_method, sample_params.scheduler,
sample_params.sample_steps, sample_params.eta, sample_params.flow_shift,
sample_params.guidance.txt_cfg,  // ← 这是 cfg_scale！
sample_params.guidance.img_cfg,
sample_params.guidance.distilled_guidance,
sample_params.guidance.slg

// 其他结构体
vae_tiling_params, cache, hires, pm_params, loras, lora_count
```

#### C. 枚举映射

必须检查转换函数：
```cpp
convert_sample_method()  // 是否覆盖所有 SAMPLE_METHOD？
convert_scheduler()      // 是否覆盖所有 SCHEDULER？
convert_hires_upscaler() // 是否覆盖所有 HIRES_UPSCALER？
convert_wtype()          // 是否覆盖所有 SD_TYPE？
```

### Step 3: 修复 AI 发现的问题

根据 AI 的检查结果，修复 `src/adapters/sdcpp_adapter.cpp`：

1. **新增字段**：在对应位置添加赋值
2. **删除字段**：删除不再存在的赋值
3. **重命名字段**：更新为新的字段名
4. **新增枚举**：在转换函数中添加 case

### Step 4: 编译验证

```bash
cd build
make -j$(nproc) myimg-cli
```

### Step 5: 功能测试

```bash
# 基础生成测试
./build/myimg-cli --model model.gguf -p "test" --steps 5 -o /tmp/test.png

# HiRes Fix 测试
./build/myimg-cli --model model.gguf -p "test" --hires -o /tmp/test_hires.png
```

---

## 常见升级问题速查

### 问题1：新增结构体字段

**症状**：编译警告 "missing initializer" 或运行时行为异常

**AI 检查方法**：对比 `sd_*_params_init()` 函数和 adapter 的赋值列表

**修复**：
```cpp
// 使用初始化函数避免遗漏
sd_ctx_params_t sd_params;
sd_ctx_params_init(&sd_params);  // 所有字段获得默认值
// 然后只覆盖需要自定义的字段
sd_params.n_threads = params.n_threads;
```

### 问题2：cfg_scale 未生效

**症状**：无论设置什么 CFG 值，生成结果都一样

**原因**：`cfg_scale` 未映射到 `sample_params.guidance.txt_cfg`

**修复**：
```cpp
gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;
```

### 问题3：Flash Attention 未生效

**症状**：显存占用没有降低

**原因**：只设置了 `flash_attn`，没设置 `diffusion_flash_attn`

**修复**：
```cpp
sd_params.flash_attn = params.flash_attn;
sd_params.diffusion_flash_attn = params.flash_attn;  // 同时设置！
```

### 问题4：新增采样方法/调度器

**症状**：编译错误 "not declared in this scope"

**修复**：在转换函数中添加 case：
```cpp
case SampleMethod::NewMethod: return NEW_METHOD_SAMPLE_METHOD;
```

---

## 预防建议

1. **始终使用 sd_*_params_init()**
   - 初始化函数会自动处理新增字段的默认值
   - 避免手动 memset 或逐字段赋值

2. **定期让 AI 检查**
   - 每月或每次 sd.cpp 更新后运行一次 AI 检查

3. **记录当前兼容版本**
   ```bash
   cd /opt/stable-diffusion.cpp
   git log --oneline -1 > ../../SD_VERSION.lock
   ```

4. **小步更新**
   - 不要一次性跨越太多 commit
   - 每次更新后编译+测试

---

## 附录：结构体定义速查

### sd_ctx_params_t
```c
typedef struct {
    const char* model_path;
    const char* clip_l_path, *clip_g_path, *clip_vision_path;
    const char* t5xxl_path, *llm_path, *llm_vision_path;
    const char* diffusion_model_path, *high_noise_diffusion_model_path;
    const char* vae_path, *taesd_path, *control_net_path;
    const sd_embedding_t* embeddings;
    uint32_t embedding_count;
    const char* photo_maker_path, *tensor_type_rules;
    bool vae_decode_only, free_params_immediately;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type, sampler_rng_type;
    enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode;
    bool offload_params_to_cpu, enable_mmap;
    bool keep_clip_on_cpu, keep_control_net_on_cpu, keep_vae_on_cpu;
    bool flash_attn, diffusion_flash_attn;  // 两者都要设置
    bool tae_preview_only, diffusion_conv_direct, vae_conv_direct;
    bool circular_x, circular_y, force_sdxl_vae_conv_scale;
    bool chroma_use_dit_mask, chroma_use_t5_mask;
    int chroma_t5_mask_pad;
    bool qwen_image_zero_cond_t;
} sd_ctx_params_t;
```

### sd_img_gen_params_t
```c
typedef struct {
    const sd_lora_t* loras;
    uint32_t lora_count;
    const char* prompt, *negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t* ref_images;
    int ref_images_count;
    bool auto_resize_ref_image, increase_ref_index;
    sd_image_t mask_image;
    int width, height;
    sd_sample_params_t sample_params;  // 包含 guidance.txt_cfg (cfg_scale)
    float strength;
    int64_t seed;
    int batch_count;
    sd_image_t control_image;
    float control_strength;
    sd_pm_params_t pm_params;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_hires_params_t hires;
} sd_img_gen_params_t;
```

---

**最后更新**: 2025-04-29
**维护者**: my-img Team
