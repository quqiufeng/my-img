# sd.cpp 集成与升级指南

> **本文档涵盖 stable-diffusion.cpp 的首次集成和后续升级全流程。**
>
> 目标：让升级成为一件可预期、低风险、可重复的操作。

---

## 1. 架构设计

### 1.1 集成模式：适配器 + 最小侵入

```
my-img (C++ Application)
├── src/adapters/sdcpp_adapter.{h,cpp}  ← 唯一接触 sd.cpp API 的地方
├── src/main.cpp                        ← CLI 入口
├── src/utils/                          ← 工具函数
└── /opt/stable-diffusion.cpp/          ← sd.cpp 源码（独立管理）
    ├── build/libstable-diffusion.a     ← 静态库
    ├── build/ggml/src/libggml.a        ← GGML 库
    └── include/stable-diffusion.h      ← C API 头文件
```

**核心原则**：
- **隔离性**：只有 `sdcpp_adapter` 直接包含 `<stable-diffusion.h>`
- **静态链接**：sd.cpp 编译为静态库
- **版本锁定**：sd.cpp 作为独立目录管理，不依赖系统包
- **最小侵入**：sd.cpp 源码改动控制在 30 行以内（最佳实践：0 行）

### 1.2 当前对 sd.cpp 的改动

截至本次升级，本地改动仅 21 行：

```diff
# 1. 恢复 8 个 static 关键字（原生代码本来就有）
-int64_t resolve_seed(int64_t seed) {
+static int64_t resolve_seed(int64_t seed) {

# 2. 删除了末尾废弃的扩展 API include
-#include "stable-diffusion-ext.cpp"
+
```

**升级时无需担心冲突**，直接 rebase 即可。

---

## 2. 首次集成

### 2.1 前置条件

```bash
# 确保目录可写（别像这次一样踩坑）
ls -ld /opt/stable-diffusion.cpp  # 必须是当前用户可写

# 如果权限不对，先修复
sudo chown -R $(whoami):$(whoami) /opt/stable-diffusion.cpp
```

### 2.2 一键集成

```bash
# 1. Clone sd.cpp
mkdir -p /opt
git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git \
    /opt/stable-diffusion.cpp

# 2. 编译静态库
cd /opt/stable-diffusion.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j$(nproc) stable-diffusion

# 3. 编译 my-img
cd /path/to/my-img
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) myimg-cli
```

或使用项目脚本：
```bash
./build.sh
```

### 2.3 CMake 关键配置

```cmake
# sd.cpp 路径
set(SDCPP_DIR /opt/stable-diffusion.cpp)
set(SDCPP_BUILD_DIR ${SDCPP_DIR}/build)

# 链接库（注意顺序！）
set(SDCPP_LINK_LIBRARIES
    ${SDCPP_BUILD_DIR}/libstable-diffusion.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml-base.a
)

# CUDA 后端（必须 --whole-archive）
if(EXISTS "${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a")
    list(APPEND SDCPP_LINK_LIBRARIES 
        "-Wl,--whole-archive"
        ${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a
        "-Wl,--no-whole-archive"
    )
endif()
```

---

## 3. 升级标准操作流程（SOP）

### 升级前检查清单

```bash
□ 1. 检查目录权限（避免 git 操作失败）
□ 2. 确保本地工作区 clean（git status）
□ 3. 记录当前版本（git log --oneline -1 > SD_VERSION.lock）
□ 4. 备份当前 build/（可选，但推荐）
```

### 3.1 执行升级

```bash
cd /opt/stable-diffusion.cpp

# 1. 添加 upstream（如未添加）
git remote add upstream https://github.com/leejet/stable-diffusion.cpp.git

# 2. 拉取上游最新代码
git fetch upstream

# 3. 快速检查 API 变化（最重要的一步！）
git diff HEAD..upstream/master -- include/stable-diffusion.h

# 4. 变基到上游（保留本地提交）
git rebase upstream/master

# 5. 更新子模块
git submodule update --init --recursive

# 6. 重新编译
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j$(nproc) stable-diffusion
```

### 3.2 编译 my-img 并修复

```bash
cd /path/to/my-img/build
cmake ..
make -j$(nproc) myimg-cli

# 如果有编译错误，处理优先级：
# 1. 适配器错误 → 修改 src/adapters/sdcpp_adapter.cpp
# 2. -Werror 错误 → 修复 unused variable/parameter
# 3. 链接错误 → 检查库路径和顺序
```

### 3.3 功能验证

```bash
# 基础生成
./build/myimg-cli --model /path/to/model.gguf -p "test" --steps 5 -o /tmp/test.png

# HiRes Fix
./build/myimg-cli --model /path/to/model.gguf -p "test" --hires -o /tmp/test_hires.png

# 测试脚本
./scripts/pre-commit.sh
```

---

## 4. 本次升级实战经验（2026-05-09）

这次升级上游 leejet 仓库新增了 3 个 commit，整个过程耗时约 15 分钟。但最有趣的并不是升级本身有多快，而是它暴露了三个我们在日常开发中容易忽视的问题。

### 4.1 故事的开始：权限这个"隐形守门员"

升级的第一步是 `git fetch upstream`，但命令立刻失败了——`Permission denied: .git/FETCH_HEAD`。原来 `/opt/stable-diffusion.cpp` 是 root 创建的，普通用户无法写入 `.git/` 目录。

这个错误和 sd.cpp 毫无关系，但它告诉我们：**升级流程的第一个检查项应该是权限，而不是代码**。我们花了 5 分钟修权限，如果一开始就 `ls -ld`，这 5 分钟本可以省掉。

### 4.2 意外的编译失败：不是适配器的问题

sd.cpp 编译很顺利，但当编译 my-img 时，突然冒出来 7 个 `-Werror` 错误：`unused variable 'h'`、`unused parameter 'radius'`……

第一反应是"上游改了 API？"但检查后发现，这些代码几个月来一直这样，只是平时没在 `-Werror` 模式下编译过。它们不是升级引入的，而是**平时积累的技术债**。升级就像一面镜子，照出了那些被我们忽略的警告。

这次我们没有新增任何适配器代码，反而修了 7 个旧文件的编译问题。如果平时就保持 `-Werror` 通过，升级过程会更纯粹。

### 4.3 预期的冲突没有发生

本地有一个 commit（`eef0ffa`）修改了 `sd_img_gen_params_t`，添加了 FreeU/SAG/DynamicCFG 三个结构体。升级前最担心的就是这部分和上游冲突。

但 `git diff HEAD..upstream/master -- include/stable-diffusion.h` 显示，上游只改了 `sd_ctx_params_t`（加了 `max_vram`），两个结构体完全不同。rebase 过程零冲突，一次通过。

这说明**最小侵入设计确实有效**：我们只加了字段，没有改逻辑，所以即使结构体在同一个头文件里，也不会和上游冲突。

### 4.4 数据回顾

- 上游新增 commit：3 个
- 适配器修改行数：**0 行**
- 实际修复内容：7 个文件的 `-Werror`（技术债）
- API 冲突：**无**
- 版本锁定：**未执行**（遗漏）

这次升级验证了我们的架构设计是健康的，但也提醒了我们：基础设施（权限、编译规范）和流程（版本锁定）比代码本身更容易成为瓶颈。

---

## 5. 快速检查法（AI/人工通用）

升级后，按以下顺序检查适配兼容性：

### 5.1 头文件 diff（30 秒）

```bash
cd /opt/stable-diffusion.cpp
git diff HEAD~1 -- include/stable-diffusion.h
```

重点看：
- [ ] `sd_ctx_params_t` 新增/删除字段
- [ ] `sd_img_gen_params_t` 新增/删除字段  
- [ ] `sd_sample_params_t` 子结构变化
- [ ] 新增/删除枚举值
- [ ] 函数签名变更

### 5.2 适配器字段覆盖检查

对照头文件，检查 `sdcpp_adapter.cpp`：

```cpp
// load_model() 中是否设置了 sd_ctx_params_t 的所有字段？
sd_ctx_params_t sd_params;
sd_ctx_params_init(&sd_params);  // ✅ 确保调用初始化
sd_params.max_vram = params.max_vram;  // ✅ 新增字段要映射
```

```cpp
// generate() 中是否设置了 sd_img_gen_params_t 的所有字段？
gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;  // ✅ cfg_scale 映射
```

### 5.3 枚举映射完整性

```cpp
// 检查是否覆盖了所有枚举值
case SampleMethod::NewMethod: return NEW_METHOD_SAMPLE_METHOD;  // 如有新增
```

### 5.4 CLI 参数传递检查

```cpp
// main.cpp → GenerationParams → sd_*_params_t
// 确保每个 CLI 参数都传递到最终的结构体
```

---

## 6. 常见问题速查

### Q1：cfg_scale 不生效

**原因**：未映射到 `sample_params.guidance.txt_cfg`

**修复**：
```cpp
gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;
```

### Q2：Flash Attention 没效果

**原因**：只设置了 `flash_attn`，没设置 `diffusion_flash_attn`

**修复**：
```cpp
sd_params.flash_attn = params.flash_attn;
sd_params.diffusion_flash_attn = params.flash_attn;  // 两者都要！
```

### Q3：新增字段导致编译警告

**原因**：手动 memset 或遗漏字段初始化

**修复**：
```cpp
// ✅ 正确：使用初始化函数
sd_ctx_params_t sd_params;
sd_ctx_params_init(&sd_params);

// ❌ 错误：手动初始化
sd_ctx_params_t sd_params = {};  // 可能遗漏新增字段
```

### Q4：链接错误 undefined reference

**排查步骤**：
1. 检查库文件是否存在
2. 检查库链接顺序（sd.cpp 有循环依赖，用 `--start-group`）
3. 检查 CUDA 是否用 `--whole-archive`

### Q5：rebase 冲突

**如果冲突在 sd.cpp 内部**：
```bash
# 1. 查看冲突文件
git status

# 2. 优先保留 upstream 版本（我们遵循最小侵入原则）
git checkout --ours <file>  # 保留我们的修改
git checkout --theirs <file>  # 保留上游的修改

# 3. 标记解决
git add .
git rebase --continue
```

**如果冲突在 include/stable-diffusion.h**：
- 对比两个版本的字段差异
- 合并时确保所有字段都保留
- 在适配器中添加新字段映射

---

## 7. 最佳实践总结

### 7.1 日常维护

```bash
# 每月执行一次检查
cd /opt/stable-diffusion.cpp
git fetch upstream
git log --oneline HEAD..upstream/master  # 查看新 commit
```

### 7.2 升级频率

- **小步快跑**：每次更新 1-3 个 commit
- **避免大跨越**：不要一次性跨越 10+ 个 commit
- **每次必测**：更新后立即编译 + 基础生成测试

### 7.3 文档化

```bash
# 升级后记录版本
cd /opt/stable-diffusion.cpp
git log --oneline -1 > /path/to/my-img/SD_VERSION.lock

# 在 CHANGELOG 中记录
echo "- 升级 sd.cpp 到 $(cat SD_VERSION.lock)" >> CHANGELOG.md
```

### 7.4 CI 建议

```yaml
# .github/workflows/ci.yml
check-sd-version:
  steps:
    - name: Check sd.cpp freshness
      run: |
        cd /opt/stable-diffusion.cpp
        git fetch upstream
        git log --oneline HEAD..upstream/master | wc -l
        # 如果 > 5，发出提醒
```

---

## 8. 附录

### A. 结构体速查

#### sd_ctx_params_t（模型加载）

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
    bool flash_attn, diffusion_flash_attn;
    bool tae_preview_only, diffusion_conv_direct, vae_conv_direct;
    bool circular_x, circular_y, force_sdxl_vae_conv_scale;
    bool chroma_use_dit_mask, chroma_use_t5_mask;
    int chroma_t5_mask_pad;
    bool qwen_image_zero_cond_t;
    float max_vram;  // ← 新增于 2026-05-09
} sd_ctx_params_t;
```

#### sd_img_gen_params_t（图像生成）

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
    sd_freeu_params_t freeu;       // ← 本地添加
    sd_sag_params_t sag;           // ← 本地添加
    sd_dynamic_cfg_params_t dynamic_cfg;  // ← 本地添加
} sd_img_gen_params_t;
```

### B. 相关命令速查

| 命令 | 用途 |
|------|------|
| `git remote add upstream ...` | 添加上游 remote |
| `git fetch upstream` | 拉取上游最新代码 |
| `git log --oneline HEAD..upstream/master` | 查看上游领先 commit |
| `git diff HEAD..upstream/master -- include/stable-diffusion.h` | 对比头文件变化 |
| `git rebase upstream/master` | 变基到上游 |
| `git submodule update --init --recursive` | 更新子模块 |
| `make -j$(nproc) stable-diffusion` | 编译 sd.cpp |
| `nm build/myimg-cli \| grep generate_image` | 检查符号是否存在 |

### C. 参考资源

- [stable-diffusion.cpp 上游仓库](https://github.com/leejet/stable-diffusion.cpp)
- [GGML 构建指南](https://github.com/ggerganov/ggml/blob/master/docs/build.md)
- [CMake target_link_libraries 文档](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)

---

**最后更新**: 2026-05-09  
**维护者**: my-img Team  
**版本**: 合并版 v1.0（原 SD_INTEGRATION.md + SD_UPGRADE_GUIDE.md）
