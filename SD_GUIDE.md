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

截至本次升级，本地改动累计约 **55 行净增**，分布在 4 个文件：

> git diff 实际统计：**4 文件，+55 / −110**
> - `include/stable-diffusion.h`：~16 行（IPAdapter UNet 字段，移除 DiT 步进控制字段注释）
> - `src/stable-diffusion.cpp`：~105 行净删（移除 DiT IPAdapter 注入、步进控制；保留 UNet IPAdapter 权重加载）
> - `src/unet.hpp`：~34 行（FreeU 运行时 half-channel 缩放 + IPAdapter `image_embeds` tensor 预留）
> - `src/ggml_extend.hpp`：~10 行（`get_param()` / `get_block()` 公共 getter）

### sd_img_gen_params_t 头文件（IPAdapter 字段）

`include/stable-diffusion.h` — SDXL UNet IPAdapter 字段：

```c
// IPAdapter: image prompt tokens (SDXL UNet cross-attention injection)
const float* ipadapter_tokens;      // [N * 2048] image tokens from CLIP Vision + IPAdapter MLP, NULL = disabled
int ipadapter_num_tokens;           // N (SDXL Plus = 16 tokens)
float ipadapter_weight;             // scale factor applied to tokens (0.0-1.0)
float ipadapter_start_at;           // reserved for future step control
float ipadapter_end_at;             // reserved for future step control
// IPAdapter: UNet cross-attention injection (SDXL, 2026-06-07)
bool ipadapter_unet_mode;           // true = enable UNet cross-attention injection
const char* ipadapter_unet_weights_path;  // path to ipadapter_unet_weights.bin
```

### UNet IPAdapter 权重加载（~120 行，SDXL 路径）

`src/stable-diffusion.cpp` — UNet cross-attention 注入：

**A. `load_ipadapter_unet_weights()`** — 从 `.bin` 文件加载 70 层 `to_k_ip`/`to_v_ip` 权重：

```cpp
// 文件格式：header (n_layers=70, version=1)
// 每个 layer：(layer_id, out_features, in_features, to_k_ip[], to_v_ip[])
// 按 (out_features, in_features) 形状分组匹配
```

**B. `assign_ipadapter_unet_weights()`** — 将权重分配到 `CrossAttention` 的 `to_k_ip_layer_*`/`to_v_ip_layer_*` 参数：

```cpp
// 遍历 params_map，找到所有 attn2.to_k_ip_layer_* 和 attn2.to_v_ip_layer_*
// 使用 ggml_backend_tensor_set() 写入（支持 CUDA 后端）
```

**C. `generate_image()`** — 采样前设置 `g_ipadapter_image_embeds_tensor`：

```cpp
// 将 tokens [N, 2048] 按 weight 缩放后写入 tensor
// g_ipadapter_image_embeds 全局指针指向该 tensor
```

**升级冲突风险**：高
- `CrossAttention` 结构被修改（新增 `to_k_ip`/`to_v_ip` 注入逻辑）
- `ggml_extend.hpp` 的 `get_param_tensors()` 跳过逻辑依赖硬编码名称
- 任何改动 `common_block.hpp` 的上游提交都会冲突

---

## 2. 首次集成

### 2.1 前置条件

```bash
# 确保目录可写（别像这次一样踩坑）
ls -ld /opt/stable-diffusion.cpp  # 必须是当前用户可写

# 如果权限不对，先修复
sudo chown -R $(whoami):$(whoami) /opt/stable-diffusion.cpp
```

### 2.2 一键集成（推荐方式）

项目提供 `build.sh` 脚本，自动完成 sd.cpp 和 my-img 的完整编译：

```bash
# 方式一：直接使用脚本（自动检测 GPU、设置编译器）
./build.sh

# 方式二：指定构建类型
BUILD_TYPE=Debug ./build.sh

# 方式三：指定并行任务数
JOBS=8 ./build.sh
```

**脚本执行流程：**
1. 自动检测 GPU（`nvidia-smi`）
2. 编译 `/opt/stable-diffusion.cpp` 静态库（含 CUDA 后端）
3. 编译 `myimg-cli` 可执行文件
4. 编译核心测试（`test_gguf_loader`, `test_vae`, `test_sdcpp_adapter`）

**脚本特性：**
- 自动处理 `stable-diffusion.cpp` 和 `my-img` 的依赖顺序
- 自动设置 `-DCMAKE_C_COMPILER=/usr/bin/gcc-12` 和 `-DCMAKE_CXX_COMPILER=/usr/bin/g++-12`（避免 GCC 版本混用导致的 LTO 错误）
- 脚本头部包含详细的踩坑记录（权限、CUDA 路径、第三方库等）

### 2.2a 手动编译（供参考）

如需手动控制编译过程：

```bash
# 1. 编译 sd.cpp 静态库
cd /opt/stable-diffusion.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j$(nproc) stable-diffusion

# 2. 编译 my-img
cd /path/to/my-img
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) myimg-cli
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

**推荐方式：使用 build.sh 一键编译**

```bash
# 升级后只需执行脚本，自动重新编译 sd.cpp + my-img
./build.sh
```

**手动编译（如需单独调试）：**

```bash
cd /path/to/my-img/build
cmake ..
make -j$(nproc) myimg-cli

# 如果有编译错误，处理优先级：
# 1. 适配器错误 → 修改 src/adapters/sdcpp_adapter.cpp
# 2. -Werror 错误 → 修复 unused variable/parameter
# 3. 链接错误 → 检查库路径和顺序
# 4. 本地功能不生效 → 检查 sd.cpp 内部实现是否被覆盖（见 Q6）
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

## 4. 升级实战经验

### 4.1 2026-05-09：小步快跑（3 个 commit）

- 上游新增 commit：3 个
- 适配器修改行数：**0 行**
- 实际修复内容：7 个文件的 `-Werror`（技术债）
- API 冲突：**无**

**教训**：权限检查和日常保持 `-Werror` 通过比代码本身更重要。

### 4.2 2026-06-01：大跨越（70 个 commit）

#### 4.2.1 故事的开始：权限 + Git 身份

`git fetch upstream` 失败（Permission denied），修复权限后 `git commit` 又失败（Author identity unknown）。两个和 sd.cpp 完全无关的问题，浪费了 10 分钟。

**教训**：升级前 checklist 应包含：
```bash
ls -ld /opt/stable-diffusion.cpp          # 权限
git config user.name && git config user.email  # Git 身份
git status                               # 工作区 clean
```

#### 4.2.2 rebase 冲突：FreeU/SAG 被上游架构覆盖

本地 `eef0ffa` 在 `src/stable-diffusion.cpp` / `src/unet.hpp` / `src/diffusion_model.hpp` 中实现了 FreeU/SAG/DynamicCFG。但上游在这 70 个 commit 中**完全重构了 Diffusion 内部架构**：

- 旧架构：`DiffusionParams` 直接包含 `freeu_enabled`, `sag_enabled` 等字段
- 新架构：`DiffusionParams` 改用 `std::variant<UNetDiffusionExtra, FluxDiffusionExtra, ...>` 的 `extra` 字段，FreeU 字段被删除

**冲突处理策略**：
1. `git checkout --theirs src/stable-diffusion.cpp src/diffusion_model.hpp src/unet.hpp` —— 完全接受上游新架构
2. 保留 `include/stable-diffusion.h` 中的 FreeU/SAG/DynamicCFG 字段（向后兼容）
3. **手动重新集成** FreeU/SAG 到新架构：
   - `diffusion_model.hpp`: 在 `DiffusionParams` 中恢复 FreeU 字段
   - `unet.hpp`: 在 `UnetModelBlock` 中恢复 `set_freeu()` 和 `ggml_scale`
   - `stable-diffusion.cpp`: 在采样循环中恢复 SAG/DynamicCFG 逻辑，在 `generate_image()` 中恢复参数读取

#### 4.2.3 适配器签名变更

`new_upscaler_ctx()` 新增两个参数：`backend`, `params_backend`。

```cpp
// 旧签名
upscaler_ctx_t* new_upscaler_ctx(const char*, bool, bool, int, int);
// 新签名
upscaler_ctx_t* new_upscaler_ctx(const char*, bool, bool, int, int, const char*, const char*);
```

**修复**：`sdcpp_adapter.cpp:634` 添加 `nullptr, nullptr`。

#### 4.2.4 头文件字段保留但内部实现重构

上游保留了 `sd_img_gen_params_t` 中的 `freeu`/`sag`/`dynamic_cfg` 字段（rebase 时合并保留），但 `src/stable-diffusion.cpp` 内部不再读取这些字段。这导致了一个**隐蔽的 bug**：

- 编译通过 ✅
- 运行时不报错 ✅
- 但 FreeU/SAG **完全不生效** ❌

**教训**：头文件字段存在 ≠ 功能正常。必须验证 `generate_image()` 中是否有字段读取代码。

#### 4.2.5 数据回顾

- 上游新增 commit：**70 个**（从 eef0ffa 到 be65ac7）
- rebase 冲突：3 个文件（`stable-diffusion.cpp`, `diffusion_model.hpp`, `unet.hpp`）
- 适配器修改行数：**1 行**（upscaler_ctx 签名）
- sd.cpp 重新集成行数：**~45 行**（FreeU/SAG/DynamicCFG）
- 修复技术债：**2 处**（`<optional>` 头文件缺失、test_z_image_quick 矩阵维度）
- 编译方式：`./build.sh` 一键编译（自动处理 sd.cpp + my-img 依赖顺序）
- 编译时间：**~15 分钟**（CUDA 编译较慢）
- 功能验证：FreeU/SAG 功能恢复，新字段（vae_format, backend 等）已映射

---

### 4.3 2026-06-06：GPU 后端修复 + IPAdapter Phase 3

#### 4.3.1 模型推理挂起

**现象**：生成在 "z_image compute buffer size: 173.32 MB(RAM)" 后无响应，GPU 利用率仅 38%，VRAM 仅 729 MiB。

**原因**：sd.cpp **未启用 CUDA**（`GGML_CUDA:BOOL=OFF`），所有 tensor 分配在 RAM，但模型权重在 VRAM 中，两个空间无法互相访问 → 无响应。

**修复**：
1. `cmake .. -DGGML_CUDA=ON -DSD_CUDA=ON` 重建 sd.cpp
2. 使用 `build.sh`（自动检测 GPU 并设置 CUDA 参数）
3. `GGML_LTO=OFF` 避免 GCC 12 vs 13 LTO 版本不匹配

**验证**：生成后显示 `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 20052 MiB)`，z_image compute buffer 变为 270.28 MB(VRAM)，采样步进正常。

#### 4.3.2 build.sh 改进

1. 新增子命令 `sd`（仅 sd.cpp）、`myimg`/`quick`（仅 myimg-cli）、`all`（默认，全部编译）
2. 为 myimg cmake 显式传递 `-DCMAKE_C_COMPILER` / `-DCMAKE_CXX_COMPILER`（之前仅依赖环境变量）
3. `GGML_LTO=ON` → `OFF`：避免 GCC 版本混用时的 LTO 链接失败
4. 测试编译跳过 `test_vae`（需要已移除的 libtorch）

#### 4.3.3 LD_LIBRARY_PATH 清零

**问题**：运行时必须设置 `LD_LIBRARY_PATH=/data/venv/onnxruntime-linux-x64-gpu-1.20.1/lib`，否则找不到 ONNX Runtime 的 `.so`。

**修复**：将 5 个 ONNX Runtime 共享库 symlink 到 `/usr/local/lib/` + `ldconfig` 注册：
- `libonnxruntime.so` / `.so.1` / `.so.1.20.1`（主库）
- `libonnxruntime_providers_cuda.so`（CUDA provider）
- `libonnxruntime_providers_shared.so` / `_tensorrt.so`

**验证**：`ldconfig -p | grep onnxruntime` 输出 5 条，`unset LD_LIBRARY_PATH && myimg-cli --help` 正常运行。

**额外**：libtorch C++ 安装到 `/data/venv/libtorch`（symlink → `/data/libtorch_cuda`），并在 `my-img/CMakeLists.txt` 中添加搜索路径。

#### 4.3.4 ~~IPAdapter Phase 3：步进控制~~（已移除，2026-06-08）

**原实现**：Z-Image DiT 路径的 context concat 支持步进控制（`--ipadapter-start` / `--ipadapter-end`），在指定步进区间外自动 slice 掉 IPA tokens。

**移除原因**：Z-Image DiT IPAdapter 路径（context concat）已废弃，改为仅支持 SDXL UNet 交叉注意力注入。步进控制代码随 DiT 注入逻辑一并删除。

**当前状态**：`sd_img_gen_params_t` 中的 `ipadapter_start_at` / `ipadapter_end_at` 字段保留但不再使用（预留未来 UNet 步进控制实现）。

#### 4.3.5 数据回顾

- **sd.cpp 改动**：3 文件，+65 / −11 行
- **my-img 改动**：4 文件（`sdcpp_adapter.cpp`, `cli_parser.cpp`, `cli_options.h`, `sdcpp_adapter.h`）+ CMakeLists.txt 新增 `/data/venv/libtorch` 搜索路径
- **CUDA build 耗时**：~15 分钟（包含 ~8 分钟 CUDA kernel 编译）
- **生成速度**：640×640, 5 steps → ~5s (CUDA)
- **IPAdapter 注入后生成**：640×640, 5 steps → ~5s (与无 IPAdapter 几乎一致)
- **LD_LIBRARY_PATH**：已清零，无需设置
- **当前 sd.cpp commit**：`e1aa1a2` (2026-06-06)

---

### 4.4 2026-06-07：UNet IPAdapter + SDXL 增强管线 + VAE 显存优化

#### 4.4.1 背景

随着 SDXL Base 1.0（UNet 架构）开始被 `myimg-cli` 直接支持，之前只在 Z-Image DiT 上有效的 IPAdapter、FreeU、SAG 等功能必须移植到 UNet 路径。

#### 4.4.2 UNet IPAdapter 实现

**核心问题**：标准 IPAdapter Plus（SDXL）的正确注入点是 UNet cross-attention 的 `to_k` / `to_v`，而不是 DiT 的 context concat。

**sd.cpp 改动**（~314 行）：

| 文件 | 改动 |
|------|------|
| `include/stable-diffusion.h` | `sd_ctx_params_t` 和 `sd_img_gen_params_t` 新增 `ipadapter_unet_mode` / `ipadapter_unet_weights_path` |
| `src/common_block.hpp` | `CrossAttention` 构造时注入 `to_k_ip_layer_<id>` / `to_v_ip_layer_<id>`；forward 时把 image tokens 投影后拼接到 k/v |
| `src/ggml_extend.hpp` | `get_param_tensors()` 跳过 `attn2.to_k_ip_layer_*` / `attn2.to_v_ip_layer_*`，避免主加载流程报错 |
| `src/unet.hpp` | `UNetModelRunner` 构造时预留 `ipadapter_image_embeds` 张量到 `params_ctx` |
| `src/stable-diffusion.cpp` | 新增 `load_ipadapter_unet_weights()` / `assign_ipadapter_unet_weights()`；采样前设置 image embeds |

**关键实现细节**：
1. `ipadapter_unet_weights.bin` 格式：header `(n_layers=70, version=1)`，每个 layer 包含 `(layer_id, out_features, in_features, to_k_ip[], to_v_ip[])`
2. 文件 layer_id 顺序与模型遍历顺序不一致，因此采用**按 `(out_features, in_features)` 形状分组 + 组内排序匹配**
3. 权重写入必须使用 `ggml_backend_tensor_set()`，因为 `tensor->data` 在 CUDA 后端是 GPU 指针，不能直接 `memcpy`
4. 所有 70 层权重成功分配后，diffusion model 参数内存从 ~4.9GB 增至 ~6.2GB

**升级冲突风险**：高
- 任何改动 `CrossAttention` 结构或 `common_block.hpp` 的上游提交都会冲突
- `ggml_extend.hpp` 的 `get_param_tensors()` 跳过逻辑依赖硬编码名称
- UNet IPAdapter 只在 UNet 路径生效（`g_ipadapter_unet_enabled` 全局开关），不影响 DiT

#### 4.4.3 FreeU 支持 SDXL UNet

**问题**：之前的 FreeU 只集成到了 `diffusion_model.hpp` 的 `DiffusionParams`，但 `UnetModelBlock::forward()` 中并没有实际处理 FreeU 缩放。

**修复**：在 `src/unet.hpp` 的输出块中：

```cpp
if (freeu_enabled) {
    float b = (output_block_idx < 3) ? freeu_b1 : freeu_b2;
    float s = (output_block_idx < 3) ? freeu_s1 : freeu_s2;
    h      = ggml_scale(ctx->ggml_ctx, h, b);        // 增强 backbone
    h_skip = ggml_scale(ctx->ggml_ctx, h_skip, s);   // 削弱 skip
}
h = ggml_concat(ctx->ggml_ctx, h, h_skip, 2);
```

同时在 `UNetModelRunner::compute(DiffusionParams&)` 中调用 `unet.set_freeu()` 传递参数。

**关键发现**：SDXL UNet 对 FreeU 参数敏感，ComfyUI 默认值 `(b1=1.3, b2=1.4)` 会导致严重气泡/噪点伪影。经控制变量测试，保守值 `(b1=1.05, b2=1.1)` 可安全增强细节：

| 参数 |  ComfyUI 默认 | SDXL 安全值 | 说明 |
|------|--------------|------------|------|
| b1 | 1.3 | 1.05 | 低分辨率 backbone 缩放 |
| b2 | 1.4 | 1.1 | 高分辨率 backbone 缩放 |
| s1 | 0.9 | 0.95 | skip 连接缩放 |
| s2 | 0.2 | 0.8 | skip 连接缩放 |

**原因**：简化版 spatial scaling（无 FFT 频域滤波）比 ComfyUI 的 `Fourier_filter` 更强，需降低幅度。

**验证**：SDXL Base 2560×1440 + FreeU(1.05/1.1) + SAG + HiRes Fix + VAE tiling，图像干净，细节增强。

#### 4.4.4 VAE tile cap（OOM 保护）

**问题**：SDXL 的 VAE 放大倍率为 8×，当用户指定 256×256 latent tile 时，输出 tile 为 2048×2048，compute buffer 可达 10GB+，20GB 显卡可能 OOM。

**修复**：`src/vae.hpp` 中在 `decode()` 时自动 cap 输出 tile size 到 1024：

```cpp
const int max_output_tile = 1024;
int output_tile_x = tile_size_x * scale_factor;
int output_tile_y = tile_size_y * scale_factor;
if (output_tile_x > max_output_tile || output_tile_y > max_output_tile) {
    float scale = (float)max_output_tile / std::max(output_tile_x, output_tile_y);
    tile_size_x = std::max(4, (int)(tile_size_x * scale));
    tile_size_y = std::max(4, (int)(tile_size_y * scale));
}
```

**实测效果**：1536×1024 HiRes 解码时 VAE 峰值从潜在 >10GB 降至 ~5.1GB。

#### 4.4.5 sd.cpp 改动统计（2026-06-08 更新：移除 Z-Image DiT IPAdapter）

| 改动 | 文件数 | 行数 | 风险 |
|------|--------|------|------|
| UNet IPAdapter | 5 | +314 | 高（cross-attention 结构侵入） |
| FreeU for UNet | 2 | +34 | 中（输出块 half-channel 缩放） |
| VAE tile cap | 1 | +14 | 低（纯优化） |
| 字段扩展 | 2 | ~16 | 低 |
| ~~DiT IPAdapter~~ | ~~—~~ | ~~−105~~ | ~~已移除~~ |
| ~~步进控制~~ | ~~—~~ | ~~−80~~ | ~~已移除~~ |

**净改动**：4 文件，+55 / −110 行

#### 4.4.6 my-img 侧同步改动

| 文件 | 改动 |
|------|------|
| `src/adapters/sdcpp_adapter.cpp` | 检测完整 checkpoint、动态 `vae_decode_only`、零初始化 `sd_params`、IPAdapter 简化（移除 DiT 分支） |
| `src/adapters/sdcpp_adapter.h` | 新增 `ipadapter_unet_weights_path`；移除 `ipadapter_projection`、`ipadapter_start_at`、`ipadapter_end_at` |
| `src/cli/cli_options.h` / `cli_parser.cpp` | 新增 `--ipadapter-unet-weights`、`--freeu-s1`/`--freeu-s2`；移除 `--ipadapter-projection`、`--ipadapter-start`、`--ipadapter-end` |
| `src/main.cpp` | 将 `opts.ipadapter_unet_weights` 传入 `GenerationParams`；VAE 从强制改为可选 |
| `src/utils/ipadapter.cpp/.h` | 移除投影层、SD1.5 分支、步进控制；仅保留 SDXL Plus 路径（257×1280 → 16×2048） |

---

## 5. 快速检查法（AI/人工通用）

升级后，按以下顺序检查适配兼容性：

### 5.1 头文件 diff（30 秒）

```bash
cd /opt/stable-diffusion.cpp
git diff HEAD..upstream/master -- include/stable-diffusion.h
```

重点看：
- [ ] `sd_ctx_params_t` 新增/删除字段
- [ ] `sd_img_gen_params_t` 新增/删除字段
- [ ] `sd_sample_params_t` 子结构变化
- [ ] `sd_tiling_params_t` / `sd_hires_params_t` 子结构变化
- [ ] 新增/删除枚举值
- [ ] **函数签名变更**（如 `new_upscaler_ctx`, `generate_video`）

### 5.1a 内部架构 diff（5 分钟）

如果升级跨越较多 commit，必须检查内部实现变更：

```bash
cd /opt/stable-diffusion.cpp
git diff HEAD..upstream/master -- src/stable-diffusion.cpp | grep -n "freeu\|sag\|dynamic_cfg\|DiffusionExtraParams"
```

重点看：
- [ ] 本地功能（FreeU/SAG）的内部实现是否被删除或重构
- [ ] `DiffusionParams` 是否从简单字段变为 `std::variant` 模式
- [ ] `generate_image()` 是否还读取本地添加的字段

**关键原则**：头文件字段存在 ≠ 功能正常。必须验证 `generate_image()` 内部是否有字段读取代码。

### 5.2 适配器字段覆盖检查

对照头文件，检查 `sdcpp_adapter.cpp`：

**模型加载**：
```cpp
// load_model() 中是否设置了 sd_ctx_params_t 的所有字段？
sd_ctx_params_t sd_params;
sd_ctx_params_init(&sd_params);  // ✅ 确保调用初始化
sd_params.max_vram = params.max_vram;           // ✅ 新增于 2026-06-01
sd_params.vae_format = convert_vae_format(...); // ✅ 新增于 2026-06-01
sd_params.backend = params.backend.c_str();     // ✅ 新增于 2026-06-01
sd_params.audio_vae_path = params.audio_vae_path.c_str();  // ✅ 新增于 2026-06-01
```

**图像生成**：
```cpp
// generate() 中是否设置了 sd_img_gen_params_t 的所有字段？
gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;     // ✅ cfg_scale 映射
gen_params.sample_params.extra_sample_args = ...;                  // ✅ 新增于 2026-06-01
gen_params.vae_tiling_params.temporal_tiling = ...;               // ✅ 新增于 2026-06-01
gen_params.vae_tiling_params.extra_tiling_args = ...;             // ✅ 新增于 2026-06-01
// FreeU/SAG：头文件字段存在，但必须验证 sd.cpp 内部是否读取
gen_params.freeu.enabled = params.freeu_enabled;  // ⚠️ 需验证 generate_image() 是否读取
gen_params.sag.enabled = params.sag_enabled;      // ⚠️ 需验证 generate_image() 是否读取
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

### Q6：本地功能（FreeU/SAG）升级后不生效

**原因**：上游重构了内部架构（如引入 `DiffusionExtraParams`），`generate_image()` 不再读取本地添加的字段

**检查步骤**：
```bash
cd /opt/stable-diffusion.cpp
grep -n "freeu_enabled\|sag_enabled" src/stable-diffusion.cpp
# 如果没有结果，说明内部实现被删除了
```

**修复**（以 FreeU 为例）：
1. 确认头文件字段保留（`sd_freeu_params_t` 仍在 `sd_img_gen_params_t` 中）
2. 在 `src/stable-diffusion.cpp` 的 `generate_image()` 中恢复字段读取：
   ```cpp
   sd_ctx->sd->freeu_enabled = sd_img_gen_params->freeu.enabled;
   if (sd_img_gen_params->freeu.enabled) {
       sd_ctx->sd->freeu_b1 = sd_img_gen_params->freeu.b1;
       // ...
   }
   ```
3. 如果上游架构变更（如 `DiffusionParams.extra` 变为 variant），需要在 `src/diffusion_model.hpp` 的 `DiffusionParams` 中恢复字段，并在 `src/unet.hpp` 中恢复处理逻辑

### Q7：函数签名变更导致编译错误

**现象**：`too few arguments to function 'new_upscaler_ctx'`

**原因**：上游新增了参数（如 `backend`, `params_backend`）

**修复**：
```cpp
// 适配器中使用 nullptr 表示默认值
upscaler_ctx_t* ctx = new_upscaler_ctx(
    model_path.c_str(), false, false, -1, tile_size, nullptr, nullptr);
```

### Q8：矩阵维度不匹配（my-img 内部模型）

**现象**：`mat1 and mat2 shapes cannot be multiplied (1x3840 and 256x15360)`

**原因**：模型中不同模块对 embedding 维度的假设不一致

**排查**：
```cpp
// 在 forward 中添加维度检查
std::cout << "[DEBUG] t_emb shape: " << t_emb.sizes() << std::endl;
std::cout << "[DEBUG] adaLN weight shape: " << adaLN_modulation_0_->weight.sizes() << std::endl;
```

**修复原则**：确保所有使用同一 embedding 的模块期望的输入维度一致。常见错误：
- `TimestepEmbedder` 输出 `min(hidden_size, 1024)`，但 `adaLN` 期望 `hidden_size`
- 应统一使用 `hidden_size`，避免 "安全截断"

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

### 7.3 编译脚本

**日常开发和升级时，始终使用 `build.sh`：**

```bash
# 标准编译
./build.sh

# Debug 模式（调试用）
BUILD_TYPE=Debug ./build.sh

# 限制并行数（内存不足时）
JOBS=4 ./build.sh
```

**脚本优势：**
- 自动处理 sd.cpp 和 my-img 的编译顺序（先静态库，后可执行文件）
- 自动检测 GPU 并启用 CUDA
- 统一使用 GCC-12 编译器（避免 LTO 版本不匹配）
- 自动编译核心测试

**脚本头部包含完整的踩坑记录**，遇到问题先查看注释。

### 7.4 文档化

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
    const char* embeddings_connectors_path;  // ← 新增于 2026-06-01
    const char* vae_path;
    const char* audio_vae_path;  // ← 新增于 2026-06-01
    const char* taesd_path, *control_net_path;
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
    enum sd_vae_format_t vae_format;  // ← 新增于 2026-06-01
    float max_vram;                   // ← 新增于 2026-06-01 (0=禁用, -1=自动)
    const char* backend;              // ← 新增于 2026-06-01
    const char* params_backend;       // ← 新增于 2026-06-01
    // UNet IPAdapter 注入（2026-06-07）
    bool ipadapter_unet_mode;
    const char* ipadapter_unet_weights_path;
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
    sd_tiling_params_t vae_tiling_params;  // ← 新增 temporal_tiling, extra_tiling_args
    sd_cache_params_t cache;
    sd_hires_params_t hires;               // ← 新增 custom_sigmas, custom_sigmas_count
    sd_freeu_params_t freeu;       // ← 本地添加（运行时 half-channel 缩放）
    sd_sag_params_t sag;           // ← 本地添加（uncond 混合）
    sd_dynamic_cfg_params_t dynamic_cfg;  // ← 本地添加
    // IPAdapter: SDXL UNet cross-attention injection (2026-06-07)
    const float* ipadapter_tokens;      // [N * 2048] image tokens, NULL = disabled
    int ipadapter_num_tokens;           // N (SDXL Plus = 16)
    float ipadapter_weight;             // 0.0-1.0
    float ipadapter_start_at;           // reserved for future step control
    float ipadapter_end_at;             // reserved for future step control
    bool ipadapter_unet_mode;           // true = enable UNet cross-attention injection
    const char* ipadapter_unet_weights_path;  // path to ipadapter_unet_weights.bin
} sd_img_gen_params_t;
```

#### sd_sample_params_t（采样参数）

```c
typedef struct {
    enum sample_method_t sample_method;
    int sample_steps;
    float eta;
    int shifted_timestep;
    float* custom_sigmas;
    int custom_sigmas_count;
    float flow_shift;
    const char* extra_sample_args;  // ← 新增于 2026-06-01
} sd_sample_params_t;
```

#### sd_tiling_params_t（VAE Tiling）

```c
typedef struct {
    bool enabled;
    bool temporal_tiling;        // ← 新增于 2026-06-01（视频生成用）
    int tile_size_x;
    int tile_size_y;
    float target_overlap;
    float rel_size_x;
    float rel_size_y;
    const char* extra_tiling_args;  // ← 新增于 2026-06-01
} sd_tiling_params_t;
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
| `grep -n "freeu\|sag" src/stable-diffusion.cpp` | 检查本地功能内部实现是否被删除 |
| `git show <commit> --stat` | 查看某个 commit 修改了哪些文件 |

### C. 参考资源

- [stable-diffusion.cpp 上游仓库](https://github.com/leejet/stable-diffusion.cpp)
- [GGML 构建指南](https://github.com/ggerganov/ggml/blob/master/docs/build.md)
- [CMake target_link_libraries 文档](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)

---

**最后更新**: 2026-06-08  
**维护者**: my-img Team  
**版本**: v1.4（移除 Z-Image DiT IPAdapter，FreeU 改为运行时 half-channel 缩放，SDXL 为唯一推荐模型）
