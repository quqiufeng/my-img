# SD Integration Guide

> **稳定扩散集成指南** - 描述 my-img 如何集成 stable-diffusion.cpp 以及更新维护流程

---

## 1. 集成架构概述

my-img 采用**适配器模式**集成 stable-diffusion.cpp（以下简称 sd.cpp）：

```
my-img (C++ Application)
├── src/adapters/sdcpp_adapter.{h,cpp}  ← 适配器层（唯一接触 sd.cpp API 的地方）
├── src/main.cpp                        ← CLI 入口
├── src/utils/                          ← 工具函数
└── third_party/stable-diffusion.cpp/   ← sd.cpp 源码（独立构建）
    ├── build/libstable-diffusion.a     ← 静态库输出
    ├── build/ggml/src/libggml.a        ← GGML 库
    └── include/stable-diffusion.h      ← C API 头文件
```

**核心原则**：
- **隔离性**：只有 `sdcpp_adapter` 直接包含 sd.cpp 的头文件
- **静态链接**：sd.cpp 编译为静态库，链接到 my-img
- **版本锁定**：sd.cpp 作为独立目录管理，不依赖系统包

---

## 2. 目录结构

### 2.1 sd.cpp 位置

```bash
third_party/stable-diffusion.cpp/          # sd.cpp 根目录
├── include/stable-diffusion.h             # C API 头文件（适配器引用）
├── src/stable-diffusion.cpp               # 核心实现
├── src/sd-engine/                         # sd.cpp 的引擎组件
│   ├── core/                              # 工作流引擎核心
│   ├── nodes/                             # 节点实现
│   └── adapter/sd_adapter.cpp             # sd.cpp 内部适配器
├── ggml/                                  # GGML 子模块
│   └── src/                               # GGML 源码
└── build/                                 # 构建输出（需先编译）
    ├── libstable-diffusion.a              # 主静态库
    ├── ggml/src/libggml.a                 # GGML 库
    ├── ggml/src/libggml-cpu.a             # CPU 后端
    ├── ggml/src/libggml-base.a            # 基础库
    └── ggml/src/ggml-cuda/libggml-cuda.a  # CUDA 后端（如启用）
```

### 2.2 my-img 侧的适配器

```bash
src/adapters/
├── sdcpp_adapter.h                        # 适配器头文件
└── sdcpp_adapter.cpp                      # 适配器实现
    ├── 枚举转换函数                        # SampleMethod/Scheduler/HiresUpscaler
    ├── 参数映射                           # GenerationParams → sd_img_gen_params_t
    ├── 图像格式转换                        # Image ↔ sd_image_t
    └── 模型管理                           # sd_ctx_t 生命周期
```

---

## 3. 构建流程

### 3.1 首次集成步骤

```bash
# 1. Clone sd.cpp（在项目根目录执行）
git clone --recursive https://github.com/leejet/stable-diffusion.git \
    third_party/stable-diffusion.cpp

# 2. 构建 sd.cpp（生成静态库）
cd third_party/stable-diffusion.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j$(nproc) stable-diffusion

# 3. 构建 my-img
cd ../../..
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) myimg-cli
```

### 3.2 使用 build.sh 脚本

项目提供一键构建脚本：

```bash
./build.sh
```

脚本逻辑：
1. 自动检测 GPU 并设置 `SD_CUDA` / `GGML_CUDA`
2. 编译 sd.cpp 静态库（`make stable-diffusion`）
3. 编译 my-img 主程序（`make myimg-cli`）
4. 编译测试程序

### 3.3 CMake 集成细节

`CMakeLists.txt` 中关键配置：

```cmake
# sd.cpp 路径
set(SDCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp)
set(SDCPP_BUILD_DIR ${SDCPP_DIR}/build)

# 检查静态库是否存在
if(EXISTS "${SDCPP_BUILD_DIR}/libstable-diffusion.a")
    set(SDCPP_FOUND TRUE)
else()
    message(FATAL_ERROR "sd.cpp not built. Run ./build.sh first.")
endif()

# 链接库列表（注意顺序！）
set(SDCPP_LINK_LIBRARIES
    ${SDCPP_BUILD_DIR}/libstable-diffusion.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${SDCPP_BUILD_DIR}/ggml/src/libggml-base.a
)

# CUDA 后端（使用 --whole-archive 防止符号丢弃）
if(EXISTS "${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a")
    list(APPEND SDCPP_LINK_LIBRARIES 
        "-Wl,--whole-archive"
        ${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a
        "-Wl,--no-whole-archive"
    )
endif()

# 使用链接组解决循环依赖
set(SDCPP_LINK_START "-Wl,--start-group")
set(SDCPP_LINK_END "-Wl,--end-group")
```

---

## 4. 适配器设计

### 4.1 核心职责

`SDCPPAdapter` 是 my-img 与 sd.cpp 之间的**唯一接口**：

| 职责 | 说明 |
|------|------|
| **类型转换** | my-img 枚举 → sd.cpp C 枚举（`SampleMethod` → `sample_method_t`） |
| **参数映射** | `GenerationParams` → `sd_img_gen_params_t` / `sd_ctx_params_t` |
| **图像转换** | `Image`（C++ 结构） ↔ `sd_image_t`（C 结构） |
| **资源管理** | RAII 封装 `sd_ctx_t` 生命周期（构造/析构/移动语义） |
| **错误处理** | 将 sd.cpp 错误码转换为 my-img 异常/返回值 |

### 4.2 关键数据流

```
用户输入 (CLI/GUI)
    ↓
GenerationParams (my-img 结构)
    ↓
sdcpp_adapter.cpp:
  - convert_sample_method()
  - convert_scheduler()
  - convert_hires_upscaler()
  - image_to_sd_image()
    ↓
sd_img_gen_params_t (sd.cpp C 结构)
    ↓
generate_image() [sd.cpp C API]
    ↓
sd_image_t* (原始图像数据)
    ↓
sd_image_to_image() → Image (my-img 结构)
    ↓
保存/显示
```

### 4.3 版本隔离

如果 sd.cpp 升级导致 API 变化，**只需修改适配器文件**：

```cpp
// src/adapters/sdcpp_adapter.cpp
// 例如 sd.cpp 新增参数时，在此添加映射

// HiRes Fix 参数映射示例
gen_params.hires.enabled = params.enable_hires;
gen_params.hires.upscaler = convert_hires_upscaler(params.hires_upscaler);
gen_params.hires.target_width = params.hires_width;
// ... 其他字段
```

---

## 5. 更新 sd.cpp（git pull）

### 5.1 标准更新流程

当 sd.cpp 上游有更新时，按以下步骤操作：

```bash
# 1. 进入 sd.cpp 目录
cd third_party/stable-diffusion.cpp

# 2. 拉取最新代码
git pull origin master

# 3. 更新子模块（ggml 等）
git submodule update --init --recursive

# 4. 重新编译 sd.cpp
rm -rf build  # 清理旧构建（重要！）
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DGGML_CUDA=ON
make -j$(nproc) stable-diffusion

# 5. 返回项目根目录，重新编译 my-img
cd ../../..
rm -rf build  # 建议清理，避免 CMake 缓存问题
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) myimg-cli
```

### 5.2 检查清单（Checklist）

更新后必须检查以下项目：

#### A. 头文件兼容性

```bash
# 检查 stable-diffusion.h 是否有新增/修改的 API
sd include/stable-diffusion.h > /tmp/sd_api_new.txt
# 对比之前的版本（如有备份）
```

重点关注：
- [ ] 新增/删除的结构体字段（`sd_hires_params_t`、`sd_img_gen_params_t` 等）
- [ ] 新增/删除的枚举值（`sample_method_t`、`scheduler_t`）
- [ ] 函数签名变更（`generate_image`、`new_sd_ctx` 等）

#### B. 编译错误处理

如果出现编译错误，按优先级处理：

1. **头文件找不到**：
   - 检查 `stable-diffusion.h` 路径是否变更
   - 更新 `CMakeLists.txt` 中的包含路径

2. **结构体字段不存在**：
   - 在适配器中添加 `#ifdef` 条件编译
   - 或使用新的字段名

3. **枚举值缺失**：
   - 在转换函数中添加新枚举映射

4. **链接错误（undefined reference）**：
   - 检查库文件路径是否正确
   - 检查库链接顺序（sd.cpp 的库有循环依赖）
   - 确保使用了 `--start-group` / `--end-group`

#### C. 运行时验证

```bash
# 运行基础测试
./build/myimg-cli \
    --model /path/to/model.gguf \
    --prompt "test" \
    --steps 5 \
    -o /tmp/test_output.png

# 验证 HiRes Fix
./build/myimg-cli \
    --model /path/to/model.gguf \
    --prompt "test" \
    --hires --hires-width 1024 --hires-height 1024 \
    -o /tmp/test_hires.png
```

### 5.3 常见更新问题

#### 问题 1：sd.cpp 新增参数

**症状**：编译提示 `sd_img_gen_params_t` 缺少字段

**解决**：
```cpp
// 在 sdcpp_adapter.cpp 的 generate() 函数中补充映射
// 示例：假设 sd.cpp 新增了 cache 参数
gen_params.cache.mode = SD_CACHE_DISABLED;  // 或映射 my-img 参数
```

#### 问题 2：sd.cpp 删除/重命名 API

**症状**：编译提示函数未定义

**解决**：
```cpp
// 在适配器中使用条件编译
#if SD_VERSION >= 2000  // 假设版本判断
    new_api_call();
#else
    old_api_call();
#endif
```

#### 问题 3：GGML 子模块更新导致链接错误

**症状**：`undefined reference to ggml_*`

**解决**：
```bash
# 强制重新初始化 GGML 子模块
cd third_party/stable-diffusion.cpp
git submodule update --force --init --recursive
```

#### 问题 4：CUDA 后端符号丢失

**症状**：运行时 CUDA 不可用，但编译通过

**解决**：
```cmake
# 确保 CMakeLists.txt 中使用 --whole-archive
list(APPEND SDCPP_LINK_LIBRARIES 
    "-Wl,--whole-archive"
    ${SDCPP_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a
    "-Wl,--no-whole-archive"
)
```

### 5.4 版本锁定策略

对于生产环境，建议锁定 sd.cpp 版本：

```bash
# 记录当前使用的 commit
cd third_party/stable-diffusion.cpp
git log --oneline -1 > ../../SD_VERSION.lock

# 在 README/文档中注明兼容版本
echo "Compatible sd.cpp commit: $(cat SD_VERSION.lock)"
```

如果更新后发现问题，可快速回退：

```bash
cd third_party/stable-diffusion.cpp
git checkout <locked_commit>
git submodule update --init --recursive
# 重新编译...
```

---

## 6. 调试技巧

### 6.1 查看 sd.cpp 日志

```cpp
// 在适配器初始化时设置日志回调
static void sd_log_callback(enum sd_log_level_t level, const char* text, void* data) {
    std::cerr << "[SD " << level << "] " << text;
}

// 注册回调
sd_set_log_callback(sd_log_callback, nullptr);
```

### 6.2 验证库链接

```bash
# 检查可执行文件链接了哪些库
ldd build/myimg-cli

# 检查特定符号是否存在
nm build/myimg-cli | grep generate_image

# 检查 CUDA 符号
nm build/myimg-cli | grep ggml_cuda
```

### 6.3 构建日志分析

```bash
# 保存完整构建日志
make -j$(nproc) 2>&1 | tee build.log

# 检查链接命令
grep "Linking CXX executable" build.log
```

---

## 7. 最佳实践

### 7.1 开发流程

1. **修改前备份**：更新 sd.cpp 前，备份当前可正常编译的版本
2. **小步更新**：不要一次性跨越太多 commit，建议每次更新后测试
3. **CI 验证**：在 CI 中同时编译 sd.cpp 和 my-img，确保兼容性

### 7.2 代码规范

- **不要**在适配器以外的地方包含 `<stable-diffusion.h>`
- **不要**直接使用 `sd_image_t` 等 C 结构，使用 `Image` 封装
- **务必**处理所有 sd.cpp 返回的错误码
- **建议**使用 `std::unique_ptr` 管理 sd.cpp 分配的资源

### 7.3 性能优化

- 使用 `enable_mmap` 减少模型加载内存
- 使用 `offload_params_to_cpu` 在 GPU 显存不足时卸载到 CPU
- 使用 `flash_attn` 加速注意力计算

---

## 8. 参考资源

- [stable-diffusion.cpp README](https://github.com/leejet/stable-diffusion.cpp/blob/master/README.md)
- [stable-diffusion.cpp API 文档](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/api.md)
- [GGML 构建指南](https://github.com/ggerganov/ggml/blob/master/docs/build.md)
- [CMake 链接顺序最佳实践](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)

---

**最后更新**: 2025-04-29
**维护者**: my-img Team
