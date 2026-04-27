# my-img 开发规范 (AGENTS.md)

> **本文档是 AI 辅助开发的核心指令集**。所有参与 my-img 开发的 AI Agent 和人类开发者必须遵循本规范。

---

## 1. 项目哲学

### 1.1 核心目标
- **零 Python**：不嵌入 Python 解释器，零 Python 依赖
- **性能优先**：C++ 的性能，Python 的便捷（最终目标）
- **功能对等**：ComfyUI 能做的，my-img 最终都要能做
- **稳定可靠**：每个功能必须有单元测试覆盖

### 1.2 质量准则
1. **先测试，后提交**：没有测试的代码不得提交
2. **先设计，后编码**：复杂功能先写设计文档
3. **先 Review，后合并**：所有代码必须经过 Review
4. **先文档，后发布**：功能未完成文档不得标记为完成

---

## 2. 开发流程

### 2.1 功能开发 checklist

每个功能必须完成以下步骤才能标记为"完成"：

```
□ 1. 需求分析
    - 明确功能目标
    - 确定 CLI 参数接口
    - 参考 ComfyUI 对应节点行为

□ 2. 设计文档
    - 更新 task.md，标记任务为"进行中"
    - 如需复杂设计，在 design.md 中添加详细说明

□ 3. 代码实现
    - 遵循 C++ 代码规范（见第 3 节）
    - 在 adapters/ 中添加 sd.cpp 封装（如需要）
    - 在 nodes/ 中添加节点实现（如需要）

□ 4. 单元测试
    - 编写 test_*.cpp（见第 4 节）
    - 测试通过率 100%
    - 边界条件覆盖

□ 5. 集成测试
    - 端到端功能验证
    - 与现有功能兼容性测试

□ 6. 性能测试（如适用）
    - VRAM 使用监控
    - 生成速度基准
    - 内存泄漏检测（valgrind）

□ 7. 文档更新
    - README.md：添加使用示例
    - task.md：标记任务为"完成"
    - 代码注释：Doxygen 风格

□ 8. 代码审查
    - 自我审查 checklist
    - 提交 PR 描述功能

□ 9. 合并发布
    - CI 通过
    - Reviewer 批准
```

### 2.2 禁止清单

**绝对禁止**：
- ❌ 提交未编译通过的代码
- ❌ 提交没有单元测试的新功能
- ❌ 提交包含密码/密钥/Token 的代码
- ❌ 提交二进制文件（>1MB）到 git
- ❌ 提交临时文件（*.tmp, *.log）
- ❌ 使用 `using namespace std;` 在头文件中
- ❌ 在 C++ 中使用裸指针（原始指针）管理资源
- ❌ 忽略编译器警告（-Werror 级别）

**尽量避免**：
- ⚠️ 全局变量（除非必要且线程安全）
- ⚠️ 宏定义（优先用 const/constexpr/enum）
- ⚠️ 递归（深度不确定时）
- ⚠️ 动态内存分配（优先用栈和 RAII）

---

## 3. C++ 代码规范

### 3.1 命名规范

```cpp
// 文件命名：小写 + 下划线
sdcpp_adapter.h          // ✅
SDCPPAdapter.h           // ❌
sdcpp-adapter.h          // ❌

// 类名：大驼峰
class SDCPPAdapter {};   // ✅
class sdcpp_adapter {};  // ❌

// 函数名：小驼峰
void generateImage();    // ✅
void GenerateImage();    // ❌
void generate_image();   // ❌（C 风格）

// 变量名：小写 + 下划线
int batch_count;         // ✅
int batchCount;          // ❌
int BatchCount;          // ❌

// 成员变量：尾部下划线（类内）
class Foo {
    int private_member_;  // ✅
    int privateMember;    // ❌
};

// 常量：k + 大驼峰
constexpr int kMaxBatchSize = 10;  // ✅
constexpr int MAX_BATCH_SIZE = 10; // ❌（宏风格）

// 枚举：EnumName + 大驼峰
enum class SampleMethod {
    Euler,              // ✅
    EulerAncestral,     // ✅
    EULER,              // ❌
};

// 宏：全大写 + 下划线（尽量少用）
#define MY_MACRO(x) ((x) * 2)  // ✅
```

### 3.2 代码格式

使用 `.clang-format`（项目根目录已提供）：

```bash
# 格式化当前目录所有代码
find src tests -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# 或在 git commit 前自动格式化
git diff --cached --name-only | grep -E '\.(cpp|h)$' | xargs clang-format -i
```

**关键规则**：
- 缩进：4 空格（不使用 Tab）
- 行宽：120 字符
- 大括号：K&R 风格（函数换行，控制语句不换行）
- 空格：运算符前后加空格

```cpp
// ✅ 正确
if (condition) {
    doSomething();
}

// ❌ 错误
if(condition)
{
    doSomething();
}
```

### 3.3 头文件规范

```cpp
// ✅ 正确的头文件结构
#pragma once  // 或 #ifndef 宏保护

// 1. C++ 标准库
#include <vector>
#include <string>

// 2. 第三方库
#include <torch/torch.h>
#include <nlohmann/json.hpp>

// 3. 本项目头文件
#include "adapters/sdcpp_adapter.h"
#include "utils/image_utils.h"

namespace myimg {

// 类声明
class MyClass {
public:
    // 构造函数
    explicit MyClass(int param);  // explicit 防止隐式转换
    
    // 析构函数（默认即可）
    ~MyClass() = default;
    
    // 禁用拷贝
    MyClass(const MyClass&) = delete;
    MyClass& operator=(const MyClass&) = delete;
    
    // 允许移动
    MyClass(MyClass&&) noexcept = default;
    MyClass& operator=(MyClass&&) noexcept = default;
    
    // 公共接口
    bool initialize();
    void process();
    
private:
    // 私有成员
    int param_;
    std::unique_ptr<Impl> impl_;  // PIMPL 模式
};

} // namespace myimg
```

### 3.4 资源管理（RAII）

```cpp
// ✅ 正确：RAII
class ModelLoader {
    std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> ctx_;
public:
    ModelLoader() : ctx_(nullptr, free_sd_ctx) {}
    
    bool load(const std::string& path) {
        ctx_.reset(new_sd_ctx(params));
        return ctx_ != nullptr;
    }
};

// ❌ 错误：裸指针
class BadLoader {
    sd_ctx_t* ctx_;  // 危险！可能内存泄漏
public:
    ~BadLoader() {
        if (ctx_) free_sd_ctx(ctx_);  // 容易忘记
    }
};
```

### 3.5 错误处理

```cpp
// ✅ 正确：使用 std::optional / std::expected (C++23)
std::optional<Image> loadImage(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        LOG_ERROR("Image not found: %s", path.c_str());
        return std::nullopt;
    }
    // ...
    return image;
}

// 调用方
auto image = loadImage("photo.png");
if (!image) {
    return false;  // 优雅处理错误
}

// ❌ 错误：返回裸指针 + nullptr
Image* loadImageBad(const std::string& path) {
    if (...) return nullptr;  // 调用方容易忘记检查
}
```

### 3.6 日志规范

使用项目统一的日志系统：

```cpp
#include "utils/log.h"

// 级别：TRACE < DEBUG < INFO < WARN < ERROR < FATAL
LOG_TRACE("Entering function: %s", __func__);
LOG_DEBUG("Processing image: %dx%d", width, height);
LOG_INFO("Model loaded in %.2f seconds", duration);
LOG_WARN("VAE tiling enabled due to low VRAM");
LOG_ERROR("Failed to allocate CUDA memory: %s", cudaGetErrorString(err));
LOG_FATAL("Critical error, aborting...");  // 会自动 abort
```

**规则**：
- 正常流程用 `INFO`
- 调试信息用 `DEBUG`（编译 Release 时会被优化掉）
- 可恢复问题用 `WARN`
- 错误用 `ERROR`
- 致命错误用 `FATAL`

---

## 4. 单元测试规范

### 4.1 测试框架

使用 Catch2（已集成到项目）：

```cpp
// tests/test_feature.cpp
#include <catch2/catch_test_macros.hpp>
#include "adapters/sdcpp_adapter.h"

using namespace myimg;

TEST_CASE("FeatureName: Scenario", "[tag]") {
    // Arrange
    SDCPPAdapter adapter;
    GenerationParams params;
    params.diffusion_model_path = "test_model.gguf";
    
    // Act
    bool result = adapter.initialize(params);
    
    // Assert
    REQUIRE(result == true);
    REQUIRE(adapter.is_initialized() == true);
}
```

### 4.2 测试命名规范

```cpp
// 文件命名
test_<feature_name>.cpp       // ✅
test_sdcpp_adapter.cpp       // ✅
test_001.cpp                 // ❌ 无意义编号

// 测试用例命名
TEST_CASE("SDCPPAdapter: Initialize with valid model", "[adapter][init]")  // ✅
TEST_CASE("Test 1", "[test]")                                               // ❌
TEST_CASE("adapter_init", "[test]")                                         // ❌

// 测试标签
// [unit]       - 单元测试
// [integration]- 集成测试
// [performance]- 性能测试
// [slow]       - 慢测试（需要模型文件）
// [gpu]        - 需要 GPU
```

### 4.3 测试覆盖率要求

**最低标准**：
- 行覆盖率：≥ 80%
- 分支覆盖率：≥ 70%
- 函数覆盖率：100%（公共接口必须全部测试）

**测试类型**：

1. **单元测试**（必须）
   - 测试单个函数/类
   - Mock 外部依赖
   - 快速执行（< 1秒）

2. **集成测试**（必须）
   - 测试模块间交互
   - 使用真实依赖（如 sd.cpp）
   - 中等速度（< 30秒）

3. **端到端测试**（关键功能）
   - 完整生成流程
   - 需要真实模型文件
   - 标记为 `[slow]`

4. **性能测试**（如适用）
   - 基准测试，防止性能退化
   - VRAM 使用监控
   - 标记为 `[performance]`

### 4.4 测试数据管理

```cpp
// ✅ 正确：使用测试固件（Fixture）
class ModelTestFixture {
protected:
    void SetUp() {
        // 加载测试模型（小尺寸）
        params_.diffusion_model_path = 
            "test_models/z_image_turbo-Q2_K.gguf";  // 最小模型
    }
    
    void TearDown() {
        // 清理资源
    }
    
    GenerationParams params_;
};

TEST_CASE_METHOD(ModelTestFixture, "Generation: Basic txt2img", "[slow][gpu]") {
    // 使用 fixture 中的 params_
    auto image = adapter.generate_single(params_);
    REQUIRE(!image.empty());
}
```

**测试模型存放**：
```
tests/
├── test_data/
│   ├── models/          # 小型测试模型（< 500MB）
│   ├── images/          # 测试图片
│   └── configs/         # 测试配置
```

> **注意**：测试模型用 `.gitignore` 排除，通过脚本下载。

### 4.5 测试运行

```bash
# 运行所有测试
cd build && ctest --output-on-failure

# 运行快速测试（排除 [slow]）
./test_runner "[unit],[integration]"

# 运行特定标签
./test_runner "[adapter]"
./test_runner "[slow]"

# 生成覆盖率报告
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make -j$(nproc)
ctest
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

---

## 5. Git 工作流

### 5.1 分支策略

```
main        - 稳定分支，随时可发布
  ↑
develop     - 开发分支，功能集成
  ↑
feature/*   - 功能分支
bugfix/*    - 修复分支
hotfix/*    - 紧急修复（从 main 切出）
```

### 5.2 提交规范

使用 **Conventional Commits**：

```bash
# 格式：<type>(<scope>): <description>
# type: feat|fix|docs|test|refactor|perf|chore

feat(adapter): add LoRA weight injection support     # ✅ 新功能
fix(vae): resolve tiling overlap artifact             # ✅ 修复
test(hires): add 2560x1440 generation test            # ✅ 测试
docs(readme): update API usage examples               # ✅ 文档
refactor(nodes): simplify sampler node interface      # ✅ 重构
perf(sampling): optimize Euler scheduler by 20%       # ✅ 性能
chore(build): update CMake minimum version to 3.20    # ✅ 构建

# ❌ 错误示例
update code                                             # 无类型
fix bug                                                 # 太笼统
WIP                                                     # 未完成
```

### 5.3 提交前检查清单

```bash
# 提交前必须执行：
./scripts/pre-commit.sh
```

脚本内容：
```bash
#!/bin/bash
# scripts/pre-commit.sh

echo "=== Pre-commit Checks ==="

# 1. 代码格式化
echo "1. Formatting code..."
find src tests -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# 2. 编译
echo "2. Building..."
cd build && make -j$(nproc) || exit 1

# 3. 运行快速测试
echo "3. Running tests..."
ctest --output-on-failure -L "unit|integration" || exit 1

# 4. 静态分析
echo "4. Static analysis..."
cppcheck --enable=all --error-exitcode=1 src/ || exit 1

# 5. 检查大文件
echo "5. Checking large files..."
find . -type f -size +1M | grep -v ".git/" | grep -v "build/" | grep -v "test_data/"

echo "=== All checks passed ==="
```

---

## 6. 文档规范

### 6.1 代码注释（Doxygen）

```cpp
/**
 * @brief 生成单张图像
 * 
 * 根据提供的参数执行完整的 txt2img 或 img2img 生成流程。
 * 自动处理模型加载、文本编码、采样和 VAE 解码。
 * 
 * @param params 生成参数，包含提示词、尺寸、种子等
 * @return 生成的图像，失败时返回空 Image
 * 
 * @note 线程安全：此函数非线程安全，每个线程应使用独立 adapter 实例
 * @warning 需要至少 8GB VRAM，否则可能触发 OOM
 * 
 * @code
 * GenerationParams params;
 * params.prompt = "a cat";
 * auto image = adapter.generate_single(params);
 * @endcode
 * 
 * @see GenerationParams
 */
Image generate_single(const GenerationParams& params);
```

### 6.2 README 更新

新增功能后必须更新：

```markdown
## 功能列表

- ✅ txt2img 文本生成
- ✅ HiRes Fix 高分辨率修复
- 🆕 LoRA 微调支持（新增）
- ⏳ ControlNet 控制网络（开发中）

## CLI 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--diffusion-model` | 扩散模型路径 | `--diffusion-model model.gguf` |
| `--lora` | LoRA 权重（新增） | `--lora style.safetensors:0.8` |
```

### 6.3 CHANGELOG 维护

```markdown
## [0.2.0] - 2026-04-28

### Added
- LoRA 权重加载和注入支持
- img2img 图像到图像生成
- 区域提示词（Regional Prompting）

### Changed
- 优化 VAE tiling 内存占用
- 提升 HiRes Fix 生成速度 15%

### Fixed
- 修复 2560x1440 分辨率 OOM 问题
```

---

## 7. 性能规范

### 7.1 VRAM 使用限制

| GPU 显存 | 最大分辨率 | 推荐设置 |
|----------|-----------|----------|
| 8 GB | 1920x1080 | 启用 VAE tiling |
| 10 GB | 2560x1440 | 启用 VAE tiling + Flash Attention |
| 12 GB | 2560x1440 | 启用 Flash Attention |
| 16 GB | 3840x2160 | 标准模式 |
| 24 GB | 3840x2160 | 标准模式，可同时加载多个 LoRA |

### 7.2 性能基准

```cpp
// tests/benchmark_generation.cpp
TEST_CASE("Performance: 1280x720 txt2img", "[performance]") {
    auto start = std::chrono::high_resolution_clock::now();
    auto image = adapter.generate_single(params_1280x720);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    // 基准：RTX 3080 10GB 应在 120 秒内完成
    REQUIRE(duration < 120);
    
    // VRAM 峰值应 < 10GB
    auto vram_peak = get_peak_vram_usage();
    REQUIRE(vram_peak < 10.0f * 1024 * 1024 * 1024);  // 10GB in bytes
}
```

### 7.3 内存泄漏检测

```bash
# 使用 valgrind 检测内存泄漏
valgrind --leak-check=full --show-leak-kinds=all \
    ./myimg-cli --diffusion-model model.gguf --prompt "test" -o test.png

# 预期输出：
# definitely lost: 0 bytes in 0 blocks
# indirectly lost: 0 bytes in 0 blocks
```

---

## 8. 安全规范

### 8.1 输入验证

```cpp
// ✅ 正确：严格验证输入
bool validate_image_path(const std::string& path) {
    // 1. 检查路径遍历攻击
    if (path.find("..") != std::string::npos) {
        LOG_ERROR("Invalid path: %s", path.c_str());
        return false;
    }
    
    // 2. 检查文件扩展名
    if (!ends_with(path, ".png") && !ends_with(path, ".jpg")) {
        LOG_ERROR("Unsupported format: %s", path.c_str());
        return false;
    }
    
    // 3. 检查文件大小（防止 DoS）
    auto size = std::filesystem::file_size(path);
    if (size > 100 * 1024 * 1024) {  // 100MB 上限
        LOG_ERROR("File too large: %s", path.c_str());
        return false;
    }
    
    return true;
}
```

### 8.2 模型文件安全

- 所有模型文件必须校验哈希（SHA256）
- 不支持加载来自网络的未验证模型
- 模型加载失败时优雅降级，不崩溃

---

## 9. 调试规范

### 9.1 核心转储

```bash
# 启用核心转储
ulimit -c unlimited

# 崩溃后分析
gdb ./myimg-cli core
(gdb) bt full          # 查看完整堆栈
(gdb) info locals      # 查看局部变量
```

### 9.2 调试日志

```cpp
// 使用分级日志控制输出量
LOG_DEBUG("详细调试信息（仅在 Debug 模式显示）");
LOG_INFO("正常流程信息");
LOG_ERROR("错误信息（总是显示）");

// 运行时调整日志级别
./myimg-cli --log-level debug  # 显示 DEBUG 及以上
./myimg-cli --log-level error  # 只显示 ERROR
```

---

## 10. 附录

### 10.1 推荐阅读

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [Effective Modern C++](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/)

### 10.2 工具链

| 工具 | 用途 | 版本 |
|------|------|------|
| GCC | 编译器 | ≥ 11 |
| CMake | 构建系统 | ≥ 3.18 |
| clang-format | 代码格式化 | ≥ 14 |
| clang-tidy | 静态分析 | ≥ 14 |
| cppcheck | 静态分析 | ≥ 2.9 |
| valgrind | 内存检测 | ≥ 3.18 |
| Catch2 | 测试框架 | v3.x |
| lcov | 覆盖率 | ≥ 1.14 |

### 10.3 CI/CD 检查清单

```yaml
# .github/workflows/ci.yml
jobs:
  build-and-test:
    steps:
      - name: Checkout
        uses: actions/checkout@v3
      
      - name: Format Check
        run: |
          find src tests -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run --Werror
      
      - name: Build
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build -j$(nproc)
      
      - name: Test
        run: |
          cd build && ctest --output-on-failure
      
      - name: Coverage
        run: |
          cmake -B build -DENABLE_COVERAGE=ON
          cmake --build build
          cd build && ctest
          lcov --capture --directory . --output-file coverage.info
          lcov --remove coverage.info '/usr/*' --output-file coverage.info
          # 要求覆盖率 ≥ 80%
          lcov --summary coverage.info | grep "lines" | awk '{if($2 < 80) exit 1}'
```

---

**最后更新**: 2026-04-27
**维护者**: my-img Team
**许可证**: MIT
