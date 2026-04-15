# sd-engine 节点开发教程

本教程介绍如何为 sd-engine 开发自定义节点。

## 目录

1. [节点基础](#节点基础)
2. [最小可运行示例](#最小可运行示例)
3. [端口类型约定](#端口类型约定)
4. [缓存与哈希](#缓存与哈希)
5. [错误处理](#错误处理)
6. [注册节点](#注册节点)
7. [最佳实践](#最佳实践)

---

## 节点基础

sd-engine 中的每个节点都必须继承自 `sdengine::Node` 基类，并实现以下纯虚方法：

- `get_class_type()` — 返回节点的类型名称（如 `"MyCustomNode"`）
- `get_inputs()` — 返回输入端口定义列表
- `get_outputs()` — 返回输出端口定义列表
- `execute()` — 执行节点逻辑

可选实现：

- `compute_hash()` — 自定义缓存哈希计算
- `get_category()` — 节点分类（默认 `"general"`）

---

## 最小可运行示例

下面是一个简单的 "整数相加" 节点：

```cpp
#include "core/node.h"
#include <any>

namespace sdengine {

class AddIntNode : public Node {
public:
    std::string get_class_type() const override {
        return "AddInt";
    }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"a", "INT", true},
            {"b", "INT", true}
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {
            {"result", "INT", true}
        };
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int a = std::any_cast<int>(inputs.at("a"));
        int b = std::any_cast<int>(inputs.at("b"));
        outputs["result"] = a + b;
        return sd_error_t::OK;
    }
};

REGISTER_NODE("AddInt", AddIntNode)

} // namespace sdengine
```

---

## 端口类型约定

sd-engine 使用字符串标识端口类型。常用类型约定如下：

| 类型名称 | C++ 对应类型 | 说明 |
|---------|-------------|------|
| `MODEL` | `sd_ctx_t*` | Stable Diffusion 模型上下文 |
| `CLIP` | `void*` / `sd_ctx_t*` | CLIP 编码器（通常从 MODEL 节点获取） |
| `VAE` | `void*` / `sd_ctx_t*` | VAE（通常从 MODEL 节点获取） |
| `CONDITIONING` | `ConditioningPtr` | 文本编码结果 |
| `LATENT` | `LatentPtr` | 潜在空间张量 |
| `IMAGE` | `ImagePtr` | 图像数据（sd_image_t） |
| `UPSCALER` | `UpscalerPtr` | 超分模型上下文 |
| `STRING` | `std::string` | 字符串 |
| `INT` | `int` | 整数 |
| `FLOAT` | `float` | 浮点数 |
| `BOOLEAN` | `bool` | 布尔值 |

---

## 缓存与哈希

`DAGExecutor` 默认启用 `ExecutionCache`。如果节点的输入没有变化，执行器会直接返回缓存结果，跳过 `execute()` 调用。

默认的 `compute_hash()` 实现会遍历所有输入值并生成字符串哈希。如果你的节点包含非确定性行为（如随机数生成），或者某些输入不应参与缓存计算，可以重写此方法：

```cpp
std::string MyNode::compute_hash(const NodeInputs& inputs) const {
    // 排除 seed 字段参与缓存
    std::string hash = id_;
    for (const auto& [k, v] : inputs) {
        if (k == "seed") continue;
        hash += k + std::any_cast<std::string>(v);
    }
    return hash;
}
```

---

## 错误处理

节点执行失败时应返回相应的 `sd_error_t`：

```cpp
sd_error_t MyNode::execute(const NodeInputs& inputs, NodeOutputs& outputs) {
    auto it = inputs.find("required_input");
    if (it == inputs.end()) {
        return sd_error_t::ERROR_MISSING_INPUT;
    }

    try {
        // 执行逻辑...
    } catch (const std::bad_alloc&) {
        return sd_error_t::ERROR_MEMORY_ALLOCATION;
    } catch (...) {
        return sd_error_t::ERROR_EXECUTION_FAILED;
    }

    return sd_error_t::OK;
}
```

---

## 注册节点

节点实现文件末尾必须使用 `REGISTER_NODE` 宏注册：

```cpp
REGISTER_NODE("ClassTypeName", CppClassName)
```

注册是自动的，不需要在 main 中手动调用。只要目标文件被链接到最终可执行文件中，节点就会在程序启动时自动注册到 `NodeRegistry`。

> **注意**：如果节点所在的 `.cpp` 文件只被编译进静态库（如 `sd-engine` 库），而没有任何可执行文件直接引用该文件中的符号，链接器可能会丢弃该目标文件，导致注册代码不执行。确保该文件被包含在库的源文件列表中（CMakeLists.txt 的 `add_library`）。

---

## 最佳实践

### 1. 使用智能指针管理 SD 资源

当创建或接收 `sd_image_t*`、`sd_latent_t*`、`sd_conditioning_t*` 时，优先使用 `sd_ptr.h` 中提供的智能指针：

```cpp
#include "core/sd_ptr.h"

// 从 C API 获取图像
sd_image_t* raw = sd_decode_latent(ctx, latent);
ImagePtr img = make_image_ptr(raw);  // 自动释放
```

### 2. 节点内部创建新 image 时使用对象池

```cpp
#include "core/sd_ptr.h"

sd_image_t* img = acquire_image();  // 从对象池获取
img->width = 512;
img->height = 512;
img->channel = 3;
img->data = malloc(512 * 512 * 3);

// 传递给下游时包装为智能指针
outputs["image"] = make_image_ptr(img);
```

### 3. 避免在 execute 中修改全局状态

节点应该是无副作用的（除了必要的模型状态修改，如 LoRA 应用）。这保证了缓存系统的正确性。

### 4. 为复杂输入提供默认值

```cpp
std::vector<PortDef> get_inputs() const override {
    return {
        {"steps", "INT", false, 20},
        {"cfg", "FLOAT", false, 7.0f}
    };
}
```

---

## 下一步

- 查看 `src/sd-engine/nodes/test_nodes.cpp` 中的简单示例
- 查看 `src/sd-engine/nodes/core_nodes.cpp` 中的复杂节点（如 `KSamplerNode`）
- 参考 `examples/workflows/` 中的 JSON 工作流格式
