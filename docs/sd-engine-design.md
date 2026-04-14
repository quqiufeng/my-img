# sd-engine: C++ 版 ComfyUI 架构设计文档

## 1. 项目愿景

### 1.1 核心目标

创建一个**纯 C++ 实现的 ComfyUI 工作流执行引擎**，能够：
- 解析并执行 ComfyUI 导出的 JSON 工作流
- 零 Python 依赖，单二进制部署
- 性能优于 Python 版本（无 GIL、内存管理更优）
- 支持 ComfyUI 核心节点生态

### 1.2 与 ComfyUI 的关系

| 维度 | ComfyUI (Python) | sd-engine (C++) |
|------|------------------|-----------------|
| 前端 | 浏览器节点编辑器 | 命令行 / JSON |
| 后端执行 | Python 解释器 | C++ 原生引擎 |
| 节点生态 | 数千个 Py 节点 | 核心节点 C++ 复刻 |
| 部署 | Python + 依赖地狱 | 单二进制 |
| 性能 | 受 GIL 限制 | 无锁并行 |
| 适用场景 | 交互式创作 | 批量生产 / 服务化 |

### 1.3 为什么这个项目有前途

1. **部署优势**：AI 图像生成服务化部署时，Python 依赖是噩梦
2. **性能优势**：批量处理时 C++ 比 Python 快 2-5 倍
3. **生态互补**：ComfyUI 负责创作，sd-engine 负责生产
4. **技术趋势**：越来越多的 AI 工具转向 C++/Rust（llama.cpp, stable-diffusion.cpp 等）

---

## 2. 核心架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户接口层                               │
├─────────────────────────────────────────────────────────────┤
│  CLI 工具          │  JSON 执行器        │  C++ API        │
│  sd-workflow       │  sd-batch           │  libsdengine    │
├─────────────────────────────────────────────────────────────┤
│                     工作流引擎层                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Workflow    │  │ DAG         │  │ ExecutionCache      │ │
│  │ Parser      │  │ Executor    │  │ (节点结果缓存)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     节点实现层                               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Loaders  │ │ Sampling │ │ Image    │ │ Conditioning │  │
│  │ 加载器   │ │ 采样     │ │ 图像处理 │ │ 条件编码     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Latent   │ │ Control  │ │ LoRA     │ │ Video        │  │
│  │ Latent操作│ │ ControlNet│ │ LoRA    │ │ 视频生成     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     底层依赖层                               │
├─────────────────────────────────────────────────────────────┤
│  stable-diffusion.cpp  │  GGML  │  stb_image  │  OpenCV    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件详解

#### 2.2.1 Workflow（工作流）

```cpp
class Workflow {
public:
    // 从 ComfyUI JSON 加载
    bool load_from_json(const std::string& path);
    bool load_from_json_string(const std::string& json);
    
    // 获取节点
    Node* get_node(const std::string& id);
    std::vector<Node*> get_all_nodes();
    
    // 拓扑排序（DAG 执行顺序）
    std::vector<std::string> topological_sort();
    
    // 执行
    bool execute(ExecutionContext& ctx);
    
private:
    std::map<std::string, std::unique_ptr<Node>> nodes_;
    std::map<std::string, std::vector<Link>> links_;  // 节点连接关系
};
```

#### 2.2.2 Node（节点基类）

```cpp
class Node {
public:
    virtual ~Node() = default;
    
    // 节点元信息
    virtual std::string get_class_type() const = 0;
    virtual std::string get_category() const = 0;
    
    // 输入输出定义
    virtual std::vector<InputDef> get_inputs() const = 0;
    virtual std::vector<OutputDef> get_outputs() const = 0;
    
    // 执行
    virtual bool execute(const NodeInputs& inputs, NodeOutputs& outputs) = 0;
    
    // 序列化（用于缓存）
    virtual std::string compute_hash(const NodeInputs& inputs);
};

// 节点注册宏
#define REGISTER_NODE(class_type, node_class) \
    static bool _registered_##node_class = []() { \
        NodeRegistry::instance().register_node(class_type, []() { \
            return std::make_unique<node_class>(); \
        }); \
        return true; \
    }();
```

#### 2.2.3 ExecutionCache（执行缓存）

ComfyUI 的核心优化：只重新执行变化的节点。

```cpp
class ExecutionCache {
public:
    // 检查是否有缓存
    bool has(const std::string& node_id, const std::string& hash);
    
    // 获取缓存结果
    NodeOutputs get(const std::string& node_id, const std::string& hash);
    
    // 存储结果
    void put(const std::string& node_id, const std::string& hash, 
             const NodeOutputs& outputs);
    
    // 清理过期缓存
    void gc();
    
private:
    struct CacheEntry {
        std::string hash;
        NodeOutputs outputs;
        std::chrono::time_point<std::chrono::steady_clock> last_access;
    };
    
    std::unordered_map<std::string, CacheEntry> cache_;
    size_t max_size_ = 1024 * 1024 * 1024;  // 1GB 上限
};
```

#### 2.2.4 DAG Executor（执行器）

```cpp
class DAGExecutor {
public:
    explicit DAGExecutor(ExecutionCache* cache = nullptr);
    
    // 执行单个节点
    bool execute_node(Node* node, const NodeInputs& inputs, NodeOutputs& outputs);
    
    // 执行完整工作流
    bool execute_workflow(Workflow* workflow, const ExecutionConfig& config);
    
    // 设置进度回调
    void set_progress_callback(ProgressCallback cb);
    
private:
    ExecutionCache* cache_;
    ProgressCallback progress_cb_;
    
    // 准备节点输入（从上游节点获取）
    bool prepare_inputs(Node* node, Workflow* workflow, NodeInputs& inputs);
};
```

---

## 3. 数据类型系统

### 3.1 ComfyUI 类型到 C++ 的映射

| ComfyUI 类型 | C++ 类型 | 说明 |
|-------------|---------|------|
| MODEL | `sd_ctx_t*` | 扩散模型上下文 |
| CLIP | `clip_ctx_t*` | CLIP 编码器上下文 |
| VAE | `vae_ctx_t*` | VAE 编解码器上下文 |
| CONDITIONING | `std::vector<SDCondition>` | 条件向量 |
| LATENT | `sd::Tensor<float>` | Latent 张量 [W,H,C,1] |
| IMAGE | `sd_image_t` | 图像数据 |
| MASK | `sd::Tensor<float>` | 遮罩张量 |
| STRING | `std::string` | 字符串 |
| INT | `int64_t` | 整数 |
| FLOAT | `float` | 浮点数 |
| BOOLEAN | `bool` | 布尔值 |

### 3.2 NodeValue（节点值包装）

```cpp
class NodeValue {
public:
    enum Type {
        MODEL, CLIP, VAE, CONDITIONING,
        LATENT, IMAGE, MASK,
        STRING, INT, FLOAT, BOOLEAN
    };
    
    Type type;
    std::any data;
    
    // 类型安全的访问
    template<typename T>
    T& as() { return std::any_cast<T&>(data); }
    
    template<typename T>
    const T& as() const { return std::any_cast<const T&>(data); }
    
    // 序列化（用于缓存）
    std::string serialize() const;
    static NodeValue deserialize(const std::string& str);
};
```

---

## 4. 核心节点实现

### 4.1 加载器节点

#### CheckpointLoaderSimple
```cpp
class CheckpointLoaderSimpleNode : public Node {
public:
    std::string get_class_type() const override { 
        return "CheckpointLoaderSimple"; 
    }
    
    std::vector<InputDef> get_inputs() const override {
        return {
            {"ckpt_name", "STRING", true, ""}  // 必需，模型路径
        };
    }
    
    std::vector<OutputDef> get_outputs() const override {
        return {
            {"MODEL", "MODEL"},
            {"CLIP", "CLIP"},
            {"VAE", "VAE"}
        };
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string ckpt_path = inputs.get<std::string>("ckpt_name");
        
        // 加载模型
        auto model = load_checkpoint(ckpt_path);
        
        outputs.set("MODEL", model.model);
        outputs.set("CLIP", model.clip);
        outputs.set("VAE", model.vae);
        
        return true;
    }
};

REGISTER_NODE("CheckpointLoaderSimple", CheckpointLoaderSimpleNode);
```

### 4.2 条件编码节点

#### CLIPTextEncode
```cpp
class CLIPTextEncodeNode : public Node {
public:
    std::string get_class_type() const override { 
        return "CLIPTextEncode"; 
    }
    
    std::vector<InputDef> get_inputs() const override {
        return {
            {"text", "STRING", true, ""},
            {"clip", "CLIP", true, nullptr}
        };
    }
    
    std::vector<OutputDef> get_outputs() const override {
        return {{"CONDITIONING", "CONDITIONING"}};
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string text = inputs.get<std::string>("text");
        clip_ctx_t* clip = inputs.get<clip_ctx_t*>("clip");
        
        // 编码文本
        auto cond = clip_encode_text(clip, text);
        
        outputs.set("CONDITIONING", cond);
        return true;
    }
};
```

### 4.3 Latent 节点

#### EmptyLatentImage
```cpp
class EmptyLatentImageNode : public Node {
public:
    std::string get_class_type() const override { 
        return "EmptyLatentImage"; 
    }
    
    std::vector<InputDef> get_inputs() const override {
        return {
            {"width", "INT", false, 512},
            {"height", "INT", false, 512},
            {"batch_size", "INT", false, 1}
        };
    }
    
    std::vector<OutputDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        int width = inputs.get<int>("width");
        int height = inputs.get<int>("height");
        int batch = inputs.get<int>("batch_size");
        
        // 创建空 latent
        auto latent = create_empty_latent(width, height, batch);
        
        outputs.set("LATENT", latent);
        return true;
    }
};
```

### 4.4 采样节点

#### KSampler
```cpp
class KSamplerNode : public Node {
public:
    std::string get_class_type() const override { return "KSampler"; }
    
    std::vector<InputDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
            {"seed", "INT", false, 0},
            {"steps", "INT", false, 20},
            {"cfg", "FLOAT", false, 8.0},
            {"sampler_name", "STRING", false, "euler"},
            {"scheduler", "STRING", false, "normal"},
            {"positive", "CONDITIONING", true, {}},
            {"negative", "CONDITIONING", true, {}},
            {"latent_image", "LATENT", true, {}}
        };
    }
    
    std::vector<OutputDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }
    
    bool execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        // 提取参数
        sd_ctx_t* model = inputs.get<sd_ctx_t*>("model");
        int seed = inputs.get<int>("seed");
        int steps = inputs.get<int>("steps");
        float cfg = inputs.get<float>("cfg");
        std::string sampler = inputs.get<std::string>("sampler_name");
        std::string scheduler = inputs.get<std::string>("scheduler");
        auto positive = inputs.get<std::vector<SDCondition>>("positive");
        auto negative = inputs.get<std::vector<SDCondition>>("negative");
        auto latent = inputs.get<sd::Tensor<float>>("latent_image");
        
        // 执行采样
        auto result = ksampler(
            model, seed, steps, cfg,
            sampler, scheduler,
            positive, negative, latent
        );
        
        outputs.set("LATENT", result);
        return true;
    }
};
```

---

## 5. 命令行工具设计

### 5.1 sd-workflow：执行单个工作流

```bash
# 执行 ComfyUI 导出的工作流
sd-workflow --workflow workflow.json --output output.png

# 覆盖工作流中的参数
sd-workflow --workflow workflow.json \
    --set "prompt=masterpiece, best quality" \
    --set "seed=42" \
    --set "steps=30"

# 批量执行（目录中的所有工作流）
sd-workflow --batch ./workflows/ --output-dir ./outputs/

# 显示执行计划（不实际执行）
sd-workflow --workflow workflow.json --dry-run

# 显示详细进度
sd-workflow --workflow workflow.json --verbose
```

### 5.2 sd-server：HTTP API 服务

```bash
# 启动服务
sd-server --port 8188 --models-dir /opt/models/

# API 端点
POST /execute          # 执行工作流 JSON
POST /upload/image     # 上传图片
GET  /models           # 列出可用模型
GET  /nodes            # 列出支持节点
```

### 5.3 sd-batch：批量处理

```bash
# 用同一个工作流处理多张图片
sd-batch --workflow upscale.json \
    --input ./input_images/ \
    --output ./output_images/
```

---

## 6. 缓存与优化

### 6.1 节点级缓存

```cpp
// 计算节点哈希（输入 + 节点类型）
std::string Node::compute_hash(const NodeInputs& inputs) {
    std::stringstream ss;
    ss << get_class_type() << "|";
    
    for (const auto& [name, value] : inputs) {
        ss << name << "=" << value.serialize() << "|";
    }
    
    return sha256(ss.str());
}

// 执行时先检查缓存
bool DAGExecutor::execute_node(Node* node, const NodeInputs& inputs, 
                               NodeOutputs& outputs) {
    std::string hash = node->compute_hash(inputs);
    std::string node_id = node->get_id();
    
    if (cache_ && cache_->has(node_id, hash)) {
        outputs = cache_->get(node_id, hash);
        return true;
    }
    
    // 实际执行
    bool success = node->execute(inputs, outputs);
    
    if (success && cache_) {
        cache_->put(node_id, hash, outputs);
    }
    
    return success;
}
```

### 6.2 内存优化

- **延迟加载**：模型只在需要时加载
- **引用计数**：共享数据（如 conditioning）用引用计数管理
- **显存池**：重用 latent 缓冲区

---

## 7. 开发路线图

### Phase 1: 核心引擎（MVP）
- [ ] Workflow JSON 解析器
- [ ] DAG 拓扑排序执行器
- [ ] 基础节点框架
- [ ] 结果缓存系统
- [ ] sd-workflow CLI 工具

**目标**：能跑通最简单的 txt2img 工作流

### Phase 2: 核心节点
- [ ] CheckpointLoaderSimple
- [ ] CLIPTextEncode
- [ ] EmptyLatentImage
- [ ] KSampler
- [ ] VAEDecode
- [ ] SaveImage
- [ ] LoadImage (img2img)
- [ ] VAEEncode

**目标**：支持 txt2img 和 img2img 完整链路

### Phase 3: 图像处理节点
- [ ] ImageScale
- [ ] ImageCrop
- [ ] ImageComposite
- [ ] UpscaleModelLoader
- [ ] ImageUpscaleWithModel

**目标**：支持高清修复流程

### Phase 4: ControlNet
- [ ] ControlNetLoader
- [ ] ControlNetApply
- [ ] CannyEdgePreprocessor
- [ ] MiDaS-DepthMapPreprocessor
- [ ] OpenPosePreprocessor

**目标**：支持 ControlNet 全流程

### Phase 5: 高级功能
- [ ] LoRA 支持
- [ ] IPAdapter 支持
- [ ] Inpaint 支持
- [ ] 视频生成（AnimateDiff）
- [ ] sd-server HTTP API

### Phase 6: 生态完善
- [ ] 自定义节点 SDK
- [ ] 工作流可视化工具
- [ ] 与 ComfyUI 的互操作（导入/导出）

---

## 8. 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| JSON 解析 | nlohmann/json | 现代 C++，易用 |
| 张量运算 | GGML (via sd.cpp) | 已集成，跨平台 |
| 图像处理 | stb_image + OpenCV | stb 轻量，OpenCV 功能全 |
| 并行计算 | std::thread + OpenMP | 标准 C++，无额外依赖 |
| 缓存存储 | 内存 + 可选 Redis | 本地用内存，分布式用 Redis |
| HTTP 服务 | cpp-httplib | 单头文件，轻量 |

---

## 9. 与 ComfyUI 的互操作

### 9.1 导入 ComfyUI 工作流

```cpp
// 处理 ComfyUI 的节点连接格式
// "clip": ["1", 1] 表示连接到节点 1 的第 2 个输出

class ComfyUIImporter {
public:
    Workflow import(const std::string& comfy_json);
    
private:
    // 节点类型映射（ComfyUI -> sd-engine）
    std::map<std::string, std::string> node_type_map_ = {
        {"CheckpointLoaderSimple", "CheckpointLoaderSimple"},
        {"CLIPTextEncode", "CLIPTextEncode"},
        {"KSampler", "KSampler"},
        // ...
    };
    
    // 不支持的节点列表
    std::set<std::string> unsupported_nodes_;
};
```

### 9.2 导出到 ComfyUI

```cpp
class ComfyUIExporter {
public:
    std::string export(const Workflow& workflow);
};
```

---

## 10. 总结

sd-engine 是一个**雄心勃勃但可行**的项目：

- **技术可行**：基于成熟的 stable-diffusion.cpp，C++ 生态已完善
- **需求明确**：ComfyUI 的 Python 依赖确实是生产部署的痛点
- **生态互补**：不与 ComfyUI 竞争，而是服务不同场景

**下一步**：开始实现 Phase 1 核心引擎？
