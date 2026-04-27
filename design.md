# my-img：纯 C++ 版 ComfyUI 实现

> **📖 项目文档**：[README.md](README.md) - 项目介绍、快速开始、功能列表
>
> **目标**：用纯 C++ + libtorch 重写 ComfyUI，彻底摆脱 Python 依赖。
>
> **理念**：将 ComfyUI 的 Python 逻辑直接翻译为 C++，保持行为一致。
>
> **第一阶段目标**：优先支持 **Z-Image** 模型（GGUF 格式），其他模型后续扩展。

---

## 1. 核心架构

### 1.1 高层设计

```
用户输入（JSON 工作流 / 命令行）
    │
    ▼
┌──────────────────────────────┐
│   工作流引擎（C++）            │  ← 节点编排、DAG 调度
│   - 工作流解析器               │
│   - DAG 执行器                 │
│   - 节点注册表                 │
└──────────────┬───────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌──────────┐      ┌─────────────────────┐
│  节点实现 │      │    推理后端适配层    │
│  (C++)   │─────▶│  (SDCPPAdapter)     │
└──────────┘      └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌────────────────┐ ┌──────────┐ ┌─────────────┐
    │stable-diffusion│ │ libtorch │ │    VAE      │
    │   .cpp (GGML)  │ │(图像处理)│ │  (编解码)   │
    │                │ │          │ │             │
    │ 模型加载/推理   │ │张量操作  │ │             │
    │ 文本编码/采样   │ │高级功能  │ │             │
    └────────────────┘ └──────────┘ └─────────────┘
```

### 1.2 设计原则

1. **零 Python**：不嵌入 Python 解释器，零 Python 依赖
2. **混合架构**：
   - **stable-diffusion.cpp**：负责模型加载、GGUF 解析、量化推理、文本编码、采样（成熟稳定）
   - **libtorch**：负责图像处理、张量操作、高级功能扩展（IPAdapter、ControlNet 等）
3. **适配层隔离**：通过 `SDCPPAdapter` 封装 sd.cpp 的 C API，升级时只需修改适配层
4. **精准复刻**：直接翻译 ComfyUI Python 逻辑到 C++
5. **支持 GGUF**：通过 sd.cpp 原生支持 GGUF 格式（第一阶段：Z-Image）
6. **格式可扩展**：后续支持 Safetensors/CKPT（第二阶段+）

---

## 2. 目录结构

```
my-img/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 项目文档
├── design.md                   # 本文档（本文档）
├── task.md                     # 开发任务和进度
├── src/
│   ├── main.cpp               # 命令行入口
│   ├── engine/                # 工作流引擎核心
│   │   ├── workflow.h         # 工作流定义 & 解析
│   │   ├── workflow.cpp
│   │   ├── executor.h         # DAG 调度 & 执行器
│   │   ├── executor.cpp
│   │   ├── node.h             # 节点基类 & 注册表
│   │   ├── node.cpp
│   │   ├── cache.h            # 执行缓存（LRU）
│   │   └── cache.cpp
│   ├── nodes/                 # 节点实现（ComfyUI 节点）
│   │   ├── loaders/           # 模型加载节点
│   │   ├── conditioning/      # 条件编码节点
│   │   ├── samplers/          # 采样器节点
│   │   ├── latent/            # 潜空间操作
│   │   ├── image/             # 图像处理
│   │   ├── ipadapter/         # IPAdapter
│   │   ├── controlnet/        # ControlNet
│   │   └── upscale/           # 放大
│   ├── adapters/              # 第三方库适配层（★关键）
│   │   ├── sdcpp_adapter.h    # stable-diffusion.cpp 适配器
│   │   └── sdcpp_adapter.cpp  # 封装 sd.cpp C API
│   ├── backend/               # 推理后端（备用/扩展）
│   │   ├── model.h            # 模型基类
│   │   ├── model.cpp
│   │   ├── z_image_model.h    # Z-Image DiT (备用实现)
│   │   └── z_image_model.cpp
│   └── utils/                 # 工具
│       ├── tensor_utils.h     # 张量操作
│       ├── image_utils.h      # 图像 I/O
│       ├── gguf_loader.h      # GGUF 格式加载器
│       └── safetensors.h      # Safetensors 格式加载器
├── tests/                      # 测试
│   ├── test_gguf_loader.cpp   # GGUF 加载测试
│   ├── test_vae.cpp           # VAE 测试
│   ├── test_hires_fix.cpp     # HiRes Fix 测试
│   ├── test_z_image_model.cpp # Z-Image DiT 测试
│   ├── test_sdcpp_adapter.cpp # sd.cpp 适配器测试
│   ├── test_txt2img.cpp       # txt2img 完整测试
│   └── test_hires_fix_real.cpp # HiRes Fix 真实测试
└── third_party/               # 第三方依赖
    ├── json/                  # nlohmann/json
    ├── stb/                   # stb_image
    └── ggml/                  # GGML（用于 GGUF 解析）
```

---

## 3. 技术栈

| 组件 | 库 | 版本 | 用途 |
|-----------|---------|---------|---------|
| **模型推理** | stable-diffusion.cpp | latest | 模型加载、GGUF 推理、文本编码、采样（核心） |
| **张量计算** | libtorch | 2.x | 图像处理、张量操作、高级功能扩展 |
| **JSON 解析** | nlohmann/json | 3.x | 工作流 JSON 解析 |
| **图像 I/O** | stb_image | 2.x | PNG/JPG 加载/保存 |
| **模型格式** | GGUF | via sd.cpp | 通过 sd.cpp 原生支持 |
| **数学** | 标准 C++ | C++17 | 基础数学运算 |
| **线程** | std::thread | C++17 | 并行节点执行 |
| **CUDA** | via sd.cpp | - | GPU 加速（GGML CUDA） |

---

## 4. 后端设计

### 4.0 stable-diffusion.cpp 适配层（核心）

**为什么使用 sd.cpp？**
- sd.cpp 已经完整支持 Z-Image、Flux、SDXL 等模型
- 原生支持 GGUF 量化和高效推理（GGML/GGUF）
- 避免重复造轮子，减少开发时间
- **升级隔离**：通过适配层封装，sd.cpp API 变化时不影响上层代码

```cpp
// adapters/sdcpp_adapter.h
class SDCPPAdapter {
public:
    // 初始化和加载模型
    bool initialize(const GenerationParams& params);
    
    // 生成图像（完整流程）
    std::vector<Image> generate(const GenerationParams& params);
    
    // 设置进度回调
    void set_progress_callback(ProgressCallback callback);
    
    // 工具函数
    static std::string get_version();
    static std::vector<std::string> get_available_sample_methods();
};
```

**适配层设计原则**：
1. **封装隔离**：所有 sd.cpp C API 调用都通过适配层
2. **类型转换**：C 结构体 <-> C++ 对象（Image、参数结构体等）
3. **资源管理**：RAII 封装 sd_ctx_t 生命周期
4. **版本兼容**：sd.cpp 升级时只需修改适配层实现

### 4.1 模型基类（备用/扩展）

### 4.1 模型基类

```cpp
// backend/model.h
class Model {
public:
    virtual ~Model() = default;
    
    // 加载模型（支持 GGUF / Safetensors / CKPT）
    virtual bool load(const std::string& path) = 0;
    
    // 移动到设备
    void to(torch::Device device) { model_->to(device); }
    
    // 获取底层模块
    torch::jit::Module* get() { return model_.get(); }
    
protected:
    std::unique_ptr<torch::jit::Module> model_;
    torch::Device device_ = torch::kCPU;
};
```

### 4.2 GGUF 模型加载器

```cpp
// utils/gguf_loader.h
class GGUFLoder {
public:
    // 加载 GGUF 文件，转换为 torch tensors
    static std::map<std::string, torch::Tensor> load(const std::string& path);
    
    // 获取元数据
    static GGUFMetadata get_metadata(const std::string& path);
    
    // 将 GGUF 权重加载到 libtorch 模块
    static void load_into_module(
        torch::nn::Module& module, 
        const std::string& gguf_path
    );
};
```

**GGUF 处理流程**：
1. 使用 GGML 库读取 GGUF 文件格式（仅解析文件结构）
2. 提取权重张量（Q4_K/Q5_K/Q6_K 等量化格式）
3. 反量化为 FP16/FP32
4. 转换为 `torch::Tensor`
5. 加载到 libtorch 模块中

**注意**：GGML 仅用于**文件解析**，不参与**推理计算**。所有推理都用 libtorch。

### 4.3 UNet 实现

```cpp
// backend/unet.h
class UNetModel : public Model {
public:
    bool load(const std::string& path) override;
    
    // 前向传播（翻译 diffusers.UNet2DConditionModel）
    torch::Tensor forward(
        torch::Tensor sample,           // 噪声潜变量 [B, C, H, W]
        torch::Tensor timestep,         // 时间步张量
        torch::Tensor encoder_hidden_states,  // CLIP 文本嵌入
        std::optional<torch::Tensor> cross_attention_kwargs = std::nullopt
    );
};
```

**关键实现细节**：
- 直接翻译 ComfyUI 的 `model_base.py` 和 `modules/diffusionmodules`
- 支持 SDXL 交叉注意力（双文本编码器）
- 支持 Flux 流匹配（不同时间步调度）
- 支持 Z-Image 架构（基于 Flux 的变体）
- 内存高效注意力（torch 的 scaled_dot_product_attention）

### 4.4 VAE 实现

```cpp
// backend/vae.h
class VAEModel : public Model {
public:
    bool load(const std::string& path) override;
    
    // 编码图像到潜变量
    torch::Tensor encode(torch::Tensor image);  // [B, 3, H, W] -> [B, C, h, w]
    
    // 解码潜变量到图像
    torch::Tensor decode(torch::Tensor latent); // [B, C, h, w] -> [B, 3, H, W]
};
```

### 4.5 CLIP 实现

```cpp
// backend/clip.h
class CLIPModel : public Model {
public:
    bool load(const std::string& path) override;
    
    // 编码文本到嵌入
    torch::Tensor encode_text(const std::string& text);
    torch::Tensor encode_text(const std::vector<std::string>& texts);
    
    // 分词
    std::vector<int> tokenize(const std::string& text);
};
```

**关键实现细节**：
- 翻译 `transformers.CLIPTextModel`
- 支持 CLIP Skip（返回中间层）
- 支持 SDXL 双 CLIP（CLIP-L + CLIP-G）
- **Z-Image 支持**：支持 Qwen 文本编码器（Z-Image 使用 Qwen 而非 CLIP）

### 4.6 采样器基类

```cpp
// backend/sampler_base.h
class Sampler {
public:
    virtual ~Sampler() = default;
    virtual std::string get_name() const = 0;
    
    // 主采样循环（翻译 ComfyUI samplers.py）
    virtual torch::Tensor sample(
        UNetModel* model,
        torch::Tensor x,                    // 初始噪声
        torch::Tensor positive_cond,        // 正条件
        torch::Tensor negative_cond,        // 负条件
        int steps,
        float cfg_scale,
        std::optional<torch::Tensor> denoise_mask = std::nullopt
    ) = 0;
    
protected:
    // 获取噪声调度（翻译 comfy/model_sampling.py）
    torch::Tensor get_sigmas(int steps, std::string scheduler = "normal");
    
    // CFG 引导
    torch::Tensor cfg_guidance(
        torch::Tensor pred_positive,
        torch::Tensor pred_negative,
        float scale
    );
};
```

---

## 5. 节点系统

### 5.1 节点基类

```cpp
// engine/node.h
class Node {
public:
    virtual ~Node() = default;
    virtual std::string get_class_type() const = 0;
    virtual std::string get_category() const = 0;
    
    // 输入/输出端口定义
    virtual std::vector<PortDef> get_inputs() const = 0;
    virtual std::vector<PortDef> get_outputs() const = 0;
    
    // 主执行
    virtual NodeOutputs execute(const NodeInputs& inputs) = 0;
    
    // 可选：本次执行的缓存键
    virtual std::string compute_hash(const NodeInputs& inputs) const {
        return get_class_type();
    }
};

// 自动注册宏
#define REGISTER_NODE(name, class_type) \
    static struct class_type##Registrar { \
        class_type##Registrar() { \
            NodeRegistry::register_node(name, []() -> std::unique_ptr<Node> { \
                return std::make_unique<class_type>(); \
            }); \
        } \
    } class_type##Instance;
```

### 5.2 示例：KSampler 节点

```cpp
// nodes/samplers/ksampler.cpp
class KSamplerNode : public Node {
public:
    std::string get_class_type() const override { return "KSampler"; }
    std::string get_category() const override { return "sampling"; }
    
    std::vector<PortDef> get_inputs() const override {
        return {
            {"model", "MODEL", true, nullptr},
            {"positive", "CONDITIONING", true, nullptr},
            {"negative", "CONDITIONING", true, nullptr},
            {"latent_image", "LATENT", false, nullptr},
            {"seed", "INT", false, 0},
            {"steps", "INT", false, 20},
            {"cfg", "FLOAT", false, 8.0},
            {"sampler_name", "STRING", false, std::string("euler")},
            {"scheduler", "STRING", false, std::string("normal")},
        };
    }
    
    std::vector<PortDef> get_outputs() const override {
        return {{"LATENT", "LATENT"}};
    }
    
    NodeOutputs execute(const NodeInputs& inputs) override {
        // 1. 获取输入
        auto model = inputs.get<UNetModel*>("model");
        auto positive = inputs.get<torch::Tensor>("positive");
        auto negative = inputs.get<torch::Tensor>("negative");
        auto latent = inputs.get_or<torch::Tensor>("latent_image", 
                                                   torch::randn({1, 4, 64, 64}));
        int seed = inputs.get_or<int>("seed", 0);
        int steps = inputs.get_or<int>("steps", 20);
        float cfg = inputs.get_or<float>("cfg", 8.0);
        auto sampler_name = inputs.get_or<std::string>("sampler_name", "euler");
        
        // 2. 设置种子
        torch::manual_seed(seed);
        
        // 3. 创建采样器
        auto sampler = SamplerRegistry::create(sampler_name);
        
        // 4. 采样
        auto result = sampler->sample(model, latent, positive, negative, steps, cfg);
        
        // 5. 返回
        return {{"LATENT", result}};
    }
};

REGISTER_NODE("KSampler", KSamplerNode);
```

### 5.3 示例：IPAdapterApply 节点

```cpp
// nodes/ipadapter/ipadapter_apply.cpp
class IPAdapterApplyNode : public Node {
public:
    std::string get_class_type() const override { return "IPAdapterApply"; }
    std::string get_category() const override { return "ipadapter"; }
    
    std::vector<PortDef> get_inputs() const override {
        return {
            {"ipadapter", "IPADAPTER", true, nullptr},
            {"image", "IMAGE", true, nullptr},
            {"weight", "FLOAT", false, 1.0},
            {"noise", "FLOAT", false, 0.0},
        };
    }
    
    std::vector<PortDef> get_outputs() const override {
        return {{"IPADAPTER_PARAMS", "IPADAPTER_PARAMS"}};
    }
    
    NodeOutputs execute(const NodeInputs& inputs) override {
        auto ipadapter = inputs.get<IPAdapterModel*>("ipadapter");
        auto image = inputs.get<torch::Tensor>("image");  // [B, H, W, 3]
        float weight = inputs.get_or<float>("weight", 1.0);
        
        // 1. CLIP Vision 编码图像
        auto image_embeds = ipadapter->encode_image(image);
        
        // 2. 投影到文本空间
        auto projected = ipadapter->project(image_embeds);
        
        // 3. 打包供 UNet 注入
        IPAdapterParams params;
        params.embeds = projected;
        params.weight = weight;
        
        return {{"IPADAPTER_PARAMS", params}};
    }
};
```

---

## 6. 工作流引擎

### 6.1 工作流解析器

```cpp
// engine/workflow.h
class Workflow {
public:
    // 解析 ComfyUI JSON 格式
    static Workflow from_json(const nlohmann::json& json);
    
    // 获取执行顺序（拓扑排序）
    std::vector<std::string> get_execution_order() const;
    
    // 按 ID 获取节点
    const NodeDef& get_node(const std::string& id) const;
    
private:
    std::map<std::string, NodeDef> nodes_;
    std::map<std::string, std::vector<std::string>> edges_;
};
```

### 6.2 DAG 执行器

```cpp
// engine/executor.h
class Executor {
public:
    // 执行工作流
    void execute(const Workflow& workflow);
    
    // 带缓存执行
    void execute_with_cache(const Workflow& workflow, Cache* cache);
    
    // 进度回调
    using ProgressCallback = std::function<void(const std::string& node_id, float progress)>;
    void set_progress_callback(ProgressCallback cb);
    
private:
    // 执行单个节点
    NodeOutputs execute_node(const NodeDef& node, const std::map<std::string, NodeOutputs>& inputs);
    
    // 解析上游节点输入
    std::map<std::string, NodeOutputs> resolve_inputs(const NodeDef& node);
};
```

### 6.3 执行流程

```
1. 解析 JSON 工作流
2. 构建 DAG（节点 + 边）
3. 拓扑排序 → 执行顺序
4. 按顺序执行每个节点：
   a. 检查缓存（如启用且哈希匹配）
   b. 解析上游节点输出作为输入
   c. 执行节点
   d. 存储输出供下游节点使用
   e. 更新进度回调
5. 返回最终输出
```

---

## 7. 模型格式支持

### 7.1 Phase 1：GGUF 格式（Z-Image）

**为什么先支持 GGUF？**
- Z-Image 模型已有 GGUF 版本（`z_image_turbo-Q5_K_M.gguf`）
- 经过验证，该模型在 RTX 3080 10GB 上运行良好
- 量化格式减少显存占用

**GGUF 加载流程**：
```cpp
// 1. 读取 GGUF 文件（使用 GGML 库解析文件格式）
auto gguf_data = GGUFLoder::load("z_image_turbo-Q5_K_M.gguf");

// 2. 反量化权重
for (auto& [name, tensor] : gguf_data) {
    if (tensor.is_quantized()) {
        tensor = tensor.dequantize();  // Q4_K/Q5_K -> FP16
    }
}

// 3. 加载到 libtorch 模块
auto model = std::make_shared<UNetModel>();
model->load_from_state_dict(gguf_data);
```

### 7.2 Phase 2+：Safetensors / CKPT

**后续支持**：
- Safetensors：ComfyUI 原生格式
- CKPT：PyTorch 原生格式
- 无需转换，直接加载

```cpp
// Safetensors 加载（预留接口）
class SafetensorsLoader {
public:
    static std::map<std::string, torch::Tensor> load(const std::string& path);
};
```

### 7.3 模型格式对比

| 格式 | Phase | 优点 | 缺点 |
|------|-------|------|------|
| **GGUF** | Phase 1 | 量化减小体积、显存友好 | 需要反量化步骤 |
| **Safetensors** | Phase 2+ | ComfyUI 原生、加载快 | 无压缩、体积大 |
| **CKPT** | Phase 2+ | PyTorch 原生 | 需要 pickle 解析 |

---

## 8. 内存管理

### 8.1 设备分配

```cpp
// 自动设备选择
class DeviceManager {
public:
    static torch::Device get_default() {
        if (torch::cuda::is_available()) {
            return torch::kCUDA;
        }
        return torch::kCPU;
    }
    
    // OOM 时卸载到 CPU
    static void offload_to_cpu(torch::nn::Module& module);
};
```

### 8.2 模型加载策略

1. **懒加载**：节点执行时才加载模型权重
2. **引用计数**：无节点使用时卸载模型
3. **CPU 卸载**：权重存 CPU，计算时移 GPU
4. **精度**：默认 FP16，可选 BF16/FP32

### 8.3 Z-Image 显存优化

针对 Z-Image（Flux 架构）：
- 文本编码器（Qwen 4B）占用大 → CPU offload
- VAE（AE）较小 → 常驻 GPU
- UNet（扩散模型）→ 按需加载

---

## 9. 实现路线图

### Phase 1：Z-Image 基础出图（Week 1-3）

**目标**：命令行生成一张 Z-Image 图片

- [ ] CMake + libtorch 配置
- [ ] GGUF 加载器（读取 Z-Image 权重）
- [ ] 反量化为 libtorch 张量
- [ ] Qwen 文本编码器（Z-Image 使用 Qwen 而非 CLIP）
- [ ] Flux UNet 前向传播
- [ ] AE VAE 编解码
- [ ] Euler 采样器
- [ ] 图像保存
- [ ] **里程碑**：`./sd-workflow --model z_image.gguf --prompt "a cat" --output cat.png`

### Phase 2：HiRes Fix（Week 4-5）

- [ ] Latent 空间放大
- [ ] 二阶段重采样
- [ ] 2560x1440 生成（和当前 img1.sh 效果一致）

### Phase 3：高级功能（Week 6-8）

- [ ] img2img（带 denoise strength）
- [ ] 额外采样器（DPM++）
- [ ] LoRA 加载

### Phase 4：IPAdapter（Week 9-10）⭐ 优先级

- [ ] CLIP Vision 模型加载
- [ ] 图像编码（CLIP Vision）
- [ ] 图像投影层
- [ ] UNet 注意力注入
- [ ] IPAdapterApply 节点
- [ ] **里程碑**：参考图人脸一致性

### Phase 5：ControlNet（Week 11-12）

- [ ] ControlNet 模型加载
- [ ] 预处理器（Canny、Depth）
- [ ] ControlNetApply 节点

### Phase 6：多模型支持（Week 13-16）

- [ ] Safetensors 加载器
- [ ] SDXL 支持
- [ ] SD3 支持
- [ ] 其他 Flux 变体

---

## 10. 命令行界面

```bash
# 基础 Z-Image 出图
./sd-workflow \
  --model z_image_turbo-Q5_K_M.gguf \
  --prompt "portrait of a woman" \
  --negative-prompt "bad quality" \
  --width 1024 --height 1024 \
  --steps 20 --cfg 1.0 \
  --output portrait.png

# 带 IPAdapter（参考图）
./sd-workflow \
  --model z_image_turbo-Q5_K_M.gguf \
  --prompt "portrait" \
  --ipadapter-model ipadapter_flux.gguf \
  --ipadapter-image reference_face.png \
  --ipadapter-weight 0.8 \
  --output portrait_with_face.png

# JSON 工作流
./sd-workflow --workflow workflow.json

# 带 HiRes Fix
./sd-workflow \
  --model z_image_turbo-Q5_K_M.gguf \
  --prompt "landscape" \
  --width 1280 --height 720 \
  --hires --hires-width 2560 --hires-height 1440 \
  --output landscape_2k.png
```

---

## 11. 与现有项目对比

| 特性 | ComfyUI (Python) | stable-diffusion.cpp (GGML) | my-img (libtorch) |
|---------|-----------------|----------------------------|-------------------|
| **Python 依赖** | ✅ 需要 | ❌ 不需要 | ❌ 不需要 |
| **模型格式** | Safetensors | GGUF | **GGUF + Safetensors** |
| **张量框架** | PyTorch | GGML | **libtorch** |
| **IPAdapter** | ✅ 可用 | ❌ 不支持 | 🚧 开发中 |
| **ControlNet** | ✅ 完整 | ⚠️ 基础 | 🚧 开发中 |
| **部署** | Docker 10GB+ | 单文件 ~50MB | 单文件 ~100MB |
| **Z-Image 支持** | ✅ 可用 | ✅ 可用 | **Phase 1 目标** |
| **HiRes Fix** | ✅ 可用 | ✅ 可用 | **Phase 2 目标** |

---

## 12. 关键设计决策

### 12.1 混合架构：sd.cpp + libtorch

**为什么不用纯 libtorch？**
- **模型初始化太慢**：6B+ 参数的 DiT 模型在 libtorch 中创建需要几十秒
- **量化推理复杂**：需要重新实现 Q4_K/Q5_K 等格式的反量化和推理
- **sd.cpp 已成熟**：完整支持 Z-Image、Flux、SDXL，原生 GGUF 推理

**为什么不用纯 sd.cpp？**
- **GGML 限制**：无法支持 IPAdapter、复杂 ControlNet 等高级特性
- **libtorch 优势**：完整 PyTorch 生态，用于图像处理、张量操作、高级功能

**最终方案**：
- **sd.cpp**：模型加载、文本编码、采样、基础推理（成熟稳定）
- **libtorch**：图像处理、张量操作、IPAdapter、ControlNet 等扩展功能
- **适配层**：隔离 sd.cpp API，升级时只需修改适配层

### 12.2 为什么保留 GGUF 格式？

- **Z-Image 已有 GGUF**：不需要重新下载
- **量化优势**：减少显存占用（RTX 3080 10GB 刚好）
- **sd.cpp 原生支持**：无需额外处理

### 12.3 为什么不嵌入 Python？

- **目标是零 Python**：不是 Python-lite
- **嵌入仍需 Python 环境**：违背初衷
- **GIL 仍存在**：性能受限
- **纯 C++ 方案**：通过 sd.cpp + libtorch 实现

---

## 13. 测试策略

### 13.1 单元测试

```cpp
// 测试 Qwen 编码
TEST(QwenTest, EncodeText) {
    QwenModel qwen;
    qwen.load("qwen_4b.gguf");
    auto embeds = qwen.encode_text("一只猫");
    EXPECT_EQ(embeds.sizes(), torch::IntArrayRef({1, 77, 4096}));
}

// 测试采样器
TEST(SamplerTest, EulerSample) {
    EulerSampler sampler;
    auto noise = torch::randn({1, 16, 64, 64});  // Flux latent
    auto result = sampler.sample(...);
    EXPECT_FALSE(torch::allclose(result, noise));
}
```

### 13.2 集成测试

- 加载真实 Z-Image 模型，生成图像
- 对比 stable-diffusion.cpp 输出（相同 seed）
- 验证 HiRes Fix 2560x1440 生成

### 13.3 验证标准

- **像素级**：与 stable-diffusion.cpp 输出 MSE < 0.01（相同 seed）
- **视觉级**：人眼观察无差异
- **性能**：与 stable-diffusion.cpp 相当或更快

---

## 14. 未来扩展

### 14.1 插件系统

```cpp
// 从共享库加载自定义节点
class PluginLoader {
public:
    void load_plugin(const std::string& path);
};
```

### 14.2 模型仓库

- 自动从 HuggingFace 下载模型
- 本地模型缓存管理

### 14.3 服务模式

```bash
./sd-workflow --server --port 8080
# HTTP API 提交工作流
```

---

## 15. 总结

**my-img** 是纯 C++ 版 ComfyUI，使用 libtorch 作为推理后端：

- **零 Python 依赖**
- **支持 GGUF 格式**（Phase 1：Z-Image）
- **纯 libtorch 推理**（无 GGML 参与计算）
- **直接翻译 ComfyUI 逻辑**
- **后续扩展 Safetensors/CKPT**

**核心价值**：保留 ComfyUI 的能力，去除 Python 的痛苦。

**第一个里程碑**：命令行生成一张 Z-Image 512x512 图片，与 stable-diffusion.cpp 输出一致。
