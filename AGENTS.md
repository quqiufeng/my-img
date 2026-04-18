# AGENTS.md - AI 开发指南

## 项目背景

my-img 是一个 C++ 复刻 ComfyUI 生态的项目，基于 stable-diffusion.cpp 构建。核心是一个工作流执行引擎（sd-engine），支持 DAG 拓扑排序执行、节点缓存、多线程并行。

## 当前开发进度

### 已完成（✅）

| 类别 | 项目 | 说明 |
|------|------|------|
| 核心引擎 | sd-engine | DAG 执行器、节点缓存、WorkflowBuilder |
| 采样 | KSampler / KSamplerAdvanced | 支持 LoRA/ControlNet/IPAdapter/Mask |
| Latent | EmptyLatentImage / VAEEncode / VAEDecode / LatentUpscale / LatentComposite | 完整实现 |
| 条件编码 | CLIPTextEncode / CLIPSetLastLayer / ConditioningCombine / Concat / Average | 完整实现 |
| 图像处理 | LoadImage / SaveImage / ImageScale / ImageCrop / ImageBlend / ImageCompositeMasked | 完整实现 |
| 图像效果 | ImageInvert / ImageColorAdjust / ImageBlur / ImageGrayscale / ImageThreshold | 完整实现 |
| 超分 | UpscaleModelLoader / ImageUpscaleWithModel | ESRGAN 支持 |
| 背景抠图 | RemBGModelLoader / ImageRemoveBackground | ONNX Runtime |
| 人脸 | FaceDetect / FaceRestoreWithModel / FaceSwap | ONNX Runtime |
| 预处理器 | CannyEdgePreprocessor | CPU 实现 |
| 高清修复 | DeepHighResFix | 原生 latent hook 实现 |

### 占位符节点（⚠️ 运行时返回 ERROR_EXECUTION_FAILED）

| 节点 | 状态 | 说明 |
|------|------|------|
| MiDaS-DepthMapPreprocessor | ⚠️ | 需 ONNX 模型（MiDaS/DPT） |
| OpenPosePreprocessor | ⚠️ | 需 ONNX 模型（OpenPose/RTMPose） |
| INPAINT_LoadInpaintModel | ⚠️ | 需 inpaint UNet 支持 |
| INPAINT_ApplyInpaint | ⚠️ | 需 inpaint mask 采样逻辑 |
| AnimateDiffLoader | ⚠️ | 需 motion module 加载 |
| AnimateDiffSampler | ⚠️ | 需多帧 latent 采样 |
| CLIPVisionEncode | ⚠️ | C API 为占位符，返回 dummy 数据 |
| IPAdapterApply | ⚠️ | 传递信息到 KSampler，但底层 sd_load_ipadapter 为占位符 |

### 已修复问题

| 问题 | 修复方式 |
|------|----------|
| 占位符节点返回假结果 | 改为返回 `ERROR_EXECUTION_FAILED` 并打印明确错误信息 |
| `new` 后无效 `nullptr` 检查 | 改为 `try-catch (std::bad_alloc)` |
| 缺失 C API 实现 | 补充 `sd_conditioning_concat/average`、`sd_load_ipadapter`、`sd_clip_vision_encode_image` 等 |

## 节点开发规范

### 1. 继承 Node 基类

```cpp
class MyNode : public Node {
  public:
    std::string get_class_type() const override { return "MyNode"; }
    std::string get_category() const override { return "category"; }
    std::vector<PortDef> get_inputs() const override { return {{"name", "TYPE", true, nullptr}}; }
    std::vector<PortDef> get_outputs() const override { return {{"OUTPUT", "TYPE"}}; }
    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        // 实现逻辑
        return sd_error_t::OK;
    }
};
REGISTER_NODE("MyNode", MyNode);
```

### 2. 错误处理

- 使用 `SD_RETURN_IF_ERROR(expr)` 宏
- 使用 `SD_RETURN_IF_NULL(ptr, err_code)` 宏
- 内存分配使用 `try-catch (std::bad_alloc)`
- 占位符节点必须返回 `ERROR_EXECUTION_FAILED` 并打印明确错误

### 3. 输入提取

```cpp
// 必需输入
LatentPtr latent;
SD_RETURN_IF_ERROR(get_input(inputs, "samples", latent));

// 可选输入
std::string method = get_input_opt<std::string>(inputs, "upscale_method", "nearest-exact");
int steps = get_input_opt<int>(inputs, "steps", 20);
```

### 4. 内存管理

- Latent/Conditioning/Image 使用智能指针：`LatentPtr`, `ConditioningPtr`, `ImagePtr`
- 新分配使用 `make_latent_ptr()`, `make_conditioning_ptr()`, `make_image_ptr()`
- 像素缓冲区使用 `make_malloc_buffer()`

## 编译流程

### 1. 编译 stable-diffusion.cpp

```bash
cd ~/my-img/build
./build_sd.sh [--cuda|--no-cuda] [--flash-attn|--no-flash] [--clean] [--jobs N]
```

### 2. 编译 my-img

```bash
cd ~/my-img/build
cmake .. -DSD_PATH=/home/dministrator/stable-diffusion.cpp
make -j$(nproc)
```

### 3. 运行测试

```bash
cd ~/my-img/build
ctest --output-on-failure
```

## 补丁管理

### 应用补丁

```bash
cd ~/my-img
./apply_patches.sh
```

### 补丁文件

- `patches/sd-engine-full.patch`：主补丁（stable-diffusion.cpp 修改）
- `patches/sd-engine-ext-header.patch`：扩展头文件补丁
- `stable-diffusion.cpp-patched/`：修改备份

### 升级 stable-diffusion.cpp 后

```bash
cd ~/stable-diffusion.cpp && git pull
cd ~/my-img && ./apply_patches.sh
cd ~/my-img/build && ./build_sd.sh
cd ~/my-img/build && cmake .. && make -j$(nproc)
```

## C API 扩展规范

在 `stable-diffusion.cpp` 中添加新 C API 时：

1. 在 `include/stable-diffusion-ext.h` 中声明
2. 在 `src/stable-diffusion.cpp` 中实现
3. 使用 `SD_API` 标记导出
4. 空指针检查必须完整
5. 内存分配使用 `try-catch (std::bad_alloc)`

## 已知限制

1. **ONNX 模型需自备**：人脸/背景抠图/LineArt 等节点需要用户自行下载 ONNX 模型
2. **部分节点为占位符**：见上方占位符节点列表
3. **测试未覆盖核心生成链路**：KSampler/VAEDecode 等关键节点缺乏端到端测试（需要大模型）
4. **日志系统已引入但未全面替换**：`log.h` 已创建，后续需逐步替换所有 `printf` 调用

## 文件结构

```
my-img/
├── CMakeLists.txt
├── README.md
├── AGENTS.md              # 本文档
├── apply_patches.sh
├── build/
│   └── build_sd.sh        # 一键编译 stable-diffusion.cpp
├── docs/
│   ├── sd-engine-design.md
│   └── face-onnx-design.md
├── src/
│   ├── sd-engine/
│   │   ├── core/          # 引擎核心
│   │   ├── face/          # 人脸处理 ONNX 模块
│   │   ├── nodes/         # 节点实现
│   │   ├── preprocessors/ # 图像预处理器
│   │   └── tools/         # CLI 工具
│   ├── sd-hires/
│   ├── sd-img2img/
│   └── sd-upscale/
├── patches/
│   ├── sd-engine-full.patch
│   └── sd-engine-ext-header.patch
├── stable-diffusion.cpp-patched/
└── tests/                 # Catch2 测试
```
