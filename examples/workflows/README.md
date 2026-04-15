# sd-engine 示例工作流

本目录包含可直接在 sd-engine 中运行的 ComfyUI 格式工作流 JSON 示例。

## 可用示例

### 1. txt2img.json
最基础的文生图工作流：
- `CheckpointLoaderSimple` 加载模型
- `CLIPTextEncode` 编码正负提示词
- `EmptyLatentImage` 创建空 latent
- `KSampler` 采样
- `VAEDecode` 解码为图像
- `SaveImage` 保存结果

### 2. img2img.json
图生图工作流：
- 加载输入图像
- `VAEEncode` 将图像编码为 latent
- `KSampler` 基于 latent 重新采样
- 解码并保存结果

## 运行方式

```bash
# 使用 sd-workflow 工具运行
sd-workflow --workflow examples/workflows/txt2img.json --output output/
```

或直接通过 C++ API 加载：

```cpp
sdengine::Workflow wf;
wf.load_from_file("examples/workflows/txt2img.json");
```

## 自定义参数

工作流 JSON 中的字面量值（如 `seed`、`steps`、`text`）可以在执行时被覆盖：

```cpp
sdengine::ExecutionConfig config;
config.overrides["5.seed"] = 12345;      // 覆盖 KSampler 的 seed
config.overrides["2.text"] = "a cat";    // 覆盖正提示词
executor.execute(&wf, config);
```

> 注意：覆盖键的格式为 `node_id.input_name`，具体取决于工作流 JSON 中的节点 ID。
