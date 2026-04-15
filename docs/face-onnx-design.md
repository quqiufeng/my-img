# 基于 ONNX Runtime 的人脸修复与换脸方案设计

## 1. 目标

在 `sd-engine` 中实现 ComfyUI 风格的人脸修复（Face Restore）和换脸（Face Swap）节点，基于已集成的 **ONNX Runtime** 可选依赖。

## 2. 核心挑战

人脸修复/换脸不是简单的端到端图像变换，完整流程包含 4 个阶段：
1. **人脸检测**（Face Detection）— 定位图像中的人脸框
2. **人脸对齐**（Face Alignment）— 根据 landmark 将人脸旋转/缩放到标准姿态
3. **人脸处理**（Restore / Swap）— ONNX 模型推理
4. **人脸贴回**（Paste Back）— 逆变换将处理后人脸融合回原图

## 3. ONNX 模型选型

| 阶段 | 实际使用模型 | 格式 | 约大小 | 来源 |
|------|-------------|------|--------|------|
| 人脸检测 | `yunet_320_320.onnx` | ONNX | ~228KB | OpenCV Zoo YuNet |
| 人脸修复 | `GFPGANv1.4.onnx` | ONNX | ~325MB | Hugging Face 社区转换版 |
| 人脸修复 | `codeformer.onnx` | ONNX | ~360MB | Hugging Face 社区转换版 |
| 人脸换脸 | `inswapper_128.onnx` | ONNX | ~529MB | InsightFace |
| 人脸换脸 | `arcface.onnx` | ONNX | ~131MB | ONNX Community |

> **注**：实际实现中选用 **YuNet** 替代 SCRFD 作为人脸检测器，因为 YuNet 在 OpenCV Zoo 中有稳定的 ONNX 导出，且输入输出格式清晰。

## 4. 代码架构

新建目录 `src/sd-engine/face/`，所有 ONNX 推理代码集中管理：

```
src/sd-engine/face/
├── face_detect.hpp / .cpp      # YuNet 人脸检测 + NMS
├── face_align.hpp / .cpp       # 5点仿射变换 + 双线性插值 warp
├── face_restore.hpp / .cpp     # GFPGAN / CodeFormer ONNX 推理
├── face_swap.hpp / .cpp        # ArcFace embedding + inswapper ONNX 推理
└── face_utils.hpp / .cpp       # 公共工具（羽化蒙版、融合、颜色校正）
```

### 4.1 人脸检测（YuNet）

输入：`[1, 3, 320, 320]`（resize + padding 到 320×320）
输出：`cls_8/16/32`, `obj_8/16/32`, `bbox_8/16/32`, `kps_8/16/32`

后处理：
- 按 stride 分组解析输出
- 生成候选框并做 NMS（非极大值抑制）
- 将相对坐标映射回原图尺寸
- 每个框包含 5 个关键点（双眼、鼻尖、嘴角）

### 4.2 人脸对齐（Face Alignment）

目标：将检测到的人脸变换为模型期望的标准姿态。

标准模板（GFPGAN/CodeFormer 使用 512×512）：
- 左眼：(192, 240)
- 右眼：(320, 240)
- 鼻尖：(256, 320)
- 嘴左：(220, 400)
- 嘴右：(292, 400)

inswapper_128 使用同一模板缩放到 128×128：
- 左眼：(48, 60)
- 右眼：(80, 60)
- 鼻尖：(64, 80)
- 嘴左：(55, 100)
- 嘴右：(73, 100)

算法：
1. 从 landmark 中提取 5 个基准点
2. 手写 2×3 仿射矩阵估计（`estimate_affine_transform_2d3`）
3. 双线性插值 warp（`crop_face`）
4. 同时计算逆矩阵用于后续贴回

> **关键**：对齐时必须同时保存**正向矩阵**和**逆矩阵**。

### 4.3 人脸修复（GFPGAN / CodeFormer）

输入：`[1, 3, 512, 512]` NCHW RGB 图像（已对齐）
输出：`[1, 3, 512, 512]` 修复后图像

预处理：
- 像素值归一化到 `[-1, 1]`：`pixel = (pixel / 255 - 0.5) / 0.5`

后处理：
- 反归一化回到 `[0, 255]`
- 逆仿射变换 warp 回原人脸区域
- 32px 高斯羽化边缘融合

CodeFormer 特殊处理：
- 部分 ONNX 导出版本包含第二个输入 `w`（fidelity 参数，类型为 `double`）
- 默认 fidelity = 0.5，范围 [0.0, 1.0]

### 4.4 人脸换脸（InsightFace inswapper_128 + ArcFace）

**ArcFace**（embedding 提取）：
- 输入：`[N, 112, 112, 3]` NHWC
- 输出：`[N, 512]` embedding 向量
- 预处理：128×128 人脸双线性插值缩放到 112×112，归一化到 `[-1, 1]`

**inswapper_128**（换脸推理）：
- 输入：`target` [1, 3, 128, 128] + `source` [1, 512] embedding
- 输出：`[1, 3, 128, 128]` 换脸结果

完整换脸流程：
1. 检测 target 和 source 图像中的人脸
2. 分别对齐到 128×128
3. ArcFace 从 source 人脸提取 512 维 embedding
4. inswapper 推理生成换脸结果
5. 逆仿射变换 + 16px 羽化边缘贴回 target 图

## 5. 节点设计

### 5.1 模型加载节点

```cpp
class FaceDetectModelLoaderNode : public Node {
    inputs:  {"model_path": "STRING"}
    outputs: {"FACE_DETECT_MODEL": "FACE_DETECT_MODEL"}
};

class FaceRestoreModelLoaderNode : public Node {
    inputs:  {"model_path": "STRING", "model_type": "STRING"}  // gfpgan / codeformer
    outputs: {"FACE_RESTORE_MODEL": "FACE_RESTORE_MODEL"}
};

class FaceSwapModelLoaderNode : public Node {
    inputs:  {
        "inswapper_path": "STRING",
        "arcface_path": "STRING"
    }
    outputs: {"FACE_SWAP_MODEL": "FACE_SWAP_MODEL"}
};
```

### 5.2 人脸检测节点

```cpp
class FaceDetectNode : public Node {
    inputs:  {
        "image": "IMAGE",
        "model": "FACE_DETECT_MODEL",
        "confidence_threshold": "FLOAT"  // 默认 0.5
    }
    outputs: {
        "IMAGE": "IMAGE",           // 带检测框的预览图
        "faces": "FACE_BBOX_LIST"   // 人脸框列表（自定义类型）
    }
};
```

### 5.3 人脸修复节点（一键版）

```cpp
class FaceRestoreWithModelNode : public Node {
    inputs: {
        "image": "IMAGE",
        "face_restore_model": "FACE_RESTORE_MODEL",
        "face_detect_model": "FACE_DETECT_MODEL",  // 可选
        "codeformer_fidelity": "FLOAT"             // 0.0 - 1.0，默认 0.5
    }
    outputs: {
        "IMAGE": "IMAGE"
    }
};
```

### 5.4 人脸换脸节点

```cpp
class FaceSwapNode : public Node {
    inputs: {
        "target_image": "IMAGE",
        "source_image": "IMAGE",
        "face_swap_model": "FACE_SWAP_MODEL",
        "face_detect_model": "FACE_DETECT_MODEL"  // 可选
    }
    outputs: {
        "IMAGE": "IMAGE"
    }
};
```

## 6. 实现阶段规划

### Phase 1：人脸检测 + 对齐基础设施 ✅
- [x] 实现 `face_detect.hpp/cpp`（YuNet ONNX 推理 + NMS）
- [x] 实现 `face_align.hpp/cpp`（5 点仿射变换 + 双线性 warp）
- [x] 实现 `face_utils.hpp/cpp`（羽化蒙版、人脸融合）
- [x] 实现 `FaceDetectModelLoaderNode` 和 `FaceDetectNode`
- [x] 下载并验证 ONNX 模型可用性

### Phase 2：人脸修复 ✅
- [x] 实现 `face_restore.hpp/cpp`（GFPGAN/CodeFormer ONNX 推理）
- [x] 实现 `FaceRestoreModelLoaderNode` 和 `FaceRestoreWithModelNode`
- [x] 边缘羽化 + 颜色校正
- [x] 单元测试（含 GFPGAN/CodeFormer 直接推理验证）

### Phase 3：人脸换脸 ✅
- [x] 下载 ArcFace ONNX 模型
- [x] 实现 `face_swap.hpp/cpp`（ArcFace embedding + inswapper ONNX 推理）
- [x] 实现 `FaceSwapModelLoaderNode` 和 `FaceSwapNode`
- [x] 单元测试（含 inswapper+ArcFace 直接推理验证）

## 7. 已知问题与限制

| 问题 | 说明 | 状态 |
|------|------|------|
| 合成人脸检测不到 | YuNet 训练于真实人脸，卡通/合成图像可能检测不到 | 预期行为 |
| 多人脸只处理第一张 | 当前 `FaceRestoreWithModel` 和 `FaceSwap` 仅处理检测到的第一张人脸 | 已知限制 |
| 手写仿射变换性能 | 大图像（4K+）处理较慢，后期可引入 OpenCV 作为可选依赖 | 待优化 |
| CodeFormer `w` 类型 | 不同 ONNX 导出版本可能要求 `float` 或 `double`，当前使用 `double` | 已适配 |

## 8. 模型下载地址

| 模型 | 下载链接 |
|------|---------|
| YuNet | `https://github.com/opencv/opencv_zoo`（或已随 OpenCV 分发） |
| GFPGAN v1.4 ONNX | `https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx` |
| CodeFormer ONNX | `https://huggingface.co/bluefoxcreation/Codeformer-ONNX/resolve/main/codeformer.onnx` |
| inswapper_128 | `https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx` |
| ArcFace | `https://huggingface.co/onnx-community/arcface-onnx/resolve/main/arcface.onnx` |

## 9. 测试状态

- **全量单元测试**：89 assertions, 19 test cases — **全部通过** ✅
- **人脸相关测试**：39 assertions, 11 test cases — **全部通过** ✅

---

*设计日期：2025-04-15*  
*完成日期：2025-04-15*
