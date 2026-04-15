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

| 阶段 | 推荐模型 | 格式 | 约大小 | 来源 |
|------|---------|------|--------|------|
| 人脸检测 | `scrfd_2.5g_bnkps.onnx` | ONNX | ~6MB | [InsightFace](https://github.com/deepinsight/insightface) |
| 人脸 Landmark | `2d106det.onnx` | ONNX | ~4MB | InsightFace |
| 人脸修复 | `gfpgan_1.4.onnx` | ONNX | ~40MB | 社区 ONNX 转换版 |
| 人脸修复 | `codeformer_0.1.onnx` | ONNX | ~120MB | 社区 ONNX 转换版 |
| 人脸换脸 | `inswapper_128.onnx` | ONNX | ~530MB | InsightFace |

> **注**：GFPGAN/CodeFormer 的 ONNX 版本需要社区转换。若找不到稳定来源，可优先实现 **CodeFormer**（效果更稳定）。

## 4. 代码架构

新建目录 `src/sd-engine/face/`，所有 ONNX 推理代码集中管理：

```
src/sd-engine/face/
├── face_detect.hpp / .cpp      # SCRFD/YuNet 人脸检测
├── face_landmark.hpp / .cpp    # 106/68 点 landmark 检测
├── face_align.hpp / .cpp       # 人脸对齐矩阵计算 + 仿射变换
├── face_restore.hpp / .cpp     # GFPGAN / CodeFormer ONNX 推理
├── face_swap.hpp / .cpp        # InsightFace inswapper ONNX 推理
└── face_utils.hpp / .cpp       # 公共工具（裁剪、融合、颜色校正）
```

### 4.1 人脸检测（SCRFD）

输入：`[1, 3, H, W]`（图像尺寸，如 640x640）
输出：人脸框 `[num_faces, 4]`（x1, y1, x2, y2）+ 置信度 + 5/10 个关键点

预处理：
- resize 到模型输入尺寸（保持长宽比，padding）
- 归一化（减去 mean，除以 std）

后处理：
- NMS（非极大值抑制）去重
- 将相对坐标映射回原图尺寸

### 4.2 人脸 Landmark（2d106det）

输入：裁剪后的人脸图像 `[1, 3, 192, 192]`
输出：`[106, 2]` 关键点坐标

作用：获取更精确的 landmark，用于计算对齐仿射矩阵。

### 4.3 人脸对齐（Face Alignment）

目标：将检测到的人脸变换为模型期望的标准姿态。

标准模板（GFPGAN/CodeFormer 通常使用 512x512）：
- 左眼：(192, 240)
- 右眼：(320, 240)
- 鼻尖：(256, 320)
- 嘴左：(220, 400)
- 嘴右：(292, 400)

算法：
1. 从 landmark 中提取 5 个基准点
2. 计算 `cv::getAffineTransform` 或 `cv::getPerspectiveTransform`
3. 由于我们没有 OpenCV，需要手写 2x3 仿射矩阵计算 + 双线性插值 warp

> **关键**：对齐时必须同时保存**正向矩阵**和**逆矩阵**，用于后续贴回。

### 4.4 人脸修复（GFPGAN / CodeFormer）

输入：`[1, 3, 512, 512]` RGB 图像（已对齐）
输出：`[1, 3, 512, 512]` 修复后图像

后处理：
- 逆仿射变换 warp 回原人脸区域
- 使用高斯羽化边缘（feathering）消除接缝
- 可选：颜色校正（匹配原图肤色）

### 4.5 人脸换脸（InsightFace inswapper_128）

输入：
- `target`: `[1, 3, 128, 128]` 目标人脸（已对齐）
- `source`: `[1, 512]` 源人脸的 embedding 向量（由另一张图通过 ArcFace 提取）

输出：`[1, 3, 128, 128]` 换脸结果

完整换脸流程：
1. 检测源图人脸 → 对齐 → ArcFace 编码为 embedding
2. 检测目标图人脸 → 对齐
3. inswapper 推理
4. 逆仿射变换贴回目标图

> **注**：ArcFace 也需要一个 ONNX 模型（约 130MB），或 inswapper 的 embedding 可直接用。

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
    inputs:  {"model_path": "STRING"}
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
        "IMAGE": "IMAGE",           // 带检测框的预览图（可选）
        "faces": "FACE_BBOX_LIST"   // 人脸框列表（自定义类型）
    }
};
```

### 5.3 人脸修复节点（一键版，最实用）

```cpp
class FaceRestoreWithModelNode : public Node {
    inputs: {
        "image": "IMAGE",
        "face_restore_model": "FACE_RESTORE_MODEL",
        "face_detect_model": "FACE_DETECT_MODEL",  // 可选，内部默认使用
        "facedetection": "STRING",                  // 检测器类型
        "codeformer_fidelity": "FLOAT",             // 0.0 - 1.0
        "restore_first": "BOOLEAN"                  // 是否先放大再修复
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
        "face_detect_model": "FACE_DETECT_MODEL"
    }
    outputs: {
        "IMAGE": "IMAGE"
    }
};
```

## 6. 实现阶段规划

### Phase 1：基础设施（1-2 天）
- [ ] 实现 `face_detect.hpp/cpp`（SCRFD ONNX 推理 + NMS）
- [ ] 实现 `face_align.hpp/cpp`（5 点仿射变换 + 双线性 warp）
- [ ] 下载并验证 ONNX 模型可用性

### Phase 2：人脸修复（1-2 天）
- [ ] 实现 `face_restore.hpp/cpp`（GFPGAN/CodeFormer ONNX 推理）
- [ ] 实现 `FaceRestoreModelLoaderNode` 和 `FaceRestoreWithModelNode`
- [ ] 边缘羽化 + 颜色校正
- [ ] 单元测试

### Phase 3：人脸换脸（2-3 天）
- [ ] 实现 ArcFace embedding 提取（或复用现有 embedding）
- [ ] 实现 `face_swap.hpp/cpp`（inswapper ONNX 推理）
- [ ] 实现 `FaceSwapModelLoaderNode` 和 `FaceSwapNode`
- [ ] 单元测试

## 7. 风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| GFPGAN ONNX 模型来源不稳定 | 无法运行修复 | 优先实现 CodeFormer，或只支持一种 |
| 手写仿射变换性能差 | 大图像处理慢 | 后期可引入 OpenCV 作为可选依赖 |
| inswapper_128.onnx 太大（530MB） | 下载困难 | 使用 256 版本（更大）或寻找更轻量模型 |
| 多人脸场景复杂 | 贴回逻辑易出错 | Phase 1 先只支持单人脸 |

## 8. 建议的启动顺序

考虑到实现复杂度和实用性，建议按以下顺序启动：

1. **先做人脸检测 + 对齐基础设施**（所有后续功能的基础）
2. **再做 FaceRestoreWithModel**（最实用、用户呼声最高的功能）
3. **最后做 FaceSwap**（模型大、流程长）

---

*设计日期：2025-04-15*
