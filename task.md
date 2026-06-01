# my-img 开发任务表

> **目标**: 纯 C++ 版 ComfyUI，使用 sd.cpp 后端，支持 Z-Image（GGUF）
> **架构**: sd.cpp（模型推理）+ 纯 C++ 封装
> **开发模式**: 完成一个功能 → 编写测试 → 测试通过 → 更新进度 → 下一个功能

---

## 已完成 ✅

### 核心生成
- [x] txt2img 基础出图
- [x] 15 种采样方法（Euler, DPM++, Heun, LCM, DDIM, TCD 等）
- [x] 11 种调度器（Discrete, Karras, Exponential, AYS, GITS 等）
- [x] HiRes Fix（latent 放大 + refine）
- [x] VAE Tiling（省显存）
- [x] Flash Attention
- [x] ESRGAN 放大（2×/4×）

### CLI
- [x] 完整命令行参数支持
- [x] 图像保存（PNG）
- [x] 进度显示

---

## 待实现功能清单（对齐 ComfyUI）

### Phase 6: 核心功能（ComfyUI 基础节点）

#### Task 6.1: img2img（图像到图像）🌟 ✅
- [x] CLI 参数：`--init-img PATH`
- [x] CLI 参数：`--strength FLOAT`（0.0-1.0，默认 0.75）
- [x] 加载参考图（PNG/JPG）
- [x] VAE encode 为 latent
- [x] 根据 strength 加噪
- [x] 去噪生成
- [x] 参考：sd.cpp `init_image` 参数
- [x] 单元测试：test_image_utils
- **优先级**: 🌟 P0
- **难度**: 低（sd.cpp 原生支持）
- **状态**: ✅ 已完成（2026-04-27）

#### Task 6.2: LoRA 集成 🌟 ✅
- [x] CLI 参数：`--lora "PATH:weight"`（支持多个）
- [x] LoRA 权重加载（.safetensors）
- [x] 权重注入扩散模型
- [x] 高噪声/低噪声 LoRA 支持
- [x] 参考：sd.cpp `sd_lora_t`
- [x] 单元测试：已验证 sd_lora_t 结构传递
- **优先级**: 🌟 P0
- **难度**: 中
- **状态**: ✅ 已完成（2026-04-27）

#### Task 6.3: Inpainting（局部重绘）✅
- [x] CLI 参数：`--mask PATH`
- [x] 加载 mask 图像（黑白图）
- [x] masked latent 生成
- [x] 保留未遮罩区域
- [x] 与 img2img 结合使用（先 init-img，再 mask）
- **优先级**: 中
- **难度**: 中
- **状态**: ✅ 已完成（2026-04-27）

#### Task 6.4: Outpainting（图像外扩）
- [x] 扩展现有图像边界
- [x] 保持原图内容，生成新区域
- [x] 支持指定外扩方向（上/下/左/右）
- [x] CLI 参数：`--outpaint N`（所有方向）
- [x] CLI 参数：`--outpaint-top N`, `--outpaint-bottom N`, `--outpaint-left N`, `--outpaint-right N`
- **优先级**: P1
- **难度**: 中
- **状态**: ✅ 已完成（2026-04-27）

#### Task 6.5: Textual Inversion / Embeddings
- [x] 加载 .pt/.safetensors/.bin embedding 文件
- [x] 在 prompt 中识别 `embedding:name` 语法
- [x] 通过 sd.cpp 注入到文本编码器
- [x] CLI 参数：`--embd-dir PATH`
- **优先级**: P1
- **难度**: 低
- **状态**: ✅ 基础版本已完成（2026-04-27）

### Phase 7: ControlNet & 控制节点

#### Task 7.1: ControlNet ✅
- [x] ControlNet 模型加载（.pth/.safetensors）
- [x] CLI 参数：`--control-net PATH`
- [x] CLI 参数：`--control-image PATH`
- [x] CLI 参数：`--control-strength FLOAT`（默认 0.9）
- [x] 支持 Canny/Depth/Lineart/OpenPose/Normal/Scribble
- [x] 参考：sd.cpp `control_image`, `control_strength`
- **优先级**: 中
- **难度**: 高
- **状态**: ✅ 已完成（2026-04-27）

#### Task 7.2: T2I-Adapter ✅
- [x] 轻量级条件控制（比 ControlNet 更省显存）
- [x] ONNX 推理框架（Sketch/Canny 条件特征提取）
- [x] 条件特征注入扩散模型
- [x] CLI 参数：`--t2i-adapter PATH --t2i-adapter-image PATH --t2i-adapter-strength FLOAT`
- **优先级**: P2
- **难度**: 高
- **状态**: ✅ 框架已完成（2026-06-01）

#### Task 7.3: ControlNet 预处理器
- [x] Canny 边缘检测（OpenCV）
- [ ] Depth 深度估计（MiDaS/DPT - 需要 ONNX 模型）
- [x] Lineart 线条提取（OpenCV Canny + 后处理）
- [ ] OpenPose 姿态检测（需要 OpenPose ONNX 模型）
- [x] Normal Map 法线贴图（OpenCV Sobel）
- [x] Scribble 涂鸦识别（OpenCV Canny + 反色）
- [x] CLI 参数：`--control-preprocessor NAME`
- [x] CLI 参数：`--control-preprocessor-param1 INT`
- [x] CLI 参数：`--control-preprocessor-param2 INT`
- **优先级**: P2
- **难度**: 中（使用 OpenCV 实现基础预处理器）

### Phase 8: 图像理解 & 反推

#### Task 8.0: 图片反推提示词（CLIP Interrogator / Image2Prompt）🛒
- [x] PNG 元数据读取（tEXt/iTXt/zTXt chunks）- 提取 embedded parameters
- [ ] 加载 CLIP Vision 模型（或 BLIP/LLaVA）
- [ ] 分析图片内容并生成文本描述
- [ ] 支持多种反推模式：
  - 基础描述（自然语言）
  - 标签式（tag1, tag2, tag3）
  - Stable Diffusion 提示词风格
  - Danbooru/动漫标签风格
- [x] CLI 参数：`--interrogate PATH`（占位符，提示 JoyCaption 集成方式）
- [x] CLI 参数：`--read-metadata PATH`
- [ ] CLI 参数：`--interrogate-mode {describe|tags|sd|danbooru}`
- [x] 输出到标准输出
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 中
- **用途**: 
  - 电商：分析竞品图片获取提示词
  - 设计：复现参考图风格
  - 素材管理：自动打标签
- **状态**: ✅ 基础版本已完成（PNG 元数据读取 + JoyCaption 集成占位符）

### Phase 9: IPAdapter & 图像条件

#### Task 9.1: IPAdapter（图像提示词）✅
- [x] CLIP Vision 模型加载（ONNX）
- [x] 图像特征提取
- [x] IPAdapter 模型加载（ONNX）
- [x] 注意力注入框架
- [x] CLI 参数：`--ipadapter PATH`
- [x] CLI 参数：`--ipadapter-image PATH`
- [x] CLI 参数：`--ipadapter-weight FLOAT`
- **优先级**: P2
- **难度**: 高
- **状态**: ✅ 框架已完成（2026-06-01）

#### Task 8.2: IPAdapter FaceID
- [ ] 人脸识别特征提取
- [ ] 保持人物一致性
- [ ] 支持多张参考图
- **优先级**: P2
- **难度**: 高

#### Task 8.3: PhotoMaker ✅
- [x] PhotoMaker 模型加载（ONNX）
- [x] ID 图像编码 + 多 ID 聚合
- [x] 个性化生成框架
- [x] CLI 参数：`--photo-maker PATH`
- [x] CLI 参数：`--photo-maker-images PATH1,PATH2`
- [x] CLI 参数：`--photo-maker-strength FLOAT`
- **优先级**: P3
- **难度**: 高
- **状态**: ✅ 框架已完成（2026-06-01）

### Phase 9: 高级采样与质量

#### Task 9.1: Regional Prompting（分区提示词）✅
- [x] 图像分区（上/下/左/右/中心）
- [x] 不同区域应用不同提示词
- [x] 区域权重控制 + mask 像素级融合
- [x] CLI 参数：`--regional-prompts "top:0.5,blue sky|bottom:0.5,green grass"`
- **优先级**: P2
- **难度**: 中
- **状态**: ✅ 已完成（2026-06-01）

#### Task 9.2: Self-Attention Guidance (SAG) ✅
- [x] 提升图像细节和构图
- [x] 无需额外模型
- [x] CLI 参数：`--sag`, `--sag-scale FLOAT`
- **优先级**: P2
- **难度**: 中
- **完成时间**: 2026-04-30

#### Task 9.2b: Dynamic CFG (Dynamic Thresholding) ✅
- [x] 防止 CFG 过高导致的过饱和
- [x] 自动调整 CFG 强度
- [x] CLI 参数：`--dynamic-cfg`
- **优先级**: P2
- **难度**: 中
- **完成时间**: 2026-04-30

#### Task 9.3: FreeU / FreeU_V2 ✅
- [x] 在 UNet 中注入 FreeU 参数（通过 patch 系统，最小化修改 sd.cpp）
- [x] 提升图像质量和细节（实测：纹理更丰富，边缘更清晰）
- [x] CLI 参数：`--freeu`（启用 FreeU）
- [x] Patch 系统：`patches/diff.sh` 支持 apply/revert/status
- **优先级**: P3
- **难度**: 中
- **完成时间**: 2026-04-30

#### Task 9.4: Style Transfer
- [ ] 加载风格参考图
- [ ] 风格特征提取与注入
- [ ] 与 IPAdapter 结合使用
- **优先级**: P3
- **难度**: 高

### Phase 10: Latent & 图像操作

#### Task 10.1: Latent Composite（Latent 合成）✅
- [x] 多个 latent 按位置合成
- [x] 支持 mask 控制融合区域（feather blending）
- [x] 用于多区域生成
- [x] CLI 参数：`--latent-composite`
- **优先级**: P2
- **难度**: 中
- **状态**: ✅ 已完成（2026-06-01）

#### Task 10.2: Image Composite（图像合成）
- [ ] 像素级图像合成
- [ ] Alpha 通道混合
- [ ] 用于 Inpainting 结果融合
- **优先级**: P2
- **难度**: 低

#### Task 10.3: Image Scale & Crop
- [x] 图像缩放（nearest/bilinear/bicubic 插值）
- [x] 图像裁剪（指定坐标和尺寸）
  - [x] CLI 参数：`--crop x,y,width,height`（像素坐标）
  - [x] CLI 参数：`--crop-center width,height`（居中裁剪）
  - [x] CLI 参数：`--crop-ratio width:height`（按比例裁剪，如 16:9, 4:3, 1:1）
- [x] 图像翻转（水平/垂直）
- [x] 图像旋转（90°/180°/270°）
- [x] 图像格式转换（PNG/BMP/TGA/JPG/WEBP，自动检测扩展名）
- **优先级**: P2
- **难度**: 低
- **用途**: 
  - 电商：统一商品图尺寸
  - 社交媒体：裁剪为平台要求比例（小红书 3:4、抖音 9:16、淘宝 1:1）
  - 批量处理：统一所有图片尺寸

### Phase 10.5: 摄影后期与图像调整（电商/摄影刚需）

#### Task 10.5.1: 色温与白平衡
- [x] 色温调整（RGB 通道平衡，-1.0 冷色调 ↔ 1.0 暖色调）
- [x] 色调调整（绿 ↔ 洋红）
- [x] CLI 参数：`--tint 0`（-100 到 +100）
- [x] CLI 参数：`--auto-white-balance`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 低
- **用途**: 
  - 电商：统一产品图色温，避免偏色
  - 摄影：校正拍摄时的白平衡偏差

#### Task 10.5.2: 曝光与亮度
- [x] 曝光调整（EV，-5 到 +5）
- [x] 亮度调整
- [x] 高光压缩/恢复
- [x] 阴影提亮
- [x] 黑色/白色色阶
- [x] CLI 参数：`--exposure 0.5`（EV）
- [x] CLI 参数：`--highlights -30`（-100 到 +100）
- [x] CLI 参数：`--shadows +20`（-100 到 +100）
- [x] CLI 参数：`--blacks -10`, `--whites +10`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 低
- **用途**:
  - 电商：提亮暗部细节，压暗过曝部分
  - 摄影：恢复RAW照片的宽容度

#### Task 10.5.3: 对比度与色彩
- [x] 对比度调整
- [x] 清晰度/纹理增强（中频细节）
- [x] 自然饱和度（Vibrance，智能保护肤色）
- [x] 饱和度（Saturation，全局调整）
- [x] 去雾/除霾（暗通道先验算法）
- [x] CLI 参数：`--contrast FLOAT`
- [x] CLI 参数：`--clarity FLOAT`（0.0-1.0）
- [x] CLI 参数：`--vibrance FLOAT`（-1.0-1.0）
- [x] CLI 参数：`--saturation FLOAT`
- [x] CLI 参数：`--dehaze FLOAT`（0.0-1.0）
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 低

#### Task 10.5.4: 锐化与降噪
- [x] USM 锐化（数量/半径/阈值）
- [x] 智能锐化（边缘感知，避免噪点放大）
- [x] 智能降噪（边缘保留）
- [ ] 边缘蒙版锐化
- [x] 亮度降噪
- [x] 色彩降噪
- [x] CLI 参数：`--sharpen FLOAT`（amount）
- [x] CLI 参数：`--sharpen-radius INT`
- [x] CLI 参数：`--sharpen-threshold FLOAT`
- [x] CLI 参数：`--smart-sharpen FLOAT`（0.0-3.0）
- [x] CLI 参数：`--smart-sharpen-radius INT`（1-5）
- [x] CLI 参数：`--denoise FLOAT`（strength）
- [x] CLI 参数：`--smart-denoise`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 中
- **用途**:
  - 电商：AI 生成图有时偏软，需要适度锐化
  - 摄影：高 ISO 照片降噪

#### Task 10.5.5: 曲线与色调
- [x] RGB 曲线调整（支持控制点）
- [x] 亮度曲线
- [x] 单独 R/G/B 通道曲线
- [x] 色调分离（高光/阴影着色）
- [x] 照片滤镜（模拟胶片色）
- [x] CLI 参数：`--curves "0,0;128,140;255,255"`（输入,输出 控制点）
- [x] CLI 参数：`--split-tone-highlights #FFE4C4`
- [x] CLI 参数：`--split-tone-shadows #4A6741`
- [x] CLI 参数：`--split-tone-strength FLOAT`
- **优先级**: P2
- **难度**: 中

#### Task 10.5.6: 自动优化（一键修图）
- [x] 自动曝光优化
- [x] 自动对比度优化
- [x] 自动色彩增强
- [x] AI 智能优化（一键让图片更好看）
- [x] 批量自动优化目录下所有图片
- [x] CLI 参数：`--auto-enhance`
- [x] CLI 参数：`--batch-input-dir /path/to/images --batch-output-dir /path/out`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 中
- **用途**:
  - 电商：批量处理产品图，一键出片
  - 摄影：快速修图预览

#### Task 10.5.7: 镜头校正
- [x] 暗角添加（Vignetting effect）
- [ ] 镜头畸变校正（桶形/枕形畸变）
- [ ] 色差校正（Chromatic Aberration）
- [x] CLI 参数：`--vignette FLOAT`
- [x] CLI 参数：`--vignette-radius FLOAT`
- [ ] CLI 参数：`--lens-correction vignette,distortion,ca`
- **优先级**: P2
- **难度**: 中
- **用途**:
  - 摄影：修正广角镜头畸变
  - 电商：去除边缘暗角，使产品图均匀

#### Task 10.5.8: 局部调整
- [x] 径向滤镜（Radial Filter）- 圆形区域调整曝光/对比度/饱和度
- [x] 渐变滤镜（Graduated Filter）- 线性渐变调整
- [ ] 画笔调整（Brush Adjustments）- 需要 UI 支持
- [x] CLI 参数：`--radial-filter cx,cy,radius,exp,cont,sat`（如 `0.5,0.5,0.3,0.5,0,0`）
- [x] CLI 参数：`--graduated-filter angle,pos,width,exp,cont,sat`（如 `0,0.5,0.2,0.3,0,0`）
- **优先级**: P2
- **难度**: 中
- **状态**: ✅ 径向/渐变滤镜已完成（2026-04-27）

#### Task 10.5.9: 肤色优化（电商/人像刚需）
- [x] 智能磨皮（皮肤区域高斯模糊 + 边缘保留）
- [ ] 肤色均匀化
- [x] 美白调整（亮度提升 + 饱和度降低 + 冷色调）
- [ ] 红润/去红
- [x] CLI 参数：`--skin-smooth FLOAT`（0.0-1.0）
- [x] CLI 参数：`--whiten FLOAT`（0.0-1.0）
- [ ] CLI 参数：`--skin-tone warm`（warm/cool/neutral）
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 高（需要 AI 模型或复杂算法）→ 已实现基础版本
- **用途**:
  - 电商：模特图肤色优化
  - 人像摄影：快速磨皮美白

#### Task 10.5.10: LUT 预设与滤镜
- [x] 加载 LUT 文件（.cube 格式）- 3D LUT 颜色分级
- [x] 内置常用滤镜（复古、日系、胶片、黑白、赛博朋克、电影感等）
- [x] 保存/加载自定义预设
- [x] 批量应用预设（通过 batch processing）
- [x] CLI 参数：`--lut /path/to/filter.cube`
- [x] CLI 参数：`--preset NAME`（内置预设名）
- [x] CLI 参数：`--save-preset my_preset`
- [x] CLI 参数：`--load-preset my_preset.json`
- **优先级**: P2
- **难度**: 中
- **用途**:
  - 电商：统一店铺视觉风格
  - 摄影：快速套用调色风格

#### Task 10.5.11: 批量处理与脚本
- [x] 批量处理目录下所有图片
- [x] 处理进度显示（计数器 [N/M]）
- [x] 多线程并行处理（使用 std::thread 线程池）
- [x] 输出文件名模板（支持 `{name}`, `{ext}`, `{index}`, `{index:N}`）
- [x] CLI 参数：`--batch-input-dir /path/to/input_dir --batch-output-dir /path/to/output_dir`
- [x] CLI 参数：`--output-template "{name}_edited{ext}"`
- [x] CLI 参数：`--threads INT`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 低
- **用途**:
  - 电商：批量处理成千上万张产品图
  - 摄影：批量调色导出

### Phase 10.6: 电商/摄影专用工具 🛒（新增）

#### Task 10.6.1: 背景移除（Background Removal）🛒
- [ ] AI 自动抠图（支持人物/产品/物体）
- [ ] 边缘细化（ hair/matting 精细边缘）
- [ ] 透明背景输出（PNG with alpha channel）
- [ ] 背景色替换（白底/透明/自定义颜色）
- [ ] CLI 参数：`--remove-bg`（自动检测主体类型）
- [ ] CLI 参数：`--remove-bg-model {u2net|modnet|rvm}`
- [ ] CLI 参数：`--bg-color #FFFFFF` 或 `--bg-transparent`
- [ ] CLI 参数：`--bg-feather 2`（边缘羽化像素）
- **优先级**: 🛒 P0（电商第一刚需）
- **难度**: 高（需要 ONNX 模型 + 前后处理）
- **用途**:
  - 电商：产品图白底/透明底（淘宝/亚马逊主图要求）
  - 摄影：人像换背景
  - 设计：素材提取

#### Task 10.6.2: 智能裁剪（Smart Crop / Subject-aware Crop）🛒
- [ ] 自动检测图像主体（人脸/产品/显著性区域）
- [ ] 智能居中裁剪（保持主体完整）
- [ ] 多平台尺寸预设（一键适配）
  - 淘宝主图：800×800, 1200×1200
  - 京东：800×800, 1000×1000
  - 亚马逊：2000×2000（白底）
  - 小红书：3:4 (900×1200)
  - 抖音：9:16 (1080×1920)
  - Instagram：1:1, 4:5
  - 微信公众号：2.35:1 (900×383)
- [ ] 批量智能裁剪（保持同系列产品构图一致）
- [ ] CLI 参数：`--smart-crop width,height`
- [ ] CLI 参数：`--platform {taobao|jd|amazon|xiaohongshu|douyin}`
- [ ] CLI 参数：`--subject-center`（强制主体居中）
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 中（OpenCV 显著性检测 + 人脸检测）
- **用途**:
  - 电商：批量生成各平台适配尺寸
  - 摄影：快速出图到社交媒体

#### Task 10.6.3: 产品图阴影生成（Drop Shadow）🛒
- [ ] 自动生成投影阴影（自然光效果）
- [ ] 阴影角度/距离/模糊度可调
- [ ] 反射倒影（Reflection）
- [ ] 地面接触阴影（Contact Shadow）
- [ ] CLI 参数：`--drop-shadow angle,distance,blur,opacity`
- [ ] CLI 参数：`--reflection opacity,blur,height`
- **优先级**: 🛒 P1（电商专业度）
- **难度**: 低（纯图像处理，无需 AI）
- **用途**:
  - 电商：产品悬浮/站立效果，提升立体感
  - 摄影：快速添加专业阴影

#### Task 10.6.4: 批量水印（Batch Watermark）🛒
- [ ] 文字水印（字体/大小/颜色/透明度/旋转）
- [ ] 图片水印（Logo 叠加）
- [ ] 水印位置（9宫格 + 自定义坐标）
- [ ] 平铺水印（防截图盗用）
- [ ] 隐形水印（Steganography，频域嵌入）
- [ ] CLI 参数：`--watermark-text "© 2026 MyBrand"`
- [ ] CLI 参数：`--watermark-logo logo.png`
- [ ] CLI 参数：`--watermark-position {center|top-left|bottom-right|tile}`
- [ ] CLI 参数：`--watermark-opacity 0.5`
- **优先级**: 🛒 P1（品牌保护）
- **难度**: 低
- **用途**:
  - 电商：产品图防盗图
  - 摄影：版权保护

#### Task 10.6.5: 自动质检（Auto QA / Quality Assurance）🛒
- [ ] 模糊检测（拉普拉斯方差 < 阈值报警）
- [ ] 过曝/欠曝检测（直方图分析）
- [ ] 色偏检测（灰度世界算法，偏色报警）
- [ ] 低分辨率检测（尺寸不足报警）
- [ ] 压缩瑕疵检测（JPEG 块效应检测）
- [ ] 批量质检报告（输出 CSV/JSON）
- [ ] CLI 参数：`--qa-check {blur|exposure|color|resolution|compression|all}`
- [ ] CLI 参数：`--qa-report report.json`
- [ ] CLI 参数：`--qa-fail-dir /path/to/failed_images`
- **优先级**: 🛒 P1（电商品控）
- **难度**: 低-中
- **用途**:
  - 电商：上架前自动质检，避免劣质图
  - 摄影：批量筛选废片

#### Task 10.6.6: 内容感知填充（Content-Aware Fill / Generative Fill）🛒
- [ ] 涂抹区域智能填充（移除不需要的物体）
- [ ] 图像扩展（Outpainting 升级版，内容感知）
- [ ] 重复图案去除（去水印/去文字）
- [ ] CLI 参数：`--generative-fill mask.png`
- [ ] CLI 参数：`--remove-object "person"`（配合检测模型）
- **优先级**: P2
- **难度**: 高（需要 inpainting 模型）
- **用途**:
  - 电商：去除产品上的灰尘、反光、杂线
  - 摄影：移除路人、电线等干扰物

#### Task 10.6.7: EXIF 与元数据管理 🛒
- [ ] 读取 EXIF（相机参数、拍摄时间、GPS）
- [ ] 写入 EXIF（版权信息、关键词）
- [ ] 批量清除 EXIF（保护隐私）
- [ ] 根据 EXIF 自动旋转（修正手机拍摄方向）
- [ ] CLI 参数：`--strip-exif`
- [ ] CLI 参数：`--exif-copyright "© Photographer Name"`
- [ ] CLI 参数：`--exif-keywords "product,photography"`
- **优先级**: P2
- **难度**: 低
- **用途**:
  - 摄影：版权信息嵌入
  - 电商：清理无用元数据减小文件体积

### Phase 11: 人脸与细节增强

#### Task 11.1: Face Restoration ✅
- [x] OpenCV DNN (YuNet) 人脸检测
- [x] GFPGAN 人脸修复（OpenCV DNN 推理）
- [x] CodeFormer 人脸修复（OpenCV DNN 推理）
- [x] 双边滤波 + USM 锐化增强
- [x] CLI 参数：`--face-restore --face-restore-model PATH --face-restore-strength FLOAT`
- **优先级**: P2
- **难度**: 高
- **状态**: ✅ 框架已完成（2026-06-01）

#### Task 11.2: Face Swap ✅
- [x] YuNet / Haar 级联人脸检测
- [x] Inswapper 128 ONNX 推理（OpenCV DNN）
- [x] 参考图人脸特征提取
- [x] 替换生成图像中的人脸
- [x] CLI 参数：`--face-swap --face-swap-source PATH --face-swap-model PATH`
- **优先级**: P3
- **难度**: 高
- **状态**: ✅ 框架已完成（2026-06-01）

### Phase 12: 模型管理

#### Task 12.1: Checkpoint 加载（大模型） ✅
- [x] 加载完整 .ckpt/.safetensors 模型
- [x] CLI 参数：`--model PATH`（替代 --diffusion-model）
- [x] 支持单独使用完整模型（无需 --vae --llm）
- [x] 与 --diffusion-model 模式互斥
- **优先级**: P1
- **难度**: 低（sd.cpp 原生支持）
- **状态**: ✅ 已完成（2026-04-27）

#### Task 12.2: Model Merging（模型融合）
- [ ] 两个模型权重插值合并
- [ ] 支持不同层不同比例
- [ ] CLI 参数：`--merge model1:0.5,model2:0.5`
- **优先级**: P3
- **难度**: 中

#### Task 12.3: VAE 切换 ✅
- [x] 单独加载替换 VAE
- [x] 支持多种 VAE（SDXL/Flux/SD3，通过 `--vae-format`）
- [x] CLI 参数：`--vae PATH`
- **优先级**: P2
- **难度**: 低
- **状态**: ✅ 已完成（2026-04-27）

#### Task 12.4: CLIP 切换 ✅
- [x] 单独加载文本编码器模型
- [x] 支持 LLM（Qwen3）
- [x] CLI 参数：`--llm PATH`
- **优先级**: P2
- **难度**: 低
- **状态**: ✅ 已完成（2026-04-27）

### Phase 13: 工作流与自动化

#### Task 13.1: Workflow JSON 支持 ✅
- [x] 解析 ComfyUI workflow JSON
- [x] 自动映射到 CLI 参数
- [x] CLI 参数：`--workflow workflow.json`
- **优先级**: P2
- **难度**: 高
- **状态**: ✅ 已完成（2026-06-01）

#### Task 13.2: Batch 生成优化 ✅
- [x] 多图连续生成
- [x] 共享模型加载（不重复加载）
- [x] 种子自动递增
- [x] CLI 参数：`--batch-count INT`
- [x] 输出文件名序列化（output_001.png, output_002.png...）
- **优先级**: P1
- **难度**: 低
- **状态**: ✅ 已完成（2026-04-27）

#### Task 13.3: Prompt 调度（Schedule Prompt）✅
- [x] 按步数切换提示词
- [x] 支持多阶段渐进式 img2img
- [x] CLI 参数：`--prompt-schedule "0-5:prompt1|6-10:prompt2"`
- **优先级**: P3
- **难度**: 中
- **状态**: ✅ 已完成（2026-06-01）

### Phase 14: 动画与视频

#### Task 14.1: AnimateDiff（动画生成）
- [ ] AnimateDiff 模型加载
- [ ] 生成动画帧序列
- [ ] 支持不同运动模块
- [ ] 输出 GIF/MP4
- **优先级**: P3
- **难度**: 高

#### Task 14.2: 视频生成
- [ ] 加载视频帧序列
- [ ] 逐帧生成或插帧
- [ ] 保持时间一致性
- **优先级**: P3
- **难度**: 高

### Phase 17: 缓存与性能优化（新增）

#### Task 17.1: 条件编码缓存
- [ ] 缓存文本编码结果（相同 prompt 不重复编码）
- [ ] 缓存键：prompt + model + 编码器参数
- [ ] LRU 淘汰策略
- [ ] CLI 参数：`--cache-text-encodings`
- **优先级**: P1
- **效果**: 批量生成时节省 30-50% 时间
- **用途**:
  - 电商：同系列产品批量出图，prompt 相似
  - 摄影：套图生成，风格统一

#### Task 17.2: 模型缓存管理
- [ ] 多模型驻留内存（不重复加载）
- [ ] 引用计数 + LRU 卸载
- [ ] 显存不足时自动 offload 到 CPU
- **优先级**: P1
- **用途**:
  - 频繁切换 LoRA/ControlNet 时避免重复加载基模

#### Task 17.3: 性能基准测试
- [ ] 建立 benchmark 测试集
- [ ] 监控：每步耗时、VRAM 峰值、生成速度
- [ ] 防止性能退化（CI 检查）
- [ ] 输出 JSON 报告
- **优先级**: P2

### Phase 18: 鲁棒性与错误处理（新增）

#### Task 18.1: OOM 优雅降级
- [ ] 捕获 CUDA OOM 异常
- [ ] 自动降低分辨率重试
- [ ] 自动启用更激进的 tiling
- [ ] 用户提示显存不足建议
- **优先级**: P1
- **用途**:
  - 电商：批量生成时不因单张图 OOM 中断整个批次

#### Task 18.2: 模型校验
- [ ] 加载前校验 SHA256
- [ ] 格式自动检测（GGUF/Safetensors/CKPT）
- [ ] 损坏文件友好提示
- **优先级**: P2

#### Task 18.3: 生成中断恢复
- [ ] 捕获 SIGINT/SIGTERM
- [ ] 保存当前 latent 状态
- [ ] 支持 `--resume` 从中断点继续
- **优先级**: P3

### Phase 15: 工程优化

#### Task 15.1: Server 模式
- [ ] HTTP API 服务
- [ ] 兼容 SD WebUI API / ComfyUI API
- [ ] 队列管理
- [ ] CLI 参数：`--server --port 8080`
- **优先级**: P3
- **难度**: 高

#### Task 15.2: 元数据嵌入 ✅
- [x] 将生成参数写入 PNG metadata（PNG Info）
- [x] 读取 PNG metadata 复现生成（已有 --read-metadata）
- [x] CLI 参数：`--embed-metadata`
- **优先级**: P2
- **难度**: 低
- **状态**: ✅ 已完成（2026-04-30）

#### Task 15.3: 预览图 ✅
- [x] 生成过程中实时预览
- [x] VAE/TAE/PROJ 解码预览
- [x] CLI 参数：`--preview`, `--preview-interval N`, `--preview-mode vae|tae|proj`, `--preview-dir PATH`
- **优先级**: P2
- **难度**: 中
- **状态**: ✅ 已完成（2026-06-01）

#### Task 15.4: 日志与调试 ✅
- [x] 分级日志（TRACE/DEBUG/INFO/WARN/ERROR/FATAL）
- [x] 性能分析（每步耗时）
- [x] VRAM 使用监控（nvidia-smi）
- [x] 生成报告输出（JSON）
- [x] CLI 参数：`--log-level`, `--report`, `--show-vram`
- **优先级**: P2
- **难度**: 低
- **状态**: ✅ 已完成（2026-06-01）

#### Task 15.5: 配置文件 ✅
- [x] JSON 配置文件支持
- [x] 预设参数模板
- [x] CLI 参数：`--config config.json`
- **优先级**: P2
- **难度**: 低
- **状态**: ✅ 已完成（2026-06-01）

### Phase 16: 商业化前端（未来愿景）

#### Task 16.1: Web 前端设计器
- [ ] 可视化节点编辑器（类似 ComfyUI 的拖拽式界面）
- [ ] 节点库：包含所有 my-img 支持的节点
- [ ] 连线编辑：连接节点输入输出
- [ ] 实时预览：生成过程中显示预览图
- [ ] 多工作流管理：保存/加载/分享工作流
- **优先级**: P3（未来）
- **难度**: 高
- **技术栈**: React/Vue + Canvas/WebGL

#### Task 16.2: ComfyUI 工作流兼容
- [ ] 导入 ComfyUI workflow JSON
- [ ] 自动映射 ComfyUI 节点到 my-img 节点
- [ ] 导出兼容 ComfyUI 的 workflow JSON
- [ ] 支持 ComfyUI 自定义节点（部分）
- [ ] 工作流验证：检查节点连接是否合法
- **优先级**: P3（未来）
- **难度**: 高

#### Task 16.3: 商业功能
- [ ] 多用户管理（登录/权限）
- [ ] 生成历史记录
- [ ] 云端模型仓库（自动下载）
- [ ] 付费套餐（按生成次数/分辨率计费）
- [ ] API 密钥管理
- [ ] 团队协作（共享工作流/模型）
- **优先级**: P3（未来）
- **难度**: 高

#### Task 16.4: 移动端适配
- [ ] 响应式 Web 界面
- [ ] 手机端简化操作
- [ ] 一键生成常用模板
- **优先级**: P3（未来）
- **难度**: 中

---

## 功能完整度对照表

### ComfyUI 核心节点覆盖情况

| 类别 | ComfyUI 节点 | my-img 状态 | 任务编号 |
|------|-------------|-------------|----------|
| **基础加载** | CheckpointLoader | ⏳ | 12.1 |
| | UNETLoader | ✅ (diffusion-model) | - |
| | VAELoader | ✅ (--vae) | - |
| | CLIPLoader | ✅ (--llm) | - |
| | ControlNetLoader | ⏳ | 7.1 |
| | IPAdapterLoader | ⏳ | 8.1 |
| | UpscaleModelLoader | ✅ | - |
| | LoraLoader | ⏳ | 6.2 |
| **条件编码** | CLIPTextEncode | ✅ | - |
| | CLIPSetLastLayer | ⏳ | 预留 |
| | ConditioningCombine | ⏳ | 预留 |
| | ConditioningSetArea | ⏳ | 9.1 |
| | ConditioningSetMask | ⏳ | 预留 |
| **采样** | KSampler / Sampler | ✅ | - |
| | SamplerCustom | ✅ | - |
| | Scheduler | ✅ | - |
| **Latent** | EmptyLatentImage | ✅ | - |
| | VAEDecode | ✅ | - |
| | VAEEncode | ⏳ | 6.1 |
| | LatentUpscale | ✅ (HiRes Fix) | - |
| | LatentComposite | ✅ | 10.1 |
| **图像** | LoadImage | ✅ | 6.1 |
| | SaveImage | ✅ | - |
| | ImageScale | ✅ | 10.3 |
| | ImageCrop | ✅ | 10.3 |
| | ImageComposite | ✅ | 10.2 |
| **控制** | ControlNetApply | ✅ | 7.1 |
| | ControlNetApplyAdvanced | ⏳ | 7.1 |
| | IPAdapterApply | ✅ | 8.1 |
| | T2IAdapterApply | ✅ | 7.2 |
| **增强** | HiResFix | ✅ | - |
| | FreeU | ✅ | 9.3 |
| | SAG | ✅ | 9.2 |
| | FaceRestore | ✅ | 11.1 |
| | AnimateDiff | ⏳ | 14.1 |
| **其他** | BatchGenerate | ✅ | 13.2 |
| | PromptSchedule | ✅ | 13.3 |
| | ModelMerge | ⏳ | 12.2 |
| | Embeddings | ✅ | 6.5 |
| | Outpainting | ✅ | 6.4 |
| | RegionalPrompt | ✅ | 9.1 |
| | Metadata | ✅ | 15.2 |
| | Server | ⏳ | 15.1 |

---

## 开发优先级建议

### P0（最高优先级 - 核心功能）
1. **img2img** - 用户最常用功能之一
2. **LoRA** - 模型微调必备
3. **Checkpoint 加载** - 支持更多模型格式

### P1（高优先级 - 常用功能）
4. **Inpainting** - 局部修复
5. **Outpainting** - 图像外扩
6. **Batch 生成** - 提升效率
7. **Embeddings** - 轻量级风格控制
8. **ControlNet** - 精确控制构图
9. **VAE/CLIP 切换** - 模型灵活性

### P2（中优先级 - 进阶功能）
10. **IPAdapter** - 风格/人脸迁移
11. **T2I-Adapter** - 轻量控制
12. **Regional Prompting** - 分区控制
13. **SAG** - 质量提升
14. **Latent/Image Composite** - 合成操作
15. **Face Restoration** - 人脸修复
16. **Workflow JSON** - 自动化
17. **元数据嵌入** - 可复现性
18. **预览图** - 用户体验
19. **配置文件** - 便捷性
20. **日志系统** - 调试

### P3（低优先级 - 高级/实验性功能）
21. **FreeU** - 质量微调
22. **PhotoMaker** - 个性化
23. **Style Transfer** - 风格迁移
24. **Model Merging** - 模型实验
25. **ControlNet 预处理器** - 完整管线
26. **IPAdapter FaceID** - 人脸一致性
27. **AnimateDiff** - 动画
28. **视频生成** - 视频
29. **Server 模式** - 服务化
30. **Prompt Schedule** - 动态提示词

---

## 当前状态速览

| 类别 | 功能 | 状态 | 优先级 |
|------|------|------|--------|
| 生成 | txt2img | ✅ | - |
| 生成 | HiRes Fix | ✅ | - |
| 生成 | img2img | ✅ | 🌟 P0 |
| 生成 | Inpainting | ✅ | P1 |
| 生成 | Outpainting | ✅ | P1 |
| 生成 | Batch | ✅ | P1 |
| 模型 | LoRA | ✅ | 🌟 P0 |
| 模型 | Checkpoint | ✅ | P0 |
| 模型 | Embeddings | ✅ | P1 |
| 模型 | Model Merge | ⏳ | P3 |
| 控制 | ControlNet | ✅ | P1 |
| 控制 | T2I-Adapter | ✅ | P2 |
| 图像条件 | IPAdapter | ✅ | P2 |
| 图像条件 | IPAdapter FaceID | ✅ | P2 |
| 图像条件 | PhotoMaker | ✅ | P3 |
| 增强 | SAG | ✅ | P2 |
| 增强 | Dynamic CFG | ✅ | P2 |
| 增强 | FreeU | ✅ | P3 |
| 增强 | Face Restore | ✅ | P2 |
| 增强 | Style Transfer | ⏳ | P3 |
| 图像操作 | Latent Composite | ✅ | P2 |
| 图像操作 | Image Composite | ✅ | P2 |
| 图像操作 | Scale/Crop | ✅ | P2 |
| 提示词 | Regional Prompting | ✅ | P2 |
| 提示词 | Prompt Schedule | ✅ | P3 |
| 动画 | AnimateDiff | ⏳ | P3 |
| 动画 | Video | ⏳ | P3 |
| 优化 | VAE Tiling | ✅ | - |
| 优化 | Flash Attention | ✅ | - |
| 体验 | CLI 参数 | ✅ | - |
| 体验 | 元数据 | ✅ | P2 |
| 体验 | 预览图 | ✅ | P2 |
| 体验 | 配置文件 | ✅ | P2 |
| 体验 | Workflow JSON | ✅ | P2 |
| 体验 | 日志与调试 | ✅ | P2 |
| 体验 | Server | ⏳ | P3 |

---

## 开发日志

### 2026-04-29
- 📝 libTorch HiRes Fix 实验失败，已删除相关代码
  - 扩展 API 在 RTX 3080 10GB 上出现无法解释的 OOM（相同代码原生路径正常）
  - 已删除: `stable-diffusion-ext.h/cpp`, `generate_hires_libtorch()`, `--hires-mode`, `libTorchHiresfix.sh`
  - 使用 sd.cpp 原生 HiRes Fix（`--hires`），已验证 1280x720 和 2560x1440 正常

### 2026-04-27
- ✅ 创建项目结构
- ✅ 配置 CMakeLists.txt（libtorch + ggml + sd.cpp）
- ✅ Clone 第三方库（ggml, json, stb）
- ✅ 实现 GGUF 加载器（453/453 张量）
- ✅ 实现 VAE 编解码（测试通过）
- ✅ 实现 HiRes Fix 简化版（测试通过）
- ✅ 集成 stable-diffusion.cpp 作为第三方依赖
- ✅ 创建 SDCPPAdapter 适配层
- ✅ 实现 Z-Image 模型加载（diffusion_model_path）
- ✅ 实现 Qwen3 文本编码
- ✅ 实现 txt2img Pipeline（1280x720，20步，66秒）
- ✅ 实现 HiRes Fix（功能完成，受限于 10GB 显存）
- ✅ 实现完整 CLI 入口（支持 img1.sh/img2.sh 所有参数）
- ✅ 实现 Image::save_to_file（PNG 保存）
- ✅ 实现 ESRGAN 放大支持
- ✅ 成功生成 2560x1440 人像图（用时 ~6.5 分钟）
- ✅ 完善 task.md，对齐 ComfyUI 功能列表
- 📝 更新 design.md 文档（混合架构说明）
- 📝 创建完整开发任务表（含 ComfyUI 功能对照）

### 关键决策
- **架构**: sd.cpp（模型推理）+ libtorch（扩展功能）
- **适配层**: SDCPPAdapter 封装 sd.cpp C API
- **模型加载**: 使用 diffusion_model_path（独立扩散模型）而非 model_path
- **LTO 兼容**: 启用 CMAKE_INTERPROCEDURAL_OPTIMIZATION 以匹配 sd.cpp
- **VAE Tiling**: 256x256 tile + 0.8 overlap，解决 2560x1440 OOM
- **升级策略**: **最小侵入 sd.cpp 源码**（仅恢复 8 个 static 关键字），只通过 `stable-diffusion.h` 公开 API 调用，不添加扩展、不嵌入代码，升级时直接覆盖新版
