# my-img 开发任务表

> **目标**: 纯 C++ 版 ComfyUI，使用 sd.cpp + libtorch 后端，支持 Z-Image（GGUF）
> **架构**: sd.cpp（模型推理）+ libtorch（扩展功能）
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
- [ ] 在 prompt 中识别 `embedding:name` 语法
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

#### Task 7.2: T2I-Adapter
- [ ] 轻量级条件控制（比 ControlNet 更省显存）
- [ ] 支持 Sketch/Keypose/Segmentation
- [ ] CLI 参数：`--t2i-adapter PATH`
- **优先级**: P2
- **难度**: 高

#### Task 7.3: ControlNet 预处理器
- [ ] Canny 边缘检测
- [ ] Depth 深度估计（MiDaS/DPT）
- [ ] Lineart 线条提取
- [ ] OpenPose 姿态检测
- [ ] Normal Map 法线贴图
- [ ] Scribble 涂鸦识别
- **优先级**: P2
- **难度**: 高（需要集成 OpenCV/ONNX）

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

#### Task 9.1: IPAdapter（图像提示词）
- [ ] CLIP Vision 模型加载
- [ ] 图像特征提取
- [ ] IPAdapter 模型加载
- [ ] 注意力注入
- [ ] CLI 参数：`--ipadapter PATH`
- [ ] CLI 参数：`--ipadapter-image PATH`
- [ ] CLI 参数：`--ipadapter-weight FLOAT`
- **优先级**: P2
- **难度**: 高

#### Task 8.2: IPAdapter FaceID
- [ ] 人脸识别特征提取
- [ ] 保持人物一致性
- [ ] 支持多张参考图
- **优先级**: P2
- **难度**: 高

#### Task 8.3: PhotoMaker
- [ ] PhotoMaker 模型加载
- [ ] ID 图像集处理
- [ ] 个性化生成
- [ ] CLI 参数：`--photo-maker PATH`
- **优先级**: P3
- **难度**: 高

### Phase 9: 高级采样与质量

#### Task 9.1: Regional Prompting（分区提示词）
- [ ] 图像分区（上/下/左/右/中心）
- [ ] 不同区域应用不同提示词
- [ ] 区域权重控制
- [ ] CLI 参数：`--regional "top:prompt1,bottom:prompt2"`
- **优先级**: P2
- **难度**: 中

#### Task 9.2: Self-Attention Guidance (SAG)
- [ ] 提升图像细节和构图
- [ ] 无需额外模型
- [ ] CLI 参数：`--sag-scale FLOAT`
- **优先级**: P2
- **难度**: 中

#### Task 9.3: FreeU / FreeU_V2
- [ ] 在 UNet 中注入 FreeU 参数
- [ ] 提升图像质量和细节
- [ ] CLI 参数：`--freeu`（启用 FreeU）
- **优先级**: P3
- **难度**: 中

#### Task 9.4: Style Transfer
- [ ] 加载风格参考图
- [ ] 风格特征提取与注入
- [ ] 与 IPAdapter 结合使用
- **优先级**: P3
- **难度**: 高

### Phase 10: Latent & 图像操作

#### Task 10.1: Latent Composite（Latent 合成）
- [ ] 多个 latent 按位置合成
- [ ] 支持 mask 控制融合区域
- [ ] 用于多区域生成
- **优先级**: P2
- **难度**: 中

#### Task 10.2: Image Composite（图像合成）
- [ ] 像素级图像合成
- [ ] Alpha 通道混合
- [ ] 用于 Inpainting 结果融合
- **优先级**: P2
- **难度**: 低

#### Task 10.3: Image Scale & Crop
- [x] 图像缩放（nearest/bilinear/bicubic 插值）
- [ ] 图像裁剪（指定坐标和尺寸）
  - CLI 参数：`--crop x,y,width,height`（像素坐标）
  - CLI 参数：`--crop-center width,height`（居中裁剪）
  - CLI 参数：`--crop-ratio width:height`（按比例裁剪，如 16:9, 4:3, 1:1）
- [x] 图像翻转（水平/垂直）
- [x] 图像旋转（90°/180°/270°）
- [x] 图像格式转换（PNG/BMP/TGA，自动检测扩展名）
- **优先级**: P2
- **难度**: 低
- **用途**: 
  - 电商：统一商品图尺寸
  - 社交媒体：裁剪为平台要求比例（小红书 3:4、抖音 9:16、淘宝 1:1）
  - 批量处理：统一所有图片尺寸

### Phase 10.5: 摄影后期与图像调整（电商/摄影刚需）

#### Task 10.5.1: 色温与白平衡
- [x] 色温调整（RGB 通道平衡，-1.0 冷色调 ↔ 1.0 暖色调）
- [ ] 色调调整（绿 ↔ 洋红）
- [x] 自动优化（auto-enhance，包含自动曝光/对比度/饱和度）
- [x] CLI 参数：`--temperature FLOAT`（-1.0 到 1.0）
- [ ] CLI 参数：`--tint 0`（-100 到 +100）
- [ ] CLI 参数：`--auto-white-balance`
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
- [ ] 黑色/白色色阶
- [x] CLI 参数：`--exposure 0.5`（EV）
- [x] CLI 参数：`--highlights -30`（-100 到 +100）
- [x] CLI 参数：`--shadows +20`（-100 到 +100）
- [ ] CLI 参数：`--blacks -10`, `--whites +10`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 低
- **用途**:
  - 电商：提亮暗部细节，压暗过曝部分
  - 摄影：恢复RAW照片的宽容度

#### Task 10.5.3: 对比度与色彩
- [x] 对比度调整
- [ ] 清晰度/纹理增强（中频细节）
- [ ] 自然饱和度（Vibrance，智能保护肤色）
- [x] 饱和度（Saturation，全局调整）
- [ ] 去雾/除霾
- [x] CLI 参数：`--contrast FLOAT`
- [ ] CLI 参数：`--clarity +20`
- [ ] CLI 参数：`--vibrance +15`
- [x] CLI 参数：`--saturation FLOAT`
- [ ] CLI 参数：`--dehaze +30`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 低

#### Task 10.5.4: 锐化与降噪
- [x] USM 锐化（数量/半径/阈值）
- [x] 智能降噪（边缘保留）
- [ ] 边缘蒙版锐化
- [ ] 亮度降噪
- [ ] 色彩降噪
- [x] CLI 参数：`--sharpen FLOAT`（amount）
- [x] CLI 参数：`--sharpen-radius INT`
- [x] CLI 参数：`--sharpen-threshold FLOAT`
- [x] CLI 参数：`--denoise FLOAT`（strength）
- [x] CLI 参数：`--smart-denoise`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 中
- **用途**:
  - 电商：AI 生成图有时偏软，需要适度锐化
  - 摄影：高 ISO 照片降噪

#### Task 10.5.5: 曲线与色调
- [x] RGB 曲线调整（支持控制点）
- [ ] 亮度曲线
- [ ] 单独 R/G/B 通道曲线
- [ ] 色调分离（高光/阴影着色）
- [ ] 照片滤镜（模拟胶片色）
- [x] CLI 参数：`--curves "0,0;128,140;255,255"`（输入,输出 控制点）
- [ ] CLI 参数：`--split-tone-highlights #FFE4C4`
- [ ] CLI 参数：`--split-tone-shadows #4A6741`
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
- [ ] 加载 LUT 文件（.cube, .3dl）
- [x] 内置常用滤镜（复古、日系、胶片、黑白、赛博朋克、电影感等）
- [ ] 保存/加载自定义预设
- [x] 批量应用预设（通过 batch processing）
- [ ] CLI 参数：`--lut /path/to/filter.cube`
- [x] CLI 参数：`--preset NAME`（内置预设名）
- [ ] CLI 参数：`--save-preset my_preset`
- **优先级**: P2
- **难度**: 中
- **用途**:
  - 电商：统一店铺视觉风格
  - 摄影：快速套用调色风格

#### Task 10.5.11: 批量处理与脚本
- [ ] 批量处理目录下所有图片
- [ ] 处理进度显示
- [ ] 多线程并行处理
- [ ] 输出文件名模板（如 `{filename}_edited.{ext}`）
- [ ] CLI 参数：`--batch /path/to/input_dir /path/to/output_dir`
- [ ] CLI 参数：`--output-template "{name}_edited{ext}"`
- [ ] CLI 参数：`--threads 4`
- **优先级**: 🛒 P1（电商刚需）
- **难度**: 低
- **用途**:
  - 电商：批量处理成千上万张产品图
  - 摄影：批量调色导出

### Phase 11: 人脸与细节增强

#### Task 11.1: Face Restoration
- [ ] GFPGAN 人脸修复
- [ ] CodeFormer 人脸修复
- [ ] CLI 参数：`--face-restore gfgan/codeformer`
- **优先级**: P2
- **难度**: 高（需要集成 GFPGAN/CodeFormer 模型）

#### Task 11.2: Face Swap
- [ ] 提取参考人脸特征
- [ ] 替换生成图像中的人脸
- [ ] 保持表情和光照一致
- **优先级**: P3
- **难度**: 高

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

#### Task 12.3: VAE 切换
- [ ] 单独加载替换 VAE
- [ ] 支持多种 VAE（SDXL/Flux/SD3）
- [ ] CLI 参数：`--vae PATH`
- **优先级**: P2
- **难度**: 低

#### Task 12.4: CLIP 切换
- [ ] 单独加载 CLIP 模型
- [ ] 支持 CLIP-L/CLIP-G/T5/LLM
- [ ] CLI 参数：`--clip-l PATH`, `--clip-g PATH`, `--t5xxl PATH`
- **优先级**: P2
- **难度**: 低

### Phase 13: 工作流与自动化

#### Task 13.1: Workflow JSON 支持
- [ ] 解析 ComfyUI workflow JSON
- [ ] 自动映射到 CLI 参数
- [ ] CLI 参数：`--workflow workflow.json`
- **优先级**: P2
- **难度**: 高

#### Task 13.2: Batch 生成优化 ✅
- [x] 多图连续生成
- [x] 共享模型加载（不重复加载）
- [x] 种子自动递增
- [x] CLI 参数：`--batch-count INT`
- [x] 输出文件名序列化（output_001.png, output_002.png...）
- **优先级**: P1
- **难度**: 低
- **状态**: ✅ 已完成（2026-04-27）

#### Task 13.3: Prompt 调度（Schedule Prompt）
- [ ] 按步数切换提示词
- [ ] 动态调整 CFG
- [ ] CLI 参数：`--prompt-schedule "0-10:prompt1,11-20:prompt2"`
- **优先级**: P3
- **难度**: 中

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

### Phase 15: 工程优化

#### Task 15.1: Server 模式
- [ ] HTTP API 服务
- [ ] 兼容 SD WebUI API / ComfyUI API
- [ ] 队列管理
- [ ] CLI 参数：`--server --port 8080`
- **优先级**: P3
- **难度**: 高

#### Task 15.2: 元数据嵌入
- [ ] 将生成参数写入 PNG metadata（PNG Info）
- [ ] 读取 PNG metadata 复现生成
- [ ] CLI 参数：`--embed-metadata`
- **优先级**: P2
- **难度**: 低

#### Task 15.3: 预览图
- [ ] 生成过程中实时预览
- [ ] TAE/TAESD 快速解码预览
- [ ] CLI 参数：`--preview-method vae`
- **优先级**: P2
- **难度**: 中

#### Task 15.4: 日志与调试
- [ ] 分级日志（DEBUG/INFO/WARN/ERROR）
- [ ] 性能分析（每步耗时）
- [ ] VRAM 使用监控
- [ ] 生成报告输出
- **优先级**: P2
- **难度**: 低

#### Task 15.5: 配置文件
- [ ] YAML/JSON 配置文件支持
- [ ] 预设参数模板
- [ ] CLI 参数：`--config config.yaml`
- **优先级**: P2
- **难度**: 低

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
| | LatentComposite | ⏳ | 10.1 |
| **图像** | LoadImage | ⏳ | 6.1 |
| | SaveImage | ✅ | - |
| | ImageScale | ⏳ | 10.3 |
| | ImageCrop | ⏳ | 10.3 |
| | ImageComposite | ⏳ | 10.2 |
| **控制** | ControlNetApply | ⏳ | 7.1 |
| | ControlNetApplyAdvanced | ⏳ | 7.1 |
| | IPAdapterApply | ⏳ | 8.1 |
| | T2IAdapterApply | ⏳ | 7.2 |
| **增强** | HiResFix | ✅ | - |
| | FreeU | ⏳ | 9.3 |
| | SAG | ⏳ | 9.2 |
| | FaceRestore | ⏳ | 11.1 |
| | AnimateDiff | ⏳ | 14.1 |
| **其他** | BatchGenerate | ⏳ | 13.2 |
| | PromptSchedule | ⏳ | 13.3 |
| | ModelMerge | ⏳ | 12.2 |
| | Embeddings | ⏳ | 6.5 |
| | Outpainting | ⏳ | 6.4 |
| | RegionalPrompt | ⏳ | 9.1 |
| | Metadata | ⏳ | 15.2 |
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
| 生成 | img2img | ⏳ | 🌟 P0 |
| 生成 | Inpainting | ⏳ | P1 |
| 生成 | Outpainting | ⏳ | P1 |
| 生成 | Batch | ⏳ | P1 |
| 模型 | LoRA | ⏳ | 🌟 P0 |
| 模型 | Checkpoint | ⏳ | P0 |
| 模型 | Embeddings | ⏳ | P1 |
| 模型 | Model Merge | ⏳ | P3 |
| 控制 | ControlNet | ⏳ | P1 |
| 控制 | T2I-Adapter | ⏳ | P2 |
| 图像条件 | IPAdapter | ⏳ | P2 |
| 图像条件 | IPAdapter FaceID | ⏳ | P2 |
| 图像条件 | PhotoMaker | ⏳ | P3 |
| 增强 | SAG | ⏳ | P2 |
| 增强 | FreeU | ⏳ | P3 |
| 增强 | Face Restore | ⏳ | P2 |
| 增强 | Style Transfer | ⏳ | P3 |
| 图像操作 | Latent Composite | ⏳ | P2 |
| 图像操作 | Image Composite | ⏳ | P2 |
| 图像操作 | Scale/Crop | ⏳ | P2 |
| 提示词 | Regional Prompting | ⏳ | P2 |
| 提示词 | Prompt Schedule | ⏳ | P3 |
| 动画 | AnimateDiff | ⏳ | P3 |
| 动画 | Video | ⏳ | P3 |
| 优化 | VAE Tiling | ✅ | - |
| 优化 | Flash Attention | ✅ | - |
| 体验 | CLI 参数 | ✅ | - |
| 体验 | 元数据 | ⏳ | P2 |
| 体验 | 预览图 | ⏳ | P2 |
| 体验 | 配置文件 | ⏳ | P2 |
| 体验 | Workflow JSON | ⏳ | P2 |
| 体验 | Server | ⏳ | P3 |

---

## 开发日志

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
