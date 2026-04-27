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

### CLI
- [x] 完整命令行参数支持
- [x] 图像保存（PNG）
- [x] ESRGAN 放大
- [x] 进度显示

---

## 待实现功能清单

### Phase 6: 核心功能扩展

#### Task 6.1: img2img（图像到图像）🌟
- [ ] CLI 参数：`--init-img PATH`
- [ ] CLI 参数：`--strength FLOAT`（0.0-1.0，默认 0.75）
- [ ] 加载参考图（PNG/JPG）
- [ ] VAE encode 为 latent
- [ ] 根据 strength 加噪
- [ ] 去噪生成
- [ ] 参考：sd.cpp `init_image` 参数
- **优先级**: 🌟 高
- **难度**: 低（sd.cpp 原生支持）

#### Task 6.2: LoRA 集成 🌟
- [ ] CLI 参数：`--lora "PATH:weight"`（支持多个）
- [ ] LoRA 权重加载（.safetensors）
- [ ] 权重注入扩散模型
- [ ] 高噪声/低噪声 LoRA 支持
- [ ] 参考：sd.cpp `sd_lora_t`
- **优先级**: 🌟 高
- **难度**: 中

#### Task 6.3: Inpainting（局部重绘）
- [ ] CLI 参数：`--mask PATH`
- [ ] 加载 mask 图像
- [ ] masked latent 生成
- [ ] 保留未遮罩区域
- **优先级**: 中
- **难度**: 中

#### Task 6.4: ControlNet
- [ ] ControlNet 模型加载
- [ ] CLI 参数：`--control-net PATH`
- [ ] CLI 参数：`--control-image PATH`
- [ ] CLI 参数：`--control-strength FLOAT`
- [ ] Canny/Depth/Lineart/OpenPose 预处理
- [ ] 参考：sd.cpp `control_image`, `control_strength`
- **优先级**: 中
- **难度**: 高

### Phase 7: 高级功能

#### Task 7.1: IPAdapter（图像提示词）
- [ ] CLIP Vision 模型加载
- [ ] 图像特征提取
- [ ] IPAdapter 模型加载
- [ ] 注意力注入
- [ ] CLI 参数：`--ipadapter PATH`
- [ ] CLI 参数：`--ipadapter-image PATH`
- **优先级**: 中
- **难度**: 高

#### Task 7.2: PhotoMaker
- [ ] PhotoMaker 模型加载
- [ ] ID 图像集处理
- [ ] 个性化生成
- **优先级**: 低
- **难度**: 高

#### Task 7.3: FreeU / FreeU_V2
- [ ] 在 UNet 中注入 FreeU 参数
- [ ] 提升图像质量和细节
- **优先级**: 低
- **难度**: 中

### Phase 8: 工作流与自动化

#### Task 8.1: Workflow JSON 支持
- [ ] 解析 ComfyUI workflow JSON
- [ ] 自动映射到 CLI 参数
- [ ] 批量生成
- **优先级**: 中
- **难度**: 高

#### Task 8.2: Prompt 调度（Schedule Prompt）
- [ ] 按步数切换提示词
- [ ] 动态调整 CFG
- **优先级**: 低
- **难度**: 中

#### Task 8.3: Batch 生成优化
- [ ] 多图连续生成
- [ ] 共享模型加载（不重复加载）
- [ ] 种子自动递增
- **优先级**: 中
- **难度**: 低

### Phase 9: 工程优化

#### Task 9.1: Server 模式
- [ ] HTTP API 服务
- [ ] 兼容 SD WebUI API
- [ ] 队列管理
- **优先级**: 低
- **难度**: 高

#### Task 9.2: 图像预处理节点
- [ ] Canny 边缘检测
- [ ] Depth 估计
- [ ] Lineart 提取
- [ ] OpenPose 姿态检测
- **优先级**: 低
- **难度**: 高

#### Task 9.3: 模型管理
- [ ] VAE 切换
- [ ] 模型热切换
- [ ] 权重类型自动检测
- **优先级**: 低
- **难度**: 中

### Phase 10: 质量与体验

#### Task 10.1: 元数据嵌入
- [ ] 将生成参数写入 PNG metadata
- [ ] 读取 PNG metadata 复现生成
- **优先级**: 低
- **难度**: 低

#### Task 10.2: 预览图
- [ ] 生成过程中实时预览
- [ ] TAE/TAESD 快速解码预览
- **优先级**: 低
- **难度**: 中

#### Task 10.3: 日志与调试
- [ ] 分级日志（DEBUG/INFO/WARN/ERROR）
- [ ] 性能分析（每步耗时）
- [ ] VRAM 使用监控
- **优先级**: 低
- **难度**: 低

---

## 开发优先级建议

### P0（最高优先级）
1. **img2img** - 用户最常用功能之一
2. **LoRA** - 模型微调必备

### P1（高优先级）
3. **Inpainting** - 局部修复
4. **Batch 生成优化** - 提升效率

### P2（中优先级）
5. **ControlNet** - 精确控制构图
6. **IPAdapter** - 风格迁移
7. **Workflow JSON** - 自动化

### P3（低优先级）
8. **Server 模式** - 服务化部署
9. **图像预处理** - 完整管线
10. **PhotoMaker/FreeU** - 锦上添花

---

## 当前状态速览

| 类别 | 功能 | 状态 | 优先级 |
|------|------|------|--------|
| 生成 | txt2img | ✅ | - |
| 生成 | HiRes Fix | ✅ | - |
| 生成 | img2img | ⏳ | 🌟 P0 |
| 生成 | Inpainting | ⏳ | P1 |
| 模型 | LoRA | ⏳ | 🌟 P0 |
| 模型 | ControlNet | ⏳ | P2 |
| 模型 | IPAdapter | ⏳ | P2 |
| 优化 | VAE Tiling | ✅ | - |
| 优化 | Flash Attention | ✅ | - |
| 优化 | Batch 生成 | ⏳ | P1 |
| 体验 | CLI 完整参数 | ✅ | - |
| 体验 | Workflow JSON | ⏳ | P2 |
| 体验 | Server 模式 | ⏳ | P3 |

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
- 📝 更新 design.md 文档（混合架构说明）
- 📝 创建完整开发任务表

### 关键决策
- **架构**: sd.cpp（模型推理）+ libtorch（扩展功能）
- **适配层**: SDCPPAdapter 封装 sd.cpp C API
- **模型加载**: 使用 diffusion_model_path（独立扩散模型）而非 model_path
- **LTO 兼容**: 启用 CMAKE_INTERPROCEDURAL_OPTIMIZATION 以匹配 sd.cpp
- **VAE Tiling**: 256x256 tile + 0.8 overlap，解决 2560x1440 OOM
