#pragma once

#include <cstring>
#include <vector>
#include <memory>
#include <functional>
#include <string>

// 包含 stable-diffusion.cpp 的头文件
#include <stable-diffusion.h>

namespace myimg {

// 图像数据结构
struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> data;

    bool empty() const { return data.empty(); }
    size_t size() const { return width * height * channels; }

    // 保存图像到文件 (支持 PNG/BMP/TGA/JPG，自动检测扩展名)
    bool save_to_file(const std::string& path) const;
};

// JPEG 质量设置 (1-100, 默认 95)
void set_jpeg_quality(int quality);
int get_jpeg_quality();

// 采样方法
enum class SampleMethod {
    Euler,
    EulerAncestral,
    Heun,
    DPM2,
    DPMPP2S_A,
    DPMPP2M,
    DPMPP2Mv2,
    IPNDM,
    IPNDM_V,
    LCM,
    DDIM_Trailing,
    TCD,
    RES_Multistep,
    RES_2S,
    ER_SDE,
};

// 调度器
enum class Scheduler {
    Discrete,
    Karras,
    Exponential,
    AYS,
    GITS,
    SGM_Uniform,
    Simple,
    Smoothstep,
    KL_Optimal,
    LCM,
    Bong_Tangent,
};

// 预测类型
enum class PredictionType {
    EPS,
    V,
    EDM_V,
    Flow,
    Flux_Flow,
    Flux2_Flow,
};

// LoRA 配置
struct LoRAConfig {
    std::string path;
    float multiplier = 1.0f;
    bool is_high_noise = false;
};

// HiRes Upscaler 类型
enum class HiresUpscaler {
    Latent,                    // 默认，latent 空间放大
    LatentNearest,
    LatentNearestExact,
    LatentAntialiased,
    LatentBicubic,
    LatentBicubicAntialiased,
    Lanczos,
    Nearest,
    Model,                     // 使用外部模型
};

// 生成参数
struct GenerationParams {
    // 模型路径
    std::string model_path;           // 主模型 (GGUF)
    std::string vae_path;             // VAE (可选)
    std::string clip_l_path;          // CLIP-L (可选)
    std::string clip_g_path;          // CLIP-G (可选)
    std::string t5xxl_path;           // T5XXL (可选)
    std::string llm_path;             // LLM 路径 (Z-Image 用)
    std::string diffusion_model_path; // 扩散模型路径 (可选)
    
    // 生成设置
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 512;
    int clip_skip = -1;
    
    // 采样设置
    SampleMethod sample_method = SampleMethod::Euler;
    Scheduler scheduler = Scheduler::Simple;
    int sample_steps = 20;
    float cfg_scale = 7.5f;
    float eta = 0.0f;
    float flow_shift = 0.0f;  // Flow prediction 的 shift (0 = use model default)
    
    // 种子和批次
    int64_t seed = -1;  // -1 表示随机
    int batch_count = 1;
    
    // img2img / Inpainting
    float strength = 1.0f;  // 1.0 = txt2img, <1.0 = img2img
    Image init_image;
    Image mask_image;  // 白色=重绘, 黑色=保留
    
    // ControlNet
    std::string control_net_path;
    Image control_image;
    float control_strength = 0.9f;
    
    // LoRA
    std::vector<LoRAConfig> loras;
    
    // HiRes Fix
    bool enable_hires = false;
    int hires_width = 1024;
    int hires_height = 1024;
    float hires_strength = 0.5f;
    int hires_sample_steps = 20;
    HiresUpscaler hires_upscaler = HiresUpscaler::Latent;
    float hires_scale = 2.0f;
    std::string hires_model_path;  // 外部放大模型路径（upscaler=Model 时使用）
    int hires_tile_size = 128;     // 放大 tile 大小
    
    // VAE Tiling
    bool vae_tiling = false;
    int vae_tile_size_x = 256;
    int vae_tile_size_y = 256;
    float vae_tile_overlap = 0.5f;
    
    // Embeddings
    std::string embedding_dir;
    
    // 系统设置
    int n_threads = -1;  // -1 = 自动
    bool offload_params_to_cpu = false;
    bool enable_mmap = true;
    bool flash_attn = false;
    
    // FreeU 参数
    bool freeu_enabled = false;
    float freeu_b1 = 1.3f;
    float freeu_b2 = 1.4f;
    float freeu_s1 = 0.9f;
    float freeu_s2 = 0.2f;

    // SAG 参数
    bool sag_enabled = false;
    float sag_scale = 1.0f;

    // Dynamic CFG 参数
    bool dynamic_cfg_enabled = false;
    float dynamic_cfg_percentile = 1.0f;
    float dynamic_cfg_mimic_scale = 10.0f;
    float dynamic_cfg_threshold_percentile = 1.0f;

    // 类型设置
    std::string wtype = "default";  // 权重类型: "f32", "f16", "q4_0", "q5_k", etc.
};

// 进度回调
using ProgressCallback = std::function<void(int step, int steps, float time)>;

// SDCPP 适配器类
class SDCPPAdapter {
public:
    SDCPPAdapter();
    ~SDCPPAdapter();
    
    // 禁止拷贝，允许移动
    SDCPPAdapter(const SDCPPAdapter&) = delete;
    SDCPPAdapter& operator=(const SDCPPAdapter&) = delete;
    SDCPPAdapter(SDCPPAdapter&&) noexcept;
    SDCPPAdapter& operator=(SDCPPAdapter&&) noexcept;
    
    // 初始化和加载
    bool initialize(const GenerationParams& params);
    bool is_initialized() const { return ctx_ != nullptr; }
    
    // 文本编码 (用于获取 conditioning)
    // 返回文本条件，可用于手动控制生成流程
    std::vector<float> encode_prompt(const std::string& prompt, int clip_skip = -1);
    
    // 生成图像 (完整流程)
    std::vector<Image> generate(const GenerationParams& params);
    
    // 生成单张图像
    Image generate_single(const GenerationParams& params);
    
    // 设置回调
    void set_progress_callback(ProgressCallback callback);
    
    // 工具函数
    static std::vector<std::string> get_available_sample_methods();
    static std::vector<std::string> get_available_schedulers();
    static std::string get_version();
    static std::string get_commit();
    
    // 图像转换工具
    static Image sd_image_to_image(const sd_image_t& sd_img);
    static sd_image_t image_to_sd_image(const Image& img);
    
    // ESRGAN 放大
    static Image upscale_with_esrgan(const Image& image, const std::string& model_path, int repeats = 1, int tile_size = 128);
    
private:
    sd_ctx_t* ctx_ = nullptr;
    ProgressCallback progress_callback_;
    
    // 内部辅助函数
    bool load_model(const GenerationParams& params);
    static void progress_callback_wrapper(int step, int steps, float time, void* data);
};

} // namespace myimg
