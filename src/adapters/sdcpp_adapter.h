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
    int jpeg_quality = 95;  // JPEG 质量 (1-100)
    std::vector<uint8_t> data;

    bool empty() const { return data.empty(); }
    size_t size() const { return width * height * channels; }

    // 保存图像到文件 (支持 PNG/BMP/TGA/JPG，自动检测扩展名)
    bool save_to_file(const std::string& path) const;
};

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
    EulerCfgPP,
    EulerACfgPP,
    EulerGE,
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
    LTX2,
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
    
    // 类型设置
    std::string wtype = "default";  // 权重类型: "f32", "f16", "q4_0", "q5_k", etc.

    // 显存限制（GB, 0 = 禁用, -1 = 自动）
    float max_vram = 0.0f;

    // VAE 格式 (auto, flux, sd3, flux2)
    std::string vae_format = "auto";

    // 后端选择 (nullptr = 默认)
    std::string backend;
    std::string params_backend;

    // 音频 VAE 路径 (视频生成用)
    std::string audio_vae_path;

    // Embeddings connectors 路径
    std::string embeddings_connectors_path;

    // 采样器额外参数
    std::string extra_sample_args;

    // VAE 时间 tiling (视频生成用)
    bool vae_temporal_tiling = false;

    // VAE tiling 额外参数
    std::string extra_tiling_args;

    // FreeU 参数 (当前上游版本暂不支持，保留字段用于向后兼容)
    bool freeu_enabled = false;
    float freeu_b1 = 1.3f;
    float freeu_b2 = 1.4f;
    float freeu_s1 = 0.9f;
    float freeu_s2 = 0.2f;

    // SAG 参数 (当前上游版本暂不支持，保留字段用于向后兼容)
    bool sag_enabled = false;
    float sag_scale = 1.0f;
};

// sd_image_t 的 RAII 封装（避免裸 malloc/free）
class SDImageGuard {
public:
    SDImageGuard() = default;
    explicit SDImageGuard(sd_image_t img) : img_(img) {}
    
    ~SDImageGuard() {
        free_data();
    }
    
    // 禁止拷贝
    SDImageGuard(const SDImageGuard&) = delete;
    SDImageGuard& operator=(const SDImageGuard&) = delete;
    
    // 允许移动
    SDImageGuard(SDImageGuard&& other) noexcept : img_(other.img_) {
        other.img_.data = nullptr;
    }
    
    SDImageGuard& operator=(SDImageGuard&& other) noexcept {
        if (this != &other) {
            free_data();
            img_ = other.img_;
            other.img_.data = nullptr;
        }
        return *this;
    }
    
    sd_image_t* get() { return &img_; }
    const sd_image_t* get() const { return &img_; }
    
    sd_image_t release() {
        sd_image_t tmp = img_;
        img_.data = nullptr;
        return tmp;
    }
    
    bool empty() const { return img_.data == nullptr; }
    
private:
    void free_data() {
        if (img_.data) {
            free(img_.data);
            img_.data = nullptr;
        }
    }
    
    sd_image_t img_ = {};
};

// sd_image_t 数组的 RAII 封装（用于 generate_image 返回的数组）
class SDImageArrayGuard {
public:
    SDImageArrayGuard() = default;
    explicit SDImageArrayGuard(sd_image_t* ptr) : ptr_(ptr) {}
    
    ~SDImageArrayGuard() {
        free_array();
    }
    
    // 禁止拷贝
    SDImageArrayGuard(const SDImageArrayGuard&) = delete;
    SDImageArrayGuard& operator=(const SDImageArrayGuard&) = delete;
    
    // 允许移动
    SDImageArrayGuard(SDImageArrayGuard&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    SDImageArrayGuard& operator=(SDImageArrayGuard&& other) noexcept {
        if (this != &other) {
            free_array();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    sd_image_t* get() { return ptr_; }
    const sd_image_t* get() const { return ptr_; }
    
    sd_image_t* release() {
        sd_image_t* tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }
    
    bool empty() const { return ptr_ == nullptr; }
    
    sd_image_t& operator[](size_t idx) { return ptr_[idx]; }
    const sd_image_t& operator[](size_t idx) const { return ptr_[idx]; }
    
private:
    void free_array() {
        if (ptr_) {
            free(ptr_);
            ptr_ = nullptr;
        }
    }
    
    sd_image_t* ptr_ = nullptr;
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
    static SDImageGuard image_to_sd_image(const Image& img);
    
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
