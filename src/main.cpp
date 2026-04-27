#include <iostream>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <random>
#include <filesystem>
#include "adapters/sdcpp_adapter.h"

namespace fs = std::filesystem;

struct CliOptions {
    // 模型路径
    std::string diffusion_model;
    std::string vae;
    std::string llm;
    std::string upscale_model;
    
    // 生成参数
    std::string prompt = "A beautiful landscape";
    std::string negative_prompt;
    int width = 1280;
    int height = 720;
    int steps = 50;
    float cfg_scale = 3.2f;
    std::string sampling_method = "euler";
    std::string scheduler = "discrete";
    int64_t seed = -1;
    int batch_count = 1;
    
    // VRAM 优化
    bool diffusion_fa = false;
    bool vae_tiling = false;
    int vae_tile_size_w = 256;
    int vae_tile_size_h = 256;
    float vae_tile_overlap = 0.8f;
    
    // HiRes Fix
    bool hires = false;
    int hires_width = 2560;
    int hires_height = 1440;
    float hires_strength = 0.30f;
    int hires_steps = 60;
    
    // ESRGAN
    int upscale_repeats = 1;
    int upscale_tile_size = 1440;
    
    // 输出
    std::string output = "output.png";
    
    // 系统
    int threads = -1;
    bool verbose = false;
};

static void print_usage(const char* argv0) {
    std::cout << "my-img - Pure C++ ComfyUI Implementation\n\n";
    std::cout << "Usage: " << argv0 << " [options]\n\n";
    std::cout << "Model Options:\n";
    std::cout << "  --diffusion-model PATH    Diffusion model path (GGUF)\n";
    std::cout << "  --vae PATH                VAE model path\n";
    std::cout << "  --llm PATH                LLM / text encoder path\n";
    std::cout << "  --upscale-model PATH      ESRGAN upscale model path\n";
    std::cout << "\nGeneration Options:\n";
    std::cout << "  -p, --prompt TEXT         Prompt (default: \"A beautiful landscape\")\n";
    std::cout << "  -n, --negative-prompt TEXT  Negative prompt\n";
    std::cout << "  -W, --width INT           Image width (default: 1280)\n";
    std::cout << "  -H, --height INT          Image height (default: 720)\n";
    std::cout << "  --steps INT               Sampling steps (default: 50)\n";
    std::cout << "  --cfg-scale FLOAT         CFG scale (default: 3.2)\n";
    std::cout << "  --sampling-method NAME    Sampling method: euler, dpm++2m, etc. (default: euler)\n";
    std::cout << "  --scheduler NAME          Scheduler: discrete, karras, etc. (default: discrete)\n";
    std::cout << "  -s, --seed INT            Seed, -1 for random (default: -1)\n";
    std::cout << "\nVRAM Optimization:\n";
    std::cout << "  --diffusion-fa            Enable Flash Attention for diffusion\n";
    std::cout << "  --vae-tiling              Enable VAE tiling\n";
    std::cout << "  --vae-tile-size WxH       VAE tile size (default: 256x256)\n";
    std::cout << "  --vae-tile-overlap FLOAT  VAE tile overlap (default: 0.8)\n";
    std::cout << "\nHiRes Fix Options:\n";
    std::cout << "  --hires                   Enable HiRes Fix\n";
    std::cout << "  --hires-width INT         HiRes target width (default: 2560)\n";
    std::cout << "  --hires-height INT        HiRes target height (default: 1440)\n";
    std::cout << "  --hires-strength FLOAT    HiRes denoising strength (default: 0.30)\n";
    std::cout << "  --hires-steps INT         HiRes sampling steps (default: 60)\n";
    std::cout << "\nUpscale Options:\n";
    std::cout << "  --upscale-repeats INT     ESRGAN upscale repeats (default: 1)\n";
    std::cout << "  --upscale-tile-size INT   ESRGAN tile size (default: 1440)\n";
    std::cout << "\nOutput Options:\n";
    std::cout << "  -o, --output PATH         Output path (default: output.png)\n";
    std::cout << "\nSystem Options:\n";
    std::cout << "  --threads INT             Number of CPU threads (default: auto)\n";
    std::cout << "  -v, --verbose             Verbose logging\n";
    std::cout << "  --help                    Show this help\n";
}

static bool parse_args(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--version") {
            std::cout << "my-img version: 0.1.0\n";
            std::cout << "sd.cpp version: " << myimg::SDCPPAdapter::get_version() << "\n";
            exit(0);
        } else if (arg == "--diffusion-model") {
            if (++i >= argc) { std::cerr << "Missing value for --diffusion-model\n"; return false; }
            opts.diffusion_model = argv[i];
        } else if (arg == "--vae") {
            if (++i >= argc) { std::cerr << "Missing value for --vae\n"; return false; }
            opts.vae = argv[i];
        } else if (arg == "--llm") {
            if (++i >= argc) { std::cerr << "Missing value for --llm\n"; return false; }
            opts.llm = argv[i];
        } else if (arg == "--upscale-model") {
            if (++i >= argc) { std::cerr << "Missing value for --upscale-model\n"; return false; }
            opts.upscale_model = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) { std::cerr << "Missing value for -p/--prompt\n"; return false; }
            opts.prompt = argv[i];
        } else if (arg == "-n" || arg == "--negative-prompt") {
            if (++i >= argc) { std::cerr << "Missing value for -n/--negative-prompt\n"; return false; }
            opts.negative_prompt = argv[i];
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) { std::cerr << "Missing value for -W/--width\n"; return false; }
            opts.width = std::stoi(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) { std::cerr << "Missing value for -H/--height\n"; return false; }
            opts.height = std::stoi(argv[i]);
        } else if (arg == "--steps") {
            if (++i >= argc) { std::cerr << "Missing value for --steps\n"; return false; }
            opts.steps = std::stoi(argv[i]);
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) { std::cerr << "Missing value for --cfg-scale\n"; return false; }
            opts.cfg_scale = std::stof(argv[i]);
        } else if (arg == "--sampling-method") {
            if (++i >= argc) { std::cerr << "Missing value for --sampling-method\n"; return false; }
            opts.sampling_method = argv[i];
        } else if (arg == "--scheduler") {
            if (++i >= argc) { std::cerr << "Missing value for --scheduler\n"; return false; }
            opts.scheduler = argv[i];
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) { std::cerr << "Missing value for -s/--seed\n"; return false; }
            opts.seed = std::stoll(argv[i]);
        } else if (arg == "--diffusion-fa") {
            opts.diffusion_fa = true;
        } else if (arg == "--vae-tiling") {
            opts.vae_tiling = true;
        } else if (arg == "--vae-tile-size") {
            if (++i >= argc) { std::cerr << "Missing value for --vae-tile-size\n"; return false; }
            std::string val = argv[i];
            size_t x = val.find('x');
            if (x != std::string::npos) {
                opts.vae_tile_size_w = std::stoi(val.substr(0, x));
                opts.vae_tile_size_h = std::stoi(val.substr(x + 1));
            } else {
                opts.vae_tile_size_w = opts.vae_tile_size_h = std::stoi(val);
            }
        } else if (arg == "--vae-tile-overlap") {
            if (++i >= argc) { std::cerr << "Missing value for --vae-tile-overlap\n"; return false; }
            opts.vae_tile_overlap = std::stof(argv[i]);
        } else if (arg == "--hires") {
            opts.hires = true;
        } else if (arg == "--hires-width") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-width\n"; return false; }
            opts.hires_width = std::stoi(argv[i]);
        } else if (arg == "--hires-height") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-height\n"; return false; }
            opts.hires_height = std::stoi(argv[i]);
        } else if (arg == "--hires-strength") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-strength\n"; return false; }
            opts.hires_strength = std::stof(argv[i]);
        } else if (arg == "--hires-steps") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-steps\n"; return false; }
            opts.hires_steps = std::stoi(argv[i]);
        } else if (arg == "--upscale-repeats") {
            if (++i >= argc) { std::cerr << "Missing value for --upscale-repeats\n"; return false; }
            opts.upscale_repeats = std::stoi(argv[i]);
        } else if (arg == "--upscale-tile-size") {
            if (++i >= argc) { std::cerr << "Missing value for --upscale-tile-size\n"; return false; }
            opts.upscale_tile_size = std::stoi(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { std::cerr << "Missing value for -o/--output\n"; return false; }
            opts.output = argv[i];
        } else if (arg == "--threads") {
            if (++i >= argc) { std::cerr << "Missing value for --threads\n"; return false; }
            opts.threads = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

static myimg::SampleMethod parse_sampling_method(const std::string& name) {
    if (name == "euler") return myimg::SampleMethod::Euler;
    if (name == "euler_a" || name == "euler-ancestral") return myimg::SampleMethod::EulerAncestral;
    if (name == "heun") return myimg::SampleMethod::Heun;
    if (name == "dpm2") return myimg::SampleMethod::DPM2;
    if (name == "dpm++2s_a") return myimg::SampleMethod::DPMPP2S_A;
    if (name == "dpm++2m") return myimg::SampleMethod::DPMPP2M;
    if (name == "dpm++2mv2") return myimg::SampleMethod::DPMPP2Mv2;
    if (name == "ipndm") return myimg::SampleMethod::IPNDM;
    if (name == "ipndm_v") return myimg::SampleMethod::IPNDM_V;
    if (name == "lcm") return myimg::SampleMethod::LCM;
    if (name == "ddim_trailing") return myimg::SampleMethod::DDIM_Trailing;
    if (name == "tcd") return myimg::SampleMethod::TCD;
    if (name == "res_multistep") return myimg::SampleMethod::RES_Multistep;
    if (name == "res_2s") return myimg::SampleMethod::RES_2S;
    if (name == "er_sde") return myimg::SampleMethod::ER_SDE;
    return myimg::SampleMethod::Euler;
}

static myimg::Scheduler parse_scheduler(const std::string& name) {
    if (name == "discrete") return myimg::Scheduler::Discrete;
    if (name == "karras") return myimg::Scheduler::Karras;
    if (name == "exponential") return myimg::Scheduler::Exponential;
    if (name == "ays") return myimg::Scheduler::AYS;
    if (name == "gits") return myimg::Scheduler::GITS;
    if (name == "sgm_uniform") return myimg::Scheduler::SGM_Uniform;
    if (name == "simple") return myimg::Scheduler::Simple;
    if (name == "smoothstep") return myimg::Scheduler::Smoothstep;
    if (name == "kl_optimal") return myimg::Scheduler::KL_Optimal;
    if (name == "lcm") return myimg::Scheduler::LCM;
    if (name == "bong_tangent") return myimg::Scheduler::Bong_Tangent;
    return myimg::Scheduler::Simple;
}

int main(int argc, char** argv) {
    CliOptions opts;
    
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // 检查必要参数
    if (opts.diffusion_model.empty()) {
        std::cerr << "Error: --diffusion-model is required\n";
        return 1;
    }
    if (opts.vae.empty()) {
        std::cerr << "Error: --vae is required\n";
        return 1;
    }
    if (opts.llm.empty()) {
        std::cerr << "Error: --llm is required\n";
        return 1;
    }
    
    // 随机种子
    if (opts.seed < 0) {
        opts.seed = std::random_device{}();
    }
    
    // 构建生成参数
    myimg::GenerationParams params;
    params.diffusion_model_path = opts.diffusion_model;
    params.vae_path = opts.vae;
    params.llm_path = opts.llm;
    params.prompt = opts.prompt;
    params.negative_prompt = opts.negative_prompt;
    params.width = opts.width;
    params.height = opts.height;
    params.sample_steps = opts.steps;
    params.cfg_scale = opts.cfg_scale;
    params.sample_method = parse_sampling_method(opts.sampling_method);
    params.scheduler = parse_scheduler(opts.scheduler);
    params.seed = opts.seed;
    params.batch_count = opts.batch_count;
    params.n_threads = opts.threads;
    params.flash_attn = opts.diffusion_fa;
    params.vae_tiling = opts.vae_tiling;
    params.vae_tile_size_x = opts.vae_tile_size_w;
    params.vae_tile_size_y = opts.vae_tile_size_h;
    params.vae_tile_overlap = opts.vae_tile_overlap;
    
    // HiRes Fix
    params.enable_hires = opts.hires;
    if (opts.hires) {
        params.hires_width = opts.hires_width;
        params.hires_height = opts.hires_height;
        params.hires_strength = opts.hires_strength;
        params.hires_sample_steps = opts.hires_steps;
    }
    
    std::cout << "========================================\n";
    std::cout << "  my-img Image Generation\n";
    std::cout << "========================================\n";
    std::cout << "Model: " << opts.diffusion_model << "\n";
    std::cout << "VAE: " << opts.vae << "\n";
    std::cout << "LLM: " << opts.llm << "\n";
    std::cout << "Size: " << opts.width << "x" << opts.height;
    if (opts.hires) {
        std::cout << " -> " << opts.hires_width << "x" << opts.hires_height;
    }
    std::cout << "\n";
    std::cout << "Steps: " << opts.steps;
    if (opts.hires) {
        std::cout << " + " << opts.hires_steps << " (HiRes)";
    }
    std::cout << "\n";
    std::cout << "CFG: " << opts.cfg_scale << "\n";
    std::cout << "Sampler: " << opts.sampling_method << " + " << opts.scheduler << "\n";
    std::cout << "Seed: " << opts.seed << "\n";
    std::cout << "Output: " << opts.output << "\n";
    std::cout << "========================================\n\n";
    
    // 初始化适配器
    myimg::SDCPPAdapter adapter;
    if (!adapter.initialize(params)) {
        std::cerr << "Failed to initialize model\n";
        return 1;
    }
    
    // 生成图像
    myimg::Image image = adapter.generate_single(params);
    if (image.empty()) {
        std::cerr << "Generation failed\n";
        return 1;
    }
    
    // ESRGAN 放大（如果指定了模型）
    if (!opts.upscale_model.empty()) {
        std::cout << "\nApplying ESRGAN upscaling...\n";
        image = myimg::SDCPPAdapter::upscale_with_esrgan(image, opts.upscale_model, opts.upscale_repeats, opts.upscale_tile_size);
        if (image.empty()) {
            std::cerr << "Upscale failed\n";
            return 1;
        }
    }
    
    // 创建输出目录
    fs::path out_path = opts.output;
    if (!out_path.parent_path().empty()) {
        fs::create_directories(out_path.parent_path());
    }
    
    // 保存图像
    if (!image.save_to_file(opts.output)) {
        std::cerr << "Failed to save image\n";
        return 1;
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Generation successful!\n";
    std::cout << "File: " << opts.output << "\n";
    std::cout << "Size: " << image.width << "x" << image.height << "\n";
    std::cout << "Seed: " << opts.seed << "\n";
    std::cout << "========================================\n";
    
    return 0;
}
