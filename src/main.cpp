#include <iostream>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <random>
#include <filesystem>
#include <sstream>
#include <cmath>
#include "adapters/sdcpp_adapter.h"
#include "utils/image_utils.h"
#include "utils/image_adjust.h"
#include "utils/png_metadata.h"
#include "utils/lut_loader.h"

namespace fs = std::filesystem;

struct CliOptions {
    // 模型路径
    std::string model;                    // 完整模型 (ckpt/safetensors)
    std::string diffusion_model;          // 独立扩散模型 (GGUF)
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
    
    // img2img / Inpainting
    std::string init_image;
    float strength = 0.75f;
    std::string mask_image;
    
    // ControlNet
    std::string control_net;
    std::string control_image;
    float control_strength = 0.9f;
    
    // LoRA
    std::vector<std::string> loras;
    
    // 输出
    std::string output = "output.png";
    
    // Embeddings
    std::string embedding_dir;
    
    // 摄影后期调整
    float temperature = 0.0f;    // -1.0 ~ 1.0
    float brightness = 0.0f;     // -1.0 ~ 1.0
    float contrast = 0.0f;       // -1.0 ~ 1.0
    float saturation = 0.0f;     // -1.0 ~ 1.0
    float exposure = 0.0f;       // EV -5.0 ~ 5.0
    float highlights = 0.0f;     // -100 ~ 100
    float shadows = 0.0f;        // -100 ~ 100
    bool auto_enhance = false;   // 一键优化
    std::string curves;          // RGB curves "in,out;in,out"
    std::string preset;          // Filter preset name
    float vignette_strength = 0.0f; // 0.0-1.0
    float vignette_radius = 0.75f;  // 0.0-1.0
    
    // 人像修饰
    float whiten_strength = 0.0f;    // 0.0-1.0
    float skin_smooth_strength = 0.0f; // 0.0-1.0
    
    // 锐化与降噪
    float sharpen_amount = 0.0f;   // 0.0-3.0
    int sharpen_radius = 1;        // 1-5
    float sharpen_threshold = 0.0f; // 0-255
    float smart_sharpen_strength = 0.0f; // 0.0-3.0
    int smart_sharpen_radius = 2;   // 1-5
    float denoise_strength = 0.0f;  // 0.0-1.0
    bool smart_denoise_flag = false; // 智能降噪
    
    // Outpainting
    int outpaint_top = 0;
    int outpaint_bottom = 0;
    int outpaint_left = 0;
    int outpaint_right = 0;
    
    // Image transformation (post-processing)
    int resize_width = 0;     // 0 = no resize
    int resize_height = 0;    // 0 = no resize
    std::string resize_mode = "bilinear"; // nearest, bilinear, bicubic
    bool flip_h = false;
    bool flip_v = false;
    int rotate = 0;           // 90, 180, 270
    
    // LUT / Color grading
    std::string lut_path;  // 3D LUT file (.cube)
    
    // Batch processing (post-processing only)
    std::string batch_input_dir;
    std::string batch_output_dir;
    
    // Image interrogation / metadata
    std::string interrogate_image;   // Image to interrogate (JoyCaption placeholder)
    std::string read_metadata_image; // Image to read PNG metadata from
    
    // Local adjustments
    std::string radial_filter;      // cx,cy,radius,exp,cont,sat
    std::string graduated_filter;   // angle,pos,width,exp,cont,sat
    
    // 系统
    int threads = -1;
    bool verbose = false;
};

static void print_usage(const char* argv0) {
    std::cout << "my-img - Pure C++ ComfyUI Implementation\n\n";
    std::cout << "Usage: " << argv0 << " [options]\n\n";
    std::cout << "Model Options:\n";
    std::cout << "  -m, --model PATH          Full model path (ckpt/safetensors)\n";
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
    std::cout << "  --batch-count INT         Number of images to generate (default: 1)\n";
    std::cout << "\nimg2img Options:\n";
    std::cout << "  -i, --init-img PATH       Initial image for img2img (default: none)\n";
    std::cout << "  --strength FLOAT          Denoising strength 0.0-1.0 (default: 0.75)\n";
    std::cout << "  --mask PATH               Mask image for inpainting (white=inpaint, black=keep)\n";
    std::cout << "\nLoRA Options:\n";
    std::cout << "  --lora PATH:weight        LoRA model path and weight (can specify multiple)\n";
    std::cout << "  --lora-model-dir PATH     Directory containing LoRA models\n";
    std::cout << "\nControlNet Options:\n";
    std::cout << "  --control-net PATH        ControlNet model path\n";
    std::cout << "  --control-image PATH      Control image (canny/depth/lineart)\n";
    std::cout << "  --control-strength FLOAT  Control strength (default: 0.9)\n";
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
    std::cout << "\nEmbedding Options:\n";
    std::cout << "  --embd-dir PATH           Embeddings directory (Textual Inversion)\n";
    std::cout << "\nOutput Options:\n";
    std::cout << "  -o, --output PATH         Output path (default: output.png)\n";
    std::cout << "\nPhoto Adjustment Options:\n";
    std::cout << "  --temperature FLOAT       Color temperature -1.0(cold) to 1.0(warm)\n";
    std::cout << "  --brightness FLOAT        Brightness -1.0 to 1.0\n";
    std::cout << "  --contrast FLOAT          Contrast -1.0 to 1.0\n";
    std::cout << "  --saturation FLOAT        Saturation -1.0 to 1.0\n";
    std::cout << "  --exposure FLOAT          Exposure EV -5.0 to 5.0\n";
    std::cout << "  --highlights FLOAT        Highlights -100 to 100\n";
    std::cout << "  --shadows FLOAT           Shadows -100 to 100\n";
    std::cout << "  --auto-enhance            Auto one-click photo enhancement\n";
    std::cout << "\nCurves Options:\n";
    std::cout << "  --curves \"in,out;in,out\"  RGB curves (0-255, e.g. \"0,0;128,140;255,255\")\n";
    std::cout << "\nVignette Options:\n";
    std::cout << "  --vignette FLOAT          Vignette strength 0.0-1.0\n";
    std::cout << "  --vignette-radius FLOAT   Vignette radius 0.0-1.0 (default: 0.75)\n";
    std::cout << "\nLocal Adjustment Options:\n";
    std::cout << "  --radial-filter cx,cy,radius,exp,cont,sat  Radial filter (e.g. 0.5,0.5,0.3,0.5,0,0)\n";
    std::cout << "  --graduated-filter angle,pos,width,exp,cont,sat  Graduated filter (e.g. 0,0.5,0.2,0.3,0,0)\n";
    std::cout << "\nFilter Presets:\n";
    std::cout << "  --preset NAME             Apply filter preset: bw, sepia, vintage, warm,\n";
    std::cout << "                            cool, dramatic, japanese, film, cyberpunk, cinematic\n";
    std::cout << "\nColor Grading:\n";
    std::cout << "  --lut PATH                Load 3D LUT file (.cube format)\n";
    std::cout << "\nImage Interrogation:\n";
    std::cout << "  --interrogate PATH        Image path for caption/description\n";
    std::cout << "                            (requires JoyCaption model - placeholder)\n";
    std::cout << "  --read-metadata PATH      Read PNG metadata (prompt/parameters)\n";
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
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { std::cerr << "Missing value for -m/--model\n"; return false; }
            opts.model = argv[i];
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
        } else if (arg == "--embd-dir") {
            if (++i >= argc) { std::cerr << "Missing value for --embd-dir\n"; return false; }
            opts.embedding_dir = argv[i];
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
        } else if (arg == "--batch-count") {
            if (++i >= argc) { std::cerr << "Missing value for --batch-count\n"; return false; }
            opts.batch_count = std::stoi(argv[i]);
        } else if (arg == "-i" || arg == "--init-img") {
            if (++i >= argc) { std::cerr << "Missing value for -i/--init-img\n"; return false; }
            opts.init_image = argv[i];
        } else if (arg == "--strength") {
            if (++i >= argc) { std::cerr << "Missing value for --strength\n"; return false; }
            opts.strength = std::stof(argv[i]);
        } else if (arg == "--mask") {
            if (++i >= argc) { std::cerr << "Missing value for --mask\n"; return false; }
            opts.mask_image = argv[i];
        } else if (arg == "--control-net") {
            if (++i >= argc) { std::cerr << "Missing value for --control-net\n"; return false; }
            opts.control_net = argv[i];
        } else if (arg == "--control-image") {
            if (++i >= argc) { std::cerr << "Missing value for --control-image\n"; return false; }
            opts.control_image = argv[i];
        } else if (arg == "--control-strength") {
            if (++i >= argc) { std::cerr << "Missing value for --control-strength\n"; return false; }
            opts.control_strength = std::stof(argv[i]);
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
        } else if (arg == "--lora") {
            if (++i >= argc) { std::cerr << "Missing value for --lora\n"; return false; }
            opts.loras.push_back(argv[i]);
        } else if (arg == "--upscale-tile-size") {
            if (++i >= argc) { std::cerr << "Missing value for --upscale-tile-size\n"; return false; }
            opts.upscale_tile_size = std::stoi(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { std::cerr << "Missing value for -o/--output\n"; return false; }
            opts.output = argv[i];
        } else if (arg == "--threads") {
            if (++i >= argc) { std::cerr << "Missing value for --threads\n"; return false; }
            opts.threads = std::stoi(argv[i]);
        } else if (arg == "--temperature") {
            if (++i >= argc) { std::cerr << "Missing value for --temperature\n"; return false; }
            opts.temperature = std::stof(argv[i]);
        } else if (arg == "--brightness") {
            if (++i >= argc) { std::cerr << "Missing value for --brightness\n"; return false; }
            opts.brightness = std::stof(argv[i]);
        } else if (arg == "--contrast") {
            if (++i >= argc) { std::cerr << "Missing value for --contrast\n"; return false; }
            opts.contrast = std::stof(argv[i]);
        } else if (arg == "--saturation") {
            if (++i >= argc) { std::cerr << "Missing value for --saturation\n"; return false; }
            opts.saturation = std::stof(argv[i]);
        } else if (arg == "--exposure") {
            if (++i >= argc) { std::cerr << "Missing value for --exposure\n"; return false; }
            opts.exposure = std::stof(argv[i]);
        } else if (arg == "--highlights") {
            if (++i >= argc) { std::cerr << "Missing value for --highlights\n"; return false; }
            opts.highlights = std::stof(argv[i]);
        } else if (arg == "--shadows") {
            if (++i >= argc) { std::cerr << "Missing value for --shadows\n"; return false; }
            opts.shadows = std::stof(argv[i]);
        } else if (arg == "--auto-enhance") {
            opts.auto_enhance = true;
        } else if (arg == "--curves") {
            if (++i >= argc) { std::cerr << "Missing value for --curves\n"; return false; }
            opts.curves = argv[i];
        } else if (arg == "--vignette") {
            if (++i >= argc) { std::cerr << "Missing value for --vignette\n"; return false; }
            opts.vignette_strength = std::stof(argv[i]);
        } else if (arg == "--vignette-radius") {
            if (++i >= argc) { std::cerr << "Missing value for --vignette-radius\n"; return false; }
            opts.vignette_radius = std::stof(argv[i]);
        } else if (arg == "--preset") {
            if (++i >= argc) { std::cerr << "Missing value for --preset\n"; return false; }
            opts.preset = argv[i];
        } else if (arg == "--lut") {
            if (++i >= argc) { std::cerr << "Missing value for --lut\n"; return false; }
            opts.lut_path = argv[i];
        } else if (arg == "--whiten") {
            if (++i >= argc) { std::cerr << "Missing value for --whiten\n"; return false; }
            opts.whiten_strength = std::stof(argv[i]);
        } else if (arg == "--skin-smooth") {
            if (++i >= argc) { std::cerr << "Missing value for --skin-smooth\n"; return false; }
            opts.skin_smooth_strength = std::stof(argv[i]);
        } else if (arg == "--sharpen") {
            if (++i >= argc) { std::cerr << "Missing value for --sharpen\n"; return false; }
            opts.sharpen_amount = std::stof(argv[i]);
        } else if (arg == "--sharpen-radius") {
            if (++i >= argc) { std::cerr << "Missing value for --sharpen-radius\n"; return false; }
            opts.sharpen_radius = std::stoi(argv[i]);
        } else if (arg == "--sharpen-threshold") {
            if (++i >= argc) { std::cerr << "Missing value for --sharpen-threshold\n"; return false; }
            opts.sharpen_threshold = std::stof(argv[i]);
        } else if (arg == "--smart-sharpen") {
            if (++i >= argc) { std::cerr << "Missing value for --smart-sharpen\n"; return false; }
            opts.smart_sharpen_strength = std::stof(argv[i]);
        } else if (arg == "--smart-sharpen-radius") {
            if (++i >= argc) { std::cerr << "Missing value for --smart-sharpen-radius\n"; return false; }
            opts.smart_sharpen_radius = std::stoi(argv[i]);
        } else if (arg == "--denoise") {
            if (++i >= argc) { std::cerr << "Missing value for --denoise\n"; return false; }
            opts.denoise_strength = std::stof(argv[i]);
        } else if (arg == "--smart-denoise") {
            opts.smart_denoise_flag = true;
        } else if (arg == "--outpaint-top") {
            if (++i >= argc) { std::cerr << "Missing value for --outpaint-top\n"; return false; }
            opts.outpaint_top = std::stoi(argv[i]);
        } else if (arg == "--outpaint-bottom") {
            if (++i >= argc) { std::cerr << "Missing value for --outpaint-bottom\n"; return false; }
            opts.outpaint_bottom = std::stoi(argv[i]);
        } else if (arg == "--outpaint-left") {
            if (++i >= argc) { std::cerr << "Missing value for --outpaint-left\n"; return false; }
            opts.outpaint_left = std::stoi(argv[i]);
        } else if (arg == "--outpaint-right") {
            if (++i >= argc) { std::cerr << "Missing value for --outpaint-right\n"; return false; }
            opts.outpaint_right = std::stoi(argv[i]);
        } else if (arg == "--outpaint") {
            if (++i >= argc) { std::cerr << "Missing value for --outpaint\n"; return false; }
            int val = std::stoi(argv[i]);
            opts.outpaint_top = opts.outpaint_bottom = opts.outpaint_left = opts.outpaint_right = val;
        } else if (arg == "--resize") {
            if (++i >= argc) { std::cerr << "Missing value for --resize\n"; return false; }
            std::string val = argv[i];
            size_t x = val.find('x');
            if (x != std::string::npos) {
                opts.resize_width = std::stoi(val.substr(0, x));
                opts.resize_height = std::stoi(val.substr(x + 1));
            }
        } else if (arg == "--resize-mode") {
            if (++i >= argc) { std::cerr << "Missing value for --resize-mode\n"; return false; }
            opts.resize_mode = argv[i];
        } else if (arg == "--flip-h") {
            opts.flip_h = true;
        } else if (arg == "--flip-v") {
            opts.flip_v = true;
        } else if (arg == "--rotate") {
            if (++i >= argc) { std::cerr << "Missing value for --rotate\n"; return false; }
            opts.rotate = std::stoi(argv[i]);
        } else if (arg == "--batch-input-dir") {
            if (++i >= argc) { std::cerr << "Missing value for --batch-input-dir\n"; return false; }
            opts.batch_input_dir = argv[i];
        } else if (arg == "--batch-output-dir") {
            if (++i >= argc) { std::cerr << "Missing value for --batch-output-dir\n"; return false; }
            opts.batch_output_dir = argv[i];
        } else if (arg == "--interrogate") {
            if (++i >= argc) { std::cerr << "Missing value for --interrogate\n"; return false; }
            opts.interrogate_image = argv[i];
        } else if (arg == "--read-metadata") {
            if (++i >= argc) { std::cerr << "Missing value for --read-metadata\n"; return false; }
            opts.read_metadata_image = argv[i];
        } else if (arg == "--radial-filter") {
            if (++i >= argc) { std::cerr << "Missing value for --radial-filter\n"; return false; }
            opts.radial_filter = argv[i];
        } else if (arg == "--graduated-filter") {
            if (++i >= argc) { std::cerr << "Missing value for --graduated-filter\n"; return false; }
            opts.graduated_filter = argv[i];
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
    
    // Read PNG metadata (no model required)
    if (!opts.read_metadata_image.empty()) {
        std::cout << "========================================\n";
        std::cout << "  PNG Metadata Reader\n";
        std::cout << "========================================\n";
        std::cout << "File: " << opts.read_metadata_image << "\n\n";
        
        if (!myimg::is_png_file(opts.read_metadata_image)) {
            std::cerr << "Error: Not a PNG file\n";
            return 1;
        }
        
        auto metadata = myimg::read_png_metadata(opts.read_metadata_image);
        if (metadata.empty()) {
            std::cout << "No metadata found in this PNG file.\n";
        } else {
            for (const auto& [key, value] : metadata) {
                std::cout << "[" << key << "]\n";
                std::cout << value << "\n\n";
            }
        }
        return 0;
    }
    
    // Image interrogation placeholder (JoyCaption integration point)
    if (!opts.interrogate_image.empty()) {
        std::cout << "========================================\n";
        std::cout << "  Image Interrogation\n";
        std::cout << "========================================\n";
        std::cout << "File: " << opts.interrogate_image << "\n\n";
        
        // First, try to read embedded metadata
        if (myimg::is_png_file(opts.interrogate_image)) {
            auto metadata = myimg::read_png_metadata(opts.interrogate_image);
            if (metadata.count("parameters")) {
                std::cout << "[Embedded Parameters]\n";
                std::cout << metadata["parameters"] << "\n\n";
            }
        }
        
        std::cout << "[JoyCaption Integration]\n";
        std::cout << "To use JoyCaption for image captioning:\n";
        std::cout << "  1. Download JoyCaption model\n";
        std::cout << "  2. Place it in models/ directory\n";
        std::cout << "  3. Use: --interrogate-model PATH --interrogate " << opts.interrogate_image << "\n";
        std::cout << "\nNote: Full JoyCaption integration requires additional model files.\n";
        return 0;
    }
    
    // 检查必要参数
    if (opts.model.empty() && opts.diffusion_model.empty()) {
        std::cerr << "Error: --model or --diffusion-model is required\n";
        return 1;
    }
    // 如果使用 diffusion-model 模式，需要 vae 和 llm
    if (!opts.diffusion_model.empty()) {
        if (opts.vae.empty()) {
            std::cerr << "Error: --vae is required when using --diffusion-model\n";
            return 1;
        }
        if (opts.llm.empty()) {
            std::cerr << "Error: --llm is required when using --diffusion-model\n";
            return 1;
        }
    }
    
    // 随机种子
    if (opts.seed < 0) {
        opts.seed = std::random_device{}();
    }
    
    // 构建生成参数
    myimg::GenerationParams params;
    if (!opts.model.empty()) {
        params.model_path = opts.model;
    }
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
    params.embedding_dir = opts.embedding_dir;
    
    // ControlNet
    params.control_net_path = opts.control_net;
    if (!opts.control_image.empty()) {
        std::cout << "[INFO] Loading control image: " << opts.control_image << "\n";
        auto ctrl_data = myimg::load_image_from_file(opts.control_image);
        if (ctrl_data.empty()) {
            std::cerr << "Error: Failed to load control image: " << opts.control_image << "\n";
            return 1;
        }
        params.control_image.width = ctrl_data.width;
        params.control_image.height = ctrl_data.height;
        params.control_image.channels = ctrl_data.channels;
        params.control_image.data = std::move(ctrl_data.data);
        params.control_strength = opts.control_strength;
    }
    
    // HiRes Fix
    params.enable_hires = opts.hires;
    if (opts.hires) {
        params.hires_width = opts.hires_width;
        params.hires_height = opts.hires_height;
        params.hires_strength = opts.hires_strength;
        params.hires_sample_steps = opts.hires_steps;
    }
    
    // Outpainting
    bool has_outpaint = opts.outpaint_top > 0 || opts.outpaint_bottom > 0 ||
                        opts.outpaint_left > 0 || opts.outpaint_right > 0;
    if (has_outpaint) {
        if (opts.init_image.empty()) {
            std::cerr << "Error: --init-img is required for outpainting\n";
            return 1;
        }
        std::cout << "[INFO] Outpainting mode: top=" << opts.outpaint_top
                  << " bottom=" << opts.outpaint_bottom
                  << " left=" << opts.outpaint_left
                  << " right=" << opts.outpaint_right << "\n";
        
        auto orig = myimg::load_image_from_file(opts.init_image);
        if (orig.empty()) {
            std::cerr << "Error: Failed to load image for outpainting: " << opts.init_image << "\n";
            return 1;
        }
        
        auto [canvas, mask] = myimg::create_outpaint_canvas(
            orig, opts.outpaint_top, opts.outpaint_bottom, opts.outpaint_left, opts.outpaint_right
        );
        
        params.init_image.width = canvas.width;
        params.init_image.height = canvas.height;
        params.init_image.channels = canvas.channels;
        params.init_image.data = std::move(canvas.data);
        
        params.mask_image.width = mask.width;
        params.mask_image.height = mask.height;
        params.mask_image.channels = mask.channels;
        params.mask_image.data = std::move(mask.data);
        
        // Update target size to expanded canvas
        params.width = params.init_image.width;
        params.height = params.init_image.height;
        
        // Outpainting usually needs higher strength
        params.strength = 1.0f;
        std::cout << "[INFO] Outpaint canvas: " << params.width << "x" << params.height << "\n";
    }
    
    // img2img
    if (!opts.init_image.empty() && !has_outpaint) {
        std::cout << "[INFO] Loading init image: " << opts.init_image << "\n";
        auto img_data = myimg::load_image_from_file(opts.init_image);
        if (img_data.empty()) {
            std::cerr << "Error: Failed to load init image: " << opts.init_image << "\n";
            return 1;
        }
        params.init_image.width = img_data.width;
        params.init_image.height = img_data.height;
        params.init_image.channels = img_data.channels;
        params.init_image.data = std::move(img_data.data);
        params.strength = opts.strength;
        std::cout << "[INFO] img2img mode, strength: " << opts.strength << "\n";
    }
    
    // Inpainting
    if (!opts.mask_image.empty() && !has_outpaint) {
        std::cout << "[INFO] Loading mask: " << opts.mask_image << "\n";
        auto mask_data = myimg::load_image_from_file(opts.mask_image);
        if (mask_data.empty()) {
            std::cerr << "Error: Failed to load mask: " << opts.mask_image << "\n";
            return 1;
        }
        params.mask_image.width = mask_data.width;
        params.mask_image.height = mask_data.height;
        params.mask_image.channels = mask_data.channels;
        params.mask_image.data = std::move(mask_data.data);
        std::cout << "[INFO] Inpainting mode\n";
    }
    
    // LoRA
    if (!opts.loras.empty()) {
        std::cout << "[INFO] Loading LoRA models:\n";
        for (const auto& lora_str : opts.loras) {
            size_t colon = lora_str.find(':');
            myimg::LoRAConfig lora;
            if (colon != std::string::npos) {
                lora.path = lora_str.substr(0, colon);
                lora.multiplier = std::stof(lora_str.substr(colon + 1));
            } else {
                lora.path = lora_str;
                lora.multiplier = 1.0f;
            }
            params.loras.push_back(lora);
            std::cout << "  - " << lora.path << " (weight: " << lora.multiplier << ")\n";
        }
    }
    
    // Batch processing mode (post-processing only)
    if (!opts.batch_input_dir.empty()) {
        if (opts.batch_output_dir.empty()) {
            std::cerr << "Error: --batch-output-dir is required when using --batch-input-dir\n";
            return 1;
        }
        
        std::cout << "========================================\n";
        std::cout << "  my-img Batch Processing\n";
        std::cout << "========================================\n";
        std::cout << "Input: " << opts.batch_input_dir << "\n";
        std::cout << "Output: " << opts.batch_output_dir << "\n";
        std::cout << "========================================\n\n";
        
        fs::create_directories(opts.batch_output_dir);
        
        int processed = 0;
        int failed = 0;
        
        for (const auto& entry : fs::directory_iterator(opts.batch_input_dir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string ext = entry.path().extension().string();
            for (auto& c : ext) c = std::tolower(c);
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp" && ext != ".tga")
                continue;
            
            std::string input_file = entry.path().string();
            std::string output_file = opts.batch_output_dir + "/" + entry.path().filename().string();
            
            std::cout << "Processing: " << entry.path().filename().string() << "\n";
            
            auto img_data = myimg::load_image_from_file(input_file);
            if (img_data.empty()) {
                std::cerr << "  Failed to load\n";
                failed++;
                continue;
            }
            
            // Apply transformations
            if (opts.resize_width > 0 && opts.resize_height > 0) {
                img_data = myimg::resize_image(img_data, opts.resize_width, opts.resize_height, opts.resize_mode);
            }
            if (opts.flip_h) {
                img_data = myimg::flip_image(img_data, true);
            }
            if (opts.flip_v) {
                img_data = myimg::flip_image(img_data, false);
            }
            if (opts.rotate != 0) {
                img_data = myimg::rotate_image(img_data, opts.rotate);
            }
            
            // Apply photo adjustments
            if (opts.temperature != 0.0f || opts.brightness != 0.0f ||
                opts.contrast != 0.0f || opts.saturation != 0.0f ||
                opts.exposure != 0.0f || opts.highlights != 0.0f ||
                opts.shadows != 0.0f || opts.auto_enhance ||
                !opts.curves.empty() ||
                opts.sharpen_amount > 0.0f || opts.denoise_strength > 0.0f ||
                opts.smart_sharpen_strength > 0.0f || opts.smart_denoise_flag ||
                opts.whiten_strength > 0.0f || opts.skin_smooth_strength > 0.0f ||
                !opts.preset.empty() ||
                opts.vignette_strength > 0.0f ||
                !opts.radial_filter.empty() ||
                !opts.graduated_filter.empty() ||
                !opts.lut_path.empty()) {
                
                auto tensor = myimg::image_data_to_tensor(img_data);
                
                if (opts.auto_enhance) {
                    tensor = myimg::auto_enhance(tensor);
                } else {
                    if (opts.temperature != 0.0f) tensor = myimg::adjust_temperature(tensor, opts.temperature);
                    if (opts.brightness != 0.0f) tensor = myimg::adjust_brightness(tensor, opts.brightness);
                    if (opts.contrast != 0.0f) tensor = myimg::adjust_contrast(tensor, opts.contrast);
                    if (opts.saturation != 0.0f) tensor = myimg::adjust_saturation(tensor, opts.saturation);
                    if (opts.exposure != 0.0f) tensor = myimg::adjust_exposure(tensor, opts.exposure);
                    if (opts.highlights != 0.0f) tensor = myimg::adjust_highlights(tensor, opts.highlights);
                    if (opts.shadows != 0.0f) tensor = myimg::adjust_shadows(tensor, opts.shadows);
                }
                
                if (opts.denoise_strength > 0.0f) {
                    tensor = myimg::denoise(tensor, opts.denoise_strength);
                }
                if (opts.smart_denoise_flag) {
                    tensor = myimg::smart_denoise(tensor, 0.5f);
                }
                if (opts.sharpen_amount > 0.0f) {
                    tensor = myimg::usm_sharpen(tensor, opts.sharpen_amount, opts.sharpen_radius, opts.sharpen_threshold);
                }
                
                // Smart sharpen
                if (opts.smart_sharpen_strength > 0.0f) {
                    tensor = myimg::smart_sharpen(tensor, opts.smart_sharpen_strength, opts.smart_sharpen_radius);
                }
                
                // RGB curves
                if (!opts.curves.empty()) {
                    tensor = myimg::apply_curves(tensor, opts.curves);
                }
                
                // Filter preset
                if (!opts.preset.empty()) {
                    tensor = myimg::apply_preset(tensor, opts.preset);
                }
                
                // Vignette
                if (opts.vignette_strength > 0.0f) {
                    tensor = myimg::vignette(tensor, opts.vignette_strength, opts.vignette_radius);
                }
                
                // 径向滤镜
                if (!opts.radial_filter.empty()) {
                    std::stringstream ss(opts.radial_filter);
                    float cx, cy, radius, exp_val, cont_val, sat_val;
                    char comma;
                    ss >> cx >> comma >> cy >> comma >> radius >> comma >> exp_val >> comma >> cont_val >> comma >> sat_val;
                    tensor = myimg::radial_filter(tensor, cx, cy, radius, exp_val, cont_val, sat_val);
                }
                
                // 渐变滤镜
                if (!opts.graduated_filter.empty()) {
                    std::stringstream ss(opts.graduated_filter);
                    float angle, pos, width, exp_val, cont_val, sat_val;
                    char comma;
                    ss >> angle >> comma >> pos >> comma >> width >> comma >> exp_val >> comma >> cont_val >> comma >> sat_val;
                    tensor = myimg::graduated_filter(tensor, angle, pos, width, exp_val, cont_val, sat_val);
                }
                
                // LUT 颜色分级
                if (!opts.lut_path.empty()) {
                    myimg::LUT3D lut;
                    if (lut.load_from_file(opts.lut_path)) {
                        tensor = lut.apply(tensor);
                    }
                }
                
                // Portrait retouching
                if (opts.whiten_strength > 0.0f) {
                    tensor = myimg::whiten(tensor, opts.whiten_strength);
                }
                if (opts.skin_smooth_strength > 0.0f) {
                    tensor = myimg::skin_smooth(tensor, opts.skin_smooth_strength);
                }
                
                img_data = myimg::tensor_to_image_data(tensor);
            }
            
            myimg::Image image;
            image.width = img_data.width;
            image.height = img_data.height;
            image.channels = img_data.channels;
            image.data = std::move(img_data.data);
            
            if (!image.save_to_file(output_file)) {
                std::cerr << "  Failed to save\n";
                failed++;
                continue;
            }
            
            processed++;
        }
        
        std::cout << "\n========================================\n";
        std::cout << "Batch processing complete!\n";
        std::cout << "Processed: " << processed << "\n";
        if (failed > 0) std::cout << "Failed: " << failed << "\n";
        std::cout << "========================================\n";
        return 0;
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
    
    // 批量生成
    fs::path out_path = opts.output;
    std::string output_dir = out_path.parent_path().string();
    std::string output_name = out_path.stem().string();
    std::string output_ext = out_path.extension().string();
    if (output_ext.empty()) output_ext = ".png";
    
    if (!output_dir.empty()) {
        fs::create_directories(output_dir);
    }
    
    std::cout << "\nGenerating " << opts.batch_count << " image(s)...\n\n";
    
    for (int i = 0; i < opts.batch_count; ++i) {
        if (opts.batch_count > 1) {
            std::cout << "--- Image " << (i + 1) << "/" << opts.batch_count << " ---\n";
        }
        
        // 递增种子
        if (i > 0) {
            params.seed = opts.seed + i;
            std::cout << "Seed: " << params.seed << "\n";
        }
        
        // 生成图像
        myimg::Image image = adapter.generate_single(params);
        if (image.empty()) {
            std::cerr << "Generation failed for image " << (i + 1) << "\n";
            continue;
        }
        
        // ESRGAN 放大
        if (!opts.upscale_model.empty()) {
            std::cout << "Applying ESRGAN upscaling...\n";
            image = myimg::SDCPPAdapter::upscale_with_esrgan(image, opts.upscale_model, opts.upscale_repeats, opts.upscale_tile_size);
            if (image.empty()) {
                std::cerr << "Upscale failed for image " << (i + 1) << "\n";
                continue;
            }
        }
        
        // 摄影后期调整
        bool has_adjustments = opts.temperature != 0.0f || opts.brightness != 0.0f ||
                               opts.contrast != 0.0f || opts.saturation != 0.0f ||
                               opts.exposure != 0.0f || opts.highlights != 0.0f ||
                               opts.shadows != 0.0f || opts.auto_enhance ||
                               !opts.curves.empty() ||
                               opts.sharpen_amount > 0.0f || opts.denoise_strength > 0.0f ||
                               opts.smart_sharpen_strength > 0.0f || opts.smart_denoise_flag ||
                               opts.whiten_strength > 0.0f || opts.skin_smooth_strength > 0.0f ||
                               !opts.preset.empty() ||
                               opts.vignette_strength > 0.0f ||
                               !opts.radial_filter.empty() ||
                               !opts.graduated_filter.empty() ||
                               !opts.lut_path.empty();
        if (has_adjustments) {
            std::cout << "Applying photo adjustments...\n";
            myimg::ImageData img_data;
            img_data.width = image.width;
            img_data.height = image.height;
            img_data.channels = image.channels;
            img_data.data = std::move(image.data);
            
            auto tensor = myimg::image_data_to_tensor(img_data);
            
            if (opts.auto_enhance) {
                tensor = myimg::auto_enhance(tensor);
            } else {
                if (opts.temperature != 0.0f) tensor = myimg::adjust_temperature(tensor, opts.temperature);
                if (opts.brightness != 0.0f) tensor = myimg::adjust_brightness(tensor, opts.brightness);
                if (opts.contrast != 0.0f) tensor = myimg::adjust_contrast(tensor, opts.contrast);
                if (opts.saturation != 0.0f) tensor = myimg::adjust_saturation(tensor, opts.saturation);
                if (opts.exposure != 0.0f) tensor = myimg::adjust_exposure(tensor, opts.exposure);
                if (opts.highlights != 0.0f) tensor = myimg::adjust_highlights(tensor, opts.highlights);
                if (opts.shadows != 0.0f) tensor = myimg::adjust_shadows(tensor, opts.shadows);
            }
            
            // 降噪（在锐化之前）
            if (opts.denoise_strength > 0.0f) {
                std::cout << "Applying denoise...\n";
                tensor = myimg::denoise(tensor, opts.denoise_strength);
            }
            if (opts.smart_denoise_flag) {
                std::cout << "Applying smart denoise...\n";
                tensor = myimg::smart_denoise(tensor, 0.5f);
            }
            
            // USM 锐化
            if (opts.sharpen_amount > 0.0f) {
                std::cout << "Applying USM sharpen...\n";
                tensor = myimg::usm_sharpen(tensor, opts.sharpen_amount, opts.sharpen_radius, opts.sharpen_threshold);
            }
            
            // 智能锐化
            if (opts.smart_sharpen_strength > 0.0f) {
                std::cout << "Applying smart sharpen...\n";
                tensor = myimg::smart_sharpen(tensor, opts.smart_sharpen_strength, opts.smart_sharpen_radius);
            }
            
            // RGB 曲线
            if (!opts.curves.empty()) {
                std::cout << "Applying curves: " << opts.curves << "\n";
                tensor = myimg::apply_curves(tensor, opts.curves);
            }
            
            // 滤镜预设
            if (!opts.preset.empty()) {
                std::cout << "Applying preset: " << opts.preset << "\n";
                tensor = myimg::apply_preset(tensor, opts.preset);
            }
            
            // 暗角
            if (opts.vignette_strength > 0.0f) {
                std::cout << "Applying vignette...\n";
                tensor = myimg::vignette(tensor, opts.vignette_strength, opts.vignette_radius);
            }
            
            // 径向滤镜
            if (!opts.radial_filter.empty()) {
                std::cout << "Applying radial filter: " << opts.radial_filter << "\n";
                // 格式: cx,cy,radius,exposure,contrast,saturation
                std::stringstream ss(opts.radial_filter);
                float cx, cy, radius, exp_val, cont_val, sat_val;
                char comma;
                ss >> cx >> comma >> cy >> comma >> radius >> comma >> exp_val >> comma >> cont_val >> comma >> sat_val;
                tensor = myimg::radial_filter(tensor, cx, cy, radius, exp_val, cont_val, sat_val);
            }
            
            // 渐变滤镜
            if (!opts.graduated_filter.empty()) {
                std::cout << "Applying graduated filter: " << opts.graduated_filter << "\n";
                // 格式: angle,position,width,exposure,contrast,saturation
                std::stringstream ss(opts.graduated_filter);
                float angle, pos, width, exp_val, cont_val, sat_val;
                char comma;
                ss >> angle >> comma >> pos >> comma >> width >> comma >> exp_val >> comma >> cont_val >> comma >> sat_val;
                tensor = myimg::graduated_filter(tensor, angle, pos, width, exp_val, cont_val, sat_val);
            }
            
            // LUT 颜色分级
            if (!opts.lut_path.empty()) {
                std::cout << "Applying LUT: " << opts.lut_path << "\n";
                myimg::LUT3D lut;
                if (lut.load_from_file(opts.lut_path)) {
                    tensor = lut.apply(tensor);
                }
            }
            
            // 人像修饰
            if (opts.whiten_strength > 0.0f) {
                std::cout << "Applying whitening...\n";
                tensor = myimg::whiten(tensor, opts.whiten_strength);
            }
            if (opts.skin_smooth_strength > 0.0f) {
                std::cout << "Applying skin smoothing...\n";
                tensor = myimg::skin_smooth(tensor, opts.skin_smooth_strength);
            }
            
            img_data = myimg::tensor_to_image_data(tensor);
            image.width = img_data.width;
            image.height = img_data.height;
            image.channels = img_data.channels;
            image.data = std::move(img_data.data);
        }
        
        // 后处理变换
        bool has_transform = opts.resize_width > 0 || opts.resize_height > 0 ||
                            opts.flip_h || opts.flip_v || opts.rotate != 0;
        if (has_transform) {
            std::cout << "Applying transformations...\n";
            myimg::ImageData img_data;
            img_data.width = image.width;
            img_data.height = image.height;
            img_data.channels = image.channels;
            img_data.data = std::move(image.data);
            
            if (opts.resize_width > 0 && opts.resize_height > 0) {
                img_data = myimg::resize_image(img_data, opts.resize_width, opts.resize_height, opts.resize_mode);
            }
            if (opts.flip_h) {
                img_data = myimg::flip_image(img_data, true);
            }
            if (opts.flip_v) {
                img_data = myimg::flip_image(img_data, false);
            }
            if (opts.rotate != 0) {
                img_data = myimg::rotate_image(img_data, opts.rotate);
            }
            
            image.width = img_data.width;
            image.height = img_data.height;
            image.channels = img_data.channels;
            image.data = std::move(img_data.data);
        }
        
        // 构建输出文件名
        std::string output_file;
        if (opts.batch_count > 1) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s_%03d%s", output_name.c_str(), i + 1, output_ext.c_str());
            output_file = (output_dir.empty() ? "" : output_dir + "/") + buf;
        } else {
            output_file = opts.output;
        }
        
        // 保存图像
        if (!image.save_to_file(output_file)) {
            std::cerr << "Failed to save image " << (i + 1) << "\n";
            continue;
        }
        
        if (opts.batch_count > 1) {
            std::cout << "Saved: " << output_file << "\n\n";
        }
    }
    
    std::cout << "========================================\n";
    std::cout << "Generation complete!\n";
    if (opts.batch_count > 1) {
        std::cout << "Generated " << opts.batch_count << " images\n";
    }
    std::cout << "Seed start: " << opts.seed << "\n";
    std::cout << "========================================\n";
    
    return 0;
}
