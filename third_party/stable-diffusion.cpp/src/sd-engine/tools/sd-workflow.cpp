// ============================================================================
// sd-engine/tools/sd-workflow.cpp
// ============================================================================
// 主命令行工具：执行 JSON 工作流或直接命令行出图
// ============================================================================

#include "adapter/sd_adapter.h"
#include "core/log.h"
#include "core/node.h"
#include "core/workflow.h"
#include "core/executor.h"
#include "nodes/node_utils.h"
#include "stable-diffusion.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace sdengine;

// ============================================================================
// 命令行参数解析
// ============================================================================

struct CLIArgs {
    // 直接命令行模式
    bool use_cli = false;
    std::string model_path;
    std::string prompt;
    std::string negative_prompt;
    std::string output = "output.png";
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfg_scale = 7.5f;
    int64_t seed = -1;  // -1 = random
    std::string sample_method = "euler";
    std::string scheduler = "discrete";
    
    // img2img
    std::string init_image;
    float strength = 0.75f;
    
    // HiRes
    bool enable_hires = false;
    int hires_width = 0;
    int hires_height = 0;
    float hires_strength = 0.5f;
    int hires_steps = 20;
    
    // Upscale
    std::string upscale_model;
    int upscale_scale = 2;
    
    // JSON 工作流模式
    std::string workflow_json;
    bool dry_run = false;
    bool verbose = false;
    bool save_json = false;
    std::string save_json_path;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\n"
              << "Direct Generation Mode:\n"
              << "  --model PATH            Diffusion model path (.gguf)\n"
              << "  --prompt TEXT           Positive prompt\n"
              << "  --negative-prompt TEXT  Negative prompt\n"
              << "  --output PATH           Output file (default: output.png)\n"
              << "  --width N               Image width (default: 512)\n"
              << "  --height N              Image height (default: 512)\n"
              << "  --steps N               Sampling steps (default: 20)\n"
              << "  --cfg-scale F           CFG scale (default: 7.5)\n"
              << "  --seed N                Random seed (-1 = random)\n"
              << "  --sample-method NAME    euler, euler_a, heun, dpm2, etc.\n"
              << "  --scheduler NAME        discrete, karras, exponential\n"
              << "\n"
              << "Img2Img Mode:\n"
              << "  --init-image PATH       Input image for img2img\n"
              << "  --strength F            Denoise strength (default: 0.75)\n"
              << "\n"
              << "HiRes Fix Mode:\n"
              << "  --hires                 Enable HiRes Fix\n"
              << "  --hires-width N         Target width\n"
              << "  --hires-height N        Target height\n"
              << "  --hires-strength F      HiRes denoise strength (default: 0.5)\n"
              << "  --hires-steps N         HiRes steps (default: 20)\n"
              << "\n"
              << "Upscale Mode:\n"
              << "  --upscale-model PATH    ESRGAN model path\n"
              << "  --upscale-scale N       Scale factor 2 or 4 (default: 2)\n"
              << "\n"
              << "JSON Workflow Mode:\n"
              << "  --workflow PATH         JSON workflow file\n"
              << "  --dry-run               Validate only, don't execute\n"
              << "  --save-json PATH        Save command as JSON workflow\n"
              << "\n"
              << "General Options:\n"
              << "  --verbose               Verbose output\n"
              << "  --help                  Show this help\n"
              << "\n"
              << "Examples:\n"
              << "  " << program << " --model model.gguf --prompt \"a cat\" --output cat.png\n"
              << "  " << program << " --model model.gguf --prompt \"a cat\" --hires --hires-width 1024 --hires-height 1024\n"
              << "  " << program << " --workflow workflow.json --verbose\n";
}

template <typename T>
static bool parse_number(const char* str, T& out) {
    try {
        if constexpr (std::is_same_v<T, int>) {
            out = std::stoi(str);
        } else if constexpr (std::is_same_v<T, float>) {
            out = std::stof(str);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            out = std::stoll(str);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Invalid number: '" << str << "' (" << e.what() << ")\n";
        return false;
    }
}

bool parse_args(int argc, char** argv, CLIArgs& args) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--model") {
            if (++i >= argc) return false;
            args.model_path = argv[i];
            args.use_cli = true;
        } else if (arg == "--prompt") {
            if (++i >= argc) return false;
            args.prompt = argv[i];
            args.use_cli = true;
        } else if (arg == "--negative-prompt") {
            if (++i >= argc) return false;
            args.negative_prompt = argv[i];
        } else if (arg == "--output") {
            if (++i >= argc) return false;
            args.output = argv[i];
        } else if (arg == "--width") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.width)) return false;
        } else if (arg == "--height") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.height)) return false;
        } else if (arg == "--steps") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.steps)) return false;
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.cfg_scale)) return false;
        } else if (arg == "--seed") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.seed)) return false;
        } else if (arg == "--sample-method") {
            if (++i >= argc) return false;
            args.sample_method = argv[i];
        } else if (arg == "--scheduler") {
            if (++i >= argc) return false;
            args.scheduler = argv[i];
        } else if (arg == "--init-image") {
            if (++i >= argc) return false;
            args.init_image = argv[i];
        } else if (arg == "--strength") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.strength)) return false;
        } else if (arg == "--hires") {
            args.enable_hires = true;
        } else if (arg == "--hires-width") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.hires_width)) return false;
        } else if (arg == "--hires-height") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.hires_height)) return false;
        } else if (arg == "--hires-strength") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.hires_strength)) return false;
        } else if (arg == "--hires-steps") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.hires_steps)) return false;
        } else if (arg == "--upscale-model") {
            if (++i >= argc) return false;
            args.upscale_model = argv[i];
        } else if (arg == "--upscale-scale") {
            if (++i >= argc) return false;
            if (!parse_number(argv[i], args.upscale_scale)) return false;
        } else if (arg == "--workflow") {
            if (++i >= argc) return false;
            args.workflow_json = argv[i];
        } else if (arg == "--dry-run") {
            args.dry_run = true;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--save-json") {
            if (++i >= argc) return false;
            args.save_json = true;
            args.save_json_path = argv[i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// 将 CLI 参数导出为 JSON 工作流
// ============================================================================

bool save_cli_to_json(const CLIArgs& args) {
    nlohmann::json j;
    int next_id = 1;
    
    auto next_node_id = [&next_id]() { return std::to_string(next_id++); };
    
    // 1. 模型加载节点
    std::string model_id = next_node_id();
    j[model_id] = {
        {"class_type", "CheckpointLoaderSimple"},
        {"inputs", {
            {"ckpt_name", args.model_path},
            {"use_gpu", true},
            {"flash_attn", true}
        }}
    };
    
    // 2. 生成节点
    std::string gen_node_id;
    if (!args.init_image.empty()) {
        // img2img: 先 LoadImage
        std::string load_img_id = next_node_id();
        j[load_img_id] = {
            {"class_type", "LoadImage"},
            {"inputs", {{"image", args.init_image}}}
        };
        
        gen_node_id = next_node_id();
        j[gen_node_id] = {
            {"class_type", "Img2Img"},
            {"inputs", {
                {"model", {model_id, 0}},
                {"init_image", {load_img_id, 0}},
                {"prompt", args.prompt},
                {"negative_prompt", args.negative_prompt},
                {"width", args.width},
                {"height", args.height},
                {"steps", args.steps},
                {"cfg_scale", args.cfg_scale},
                {"strength", args.strength},
                {"sample_method", args.sample_method},
                {"scheduler", args.scheduler},
                {"seed", args.seed}
            }}
        };
    } else {
        // txt2img
        gen_node_id = next_node_id();
        j[gen_node_id] = {
            {"class_type", "Txt2Img"},
            {"inputs", {
                {"model", {model_id, 0}},
                {"prompt", args.prompt},
                {"negative_prompt", args.negative_prompt},
                {"width", args.width},
                {"height", args.height},
                {"steps", args.steps},
                {"cfg_scale", args.cfg_scale},
                {"sample_method", args.sample_method},
                {"scheduler", args.scheduler},
                {"seed", args.seed}
            }}
        };
    }
    
    // 3. Upscale (optional)
    std::string last_img_id = gen_node_id;
    if (!args.upscale_model.empty()) {
        std::string upscale_id = next_node_id();
        j[upscale_id] = {
            {"class_type", "Upscale"},
            {"inputs", {
                {"image", {last_img_id, 0}},
                {"upscale_model", args.upscale_model},
                {"scale", args.upscale_scale}
            }}
        };
        last_img_id = upscale_id;
    }
    
    // 4. SaveImage
    std::string save_id = next_node_id();
    std::string prefix = args.output;
    auto pos = prefix.find_last_of('.');
    if (pos != std::string::npos) {
        prefix = prefix.substr(0, pos);
    }
    j[save_id] = {
        {"class_type", "SaveImage"},
        {"inputs", {
            {"images", {last_img_id, 0}},
            {"filename_prefix", prefix}
        }}
    };
    
    std::ofstream file(args.save_json_path);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open " << args.save_json_path << " for writing\n";
        return false;
    }
    file << j.dump(2);
    std::cout << "Workflow saved to: " << args.save_json_path << "\n";
    return true;
}

// ============================================================================
// 直接命令行出图
// ============================================================================

int run_cli_mode(const CLIArgs& args) {
    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        return 1;
    }
    if (args.prompt.empty()) {
        std::cerr << "Error: --prompt is required\n";
        return 1;
    }
    
    // 初始化 SDAdapter
    SDAdapter adapter;
    SDAdapterConfig config;
    config.diffusion_model_path = args.model_path;
    config.use_gpu = true;
    config.flash_attn = true;
    
    if (!adapter.init(config)) {
        std::cerr << "Error: Failed to load model\n";
        return 1;
    }
    
    // 构建生成参数
    GenerateParams params;
    params.prompt = args.prompt;
    params.negative_prompt = args.negative_prompt;
    params.width = args.width;
    params.height = args.height;
    params.sample_steps = args.steps;
    params.cfg_scale = args.cfg_scale;
    params.sample_method = str_to_sample_method(args.sample_method.c_str());
    params.scheduler = str_to_scheduler(args.scheduler.c_str());
    params.seed = args.seed >= 0 ? args.seed : (int64_t)time(nullptr);
    
    // img2img
    if (!args.init_image.empty()) {
        int w, h, c;
        auto init_img = load_image_from_file(args.init_image, &w, &h, &c);
        if (!init_img) {
            std::cerr << "Error: Failed to load init image\n";
            return 1;
        }
        params.init_image = init_img;
        params.strength = args.strength;
    }
    
    // HiRes
    if (args.enable_hires) {
        params.enable_hires = true;
        params.hires_width = args.hires_width > 0 ? args.hires_width : args.width * 2;
        params.hires_height = args.hires_height > 0 ? args.hires_height : args.height * 2;
        params.hires_strength = args.hires_strength;
        params.hires_steps = args.hires_steps;
        params.width = params.hires_width / 2;
        params.height = params.hires_height / 2;
    }
    
    LOG_INFO("Generating image...\n");
    auto image = adapter.generate(params);
    if (!image) {
        std::cerr << "Error: Generation failed\n";
        return 1;
    }
    
    // Upscale (optional)
    if (!args.upscale_model.empty()) {
        if (!adapter.init_upscaler({args.upscale_model, (uint32_t)args.upscale_scale})) {
            std::cerr << "Warning: Failed to load upscaler, skipping upscale\n";
        } else {
            image = adapter.upscale(image.get());
            if (!image) {
                std::cerr << "Warning: Upscale failed, using original image\n";
            }
        }
    }
    
    // 保存
    auto err = save_image_to_file(image.get(), args.output);
    if (is_error(err)) {
        std::cerr << "Error: Failed to save image\n";
        return 1;
    }
    
    std::cout << "Image saved to: " << args.output << "\n";
    return 0;
}

// ============================================================================
// JSON 工作流模式
// ============================================================================

int run_workflow_mode(const CLIArgs& args) {
    Workflow workflow;
    if (!workflow.load_from_file(args.workflow_json)) {
        std::cerr << "Error: Failed to load workflow: " << args.workflow_json << "\n";
        return 1;
    }
    
    std::string error_msg;
    if (!workflow.validate(error_msg)) {
        std::cerr << "Error: Invalid workflow: " << error_msg << "\n";
        return 1;
    }
    
    if (args.dry_run) {
        std::cout << "Workflow validation passed.\n";
        return 0;
    }
    
    ExecutionCache cache;
    DAGExecutor executor(&cache);
    
    ExecutionConfig exec_config;
    exec_config.verbose = args.verbose;
    
    LOG_INFO("Executing workflow: %s\n", args.workflow_json.c_str());
    auto err = executor.execute(&workflow, exec_config);
    
    if (is_error(err)) {
        std::cerr << "Error: Workflow execution failed\n";
        return 1;
    }
    
    std::cout << "Workflow executed successfully.\n";
    return 0;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    CLIArgs args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // 设置日志级别
    if (args.verbose) {
        // Logger 默认就是 INFO 级别
    }
    
    // 模式选择
    if (!args.workflow_json.empty()) {
        return run_workflow_mode(args);
    } else if (args.use_cli) {
        if (args.save_json) {
            if (!save_cli_to_json(args)) {
                return 1;
            }
        }
        return run_cli_mode(args);
    } else if (args.save_json) {
        std::cerr << "Error: --save-json requires --model and --prompt\n";
        return 1;
    } else {
        std::cerr << "Error: No mode specified. Use --model for direct generation or --workflow for JSON workflow.\n";
        print_usage(argv[0]);
        return 1;
    }
}
