#include <iostream>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <random>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <nlohmann/json.hpp>
#include "cli/cli_options.h"
#include "cli/cli_parser.h"
#include "pipeline/photo_adjustment.h"
#include "adapters/sdcpp_adapter.h"
#include "utils/image_utils.h"
#include "utils/image_adjust.h"
#include "utils/png_metadata.h"
#include "utils/lut_loader.h"
#include "utils/dehaze.h"
#include "utils/log.h"
#include "utils/vram_monitor.h"
#include "utils/report_generator.h"
#ifdef HAVE_OPENCV
#include "utils/controlnet_preprocessors.h"
#endif

namespace fs = std::filesystem;
using namespace myimg;

// Helper: apply crop_ratio to ImageData
static bool apply_crop_ratio(ImageData& img_data, const std::string& crop_ratio) {
    size_t colon = crop_ratio.find(':');
    if (colon == std::string::npos) return false;
    try {
        float ratio_w = std::stof(crop_ratio.substr(0, colon));
        float ratio_h = std::stof(crop_ratio.substr(colon + 1));
        float img_ratio = (float)img_data.width / img_data.height;
        float target_ratio = ratio_w / ratio_h;
        int w, h, x, y;
        if (img_ratio > target_ratio) {
            h = img_data.height;
            w = (int)(h * target_ratio);
            x = (img_data.width - w) / 2;
            y = 0;
        } else {
            w = img_data.width;
            h = (int)(w / target_ratio);
            x = 0;
            y = (img_data.height - h) / 2;
        }
        img_data = myimg::crop_image(img_data, x, y, w, h);
        return true;
    } catch (const std::exception&) {
        LOG_ERROR("Invalid crop ratio: %s", crop_ratio.c_str());
        return false;
    }
}

int main(int argc, char** argv) {
    CliOptions opts;
    
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // 设置日志级别
    if (opts.log_level == "trace") Logger::instance().set_level(LogLevel::Trace);
    else if (opts.log_level == "debug") Logger::instance().set_level(LogLevel::Debug);
    else if (opts.log_level == "info") Logger::instance().set_level(LogLevel::Info);
    else if (opts.log_level == "warn") Logger::instance().set_level(LogLevel::Warn);
    else if (opts.log_level == "error") Logger::instance().set_level(LogLevel::Error);
    else if (opts.log_level == "fatal") Logger::instance().set_level(LogLevel::Fatal);
    
    // 加载配置文件（命令行参数会覆盖配置文件）
    if (!opts.config_file.empty()) {
        LOG_INFO("Loading config from: %s", opts.config_file.c_str());
        if (!load_config_file(opts)) {
            return 1;
        }
    }
    
    // 解析 ComfyUI Workflow JSON
    if (!opts.workflow_file.empty()) {
        LOG_INFO("Loading workflow from: %s", opts.workflow_file.c_str());
        WorkflowParser parser;
        if (!parser.parse(opts.workflow_file)) {
            return 1;
        }
        if (!parser.to_cli_options(opts)) {
            return 1;
        }
    }
    
    // Read PNG metadata (no model required)
    if (!opts.read_metadata_image.empty()) {
        LOG_INFO("========================================");
        LOG_INFO("  PNG Metadata Reader");
        LOG_INFO("========================================");
        std::cout << "File: " << opts.read_metadata_image << "\n\n";
        
        if (!myimg::is_png_file(opts.read_metadata_image)) {
            LOG_ERROR("Error: Not a PNG file");
            return 1;
        }
        
        auto metadata = myimg::read_png_metadata(opts.read_metadata_image);
        if (metadata.empty()) {
            LOG_INFO("No metadata found in this PNG file.");
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
        LOG_INFO("========================================");
        LOG_INFO("  Image Interrogation");
        LOG_INFO("========================================");
        std::cout << "File: " << opts.interrogate_image << "\n\n";
        
        // First, try to read embedded metadata
        if (myimg::is_png_file(opts.interrogate_image)) {
            auto metadata = myimg::read_png_metadata(opts.interrogate_image);
            if (metadata.count("parameters")) {
                LOG_INFO("[Embedded Parameters]");
                std::cout << metadata["parameters"] << "\n\n";
            }
        }
        
        LOG_INFO("[JoyCaption Integration]");
        LOG_INFO("To use JoyCaption for image captioning:");
        LOG_INFO("  1. Download JoyCaption model");
        LOG_INFO("  2. Place it in models/ directory");
        std::cout << "  3. Use: --interrogate-model PATH --interrogate " << opts.interrogate_image << "\n";
        LOG_INFO("\nNote: Full JoyCaption integration requires additional model files.");
        return 0;
    }
    
    // ControlNet preprocessor (post-processing only, no model required)
    if (!opts.control_preprocessor.empty()) {
#ifdef HAVE_OPENCV
        if (opts.init_image.empty()) {
            LOG_ERROR("Error: --control-preprocessor requires --init-img");
            return 1;
        }
        
        LOG_INFO("========================================");
        std::cout << "  ControlNet Preprocessor: " << opts.control_preprocessor << "\n";
        LOG_INFO("========================================");
        
        auto img_data = myimg::load_image_from_file(opts.init_image);
        if (img_data.empty()) {
            LOG_ERROR("Error: Failed to load image");
            return 1;
        }
        
        std::string model_path;
        if (opts.control_preprocessor == "depth" || opts.control_preprocessor == "Depth") {
            model_path = opts.depth_model;
        } else if (opts.control_preprocessor == "openpose" || opts.control_preprocessor == "OpenPose") {
            model_path = opts.openpose_model;
        }
        
        auto result = myimg::apply_preprocessor(img_data, opts.control_preprocessor,
                                                 opts.control_preprocessor_param1,
                                                 opts.control_preprocessor_param2,
                                                 model_path);
        if (result.empty()) {
            LOG_ERROR("Error: Preprocessor failed");
            return 1;
        }
        
        // Save result
        std::string output_file = opts.output;
        if (output_file == "output.png") {
            output_file = "control_" + opts.control_preprocessor + ".png";
        }
        
        myimg::Image image;
        image.width = result.width;
        image.height = result.height;
        image.channels = result.channels;
        image.jpeg_quality = opts.jpeg_quality;
        image.data = std::move(result.data);
        
        if (image.save_to_file(output_file)) {
            std::cout << "Saved to: " << output_file << "\n";
        } else {
            LOG_ERROR("Error: Failed to save result");
            return 1;
        }
        return 0;
#else
        LOG_ERROR("Error: ControlNet preprocessors require OpenCV. Please install OpenCV.");
        return 1;
#endif
    }
    
    // Save preset (no model required)
    if (!opts.save_preset_name.empty()) {
        if (!save_preset(opts, opts.save_preset_name)) {
            return 1;
        }
        // If only saving preset (no generation/batch), exit
        if (opts.batch_input_dir.empty() && opts.model.empty() && opts.diffusion_model.empty()) {
            return 0;
        }
    }
    
    // Batch processing mode doesn't require model parameters
    bool batch_mode = !opts.batch_input_dir.empty();
    
    // 检查必要参数 (skip for batch mode)
    if (!batch_mode && opts.model.empty() && opts.diffusion_model.empty()) {
        LOG_ERROR("Error: --model or --diffusion-model is required");
        return 1;
    }
    // 如果使用 diffusion-model 模式，需要文本编码器 (skip for batch mode)
    // VAE 是可选的：SDXL 等完整 checkpoint 内置 VAE；Z-Image 等需要外部 VAE
    if (!batch_mode && !opts.diffusion_model.empty()) {
        if (opts.llm.empty() && (opts.clip_l.empty() || opts.clip_g.empty())) {
            LOG_ERROR("Error: --llm is required when using --diffusion-model (or provide --clip-l and --clip-g for SDXL)");
            return 1;
        }
    }
    
    // 随机种子
    if (opts.seed < 0) {
        opts.seed = std::random_device{}();
    }
    
    // 解析 prompt 中的 embedding 语法
    std::vector<std::string> referenced_embeddings;
    std::string processed_prompt = parse_embedding_syntax(opts.prompt, referenced_embeddings);
    std::string processed_neg_prompt = parse_embedding_syntax(opts.negative_prompt, referenced_embeddings);
    
    if (!referenced_embeddings.empty()) {
        std::cout << "[Embedding] Referenced embeddings: ";
        for (size_t i = 0; i < referenced_embeddings.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << referenced_embeddings[i];
        }
        LOG_INFO("");
    }
    
    // 构建生成参数
    myimg::GenerationParams params;
    if (!opts.model.empty()) {
        params.model_path = opts.model;
    }
    params.diffusion_model_path = opts.diffusion_model;
    params.vae_path = opts.vae;
    params.clip_l_path = opts.clip_l;
    params.clip_g_path = opts.clip_g;
    params.llm_path = opts.llm;
    params.prompt = processed_prompt;
    params.negative_prompt = processed_neg_prompt;
    params.width = opts.width;
    params.height = opts.height;
    params.sample_steps = opts.steps;
    params.cfg_scale = opts.cfg_scale;
    params.sample_method = parse_sampling_method(opts.sampling_method);
    params.scheduler = parse_scheduler(opts.scheduler);
    params.seed = opts.seed;
    params.batch_count = opts.batch_count;
    params.n_threads = opts.threads;
    params.max_vram = opts.max_vram;
    params.flash_attn = opts.diffusion_fa;
    params.vae_tiling = opts.vae_tiling;
    params.vae_tile_size_x = opts.vae_tile_size_w;
    params.vae_tile_size_y = opts.vae_tile_size_h;
    params.vae_tile_overlap = opts.vae_tile_overlap;
    params.embedding_dir = opts.embedding_dir;
    
    // FreeU
    params.freeu_enabled = opts.freeu;
    params.freeu_b1 = opts.freeu_b1;
    params.freeu_b2 = opts.freeu_b2;
    params.freeu_s1 = opts.freeu_s1;
    params.freeu_s2 = opts.freeu_s2;

    // SAG
    params.sag_enabled = opts.sag;
    params.sag_scale = opts.sag_scale;

    // ControlNet
    params.control_net_path = opts.control_net;
    if (!opts.control_image.empty()) {
        std::cout << "[INFO] Loading control image: " << opts.control_image << "\n";
        auto ctrl_data = myimg::load_image_from_file(opts.control_image);
        if (ctrl_data.empty()) {
            LOG_ERROR("Failed to load control image: %s", opts.control_image.c_str());
            return 1;
        }
        params.control_image.width = ctrl_data.width;
        params.control_image.height = ctrl_data.height;
        params.control_image.channels = ctrl_data.channels;
        params.control_image.data = std::move(ctrl_data.data);
        params.control_strength = opts.control_strength;
    }
    
    // 图像后处理 (clarity / sharpen / smart-sharpen / edge-sharpen)
    params.postproc_clarity = opts.clarity;
    params.postproc_sharpen_amount = opts.sharpen_amount;
    params.postproc_sharpen_radius = opts.sharpen_radius;
    params.postproc_sharpen_threshold = opts.sharpen_threshold;
    params.postproc_smart_sharpen_strength = opts.smart_sharpen_strength;
    params.postproc_smart_sharpen_radius = opts.smart_sharpen_radius;
    params.postproc_edge_sharpen_amount = opts.edge_sharpen_amount;
    params.postproc_edge_sharpen_radius = opts.edge_sharpen_radius;
    params.postproc_edge_sharpen_threshold = opts.edge_sharpen_threshold;

    // HiRes Fix
    params.enable_hires = opts.hires;
    if (opts.hires) {
        params.hires_width = opts.hires_width;
        params.hires_height = opts.hires_height;
        params.hires_strength = opts.hires_strength;
        params.hires_sample_steps = opts.hires_steps;
        params.hires_scale = opts.hires_scale;
        params.hires_tile_size = opts.hires_tile_size;
        if (!opts.hires_model_path.empty()) {
            params.hires_model_path = opts.hires_model_path;
        }
        // 解析 upscaler 名称
        std::string upscaler_lower = opts.hires_upscaler;
        for (auto& c : upscaler_lower) c = std::tolower(c);
        if (upscaler_lower == "latent") params.hires_upscaler = myimg::HiresUpscaler::Latent;
        else if (upscaler_lower == "latent-nearest") params.hires_upscaler = myimg::HiresUpscaler::LatentNearest;
        else if (upscaler_lower == "latent-nearest-exact") params.hires_upscaler = myimg::HiresUpscaler::LatentNearestExact;
        else if (upscaler_lower == "latent-antialiased") params.hires_upscaler = myimg::HiresUpscaler::LatentAntialiased;
        else if (upscaler_lower == "latent-bicubic") params.hires_upscaler = myimg::HiresUpscaler::LatentBicubic;
        else if (upscaler_lower == "latent-bicubic-antialiased") params.hires_upscaler = myimg::HiresUpscaler::LatentBicubicAntialiased;
        else if (upscaler_lower == "lanczos") params.hires_upscaler = myimg::HiresUpscaler::Lanczos;
        else if (upscaler_lower == "nearest") params.hires_upscaler = myimg::HiresUpscaler::Nearest;
        else if (upscaler_lower == "model") params.hires_upscaler = myimg::HiresUpscaler::Model;
        else {
            LOG_WARN("Unknown hires upscaler '%s', using 'latent'", opts.hires_upscaler.c_str());
            params.hires_upscaler = myimg::HiresUpscaler::Latent;
        }
    }
    
    // Outpainting
    bool has_outpaint = opts.outpaint_top > 0 || opts.outpaint_bottom > 0 ||
                        opts.outpaint_left > 0 || opts.outpaint_right > 0;
    if (has_outpaint) {
        if (opts.init_image.empty()) {
            LOG_ERROR("Error: --init-img is required for outpainting");
            return 1;
        }
        std::cout << "[INFO] Outpainting mode: top=" << opts.outpaint_top
                  << " bottom=" << opts.outpaint_bottom
                  << " left=" << opts.outpaint_left
                  << " right=" << opts.outpaint_right << "\n";
        
        auto orig = myimg::load_image_from_file(opts.init_image);
        if (orig.empty()) {
            LOG_ERROR("Failed to load image for outpainting: %s", opts.init_image.c_str());
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
            LOG_ERROR("Failed to load init image: %s", opts.init_image.c_str());
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
            LOG_ERROR("Failed to load mask: %s", opts.mask_image.c_str());
            return 1;
        }
        params.mask_image.width = mask_data.width;
        params.mask_image.height = mask_data.height;
        params.mask_image.channels = mask_data.channels;
        params.mask_image.data = std::move(mask_data.data);
        LOG_INFO("[INFO] Inpainting mode");
    }
    
    // LoRA
    if (!opts.loras.empty()) {
        LOG_INFO("[INFO] Loading LoRA models:");
        for (const auto& lora_str : opts.loras) {
            size_t colon = lora_str.find(':');
            myimg::LoRAConfig lora;
            if (colon != std::string::npos) {
                lora.path = lora_str.substr(0, colon);
                try {
                    lora.multiplier = std::stof(lora_str.substr(colon + 1));
                } catch (const std::exception&) {
                    LOG_ERROR("Invalid LoRA weight: %s", lora_str.substr(colon + 1).c_str());
                    return 1;
                }
            } else {
                lora.path = lora_str;
                lora.multiplier = 1.0f;
            }
            params.loras.push_back(lora);
            std::cout << "  - " << lora.path << " (weight: " << lora.multiplier << ")\n";
        }
    }
    
    // Advanced Features
    params.prompt_schedule = opts.prompt_schedule;
    params.regional_prompts = opts.regional_prompts;
    
    // Face Restoration
    params.face_restoration = opts.face_restoration;
    params.face_restore_model = opts.face_restore_model;
    params.face_restore_fidelity = opts.face_restore_fidelity;
    
    // IPAdapter
    params.ipadapter = opts.ipadapter;
    params.ipadapter_model = opts.ipadapter_model;
    params.ipadapter_clip_vision = opts.ipadapter_clip_vision;
    params.ipadapter_unet_weights_path = opts.ipadapter_unet_weights;
    params.ipadapter_image = opts.ipadapter_image;
    params.ipadapter_weight = opts.ipadapter_weight;
    
    // T2I-Adapter
    params.t2i_adapter = opts.t2i_adapter;
    params.t2i_adapter_model = opts.t2i_adapter_model;
    params.t2i_adapter_image = opts.t2i_adapter_image;
    params.t2i_adapter_strength = opts.t2i_adapter_strength;
    
    // Face Swap
    params.face_swap = opts.face_swap;
    params.face_swap_source = opts.face_swap_source;
    params.face_swap_detection_model = opts.face_swap_detection_model;
    params.face_swap_model = opts.face_swap_model;
    
    // PhotoMaker
    params.photo_maker = opts.photo_maker;
    params.photo_maker_model = opts.photo_maker_model;
    params.photo_maker_id_images = opts.photo_maker_id_images;
    params.photo_maker_id_weight = opts.photo_maker_id_weight;
    
    // Style Transfer
    params.style_transfer = opts.style_transfer;
    params.style_transfer_model = opts.style_transfer_model;
    params.style_transfer_image = opts.style_transfer_image;
    params.style_transfer_strength = opts.style_transfer_strength;
    params.style_transfer_block = opts.style_transfer_block;
    params.style_transfer_preserve_content = opts.style_transfer_preserve_content;
    
    // Load preset if specified
    if (!opts.load_preset_path.empty()) {
        if (!load_preset(opts, opts.load_preset_path)) {
            return 1;
        }
    }
    
    // Batch processing mode (post-processing only)
    if (!opts.batch_input_dir.empty()) {
        if (opts.batch_output_dir.empty()) {
            LOG_ERROR("Error: --batch-output-dir is required when using --batch-input-dir");
            return 1;
        }
        
        LOG_INFO("========================================");
        LOG_INFO("  my-img Batch Processing");
        LOG_INFO("========================================");
        std::cout << "Input: " << opts.batch_input_dir << "\n";
        std::cout << "Output: " << opts.batch_output_dir << "\n";
        if (opts.threads > 0) {
            std::cout << "Threads: " << opts.threads << "\n";
        }
        LOG_INFO("========================================\n");
        
        fs::create_directories(opts.batch_output_dir);
        
        // Collect all files
        struct FileTask {
            std::string input_file;
            std::string filename;
            int index;
        };
        std::vector<FileTask> tasks;
        int file_index = 0;
        
        for (const auto& entry : fs::directory_iterator(opts.batch_input_dir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string ext = entry.path().extension().string();
            for (auto& c : ext) c = std::tolower(c);
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp" && ext != ".tga")
                continue;
            
            file_index++;
            tasks.push_back({
                entry.path().string(),
                entry.path().filename().string(),
                file_index
            });
        }
        
        int total_files = tasks.size();
        if (total_files == 0) {
            LOG_INFO("No image files found in input directory.");
            return 0;
        }
        
        std::atomic<int> processed{0};
        std::atomic<int> failed{0};
        std::atomic<int> current_index{0};
        std::mutex print_mutex;
        
        auto process_file = [&](const FileTask& task) {
            std::string output_filename = expand_output_template(opts.output_template, task.filename, task.index);
            std::string output_file = opts.batch_output_dir + "/" + output_filename;
            
            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[" << task.index << "/" << total_files << "] Processing: " << task.filename;
                if (output_filename != task.filename) {
                    std::cout << " -> " << output_filename;
                }
                LOG_INFO("");
            }
            
            auto img_data = myimg::load_image_from_file(task.input_file);
            if (img_data.empty()) {
                std::lock_guard<std::mutex> lock(print_mutex);
                LOG_ERROR("  Failed to load");
                failed++;
                return;
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
            
            // Apply cropping
            if (!opts.crop.empty()) {
                std::stringstream ss(opts.crop);
                int x, y, w, h;
                char comma;
                ss >> x >> comma >> y >> comma >> w >> comma >> h;
                img_data = myimg::crop_image(img_data, x, y, w, h);
            } else if (!opts.crop_center.empty()) {
                std::stringstream ss(opts.crop_center);
                int w, h;
                char comma;
                ss >> w >> comma >> h;
                int x = (img_data.width - w) / 2;
                int y = (img_data.height - h) / 2;
                img_data = myimg::crop_image(img_data, x, y, w, h);
            } else if (!opts.crop_ratio.empty()) {
                apply_crop_ratio(img_data, opts.crop_ratio);
            }

            img_data = apply_photo_adjustments(img_data, opts);
            
            myimg::Image image;
            image.width = img_data.width;
            image.height = img_data.height;
            image.channels = img_data.channels;
            image.jpeg_quality = opts.jpeg_quality;
            image.data = std::move(img_data.data);
            
            if (!image.save_to_file(output_file)) {
                std::lock_guard<std::mutex> lock(print_mutex);
                LOG_ERROR("  Failed to save");
                failed++;
                return;
            }
            
            processed++;
        };
        
        // Determine thread count
        int num_threads = opts.threads;
        if (num_threads <= 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }
        num_threads = std::min(num_threads, total_files);
        
        // Launch worker threads
        std::vector<std::thread> workers;
        for (int t = 0; t < num_threads; ++t) {
            workers.emplace_back([&]() {
                while (true) {
                    int idx = current_index.fetch_add(1);
                    if (idx >= total_files) break;
                    process_file(tasks[idx]);
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& worker : workers) {
            worker.join();
        }
        
        LOG_INFO("\n========================================");
        LOG_INFO("Batch processing complete!");
        std::cout << "Processed: " << processed.load() << "\n";
        if (failed.load() > 0) std::cout << "Failed: " << failed.load() << "\n";
        LOG_INFO("========================================");
        return 0;
    }
    
    LOG_INFO("========================================");
    LOG_INFO("  my-img Image Generation");
    LOG_INFO("========================================");
    std::cout << "Model: " << opts.diffusion_model << "\n";
    std::cout << "VAE: " << opts.vae << "\n";
    std::cout << "LLM: " << opts.llm << "\n";
    std::cout << "Size: " << opts.width << "x" << opts.height;
    if (opts.hires) {
        std::cout << " -> " << opts.hires_width << "x" << opts.hires_height;
    }
    LOG_INFO("");
    std::cout << "Steps: " << opts.steps;
    if (opts.hires) {
        std::cout << " + " << opts.hires_steps << " (HiRes)";
    }
    LOG_INFO("");
    std::cout << "CFG: " << opts.cfg_scale << "\n";
    std::cout << "Sampler: " << opts.sampling_method << " + " << opts.scheduler << "\n";
    std::cout << "Seed: " << opts.seed << "\n";
    std::cout << "Output: " << opts.output << "\n";
    LOG_INFO("========================================\n");
    
    // 初始化报告生成器
    ReportGenerator report_gen;
    if (!opts.report_path.empty()) {
        report_gen.start_generation(params);
    }
    
    // 初始化适配器
    myimg::SDCPPAdapter adapter;
    if (!adapter.initialize(params)) {
        LOG_ERROR("Failed to initialize model");
        return 1;
    }
    
    // 设置进度回调
    adapter.set_progress_callback([&opts, &report_gen](int step, int steps, float time) {
        float progress = (float)step / steps * 100.0f;
        std::cout << "\r  Progress: " << step << "/" << steps << " (" << (int)progress << "%) - " << time << "s/step";
        
        if (opts.show_vram) {
            float vram_mb = VRAMMonitor::get_used_vram_mb();
            std::cout << " [VRAM: " << std::fixed << std::setprecision(1) << vram_mb << " MB]";
        }
        std::cout << std::flush;
        
        if (!opts.report_path.empty()) {
            report_gen.record_step(step, time, VRAMMonitor::get_used_vram_mb());
        }
    });
    
    // 设置预览回调
    if (opts.preview) {
        fs::create_directories(opts.preview_dir);
        adapter.set_preview_callback([&opts](int step, const myimg::Image& image, bool is_noisy) {
            (void)is_noisy; // 未使用但保留用于未来扩展
            if (image.empty()) return;
            std::string preview_path = opts.preview_dir + "/preview_step_" + std::to_string(step) + ".png";
            image.save_to_file(preview_path);
            LOG_INFO("Preview saved: %s", preview_path.c_str());
        }, opts.preview_interval, opts.preview_mode);
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
        image.jpeg_quality = opts.jpeg_quality;
        if (image.empty()) {
            LOG_ERROR("Generation failed for image %d", i + 1);
            continue;
        }
        
        // ESRGAN 放大
        if (!opts.upscale_model.empty()) {
            LOG_INFO("Applying ESRGAN upscaling...");
            image = myimg::SDCPPAdapter::upscale_with_esrgan(image, opts.upscale_model, opts.upscale_repeats, opts.upscale_tile_size);
            if (image.empty()) {
                LOG_ERROR("Upscale failed for image %d", i + 1);
                continue;
            }
        }
        
        // 摄影后期调整
        myimg::ImageData img_data;
        img_data.width = image.width;
        img_data.height = image.height;
        img_data.channels = image.channels;
        img_data.data = std::move(image.data);
        
        img_data = apply_photo_adjustments(img_data, opts);
        
        image.width = img_data.width;
        image.height = img_data.height;
        image.channels = img_data.channels;
        image.data = std::move(img_data.data);
        
        // 后处理变换
        bool has_transform = opts.resize_width > 0 || opts.resize_height > 0 ||
                            opts.flip_h || opts.flip_v || opts.rotate != 0 ||
                            !opts.crop.empty() || !opts.crop_center.empty() || !opts.crop_ratio.empty();
        if (has_transform) {
            LOG_INFO("Applying transformations...");
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
            
            // Apply cropping
            if (!opts.crop.empty()) {
                std::stringstream ss(opts.crop);
                int x, y, w, h;
                char comma;
                ss >> x >> comma >> y >> comma >> w >> comma >> h;
                img_data = myimg::crop_image(img_data, x, y, w, h);
            } else if (!opts.crop_center.empty()) {
                std::stringstream ss(opts.crop_center);
                int w, h;
                char comma;
                ss >> w >> comma >> h;
                int x = (img_data.width - w) / 2;
                int y = (img_data.height - h) / 2;
                img_data = myimg::crop_image(img_data, x, y, w, h);
            } else if (!opts.crop_ratio.empty()) {
                apply_crop_ratio(img_data, opts.crop_ratio);
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
            LOG_ERROR("Failed to save image %d", i + 1);
            continue;
        }
        
        // 嵌入 PNG 元数据
        if (opts.embed_metadata) {
            std::map<std::string, std::string> metadata;
            metadata["prompt"] = opts.prompt;
            if (!opts.negative_prompt.empty()) {
                metadata["negative_prompt"] = opts.negative_prompt;
            }
            metadata["seed"] = std::to_string(opts.seed + i);
            metadata["cfg_scale"] = std::to_string(opts.cfg_scale);
            metadata["steps"] = std::to_string(opts.steps);
            metadata["width"] = std::to_string(opts.width);
            metadata["height"] = std::to_string(opts.height);
            metadata["sampler"] = opts.sampling_method;
            metadata["scheduler"] = opts.scheduler;
            if (!opts.diffusion_model.empty()) {
                metadata["model"] = opts.diffusion_model;
            }
            if (opts.freeu) {
                metadata["freeu"] = "true";
            }
            if (opts.sag) {
                metadata["sag"] = "true";
            }
            if (opts.hires) {
                metadata["hires"] = "true";
                metadata["hires_width"] = std::to_string(opts.hires_width);
                metadata["hires_height"] = std::to_string(opts.hires_height);
                metadata["hires_strength"] = std::to_string(opts.hires_strength);
            }
            if (myimg::write_png_metadata(output_file, metadata)) {
                LOG_INFO("  Metadata embedded");
            } else {
                LOG_ERROR("  Warning: Failed to embed metadata");
            }
        }
        
        if (opts.batch_count > 1) {
            std::cout << "Saved: " << output_file << "\n\n";
        }
    }
    
    LOG_INFO("========================================");
    LOG_INFO("Generation complete!");
    if (opts.batch_count > 1) {
        std::cout << "Generated " << opts.batch_count << " images\n";
    }
    std::cout << "Seed start: " << opts.seed << "\n";
    
    // 保存生成报告
    if (!opts.report_path.empty()) {
        report_gen.end_generation(opts.output, opts.width, opts.height);
        report_gen.get_report().save_to_file(opts.report_path);
    }
    
    // 显示VRAM峰值
    if (opts.show_vram) {
        float peak_vram = VRAMMonitor::get_peak_vram_mb();
        if (peak_vram > 0.0f) {
            std::cout << "Peak VRAM: " << std::fixed << std::setprecision(1) << peak_vram << " MB\n";
        }
    }
    
    LOG_INFO("========================================");
    
    return 0;
}
