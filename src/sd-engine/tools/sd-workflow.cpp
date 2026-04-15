// ============================================================================
// sd-engine/tools/sd-workflow.cpp
// ============================================================================
// 
// sd-workflow: ComfyUI JSON 工作流执行器 + 快速模式
// ============================================================================

#include "core/workflow.h"
#include "core/executor.h"
#include "core/cache.h"
#include "core/init.h"
#include "core/workflow_builder.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace sdengine;

static void print_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("\n=== Workflow Mode ===\n");
    printf("  --workflow <file>    ComfyUI workflow JSON file\n");
    printf("  --dry-run            Parse and validate only\n");
    printf("  --verbose            Show detailed execution info\n");
    printf("\n=== Quick Mode: txt2img ===\n");
    printf("  --txt2img            Enable txt2img quick mode\n");
    printf("  --model <path>      Checkpoint path\n");
    printf("  --prompt <text>     Positive prompt\n");
    printf("  --negative <text>   Negative prompt\n");
    printf("  --width <n>          Image width (default: 512)\n");
    printf("  --height <n>         Image height (default: 512)\n");
    printf("  --seed <n>           Random seed (default: 0)\n");
    printf("  --steps <n>          Sampling steps (default: 20)\n");
    printf("  --cfg <f>            CFG scale (default: 7.5)\n");
    printf("  --output <prefix>   Output filename prefix\n");
    printf("\n=== Quick Mode: img2img ===\n");
    printf("  --img2img            Enable img2img quick mode\n");
    printf("  --input <image>      Input image path\n");
    printf("  --denoise <f>        Denoise strength (default: 0.75)\n");
    printf("\n=== Quick Mode: image process ===\n");
    printf("  --process            Enable image process quick mode\n");
    printf("  --scale-w <n>        Target width\n");
    printf("  --scale-h <n>        Target height\n");
    printf("  --crop-x <n>         Crop X\n");
    printf("  --crop-y <n>         Crop Y\n");
    printf("  --crop-w <n>         Crop width\n");
    printf("  --crop-h <n>         Crop height\n");
    printf("\n=== Quick Mode: Deep HighRes Fix ===\n");
    printf("  --deep-hires         Enable Deep HighRes Fix quick mode\n");
    printf("  --target-width <n>   Target width (default: 1024)\n");
    printf("  --target-height <n>  Target height (default: 1024)\n");
    printf("  --vae-tiling         Enable VAE tiling for large images\n");
    printf("\n=== Other ===\n");
    printf("  --save-json <file>  Save generated workflow to JSON\n");
    printf("  --list-nodes         List supported node types\n");
    printf("  --help               Show this help\n");
}

static void print_supported_nodes() {
    printf("Supported node types:\n");
    auto nodes = NodeRegistry::instance().get_supported_nodes();
    for (const auto& type : nodes) {
        printf("  - %s\n", type.c_str());
    }
    if (nodes.empty()) {
        printf("  (none registered yet)\n");
    }
}

static bool execute_workflow(Workflow& workflow, bool dry_run, bool verbose) {
    printf("Loaded %zu nodes\n", workflow.get_all_node_ids().size());
    
    std::string error_msg;
    if (!workflow.validate(error_msg)) {
        printf("Error: Invalid workflow - %s\n", error_msg.c_str());
        return false;
    }
    
    auto order = workflow.topological_sort();
    printf("Execution order (%zu nodes):\n", order.size());
    for (size_t i = 0; i < order.size(); i++) {
        Node* node = workflow.get_node(order[i]);
        if (node) {
            printf("  %zu. [%s] %s\n", i + 1, order[i].c_str(), node->get_class_type().c_str());
        }
    }
    
    if (dry_run) {
        printf("\nDry run completed successfully.\n");
        return true;
    }
    
    printf("\nExecuting workflow...\n");
    
    ExecutionCache cache;
    DAGExecutor executor(&cache);
    
    if (verbose) {
        executor.set_progress_callback([](const std::string& node_id, int current, int total) {
            printf("\rProgress: %d/%d", current, total);
            fflush(stdout);
        });
    }
    
    executor.set_error_callback([](const std::string& msg) {
        printf("\nError: %s\n", msg.c_str());
    });
    
    ExecutionConfig config;
    config.use_cache = true;
    config.verbose = verbose;
    
    sd_error_t result = executor.execute(&workflow, config);
    bool success = is_ok(result);
    
    printf("\n");
    
    if (success) {
        printf("Workflow executed successfully!\n");
    } else {
        printf("Workflow execution failed.\n");
    }
    
    return success;
}

int main(int argc, char** argv) {
    sdengine::init_builtin_nodes();
    
    // Workflow mode params
    const char* workflow_file = nullptr;
    bool dry_run = false;
    bool verbose = false;
    bool list_nodes = false;
    
    // Quick mode params
    bool quick_txt2img = false;
    bool quick_img2img = false;
    bool quick_process = false;
    bool quick_deep_hires = false;
    const char* model_path = nullptr;
    const char* prompt = "";
    const char* negative_prompt = "";
    const char* input_image = nullptr;
    const char* output_prefix = "output";
    const char* save_json = nullptr;
    int width = 512;
    int height = 512;
    int seed = 0;
    int steps = 20;
    float cfg = 7.5f;
    float denoise = 0.75f;
    int scale_w = 0, scale_h = 0;
    int crop_x = -1, crop_y = -1, crop_w = -1, crop_h = -1;
    int target_width = 1024;
    int target_height = 1024;
    bool vae_tiling = false;
    
    for (int i = 1; i < argc; i++) {
        // Workflow mode
        if (strcmp(argv[i], "--workflow") == 0 && i + 1 < argc) {
            workflow_file = argv[++i];
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            dry_run = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--list-nodes") == 0) {
            list_nodes = true;
        }
        // Quick mode flags
        else if (strcmp(argv[i], "--txt2img") == 0) {
            quick_txt2img = true;
        } else if (strcmp(argv[i], "--img2img") == 0) {
            quick_img2img = true;
        } else if (strcmp(argv[i], "--process") == 0) {
            quick_process = true;
        }
        // Common params
        else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--negative") == 0 && i + 1 < argc) {
            negative_prompt = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_image = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (strcmp(argv[i], "--save-json") == 0 && i + 1 < argc) {
            save_json = argv[++i];
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) {
            cfg = atof(argv[++i]);
        } else if (strcmp(argv[i], "--denoise") == 0 && i + 1 < argc) {
            denoise = atof(argv[++i]);
        } else if (strcmp(argv[i], "--scale-w") == 0 && i + 1 < argc) {
            scale_w = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--scale-h") == 0 && i + 1 < argc) {
            scale_h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--crop-x") == 0 && i + 1 < argc) {
            crop_x = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--crop-y") == 0 && i + 1 < argc) {
            crop_y = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--crop-w") == 0 && i + 1 < argc) {
            crop_w = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--crop-h") == 0 && i + 1 < argc) {
            crop_h = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--deep-hires") == 0) {
            quick_deep_hires = true;
        } else if (strcmp(argv[i], "--target-width") == 0 && i + 1 < argc) {
            target_width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--target-height") == 0 && i + 1 < argc) {
            target_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vae-tiling") == 0) {
            vae_tiling = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (list_nodes) {
        print_supported_nodes();
        return 0;
    }
    
    // Determine mode
    int quick_modes = (quick_txt2img ? 1 : 0) + (quick_img2img ? 1 : 0) + 
                      (quick_process ? 1 : 0) + (quick_deep_hires ? 1 : 0);
    
    if (quick_modes > 1) {
        printf("Error: Only one quick mode can be used at a time\n");
        return 1;
    }
    
    std::string json_str;
    bool is_quick_mode = (quick_modes == 1);
    
    if (is_quick_mode) {
        if (quick_txt2img) {
            if (!model_path) {
                printf("Error: --model is required for txt2img mode\n");
                return 1;
            }
            json_str = Txt2ImgBuilder::build(model_path, prompt, negative_prompt,
                                               width, height, seed, steps, cfg, output_prefix);
            printf("Generated txt2img workflow\n");
        } else if (quick_img2img) {
            if (!model_path || !input_image) {
                printf("Error: --model and --input are required for img2img mode\n");
                return 1;
            }
            json_str = Img2ImgBuilder::build(model_path, input_image, prompt, negative_prompt,
                                               denoise, seed, steps, cfg, output_prefix);
            printf("Generated img2img workflow\n");
        } else if (quick_process) {
            if (!input_image) {
                printf("Error: --input is required for process mode\n");
                return 1;
            }
            json_str = ImageProcessBuilder::build(input_image, scale_w, scale_h,
                                                   crop_x, crop_y, crop_w, crop_h, output_prefix);
            printf("Generated image process workflow\n");
        } else if (quick_deep_hires) {
            if (!model_path) {
                printf("Error: --model is required for deep-hires mode\n");
                return 1;
            }
            std::string img_input = input_image ? input_image : "";
            float strength_val = img_input.empty() ? 1.0f : denoise;
            json_str = DeepHiresBuilder::build(model_path, prompt, negative_prompt,
                                               target_width, target_height, seed, steps, cfg,
                                               img_input, strength_val, vae_tiling, output_prefix);
            printf("Generated Deep HighRes Fix workflow\n");
        }
        
        if (save_json) {
            WorkflowBuilder builder;
            // Re-parse and save (or we could directly write json_str)
            // For simplicity, just write the string
            FILE* f = fopen(save_json, "w");
            if (f) {
                fprintf(f, "%s", json_str.c_str());
                fclose(f);
                printf("Saved workflow to: %s\n", save_json);
            } else {
                printf("Warning: Failed to save workflow to %s\n", save_json);
            }
        }
    } else if (workflow_file) {
        // Normal workflow file mode
        printf("Loading workflow: %s\n", workflow_file);
    } else {
        printf("Error: Either --workflow or a quick mode (--txt2img/--img2img/--process) is required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    Workflow workflow;
    bool loaded = false;
    
    if (is_quick_mode) {
        loaded = workflow.load_from_string(json_str);
    } else {
        loaded = workflow.load_from_file(workflow_file);
    }
    
    if (!loaded) {
        printf("Error: Failed to load workflow\n");
        return 1;
    }
    
    bool success = execute_workflow(workflow, dry_run, verbose);
    return success ? 0 : 1;
}
