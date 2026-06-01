#include "cli/cli_parser.h"
#include "cli/cli_options.h"
#include "utils/log.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

namespace myimg {

void print_usage(const char* argv0) {
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
    std::cout << "  --hires-upscaler NAME     HiRes upscaler: latent, latent-nearest, latent-nearest-exact,\n";
    std::cout << "                            latent-antialiased, latent-bicubic, latent-bicubic-antialiased,\n";
    std::cout << "                            lanczos, nearest, model (default: latent)\n";
    std::cout << "  --hires-scale FLOAT       HiRes scale factor (default: 2.0)\n";
    std::cout << "  --hires-model PATH        External upscaler model path (for model upscaler)\n";
    std::cout << "  --hires-tile-size INT     HiRes upscale tile size (default: 128)\n";
    std::cout << "\nAdvanced Options:\n";
    std::cout << "  --vae-format FORMAT       VAE format: auto, flux, sd3, flux2 (default: auto)\n";
    std::cout << "  --backend SPEC            Backend spec (e.g. cuda:0)\n";
    std::cout << "  --params-backend SPEC     Params backend spec\n";
    std::cout << "  --audio-vae PATH          Audio VAE path (for video generation)\n";
    std::cout << "  --embeddings-connectors PATH  Embeddings connectors path\n";
    std::cout << "  --extra-sample-args ARGS  Extra sampler arguments\n";
    std::cout << "  --vae-temporal-tiling     Enable VAE temporal tiling (for video)\n";
    std::cout << "  --extra-tiling-args ARGS  Extra VAE tiling arguments\n";
    std::cout << "\nUpscale Options:\n";
    std::cout << "  --upscale-repeats INT     ESRGAN upscale repeats (default: 1)\n";
    std::cout << "  --upscale-tile-size INT   ESRGAN tile size (default: 1440)\n";
    std::cout << "\nEmbedding Options:\n";
    std::cout << "  --embd-dir PATH           Embeddings directory (Textual Inversion)\n";
    std::cout << "\nConfig Options:\n";
    std::cout << "  --config PATH             Load generation parameters from JSON config file\n";
    std::cout << "\nPreview Options:\n";
    std::cout << "  --preview                 Enable preview during generation\n";
    std::cout << "  --preview-interval INT    Preview save interval in steps (default: 1)\n";
    std::cout << "  --preview-mode MODE       Preview mode: vae, tae, proj (default: vae)\n";
    std::cout << "  --preview-dir PATH        Preview output directory (default: /tmp/myimg-preview)\n";
    std::cout << "\nLog & Report Options:\n";
    std::cout << "  --log-level LEVEL         Log level: trace, debug, info, warn, error, fatal (default: info)\n";
    std::cout << "  --report PATH             Save generation report to JSON file\n";
    std::cout << "  --show-vram               Show VRAM usage during generation\n";
    std::cout << "\nOutput Options:\n";
    std::cout << "  -o, --output PATH         Output path (default: output.png)\n";
    std::cout << "  --embed-metadata          Embed generation parameters in PNG metadata\n";
    std::cout << "  --quality INT             JPEG quality 1-100 (default: 95)\n";
    std::cout << "\nImage Transformation:\n";
    std::cout << "  --resize WxH              Resize image (e.g. 1920x1080)\n";
    std::cout << "  --resize-mode MODE        Resize mode: nearest, bilinear, bicubic\n";
    std::cout << "  --flip-h                  Flip horizontally\n";
    std::cout << "  --flip-v                  Flip vertically\n";
    std::cout << "  --rotate DEG              Rotate 90, 180, or 270 degrees\n";
    std::cout << "  --crop x,y,w,h            Crop image (pixel coordinates)\n";
    std::cout << "  --crop-center w,h         Center crop to specified size\n";
    std::cout << "  --crop-ratio w:h          Crop to aspect ratio (e.g. 16:9, 1:1, 4:3)\n";
    std::cout << "\nControlNet Preprocessors:\n";
    std::cout << "  --control-preprocessor NAME  Preprocessor: canny, lineart, normal, scribble, depth, openpose\n";
    std::cout << "  --control-preprocessor-param1 INT  Parameter 1 (threshold)\n";
    std::cout << "  --control-preprocessor-param2 INT  Parameter 2 (Canny high threshold)\n";
    std::cout << "  --depth-model PATH        Path to MiDaS depth model (.pt)\n";
    std::cout << "  --openpose-model PATH     Path to OpenPose body model (.pt)\n";
    std::cout << "\nBatch Processing:\n";
    std::cout << "  --batch-input-dir PATH    Input directory for batch processing\n";
    std::cout << "  --batch-output-dir PATH   Output directory for batch processing\n";
    std::cout << "  --output-template TPL     Output filename template (default: {name}{ext})\n";
    std::cout << "                            Placeholders: {name}, {ext}, {index}, {index:N}\n";
    std::cout << "\nAdvanced Generation Options:\n";
    std::cout << "  --prompt-schedule STR     Prompt schedule: \"0-10:prompt1|11-20:prompt2\"\n";
    std::cout << "  --regional-prompts STR    Regional prompts: \"top:0.3,prompt1|bottom:0.3,prompt2\"\n";
    std::cout << "\nFace Restoration:\n";
    std::cout << "  --face-restore            Enable face restoration\n";
    std::cout << "  --face-restore-model PATH GFPGAN/CodeFormer model path\n";
    std::cout << "  --face-restore-fidelity F Fidelity 0.0-1.0 (default: 0.5)\n";
    std::cout << "\nIPAdapter (Image Prompt):\n";
    std::cout << "  --ipadapter               Enable IPAdapter\n";
    std::cout << "  --ipadapter-model PATH    IPAdapter model path\n";
    std::cout << "  --ipadapter-clip-vision PATH  CLIP Vision model path\n";
    std::cout << "  --ipadapter-image PATH    Reference image path\n";
    std::cout << "  --ipadapter-weight FLOAT  Weight 0.0-1.0 (default: 1.0)\n";
    std::cout << "  --ipadapter-start FLOAT   Start step ratio 0.0-1.0 (default: 0.0)\n";
    std::cout << "  --ipadapter-end FLOAT     End step ratio 0.0-1.0 (default: 1.0)\n";
    std::cout << "\nT2I-Adapter:\n";
    std::cout << "  --t2i-adapter             Enable T2I-Adapter\n";
    std::cout << "  --t2i-adapter-model PATH  T2I-Adapter model path\n";
    std::cout << "  --t2i-adapter-image PATH  Condition image path\n";
    std::cout << "  --t2i-adapter-strength F  Strength 0.0-1.0 (default: 1.0)\n";
    std::cout << "\nFace Swap:\n";
    std::cout << "  --face-swap               Enable face swap\n";
    std::cout << "  --face-swap-source PATH   Source face image\n";
    std::cout << "  --face-swap-detection-model PATH  Face detection model\n";
    std::cout << "  --face-swap-model PATH    Face swap model\n";
    std::cout << "\nPhotoMaker:\n";
    std::cout << "  --photomaker              Enable PhotoMaker\n";
    std::cout << "  --photomaker-model PATH   PhotoMaker model path\n";
    std::cout << "  --photomaker-id-images LIST  Comma-separated ID images\n";
    std::cout << "  --photomaker-id-weight F  ID weight 0.0-1.0 (default: 1.0)\n";
    std::cout << "\nPhoto Adjustment Options:\n";
    std::cout << "  --temperature FLOAT       Color temperature -1.0(cold) to 1.0(warm)\n";
    std::cout << "  --brightness FLOAT        Brightness -1.0 to 1.0\n";
    std::cout << "  --contrast FLOAT          Contrast -1.0 to 1.0\n";
    std::cout << "  --saturation FLOAT        Saturation -1.0 to 1.0\n";
    std::cout << "  --exposure FLOAT          Exposure EV -5.0 to 5.0\n";
    std::cout << "  --highlights FLOAT        Highlights -100 to 100\n";
    std::cout << "  --shadows FLOAT           Shadows -100 to 100\n";
    std::cout << "  --auto-enhance            Auto one-click photo enhancement\n";
    std::cout << "\nVibrance & Clarity:\n";
    std::cout << "  --vibrance FLOAT          Vibrance -1.0 to 1.0 (smart saturation)\n";
    std::cout << "  --clarity FLOAT           Clarity/texture enhancement 0.0-1.0\n";
    std::cout << "  --split-tone-highlights HEX  Highlight color for split toning (default: #FFE4C4)\n";
    std::cout << "  --split-tone-shadows HEX     Shadow color for split toning (default: #4A6741)\n";
    std::cout << "  --split-tone-strength FLOAT  Split tone strength 0.0-1.0 (default: 0)\n";
    std::cout << "\nColor Balance:\n";
    std::cout << "  --tint FLOAT              Tint -1.0(magenta) to 1.0(green)\n";
    std::cout << "  --auto-white-balance      Auto white balance (gray world)\n";
    std::cout << "\nLevels Options:\n";
    std::cout << "  --blacks FLOAT            Black level -100 to 100 (neg=crush, pos=lift)\n";
    std::cout << "  --whites FLOAT            White level -100 to 100 (neg=reduce, pos=boost)\n";
    std::cout << "\nCurves Options:\n";
    std::cout << "  --curves \"in,out;in,out\"  RGB curves (0-255, e.g. \"0,0;128,140;255,255\")\n";
    std::cout << "  --brightness-curves \"in,out;...\"  Brightness curves (applies to luma only)\n";
    std::cout << "  --r-curves \"in,out;...\"  Red channel curves\n";
    std::cout << "  --g-curves \"in,out;...\"  Green channel curves\n";
    std::cout << "  --b-curves \"in,out;...\"  Blue channel curves\n";
    std::cout << "\nVignette Options:\n";
    std::cout << "  --vignette FLOAT          Vignette strength 0.0-1.0\n";
    std::cout << "  --vignette-radius FLOAT   Vignette radius 0.0-1.0 (default: 0.75)\n";
    std::cout << "\nLocal Adjustment Options:\n";
    std::cout << "  --radial-filter cx,cy,radius,exp,cont,sat  Radial filter (e.g. 0.5,0.5,0.3,0.5,0,0)\n";
    std::cout << "  --graduated-filter angle,pos,width,exp,cont,sat  Graduated filter (e.g. 0,0.5,0.2,0.3,0,0)\n";
    std::cout << "\nFilter Presets:\n";
    std::cout << "  --preset NAME             Apply filter preset: bw, sepia, vintage, warm,\n";
    std::cout << "                            cool, dramatic, japanese, film, cyberpunk, cinematic,\n";
    std::cout << "                            portra, velvia, provia, trix, kodachrome, instant\n";
    std::cout << "\nColor Grading:\n";
    std::cout << "  --lut PATH                Load 3D LUT file (.cube format)\n";
    std::cout << "\nImage Restoration:\n";
    std::cout << "  --dehaze FLOAT            Dehaze strength 0.0-1.0 (default: 0)\n";
    std::cout << "\nSharpening & Denoising:\n";
    std::cout << "  --sharpen FLOAT           USM sharpen amount 0.0-3.0\n";
    std::cout << "  --sharpen-radius INT      USM sharpen radius 1-5 (default: 1)\n";
    std::cout << "  --sharpen-threshold FLOAT USM sharpen threshold 0-255 (default: 0)\n";
    std::cout << "  --smart-sharpen FLOAT     Smart sharpening (edge-aware) 0.0-3.0\n";
    std::cout << "  --smart-sharpen-radius INT  Smart sharpen radius 1-5 (default: 2)\n";
    std::cout << "  --edge-sharpen FLOAT      Edge-mask sharpening 0.0-3.0 (avoids halos)\n";
    std::cout << "  --edge-sharpen-radius INT   Edge sharpen radius 1-5 (default: 2)\n";
    std::cout << "  --edge-sharpen-threshold FLOAT  Edge threshold 0.0-1.0 (default: 0.3)\n";
    std::cout << "  --denoise FLOAT           Denoise strength 0.0-1.0\n";
    std::cout << "  --luminance-denoise FLOAT Luminance noise reduction 0.0-1.0\n";
    std::cout << "  --color-denoise FLOAT     Color noise reduction 0.0-1.0\n";
    std::cout << "\nPortrait Retouching:\n";
    std::cout << "  --whiten FLOAT            Teeth/eye whitening 0.0-1.0\n";
    std::cout << "  --skin-smooth FLOAT       Skin smoothing 0.0-1.0\n";
    std::cout << "  --skin-tone MODE          Skin tone: warm, cool, neutral\n";
    std::cout << "  --skin-tone-strength FLOAT Skin tone strength 0.0-1.0\n";
    std::cout << "  --skin-even FLOAT         Skin tone evening 0.0-1.0\n";
    std::cout << "\nImage Interrogation:\n";
    std::cout << "  --interrogate PATH        Image path for caption/description\n";
    std::cout << "                            (requires JoyCaption model - placeholder)\n";
    std::cout << "  --read-metadata PATH      Read PNG metadata (prompt/parameters)\n";
    std::cout << "\nPreset Options:\n";
    std::cout << "  --save-preset NAME        Save current settings as preset (.json)\n";
    std::cout << "  --load-preset PATH        Load settings from preset file\n";
    std::cout << "\nSystem Options:\n";
    std::cout << "  --threads INT             Number of CPU threads (default: auto)\n";
    std::cout << "  --max-vram FLOAT          Max VRAM limit in GB (default: 0 = unlimited)\n";
    std::cout << "  -v, --verbose             Verbose logging\n";
    std::cout << "  --help                    Show this help\n";
}

// Parse embedding syntax in prompt: replaces "embedding:name" with "name"
// and returns list of referenced embedding names

std::string parse_embedding_syntax(const std::string& prompt, std::vector<std::string>& referenced_embeddings) {
    std::string result = prompt;
    size_t pos = 0;
    
    while ((pos = result.find("embedding:", pos)) != std::string::npos) {
        size_t start = pos;
        size_t end = pos + 10; // length of "embedding:"
        
        // Find end of embedding name (space, comma, or end of string)
        while (end < result.size() && result[end] != ' ' && result[end] != ',' && result[end] != '\t') {
            end++;
        }
        
        if (end > start + 10) {
            std::string emb_name = result.substr(start + 10, end - start - 10);
            referenced_embeddings.push_back(emb_name);
            
            // Replace "embedding:name" with just "name"
            result.replace(start, end - start, emb_name);
            pos = start + emb_name.size();
        } else {
            pos = end;
        }
    }
    
    return result;
}

// Expand output filename template
// Supported placeholders: {name}, {ext}, {index}, {index:N} (zero-padded)

std::string expand_output_template(const std::string& template_str, 
                                           const std::string& filename,
                                           int index) {
    std::string result = template_str.empty() ? "{name}{ext}" : template_str;
    
    // Extract name and extension
    size_t dot_pos = filename.rfind('.');
    std::string name = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename;
    std::string ext = (dot_pos != std::string::npos) ? filename.substr(dot_pos) : "";
    
    // Replace {name}
    size_t pos = 0;
    while ((pos = result.find("{name}", pos)) != std::string::npos) {
        result.replace(pos, 6, name);
        pos += name.size();
    }
    
    // Replace {ext}
    pos = 0;
    while ((pos = result.find("{ext}", pos)) != std::string::npos) {
        result.replace(pos, 5, ext);
        pos += ext.size();
    }
    
    // Replace {index:N} with zero-padded index
    pos = 0;
    while ((pos = result.find("{index:", pos)) != std::string::npos) {
        size_t end = result.find('}', pos);
        if (end != std::string::npos) {
            try {
                int padding = std::stoi(result.substr(pos + 7, end - pos - 7));
                std::ostringstream oss;
                oss << std::setw(padding) << std::setfill('0') << index;
                std::string idx_str = oss.str();
                result.replace(pos, end - pos + 1, idx_str);
                pos += idx_str.size();
            } catch (const std::exception&) {
                LOG_WARN("Invalid index padding in template, skipping");
                break;
            }
        } else {
            break;
        }
    }
    
    // Replace {index} (no padding)
    pos = 0;
    while ((pos = result.find("{index}", pos)) != std::string::npos) {
        std::string idx_str = std::to_string(index);
        result.replace(pos, 7, idx_str);
        pos += idx_str.size();
    }
    
    return result;
}

// Save current options as a preset JSON file

bool save_preset(const CliOptions& opts, const std::string& preset_name) {
    nlohmann::json j;
    
    // Photo adjustments
    j["temperature"] = opts.temperature;
    j["brightness"] = opts.brightness;
    j["contrast"] = opts.contrast;
    j["saturation"] = opts.saturation;
    j["exposure"] = opts.exposure;
    j["highlights"] = opts.highlights;
    j["shadows"] = opts.shadows;
    j["auto_enhance"] = opts.auto_enhance;
    j["vibrance"] = opts.vibrance;
    j["clarity"] = opts.clarity;
    j["split_tone_highlights"] = opts.split_tone_highlights;
    j["split_tone_shadows"] = opts.split_tone_shadows;
    j["split_tone_strength"] = opts.split_tone_strength;
    j["tint"] = opts.tint;
    j["auto_white_balance"] = opts.auto_white_balance;
    j["blacks"] = opts.blacks;
    j["whites"] = opts.whites;
    j["curves"] = opts.curves;
    j["brightness_curves"] = opts.brightness_curves;
    j["r_curves"] = opts.r_curves;
    j["g_curves"] = opts.g_curves;
    j["b_curves"] = opts.b_curves;
    j["preset"] = opts.preset;
    j["vignette_strength"] = opts.vignette_strength;
    j["vignette_radius"] = opts.vignette_radius;
    j["radial_filter"] = opts.radial_filter;
    j["graduated_filter"] = opts.graduated_filter;
    j["lut_path"] = opts.lut_path;
    j["dehaze_strength"] = opts.dehaze_strength;
    j["sharpen_amount"] = opts.sharpen_amount;
    j["sharpen_radius"] = opts.sharpen_radius;
    j["sharpen_threshold"] = opts.sharpen_threshold;
    j["denoise_strength"] = opts.denoise_strength;
    j["luminance_denoise_strength"] = opts.luminance_denoise_strength;
    j["color_denoise_strength"] = opts.color_denoise_strength;
    j["whiten_strength"] = opts.whiten_strength;
    j["skin_smooth_strength"] = opts.skin_smooth_strength;
    j["skin_tone"] = opts.skin_tone;
    j["skin_tone_strength"] = opts.skin_tone_strength;
    j["skin_even_strength"] = opts.skin_even_strength;
    j["resize_width"] = opts.resize_width;
    j["resize_height"] = opts.resize_height;
    j["resize_mode"] = opts.resize_mode;
    j["flip_h"] = opts.flip_h;
    j["flip_v"] = opts.flip_v;
    j["rotate"] = opts.rotate;
    j["crop"] = opts.crop;
    j["crop_center"] = opts.crop_center;
    j["crop_ratio"] = opts.crop_ratio;
    j["jpeg_quality"] = opts.jpeg_quality;
    
    std::string preset_path = preset_name;
    if (preset_path.find('.') == std::string::npos) {
        preset_path += ".json";
    }
    
    std::ofstream file(preset_path);
    if (!file) {
        LOG_ERROR("Failed to save preset to %s", preset_path.c_str());
        return false;
    }
    file << j.dump(2);
    std::cout << "Preset saved to: " << preset_path << "\n";
    return true;
}

// Load preset from JSON file

bool load_preset(CliOptions& opts, const std::string& preset_path) {
    std::ifstream file(preset_path);
    if (!file) {
        LOG_ERROR("Failed to load preset from %s", preset_path.c_str());
        return false;
    }
    
    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        LOG_ERROR("Invalid preset JSON: %s", e.what());
        return false;
    }
    
    // Helper to safely get values
    auto get_float = [&j](const std::string& key, float& val) {
        if (j.contains(key) && !j[key].is_null()) val = j[key].get<float>();
    };
    auto get_int = [&j](const std::string& key, int& val) {
        if (j.contains(key) && !j[key].is_null()) val = j[key].get<int>();
    };
    auto get_bool = [&j](const std::string& key, bool& val) {
        if (j.contains(key) && !j[key].is_null()) val = j[key].get<bool>();
    };
    auto get_string = [&j](const std::string& key, std::string& val) {
        if (j.contains(key) && !j[key].is_null()) val = j[key].get<std::string>();
    };
    
    get_float("temperature", opts.temperature);
    get_float("brightness", opts.brightness);
    get_float("contrast", opts.contrast);
    get_float("saturation", opts.saturation);
    get_float("exposure", opts.exposure);
    get_float("highlights", opts.highlights);
    get_float("shadows", opts.shadows);
    get_bool("auto_enhance", opts.auto_enhance);
    get_float("vibrance", opts.vibrance);
    get_float("clarity", opts.clarity);
    get_string("split_tone_highlights", opts.split_tone_highlights);
    get_string("split_tone_shadows", opts.split_tone_shadows);
    get_float("split_tone_strength", opts.split_tone_strength);
    get_float("tint", opts.tint);
    get_bool("auto_white_balance", opts.auto_white_balance);
    get_float("blacks", opts.blacks);
    get_float("whites", opts.whites);
    get_string("curves", opts.curves);
    get_string("brightness_curves", opts.brightness_curves);
    get_string("r_curves", opts.r_curves);
    get_string("g_curves", opts.g_curves);
    get_string("b_curves", opts.b_curves);
    get_string("preset", opts.preset);
    get_float("vignette_strength", opts.vignette_strength);
    get_float("vignette_radius", opts.vignette_radius);
    get_string("radial_filter", opts.radial_filter);
    get_string("graduated_filter", opts.graduated_filter);
    get_string("lut_path", opts.lut_path);
    get_float("dehaze_strength", opts.dehaze_strength);
    get_float("sharpen_amount", opts.sharpen_amount);
    get_int("sharpen_radius", opts.sharpen_radius);
    get_float("sharpen_threshold", opts.sharpen_threshold);
    get_float("denoise_strength", opts.denoise_strength);
    get_float("luminance_denoise_strength", opts.luminance_denoise_strength);
    get_float("color_denoise_strength", opts.color_denoise_strength);
    get_float("whiten_strength", opts.whiten_strength);
    get_float("skin_smooth_strength", opts.skin_smooth_strength);
    get_string("skin_tone", opts.skin_tone);
    get_float("skin_tone_strength", opts.skin_tone_strength);
    get_float("skin_even_strength", opts.skin_even_strength);
    get_int("resize_width", opts.resize_width);
    get_int("resize_height", opts.resize_height);
    get_string("resize_mode", opts.resize_mode);
    get_bool("flip_h", opts.flip_h);
    get_bool("flip_v", opts.flip_v);
    get_int("rotate", opts.rotate);
    get_string("crop", opts.crop);
    get_string("crop_center", opts.crop_center);
    get_string("crop_ratio", opts.crop_ratio);
    get_int("jpeg_quality", opts.jpeg_quality);
    
    std::cout << "Preset loaded from: " << preset_path << "\n";
    return true;
}

[[maybe_unused]] static bool load_config(const std::string& config_path, CliOptions& opts) {
    std::ifstream file(config_path);
    if (!file) {
        LOG_ERROR("Cannot open config file: %s", config_path.c_str());
        return false;
    }

    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }

    auto get_string = [&](const char* key, std::string& target) {
        if (j.contains(key) && j[key].is_string()) target = j[key].get<std::string>();
    };
    auto get_int = [&](const char* key, int& target) {
        if (j.contains(key) && j[key].is_number_integer()) target = j[key].get<int>();
    };
    auto get_float = [&](const char* key, float& target) {
        if (j.contains(key) && j[key].is_number()) target = j[key].get<float>();
    };
    auto get_bool = [&](const char* key, bool& target) {
        if (j.contains(key) && j[key].is_boolean()) target = j[key].get<bool>();
    };

    get_string("model", opts.model);
    get_string("diffusion_model", opts.diffusion_model);
    get_string("vae", opts.vae);
    get_string("llm", opts.llm);
    get_string("upscale_model", opts.upscale_model);
    get_string("prompt", opts.prompt);
    get_string("negative_prompt", opts.negative_prompt);
    get_int("width", opts.width);
    get_int("height", opts.height);
    get_int("steps", opts.steps);
    get_float("cfg_scale", opts.cfg_scale);
    get_string("sampling_method", opts.sampling_method);
    get_string("scheduler", opts.scheduler);
    if (j.contains("seed")) {
        if (j["seed"].is_number_integer()) opts.seed = j["seed"].get<int64_t>();
    }
    get_int("batch_count", opts.batch_count);
    get_bool("diffusion_fa", opts.diffusion_fa);
    get_bool("vae_tiling", opts.vae_tiling);
    get_int("vae_tile_size_w", opts.vae_tile_size_w);
    get_int("vae_tile_size_h", opts.vae_tile_size_h);
    get_float("vae_tile_overlap", opts.vae_tile_overlap);
    get_bool("hires", opts.hires);
    get_int("hires_width", opts.hires_width);
    get_int("hires_height", opts.hires_height);
    get_float("hires_strength", opts.hires_strength);
    get_int("hires_steps", opts.hires_steps);
    get_string("hires_upscaler", opts.hires_upscaler);
    get_float("hires_scale", opts.hires_scale);
    get_string("hires_model_path", opts.hires_model_path);
    get_int("hires_tile_size", opts.hires_tile_size);
    get_string("vae_format", opts.vae_format);
    get_string("backend", opts.backend);
    get_string("params_backend", opts.params_backend);
    get_string("audio_vae_path", opts.audio_vae_path);
    get_string("embeddings_connectors_path", opts.embeddings_connectors_path);
    get_string("extra_sample_args", opts.extra_sample_args);
    get_bool("vae_temporal_tiling", opts.vae_temporal_tiling);
    get_string("extra_tiling_args", opts.extra_tiling_args);
    get_bool("embed_metadata", opts.embed_metadata);
    get_string("output", opts.output);
    get_string("init_image", opts.init_image);
    get_float("strength", opts.strength);
    get_string("mask_image", opts.mask_image);
    get_string("control_net", opts.control_net);
    get_string("control_image", opts.control_image);
    get_float("control_strength", opts.control_strength);
    get_string("embedding_dir", opts.embedding_dir);

    if (j.contains("loras") && j["loras"].is_array()) {
        for (const auto& lora : j["loras"]) {
            if (lora.is_string()) opts.loras.push_back(lora.get<std::string>());
        }
    }

    std::cout << "Config loaded from: " << config_path << "\n";
    return true;
}

// Helper for safe type conversion with error handling
template <typename T>
static bool safe_convert(const char* str, T& out, const std::string& arg_name) {
    try {
        if constexpr (std::is_same_v<T, int>) {
            out = std::stoi(str);
        } else if constexpr (std::is_same_v<T, float>) {
            out = std::stof(str);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            out = std::stoll(str);
        }
        return true;
    } catch (const std::exception&) {
        LOG_ERROR("Invalid value for %s: %s", arg_name.c_str(), str);
        return false;
    }
}


bool parse_args(int argc, char** argv, CliOptions& opts) {
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
            if (++i >= argc) { LOG_ERROR("Missing value for -m/--model"); return false; }
            opts.model = argv[i];
        } else if (arg == "--diffusion-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --diffusion-model"); return false; }
            opts.diffusion_model = argv[i];
        } else if (arg == "--vae") {
            if (++i >= argc) { LOG_ERROR("Missing value for --vae"); return false; }
            opts.vae = argv[i];
        } else if (arg == "--llm") {
            if (++i >= argc) { LOG_ERROR("Missing value for --llm"); return false; }
            opts.llm = argv[i];
        } else if (arg == "--upscale-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --upscale-model"); return false; }
            opts.upscale_model = argv[i];
        } else if (arg == "--embd-dir") {
            if (++i >= argc) { LOG_ERROR("Missing value for --embd-dir"); return false; }
            opts.embedding_dir = argv[i];
        } else if (arg == "--config") {
            if (++i >= argc) { LOG_ERROR("Missing value for --config"); return false; }
            opts.config_file = argv[i];
        } else if (arg == "--workflow") {
            if (++i >= argc) { LOG_ERROR("Missing value for --workflow"); return false; }
            opts.workflow_file = argv[i];
        } else if (arg == "--preview") {
            opts.preview = true;
        } else if (arg == "--preview-interval") {
            if (++i >= argc) { LOG_ERROR("Missing value for --preview-interval"); return false; }
            if (!safe_convert(argv[i], opts.preview_interval, "--preview-interval")) return false;
        } else if (arg == "--preview-mode") {
            if (++i >= argc) { LOG_ERROR("Missing value for --preview-mode"); return false; }
            opts.preview_mode = argv[i];
        } else if (arg == "--preview-dir") {
            if (++i >= argc) { LOG_ERROR("Missing value for --preview-dir"); return false; }
            opts.preview_dir = argv[i];
        } else if (arg == "--log-level") {
            if (++i >= argc) { LOG_ERROR("Missing value for --log-level"); return false; }
            opts.log_level = argv[i];
        } else if (arg == "--report") {
            if (++i >= argc) { LOG_ERROR("Missing value for --report"); return false; }
            opts.report_path = argv[i];
        } else if (arg == "--show-vram") {
            opts.show_vram = true;
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) { LOG_ERROR("Missing value for -p/--prompt"); return false; }
            opts.prompt = argv[i];
        } else if (arg == "-n" || arg == "--negative-prompt") {
            if (++i >= argc) { LOG_ERROR("Missing value for -n/--negative-prompt"); return false; }
            opts.negative_prompt = argv[i];
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) { LOG_ERROR("Missing value for -W/--width"); return false; }
            if (!safe_convert(argv[i], opts.width, "--width")) return false;
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) { LOG_ERROR("Missing value for -H/--height"); return false; }
            if (!safe_convert(argv[i], opts.height, "--height")) return false;
        } else if (arg == "--steps") {
            if (++i >= argc) { LOG_ERROR("Missing value for --steps"); return false; }
            if (!safe_convert(argv[i], opts.steps, "--steps")) return false;
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) { LOG_ERROR("Missing value for --cfg-scale"); return false; }
            if (!safe_convert(argv[i], opts.cfg_scale, "--cfg-scale")) return false;
        } else if (arg == "--sampling-method") {
            if (++i >= argc) { LOG_ERROR("Missing value for --sampling-method"); return false; }
            opts.sampling_method = argv[i];
        } else if (arg == "--scheduler") {
            if (++i >= argc) { LOG_ERROR("Missing value for --scheduler"); return false; }
            opts.scheduler = argv[i];
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) { LOG_ERROR("Missing value for -s/--seed"); return false; }
            if (!safe_convert(argv[i], opts.seed, "--seed")) return false;
        } else if (arg == "--batch-count") {
            if (++i >= argc) { LOG_ERROR("Missing value for --batch-count"); return false; }
            if (!safe_convert(argv[i], opts.batch_count, "--batch-count")) return false;
        } else if (arg == "-i" || arg == "--init-img") {
            if (++i >= argc) { LOG_ERROR("Missing value for -i/--init-img"); return false; }
            opts.init_image = argv[i];
        } else if (arg == "--strength") {
            if (++i >= argc) { LOG_ERROR("Missing value for --strength"); return false; }
            if (!safe_convert(argv[i], opts.strength, "--strength")) return false;
        } else if (arg == "--mask") {
            if (++i >= argc) { LOG_ERROR("Missing value for --mask"); return false; }
            opts.mask_image = argv[i];
        } else if (arg == "--control-net") {
            if (++i >= argc) { LOG_ERROR("Missing value for --control-net"); return false; }
            opts.control_net = argv[i];
        } else if (arg == "--control-image") {
            if (++i >= argc) { LOG_ERROR("Missing value for --control-image"); return false; }
            opts.control_image = argv[i];
        } else if (arg == "--control-strength") {
            if (++i >= argc) { LOG_ERROR("Missing value for --control-strength"); return false; }
            if (!safe_convert(argv[i], opts.control_strength, "--control-strength")) return false;
        } else if (arg == "--diffusion-fa") {
            opts.diffusion_fa = true;
        } else if (arg == "--vae-tiling") {
            opts.vae_tiling = true;
        } else if (arg == "--vae-tile-size") {
            if (++i >= argc) { LOG_ERROR("Missing value for --vae-tile-size"); return false; }
            std::string val = argv[i];
            size_t x = val.find('x');
            if (x != std::string::npos) {
                if (!safe_convert(val.substr(0, x).c_str(), opts.vae_tile_size_w, "--vae-tile-size")) return false;
                if (!safe_convert(val.substr(x + 1).c_str(), opts.vae_tile_size_h, "--vae-tile-size")) return false;
            } else {
                int tile_size;
                if (!safe_convert(val.c_str(), tile_size, "--vae-tile-size")) return false;
                opts.vae_tile_size_w = opts.vae_tile_size_h = tile_size;
            }
        } else if (arg == "--vae-tile-overlap") {
            if (++i >= argc) { LOG_ERROR("Missing value for --vae-tile-overlap"); return false; }
            if (!safe_convert(argv[i], opts.vae_tile_overlap, "--vae-tile-overlap")) return false;
        } else if (arg == "--hires") {
            opts.hires = true;
        } else if (arg == "--hires-width") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-width"); return false; }
            if (!safe_convert(argv[i], opts.hires_width, "--hires-width")) return false;
        } else if (arg == "--hires-height") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-height"); return false; }
            if (!safe_convert(argv[i], opts.hires_height, "--hires-height")) return false;
        } else if (arg == "--hires-strength") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-strength"); return false; }
            if (!safe_convert(argv[i], opts.hires_strength, "--hires-strength")) return false;
        } else if (arg == "--hires-steps") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-steps"); return false; }
            if (!safe_convert(argv[i], opts.hires_steps, "--hires-steps")) return false;
        } else if (arg == "--hires-upscaler") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-upscaler"); return false; }
            opts.hires_upscaler = argv[i];
        } else if (arg == "--hires-scale") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-scale"); return false; }
            if (!safe_convert(argv[i], opts.hires_scale, "--hires-scale")) return false;
        } else if (arg == "--hires-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-model"); return false; }
            opts.hires_model_path = argv[i];
        } else if (arg == "--hires-tile-size") {
            if (++i >= argc) { LOG_ERROR("Missing value for --hires-tile-size"); return false; }
            if (!safe_convert(argv[i], opts.hires_tile_size, "--hires-tile-size")) return false;
        } else if (arg == "--vae-format") {
            if (++i >= argc) { LOG_ERROR("Missing value for --vae-format"); return false; }
            opts.vae_format = argv[i];
        } else if (arg == "--backend") {
            if (++i >= argc) { LOG_ERROR("Missing value for --backend"); return false; }
            opts.backend = argv[i];
        } else if (arg == "--params-backend") {
            if (++i >= argc) { LOG_ERROR("Missing value for --params-backend"); return false; }
            opts.params_backend = argv[i];
        } else if (arg == "--audio-vae") {
            if (++i >= argc) { LOG_ERROR("Missing value for --audio-vae"); return false; }
            opts.audio_vae_path = argv[i];
        } else if (arg == "--embeddings-connectors") {
            if (++i >= argc) { LOG_ERROR("Missing value for --embeddings-connectors"); return false; }
            opts.embeddings_connectors_path = argv[i];
        } else if (arg == "--extra-sample-args") {
            if (++i >= argc) { LOG_ERROR("Missing value for --extra-sample-args"); return false; }
            opts.extra_sample_args = argv[i];
        } else if (arg == "--vae-temporal-tiling") {
            opts.vae_temporal_tiling = true;
        } else if (arg == "--extra-tiling-args") {
            if (++i >= argc) { LOG_ERROR("Missing value for --extra-tiling-args"); return false; }
            opts.extra_tiling_args = argv[i];
        } else if (arg == "--upscale-repeats") {
            if (++i >= argc) { LOG_ERROR("Missing value for --upscale-repeats"); return false; }
            if (!safe_convert(argv[i], opts.upscale_repeats, "--upscale-repeats")) return false;
        } else if (arg == "--lora") {
            if (++i >= argc) { LOG_ERROR("Missing value for --lora"); return false; }
            opts.loras.push_back(argv[i]);
        } else if (arg == "--upscale-tile-size") {
            if (++i >= argc) { LOG_ERROR("Missing value for --upscale-tile-size"); return false; }
            if (!safe_convert(argv[i], opts.upscale_tile_size, "--upscale-tile-size")) return false;
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { LOG_ERROR("Missing value for -o/--output"); return false; }
            opts.output = argv[i];
        } else if (arg == "--embed-metadata") {
            opts.embed_metadata = true;
        } else if (arg == "--quality") {
            if (++i >= argc) { LOG_ERROR("Missing value for --quality"); return false; }
            if (!safe_convert(argv[i], opts.jpeg_quality, "--jpeg-quality")) return false;
        } else if (arg == "--threads") {
            if (++i >= argc) { LOG_ERROR("Missing value for --threads"); return false; }
            if (!safe_convert(argv[i], opts.threads, "--threads")) return false;
        } else if (arg == "--max-vram") {
            if (++i >= argc) { LOG_ERROR("Missing value for --max-vram"); return false; }
            if (!safe_convert(argv[i], opts.max_vram, "--max-vram")) return false;
        } else if (arg == "--temperature") {
            if (++i >= argc) { LOG_ERROR("Missing value for --temperature"); return false; }
            if (!safe_convert(argv[i], opts.temperature, "--temperature")) return false;
        } else if (arg == "--brightness") {
            if (++i >= argc) { LOG_ERROR("Missing value for --brightness"); return false; }
            if (!safe_convert(argv[i], opts.brightness, "--brightness")) return false;
        } else if (arg == "--contrast") {
            if (++i >= argc) { LOG_ERROR("Missing value for --contrast"); return false; }
            if (!safe_convert(argv[i], opts.contrast, "--contrast")) return false;
        } else if (arg == "--saturation") {
            if (++i >= argc) { LOG_ERROR("Missing value for --saturation"); return false; }
            if (!safe_convert(argv[i], opts.saturation, "--saturation")) return false;
        } else if (arg == "--exposure") {
            if (++i >= argc) { LOG_ERROR("Missing value for --exposure"); return false; }
            if (!safe_convert(argv[i], opts.exposure, "--exposure")) return false;
        } else if (arg == "--highlights") {
            if (++i >= argc) { LOG_ERROR("Missing value for --highlights"); return false; }
            if (!safe_convert(argv[i], opts.highlights, "--highlights")) return false;
        } else if (arg == "--shadows") {
            if (++i >= argc) { LOG_ERROR("Missing value for --shadows"); return false; }
            if (!safe_convert(argv[i], opts.shadows, "--shadows")) return false;
        } else if (arg == "--auto-enhance") {
            opts.auto_enhance = true;
        } else if (arg == "--vibrance") {
            if (++i >= argc) { LOG_ERROR("Missing value for --vibrance"); return false; }
            if (!safe_convert(argv[i], opts.vibrance, "--vibrance")) return false;
        } else if (arg == "--clarity") {
            if (++i >= argc) { LOG_ERROR("Missing value for --clarity"); return false; }
            if (!safe_convert(argv[i], opts.clarity, "--clarity")) return false;
        } else if (arg == "--split-tone-highlights") {
            if (++i >= argc) { LOG_ERROR("Missing value for --split-tone-highlights"); return false; }
            opts.split_tone_highlights = argv[i];
        } else if (arg == "--split-tone-shadows") {
            if (++i >= argc) { LOG_ERROR("Missing value for --split-tone-shadows"); return false; }
            opts.split_tone_shadows = argv[i];
        } else if (arg == "--split-tone-strength") {
            if (++i >= argc) { LOG_ERROR("Missing value for --split-tone-strength"); return false; }
            if (!safe_convert(argv[i], opts.split_tone_strength, "--split-tone-strength")) return false;
        } else if (arg == "--tint") {
            if (++i >= argc) { LOG_ERROR("Missing value for --tint"); return false; }
            if (!safe_convert(argv[i], opts.tint, "--tint")) return false;
        } else if (arg == "--auto-white-balance") {
            opts.auto_white_balance = true;
        } else if (arg == "--blacks") {
            if (++i >= argc) { LOG_ERROR("Missing value for --blacks"); return false; }
            if (!safe_convert(argv[i], opts.blacks, "--blacks")) return false;
        } else if (arg == "--whites") {
            if (++i >= argc) { LOG_ERROR("Missing value for --whites"); return false; }
            if (!safe_convert(argv[i], opts.whites, "--whites")) return false;
        } else if (arg == "--curves") {
            if (++i >= argc) { LOG_ERROR("Missing value for --curves"); return false; }
            opts.curves = argv[i];
        } else if (arg == "--vignette") {
            if (++i >= argc) { LOG_ERROR("Missing value for --vignette"); return false; }
            if (!safe_convert(argv[i], opts.vignette_strength, "--vignette-strength")) return false;
        } else if (arg == "--vignette-radius") {
            if (++i >= argc) { LOG_ERROR("Missing value for --vignette-radius"); return false; }
            if (!safe_convert(argv[i], opts.vignette_radius, "--vignette-radius")) return false;
        } else if (arg == "--preset") {
            if (++i >= argc) { LOG_ERROR("Missing value for --preset"); return false; }
            opts.preset = argv[i];
        } else if (arg == "--lut") {
            if (++i >= argc) { LOG_ERROR("Missing value for --lut"); return false; }
            opts.lut_path = argv[i];
        } else if (arg == "--whiten") {
            if (++i >= argc) { LOG_ERROR("Missing value for --whiten"); return false; }
            if (!safe_convert(argv[i], opts.whiten_strength, "--whiten-strength")) return false;
        } else if (arg == "--skin-smooth") {
            if (++i >= argc) { LOG_ERROR("Missing value for --skin-smooth"); return false; }
            if (!safe_convert(argv[i], opts.skin_smooth_strength, "--skin-smooth-strength")) return false;
        } else if (arg == "--skin-tone") {
            if (++i >= argc) { LOG_ERROR("Missing value for --skin-tone"); return false; }
            opts.skin_tone = argv[i];
        } else if (arg == "--skin-tone-strength") {
            if (++i >= argc) { LOG_ERROR("Missing value for --skin-tone-strength"); return false; }
            if (!safe_convert(argv[i], opts.skin_tone_strength, "--skin-tone-strength")) return false;
        } else if (arg == "--skin-even") {
            if (++i >= argc) { LOG_ERROR("Missing value for --skin-even"); return false; }
            if (!safe_convert(argv[i], opts.skin_even_strength, "--skin-even-strength")) return false;
        } else if (arg == "--dehaze") {
            if (++i >= argc) { LOG_ERROR("Missing value for --dehaze"); return false; }
            if (!safe_convert(argv[i], opts.dehaze_strength, "--dehaze-strength")) return false;
        } else if (arg == "--sharpen") {
            if (++i >= argc) { LOG_ERROR("Missing value for --sharpen"); return false; }
            if (!safe_convert(argv[i], opts.sharpen_amount, "--sharpen-amount")) return false;
        } else if (arg == "--sharpen-radius") {
            if (++i >= argc) { LOG_ERROR("Missing value for --sharpen-radius"); return false; }
            if (!safe_convert(argv[i], opts.sharpen_radius, "--sharpen-radius")) return false;
        } else if (arg == "--sharpen-threshold") {
            if (++i >= argc) { LOG_ERROR("Missing value for --sharpen-threshold"); return false; }
            if (!safe_convert(argv[i], opts.sharpen_threshold, "--sharpen-threshold")) return false;
        } else if (arg == "--smart-sharpen") {
            if (++i >= argc) { LOG_ERROR("Missing value for --smart-sharpen"); return false; }
            if (!safe_convert(argv[i], opts.smart_sharpen_strength, "--smart-sharpen-strength")) return false;
        } else if (arg == "--smart-sharpen-radius") {
            if (++i >= argc) { LOG_ERROR("Missing value for --smart-sharpen-radius"); return false; }
            if (!safe_convert(argv[i], opts.smart_sharpen_radius, "--smart-sharpen-radius")) return false;
        } else if (arg == "--edge-sharpen") {
            if (++i >= argc) { LOG_ERROR("Missing value for --edge-sharpen"); return false; }
            if (!safe_convert(argv[i], opts.edge_sharpen_amount, "--edge-sharpen-amount")) return false;
        } else if (arg == "--edge-sharpen-radius") {
            if (++i >= argc) { LOG_ERROR("Missing value for --edge-sharpen-radius"); return false; }
            if (!safe_convert(argv[i], opts.edge_sharpen_radius, "--edge-sharpen-radius")) return false;
        } else if (arg == "--edge-sharpen-threshold") {
            if (++i >= argc) { LOG_ERROR("Missing value for --edge-sharpen-threshold"); return false; }
            if (!safe_convert(argv[i], opts.edge_sharpen_threshold, "--edge-sharpen-threshold")) return false;
        } else if (arg == "--denoise") {
            if (++i >= argc) { LOG_ERROR("Missing value for --denoise"); return false; }
            if (!safe_convert(argv[i], opts.denoise_strength, "--denoise-strength")) return false;
        } else if (arg == "--luminance-denoise") {
            if (++i >= argc) { LOG_ERROR("Missing value for --luminance-denoise"); return false; }
            if (!safe_convert(argv[i], opts.luminance_denoise_strength, "--luminance-denoise-strength")) return false;
        } else if (arg == "--color-denoise") {
            if (++i >= argc) { LOG_ERROR("Missing value for --color-denoise"); return false; }
            if (!safe_convert(argv[i], opts.color_denoise_strength, "--color-denoise-strength")) return false;
        } else if (arg == "--brightness-curves") {
            if (++i >= argc) { LOG_ERROR("Missing value for --brightness-curves"); return false; }
            opts.brightness_curves = argv[i];
        } else if (arg == "--r-curves") {
            if (++i >= argc) { LOG_ERROR("Missing value for --r-curves"); return false; }
            opts.r_curves = argv[i];
        } else if (arg == "--g-curves") {
            if (++i >= argc) { LOG_ERROR("Missing value for --g-curves"); return false; }
            opts.g_curves = argv[i];
        } else if (arg == "--b-curves") {
            if (++i >= argc) { LOG_ERROR("Missing value for --b-curves"); return false; }
            opts.b_curves = argv[i];
        } else if (arg == "--outpaint-top") {
            if (++i >= argc) { LOG_ERROR("Missing value for --outpaint-top"); return false; }
            if (!safe_convert(argv[i], opts.outpaint_top, "--outpaint-top")) return false;
        } else if (arg == "--outpaint-bottom") {
            if (++i >= argc) { LOG_ERROR("Missing value for --outpaint-bottom"); return false; }
            if (!safe_convert(argv[i], opts.outpaint_bottom, "--outpaint-bottom")) return false;
        } else if (arg == "--outpaint-left") {
            if (++i >= argc) { LOG_ERROR("Missing value for --outpaint-left"); return false; }
            if (!safe_convert(argv[i], opts.outpaint_left, "--outpaint-left")) return false;
        } else if (arg == "--outpaint-right") {
            if (++i >= argc) { LOG_ERROR("Missing value for --outpaint-right"); return false; }
            if (!safe_convert(argv[i], opts.outpaint_right, "--outpaint-right")) return false;
        } else if (arg == "--outpaint") {
            if (++i >= argc) { LOG_ERROR("Missing value for --outpaint"); return false; }
            int val;
            if (!safe_convert(argv[i], val, "--outpaint")) return false;
            opts.outpaint_top = opts.outpaint_bottom = opts.outpaint_left = opts.outpaint_right = val;
        } else if (arg == "--resize") {
            if (++i >= argc) { LOG_ERROR("Missing value for --resize"); return false; }
            std::string val = argv[i];
            size_t x = val.find('x');
            if (x != std::string::npos) {
                if (!safe_convert(val.substr(0, x).c_str(), opts.resize_width, "--resize")) return false;
                if (!safe_convert(val.substr(x + 1).c_str(), opts.resize_height, "--resize")) return false;
            }
        } else if (arg == "--resize-mode") {
            if (++i >= argc) { LOG_ERROR("Missing value for --resize-mode"); return false; }
            opts.resize_mode = argv[i];
        } else if (arg == "--flip-h") {
            opts.flip_h = true;
        } else if (arg == "--flip-v") {
            opts.flip_v = true;
        } else if (arg == "--rotate") {
            if (++i >= argc) { LOG_ERROR("Missing value for --rotate"); return false; }
            if (!safe_convert(argv[i], opts.rotate, "--rotate")) return false;
        } else if (arg == "--crop") {
            if (++i >= argc) { LOG_ERROR("Missing value for --crop"); return false; }
            opts.crop = argv[i];
        } else if (arg == "--crop-center") {
            if (++i >= argc) { LOG_ERROR("Missing value for --crop-center"); return false; }
            opts.crop_center = argv[i];
        } else if (arg == "--crop-ratio") {
            if (++i >= argc) { LOG_ERROR("Missing value for --crop-ratio"); return false; }
            opts.crop_ratio = argv[i];
        } else if (arg == "--control-preprocessor") {
            if (++i >= argc) { LOG_ERROR("Missing value for --control-preprocessor"); return false; }
            opts.control_preprocessor = argv[i];
        } else if (arg == "--control-preprocessor-param1") {
            if (++i >= argc) { LOG_ERROR("Missing value for --control-preprocessor-param1"); return false; }
            if (!safe_convert(argv[i], opts.control_preprocessor_param1, "--control-preprocessor-param1")) return false;
        } else if (arg == "--control-preprocessor-param2") {
            if (++i >= argc) { LOG_ERROR("Missing value for --control-preprocessor-param2"); return false; }
            if (!safe_convert(argv[i], opts.control_preprocessor_param2, "--control-preprocessor-param2")) return false;
        } else if (arg == "--depth-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --depth-model"); return false; }
            opts.depth_model = argv[i];
        } else if (arg == "--openpose-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --openpose-model"); return false; }
            opts.openpose_model = argv[i];
        } else if (arg == "--save-preset") {
            if (++i >= argc) { LOG_ERROR("Missing value for --save-preset"); return false; }
            opts.save_preset_name = argv[i];
        } else if (arg == "--load-preset") {
            if (++i >= argc) { LOG_ERROR("Missing value for --load-preset"); return false; }
            opts.load_preset_path = argv[i];
        } else if (arg == "--interrogate") {
            if (++i >= argc) { LOG_ERROR("Missing value for --interrogate"); return false; }
            opts.interrogate_image = argv[i];
        } else if (arg == "--read-metadata") {
            if (++i >= argc) { LOG_ERROR("Missing value for --read-metadata"); return false; }
            opts.read_metadata_image = argv[i];
        } else if (arg == "--radial-filter") {
            if (++i >= argc) { LOG_ERROR("Missing value for --radial-filter"); return false; }
            opts.radial_filter = argv[i];
        } else if (arg == "--graduated-filter") {
            if (++i >= argc) { LOG_ERROR("Missing value for --graduated-filter"); return false; }
            opts.graduated_filter = argv[i];
        } else if (arg == "--prompt-schedule") {
            if (++i >= argc) { LOG_ERROR("Missing value for --prompt-schedule"); return false; }
            opts.prompt_schedule = argv[i];
        } else if (arg == "--regional-prompts") {
            if (++i >= argc) { LOG_ERROR("Missing value for --regional-prompts"); return false; }
            opts.regional_prompts = argv[i];
        } else if (arg == "--face-restore") {
            opts.face_restoration = true;
        } else if (arg == "--face-restore-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --face-restore-model"); return false; }
            opts.face_restore_model = argv[i];
        } else if (arg == "--face-restore-fidelity") {
            if (++i >= argc) { LOG_ERROR("Missing value for --face-restore-fidelity"); return false; }
            if (!safe_convert(argv[i], opts.face_restore_fidelity, "--face-restore-fidelity")) return false;
        } else if (arg == "--ipadapter") {
            opts.ipadapter = true;
        } else if (arg == "--ipadapter-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --ipadapter-model"); return false; }
            opts.ipadapter_model = argv[i];
        } else if (arg == "--ipadapter-clip-vision") {
            if (++i >= argc) { LOG_ERROR("Missing value for --ipadapter-clip-vision"); return false; }
            opts.ipadapter_clip_vision = argv[i];
        } else if (arg == "--ipadapter-image") {
            if (++i >= argc) { LOG_ERROR("Missing value for --ipadapter-image"); return false; }
            opts.ipadapter_image = argv[i];
        } else if (arg == "--ipadapter-weight") {
            if (++i >= argc) { LOG_ERROR("Missing value for --ipadapter-weight"); return false; }
            if (!safe_convert(argv[i], opts.ipadapter_weight, "--ipadapter-weight")) return false;
        } else if (arg == "--ipadapter-start") {
            if (++i >= argc) { LOG_ERROR("Missing value for --ipadapter-start"); return false; }
            if (!safe_convert(argv[i], opts.ipadapter_start_at, "--ipadapter-start")) return false;
        } else if (arg == "--ipadapter-end") {
            if (++i >= argc) { LOG_ERROR("Missing value for --ipadapter-end"); return false; }
            if (!safe_convert(argv[i], opts.ipadapter_end_at, "--ipadapter-end")) return false;
        } else if (arg == "--t2i-adapter") {
            opts.t2i_adapter = true;
        } else if (arg == "--t2i-adapter-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --t2i-adapter-model"); return false; }
            opts.t2i_adapter_model = argv[i];
        } else if (arg == "--t2i-adapter-image") {
            if (++i >= argc) { LOG_ERROR("Missing value for --t2i-adapter-image"); return false; }
            opts.t2i_adapter_image = argv[i];
        } else if (arg == "--t2i-adapter-strength") {
            if (++i >= argc) { LOG_ERROR("Missing value for --t2i-adapter-strength"); return false; }
            if (!safe_convert(argv[i], opts.t2i_adapter_strength, "--t2i-adapter-strength")) return false;
        } else if (arg == "--face-swap") {
            opts.face_swap = true;
        } else if (arg == "--face-swap-source") {
            if (++i >= argc) { LOG_ERROR("Missing value for --face-swap-source"); return false; }
            opts.face_swap_source = argv[i];
        } else if (arg == "--face-swap-detection-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --face-swap-detection-model"); return false; }
            opts.face_swap_detection_model = argv[i];
        } else if (arg == "--face-swap-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --face-swap-model"); return false; }
            opts.face_swap_model = argv[i];
        } else if (arg == "--photomaker") {
            opts.photo_maker = true;
        } else if (arg == "--photomaker-model") {
            if (++i >= argc) { LOG_ERROR("Missing value for --photomaker-model"); return false; }
            opts.photo_maker_model = argv[i];
        } else if (arg == "--photomaker-id-images") {
            if (++i >= argc) { LOG_ERROR("Missing value for --photomaker-id-images"); return false; }
            std::stringstream ss(argv[i]);
            std::string img;
            while (std::getline(ss, img, ',')) {
                if (!img.empty()) opts.photo_maker_id_images.push_back(img);
            }
        } else if (arg == "--photomaker-id-weight") {
            if (++i >= argc) { LOG_ERROR("Missing value for --photomaker-id-weight"); return false; }
            if (!safe_convert(argv[i], opts.photo_maker_id_weight, "--photomaker-id-weight")) return false;
        } else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        } else {
            LOG_ERROR("Unknown argument: %s", arg.c_str());
            return false;
        }
    }
    return true;
}


SampleMethod parse_sampling_method(const std::string& name) {
    if (name == "euler") return SampleMethod::Euler;
    if (name == "euler_a" || name == "euler-ancestral") return SampleMethod::EulerAncestral;
    if (name == "heun") return SampleMethod::Heun;
    if (name == "dpm2") return SampleMethod::DPM2;
    if (name == "dpm++2s_a") return SampleMethod::DPMPP2S_A;
    if (name == "dpm++2m") return SampleMethod::DPMPP2M;
    if (name == "dpm++2mv2") return SampleMethod::DPMPP2Mv2;
    if (name == "ipndm") return SampleMethod::IPNDM;
    if (name == "ipndm_v") return SampleMethod::IPNDM_V;
    if (name == "lcm") return SampleMethod::LCM;
    if (name == "ddim_trailing") return SampleMethod::DDIM_Trailing;
    if (name == "tcd") return SampleMethod::TCD;
    if (name == "res_multistep") return SampleMethod::RES_Multistep;
    if (name == "res_2s") return SampleMethod::RES_2S;
    if (name == "er_sde") return SampleMethod::ER_SDE;
    return SampleMethod::Euler;
}


Scheduler parse_scheduler(const std::string& name) {
    if (name == "discrete") return Scheduler::Discrete;
    if (name == "karras") return Scheduler::Karras;
    if (name == "exponential") return Scheduler::Exponential;
    if (name == "ays") return Scheduler::AYS;
    if (name == "gits") return Scheduler::GITS;
    if (name == "sgm_uniform") return Scheduler::SGM_Uniform;
    if (name == "simple") return Scheduler::Simple;
    if (name == "smoothstep") return Scheduler::Smoothstep;
    if (name == "kl_optimal") return Scheduler::KL_Optimal;
    if (name == "lcm") return Scheduler::LCM;
    if (name == "bong_tangent") return Scheduler::Bong_Tangent;
    return Scheduler::Simple;
}


bool load_config_file(CliOptions& opts) {
    if (opts.config_file.empty()) return true;
    
    std::ifstream file(opts.config_file);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open config file: %s", opts.config_file.c_str());
        return false;
    }
    
    try {
        nlohmann::json config;
        file >> config;
        
        // 模型路径
        if (config.contains("model")) opts.model = config["model"].get<std::string>();
        if (config.contains("diffusion_model")) opts.diffusion_model = config["diffusion_model"].get<std::string>();
        if (config.contains("vae")) opts.vae = config["vae"].get<std::string>();
        if (config.contains("llm")) opts.llm = config["llm"].get<std::string>();
        if (config.contains("upscale_model")) opts.upscale_model = config["upscale_model"].get<std::string>();
        
        // 生成参数
        if (config.contains("prompt")) opts.prompt = config["prompt"].get<std::string>();
        if (config.contains("negative_prompt")) opts.negative_prompt = config["negative_prompt"].get<std::string>();
        if (config.contains("width")) opts.width = config["width"].get<int>();
        if (config.contains("height")) opts.height = config["height"].get<int>();
        if (config.contains("steps")) opts.steps = config["steps"].get<int>();
        if (config.contains("cfg_scale")) opts.cfg_scale = config["cfg_scale"].get<float>();
        if (config.contains("sampling_method")) opts.sampling_method = config["sampling_method"].get<std::string>();
        if (config.contains("scheduler")) opts.scheduler = config["scheduler"].get<std::string>();
        if (config.contains("seed")) opts.seed = config["seed"].get<int64_t>();
        if (config.contains("batch_count")) opts.batch_count = config["batch_count"].get<int>();
        
        // img2img
        if (config.contains("init_image")) opts.init_image = config["init_image"].get<std::string>();
        if (config.contains("strength")) opts.strength = config["strength"].get<float>();
        if (config.contains("mask_image")) opts.mask_image = config["mask_image"].get<std::string>();
        
        // VRAM 优化
        if (config.contains("diffusion_fa")) opts.diffusion_fa = config["diffusion_fa"].get<bool>();
        if (config.contains("vae_tiling")) opts.vae_tiling = config["vae_tiling"].get<bool>();
        if (config.contains("vae_tile_size_w")) opts.vae_tile_size_w = config["vae_tile_size_w"].get<int>();
        if (config.contains("vae_tile_size_h")) opts.vae_tile_size_h = config["vae_tile_size_h"].get<int>();
        if (config.contains("vae_tile_overlap")) opts.vae_tile_overlap = config["vae_tile_overlap"].get<float>();
        
        // 高级功能
        if (config.contains("hires")) opts.hires = config["hires"].get<bool>();
        if (config.contains("hires_width")) opts.hires_width = config["hires_width"].get<int>();
        if (config.contains("hires_height")) opts.hires_height = config["hires_height"].get<int>();
        if (config.contains("hires_strength")) opts.hires_strength = config["hires_strength"].get<float>();
        if (config.contains("hires_steps")) opts.hires_steps = config["hires_steps"].get<int>();
        
        if (config.contains("freeu")) opts.freeu = config["freeu"].get<bool>();
        if (config.contains("freeu_b1")) opts.freeu_b1 = config["freeu_b1"].get<float>();
        if (config.contains("freeu_b2")) opts.freeu_b2 = config["freeu_b2"].get<float>();
        if (config.contains("freeu_s1")) opts.freeu_s1 = config["freeu_s1"].get<float>();
        if (config.contains("freeu_s2")) opts.freeu_s2 = config["freeu_s2"].get<float>();
        
        if (config.contains("sag")) opts.sag = config["sag"].get<bool>();
        if (config.contains("sag_scale")) opts.sag_scale = config["sag_scale"].get<float>();
        
        // 输出
        if (config.contains("output")) opts.output = config["output"].get<std::string>();
        if (config.contains("batch_output_dir")) opts.batch_output_dir = config["batch_output_dir"].get<std::string>();
        if (config.contains("jpeg_quality")) opts.jpeg_quality = config["jpeg_quality"].get<int>();
        
        LOG_INFO("Loaded config from %s", opts.config_file.c_str());
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }
}


bool save_config_file(const CliOptions& opts, const std::string& path) {
    nlohmann::json config;
    
    // 模型路径
    if (!opts.model.empty()) config["model"] = opts.model;
    if (!opts.diffusion_model.empty()) config["diffusion_model"] = opts.diffusion_model;
    if (!opts.vae.empty()) config["vae"] = opts.vae;
    if (!opts.llm.empty()) config["llm"] = opts.llm;
    if (!opts.upscale_model.empty()) config["upscale_model"] = opts.upscale_model;
    
    // 生成参数
    config["prompt"] = opts.prompt;
    if (!opts.negative_prompt.empty()) config["negative_prompt"] = opts.negative_prompt;
    config["width"] = opts.width;
    config["height"] = opts.height;
    config["steps"] = opts.steps;
    config["cfg_scale"] = opts.cfg_scale;
    config["sampling_method"] = opts.sampling_method;
    config["scheduler"] = opts.scheduler;
    config["seed"] = opts.seed;
    config["batch_count"] = opts.batch_count;
    
    // img2img
    if (!opts.init_image.empty()) config["init_image"] = opts.init_image;
    if (opts.strength != 0.75f) config["strength"] = opts.strength;
    if (!opts.mask_image.empty()) config["mask_image"] = opts.mask_image;
    
    // VRAM 优化
    config["diffusion_fa"] = opts.diffusion_fa;
    config["vae_tiling"] = opts.vae_tiling;
    if (opts.vae_tile_size_w != 256) config["vae_tile_size_w"] = opts.vae_tile_size_w;
    if (opts.vae_tile_size_h != 256) config["vae_tile_size_h"] = opts.vae_tile_size_h;
    if (opts.vae_tile_overlap != 0.8f) config["vae_tile_overlap"] = opts.vae_tile_overlap;
    
    // 高级功能
    if (opts.hires) {
        config["hires"] = true;
        config["hires_width"] = opts.hires_width;
        config["hires_height"] = opts.hires_height;
        config["hires_strength"] = opts.hires_strength;
        config["hires_steps"] = opts.hires_steps;
    }
    
    if (opts.freeu) {
        config["freeu"] = true;
        config["freeu_b1"] = opts.freeu_b1;
        config["freeu_b2"] = opts.freeu_b2;
        config["freeu_s1"] = opts.freeu_s1;
        config["freeu_s2"] = opts.freeu_s2;
    }
    
    if (opts.sag) {
        config["sag"] = true;
        config["sag_scale"] = opts.sag_scale;
    }
    
    // 输出
    if (!opts.output.empty()) config["output"] = opts.output;
    if (!opts.batch_output_dir.empty()) config["batch_output_dir"] = opts.batch_output_dir;
    if (opts.jpeg_quality != 95) config["jpeg_quality"] = opts.jpeg_quality;
    
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to write config file: %s", path.c_str());
        return false;
    }
    
    file << config.dump(2);
    LOG_INFO("Saved config to %s", path.c_str());
    return true;
}

} // namespace myimg
