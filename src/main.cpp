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
#include "adapters/sdcpp_adapter.h"
#include "utils/image_utils.h"
#include "utils/image_adjust.h"
#include "utils/png_metadata.h"
#include "utils/lut_loader.h"
#include "utils/dehaze.h"
#ifdef HAVE_OPENCV
#include "utils/controlnet_preprocessors.h"
#endif

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
    std::string hires_upscaler = "latent";
    float hires_scale = 2.0f;
    std::string hires_model_path;
    int hires_tile_size = 128;

    
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
    float vibrance = 0.0f;       // -1.0 ~ 1.0 (smart saturation)
    float clarity = 0.0f;        // 0.0 ~ 1.0 (texture enhancement)
    std::string split_tone_highlights;  // Hex color for highlights (e.g., "#FFE4C4")
    std::string split_tone_shadows;     // Hex color for shadows (e.g., "#4A6741")
    float split_tone_strength = 0.0f;   // 0.0 ~ 1.0
    float tint = 0.0f;           // -1.0 ~ 1.0 (green/magenta)
    bool auto_white_balance = false;    // Auto white balance
    float blacks = 0.0f;         // -100 ~ 100 (black level)
    float whites = 0.0f;         // -100 ~ 100 (white level)
    std::string curves;          // RGB curves "in,out;in,out"
    std::string brightness_curves;  // Brightness curves "in,out;in,out"
    std::string r_curves;           // Red channel curves "in,out;in,out"
    std::string g_curves;           // Green channel curves "in,out;in,out"
    std::string b_curves;           // Blue channel curves "in,out;in,out"
    std::string preset;          // Filter preset name
    float vignette_strength = 0.0f; // 0.0-1.0
    float vignette_radius = 0.75f;  // 0.0-1.0
    
    // 人像修饰
    float whiten_strength = 0.0f;    // 0.0-1.0
    float skin_smooth_strength = 0.0f; // 0.0-1.0
    std::string skin_tone;           // warm, cool, neutral
    float skin_tone_strength = 0.0f; // 0.0-1.0
    float skin_even_strength = 0.0f; // 0.0-1.0
    
    // 图像修复
    float dehaze_strength = 0.0f;  // 0.0-1.0
    
    // 锐化与降噪
    float sharpen_amount = 0.0f;   // 0.0-3.0
    int sharpen_radius = 1;        // 1-5
    float sharpen_threshold = 0.0f; // 0-255
    float smart_sharpen_strength = 0.0f; // 0.0-3.0
    int smart_sharpen_radius = 2;   // 1-5
    float edge_sharpen_amount = 0.0f; // 0.0-3.0 (edge-mask sharpening)
    int edge_sharpen_radius = 2;    // 1-5
    float edge_sharpen_threshold = 0.3f; // 0-1 (edge detection threshold)
    float denoise_strength = 0.0f;  // 0.0-1.0
    bool smart_denoise_flag = false; // 智能降噪 (disabled)
    float luminance_denoise_strength = 0.0f; // 0.0-1.0
    float color_denoise_strength = 0.0f;     // 0.0-1.0
    
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
    std::string crop;         // x,y,w,h
    std::string crop_center;  // w,h
    std::string crop_ratio;   // w:h (e.g. 16:9, 1:1)
    
    // ControlNet preprocessor
    std::string control_preprocessor;  // canny, lineart, normal, scribble, depth, openpose
    int control_preprocessor_param1 = 0;  // threshold1 for canny, threshold for lineart/scribble
    int control_preprocessor_param2 = 0;  // threshold2 for canny
    std::string depth_model;   // Path to MiDaS model (default: /opt/image/model/midas_dpt_hybrid.pt)
    std::string openpose_model; // Path to OpenPose model (default: /opt/image/model/openpose_body.pt)
    
    // Output quality
    int jpeg_quality = 95;  // JPEG quality (1-100)
    
    // LUT / Color grading
    std::string lut_path;  // 3D LUT file (.cube)
    
    // Batch processing (post-processing only)
    std::string batch_input_dir;
    std::string batch_output_dir;
    std::string output_template;  // e.g. "{name}_edited{ext}" or "{index:04d}{ext}"
    
    // Presets
    std::string save_preset_name;   // Save current settings as preset
    std::string load_preset_path;   // Load settings from preset file
    
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
    std::cout << "  --hires-upscaler NAME     HiRes upscaler: latent, latent-nearest, latent-nearest-exact,\n";
    std::cout << "                            latent-antialiased, latent-bicubic, latent-bicubic-antialiased,\n";
    std::cout << "                            lanczos, nearest, model (default: latent)\n";
    std::cout << "  --hires-scale FLOAT       HiRes scale factor (default: 2.0)\n";
    std::cout << "  --hires-model PATH        External upscaler model path (for model upscaler)\n";
    std::cout << "  --hires-tile-size INT     HiRes upscale tile size (default: 128)\n";
    std::cout << "\nUpscale Options:\n";
    std::cout << "  --upscale-repeats INT     ESRGAN upscale repeats (default: 1)\n";
    std::cout << "  --upscale-tile-size INT   ESRGAN tile size (default: 1440)\n";
    std::cout << "\nEmbedding Options:\n";
    std::cout << "  --embd-dir PATH           Embeddings directory (Textual Inversion)\n";
    std::cout << "\nOutput Options:\n";
    std::cout << "  -o, --output PATH         Output path (default: output.png)\n";
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
    std::cout << "  -v, --verbose             Verbose logging\n";
    std::cout << "  --help                    Show this help\n";
}

// Parse embedding syntax in prompt: replaces "embedding:name" with "name"
// and returns list of referenced embedding names
static std::string parse_embedding_syntax(const std::string& prompt, std::vector<std::string>& referenced_embeddings) {
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
static std::string expand_output_template(const std::string& template_str, 
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
            int padding = std::stoi(result.substr(pos + 7, end - pos - 7));
            std::ostringstream oss;
            oss << std::setw(padding) << std::setfill('0') << index;
            std::string idx_str = oss.str();
            result.replace(pos, end - pos + 1, idx_str);
            pos += idx_str.size();
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
static bool save_preset(const CliOptions& opts, const std::string& preset_name) {
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
        std::cerr << "Error: Failed to save preset to " << preset_path << "\n";
        return false;
    }
    file << j.dump(2);
    std::cout << "Preset saved to: " << preset_path << "\n";
    return true;
}

// Load preset from JSON file
static bool load_preset(CliOptions& opts, const std::string& preset_path) {
    std::ifstream file(preset_path);
    if (!file) {
        std::cerr << "Error: Failed to load preset from " << preset_path << "\n";
        return false;
    }
    
    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid preset JSON: " << e.what() << "\n";
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
        } else if (arg == "--hires-upscaler") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-upscaler\n"; return false; }
            opts.hires_upscaler = argv[i];
        } else if (arg == "--hires-scale") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-scale\n"; return false; }
            opts.hires_scale = std::stof(argv[i]);
        } else if (arg == "--hires-model") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-model\n"; return false; }
            opts.hires_model_path = argv[i];
        } else if (arg == "--hires-tile-size") {
            if (++i >= argc) { std::cerr << "Missing value for --hires-tile-size\n"; return false; }
            opts.hires_tile_size = std::stoi(argv[i]);
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
        } else if (arg == "--quality") {
            if (++i >= argc) { std::cerr << "Missing value for --quality\n"; return false; }
            opts.jpeg_quality = std::stoi(argv[i]);
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
        } else if (arg == "--vibrance") {
            if (++i >= argc) { std::cerr << "Missing value for --vibrance\n"; return false; }
            opts.vibrance = std::stof(argv[i]);
        } else if (arg == "--clarity") {
            if (++i >= argc) { std::cerr << "Missing value for --clarity\n"; return false; }
            opts.clarity = std::stof(argv[i]);
        } else if (arg == "--split-tone-highlights") {
            if (++i >= argc) { std::cerr << "Missing value for --split-tone-highlights\n"; return false; }
            opts.split_tone_highlights = argv[i];
        } else if (arg == "--split-tone-shadows") {
            if (++i >= argc) { std::cerr << "Missing value for --split-tone-shadows\n"; return false; }
            opts.split_tone_shadows = argv[i];
        } else if (arg == "--split-tone-strength") {
            if (++i >= argc) { std::cerr << "Missing value for --split-tone-strength\n"; return false; }
            opts.split_tone_strength = std::stof(argv[i]);
        } else if (arg == "--tint") {
            if (++i >= argc) { std::cerr << "Missing value for --tint\n"; return false; }
            opts.tint = std::stof(argv[i]);
        } else if (arg == "--auto-white-balance") {
            opts.auto_white_balance = true;
        } else if (arg == "--blacks") {
            if (++i >= argc) { std::cerr << "Missing value for --blacks\n"; return false; }
            opts.blacks = std::stof(argv[i]);
        } else if (arg == "--whites") {
            if (++i >= argc) { std::cerr << "Missing value for --whites\n"; return false; }
            opts.whites = std::stof(argv[i]);
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
        } else if (arg == "--skin-tone") {
            if (++i >= argc) { std::cerr << "Missing value for --skin-tone\n"; return false; }
            opts.skin_tone = argv[i];
        } else if (arg == "--skin-tone-strength") {
            if (++i >= argc) { std::cerr << "Missing value for --skin-tone-strength\n"; return false; }
            opts.skin_tone_strength = std::stof(argv[i]);
        } else if (arg == "--skin-even") {
            if (++i >= argc) { std::cerr << "Missing value for --skin-even\n"; return false; }
            opts.skin_even_strength = std::stof(argv[i]);
        } else if (arg == "--dehaze") {
            if (++i >= argc) { std::cerr << "Missing value for --dehaze\n"; return false; }
            opts.dehaze_strength = std::stof(argv[i]);
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
        } else if (arg == "--edge-sharpen") {
            if (++i >= argc) { std::cerr << "Missing value for --edge-sharpen\n"; return false; }
            opts.edge_sharpen_amount = std::stof(argv[i]);
        } else if (arg == "--edge-sharpen-radius") {
            if (++i >= argc) { std::cerr << "Missing value for --edge-sharpen-radius\n"; return false; }
            opts.edge_sharpen_radius = std::stoi(argv[i]);
        } else if (arg == "--edge-sharpen-threshold") {
            if (++i >= argc) { std::cerr << "Missing value for --edge-sharpen-threshold\n"; return false; }
            opts.edge_sharpen_threshold = std::stof(argv[i]);
        } else if (arg == "--denoise") {
            if (++i >= argc) { std::cerr << "Missing value for --denoise\n"; return false; }
            opts.denoise_strength = std::stof(argv[i]);
        } else if (arg == "--luminance-denoise") {
            if (++i >= argc) { std::cerr << "Missing value for --luminance-denoise\n"; return false; }
            opts.luminance_denoise_strength = std::stof(argv[i]);
        } else if (arg == "--color-denoise") {
            if (++i >= argc) { std::cerr << "Missing value for --color-denoise\n"; return false; }
            opts.color_denoise_strength = std::stof(argv[i]);
        } else if (arg == "--brightness-curves") {
            if (++i >= argc) { std::cerr << "Missing value for --brightness-curves\n"; return false; }
            opts.brightness_curves = argv[i];
        } else if (arg == "--r-curves") {
            if (++i >= argc) { std::cerr << "Missing value for --r-curves\n"; return false; }
            opts.r_curves = argv[i];
        } else if (arg == "--g-curves") {
            if (++i >= argc) { std::cerr << "Missing value for --g-curves\n"; return false; }
            opts.g_curves = argv[i];
        } else if (arg == "--b-curves") {
            if (++i >= argc) { std::cerr << "Missing value for --b-curves\n"; return false; }
            opts.b_curves = argv[i];
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
        } else if (arg == "--crop") {
            if (++i >= argc) { std::cerr << "Missing value for --crop\n"; return false; }
            opts.crop = argv[i];
        } else if (arg == "--crop-center") {
            if (++i >= argc) { std::cerr << "Missing value for --crop-center\n"; return false; }
            opts.crop_center = argv[i];
        } else if (arg == "--crop-ratio") {
            if (++i >= argc) { std::cerr << "Missing value for --crop-ratio\n"; return false; }
            opts.crop_ratio = argv[i];
        } else if (arg == "--control-preprocessor") {
            if (++i >= argc) { std::cerr << "Missing value for --control-preprocessor\n"; return false; }
            opts.control_preprocessor = argv[i];
        } else if (arg == "--control-preprocessor-param1") {
            if (++i >= argc) { std::cerr << "Missing value for --control-preprocessor-param1\n"; return false; }
            opts.control_preprocessor_param1 = std::stoi(argv[i]);
        } else if (arg == "--control-preprocessor-param2") {
            if (++i >= argc) { std::cerr << "Missing value for --control-preprocessor-param2\n"; return false; }
            opts.control_preprocessor_param2 = std::stoi(argv[i]);
        } else if (arg == "--depth-model") {
            if (++i >= argc) { std::cerr << "Missing value for --depth-model\n"; return false; }
            opts.depth_model = argv[i];
        } else if (arg == "--openpose-model") {
            if (++i >= argc) { std::cerr << "Missing value for --openpose-model\n"; return false; }
            opts.openpose_model = argv[i];
        } else if (arg == "--save-preset") {
            if (++i >= argc) { std::cerr << "Missing value for --save-preset\n"; return false; }
            opts.save_preset_name = argv[i];
        } else if (arg == "--load-preset") {
            if (++i >= argc) { std::cerr << "Missing value for --load-preset\n"; return false; }
            opts.load_preset_path = argv[i];
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
    
    // ControlNet preprocessor (post-processing only, no model required)
    if (!opts.control_preprocessor.empty()) {
#ifdef HAVE_OPENCV
        if (opts.init_image.empty()) {
            std::cerr << "Error: --control-preprocessor requires --init-img\n";
            return 1;
        }
        
        std::cout << "========================================\n";
        std::cout << "  ControlNet Preprocessor: " << opts.control_preprocessor << "\n";
        std::cout << "========================================\n";
        
        auto img_data = myimg::load_image_from_file(opts.init_image);
        if (img_data.empty()) {
            std::cerr << "Error: Failed to load image\n";
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
            std::cerr << "Error: Preprocessor failed\n";
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
        image.data = std::move(result.data);
        
        if (image.save_to_file(output_file)) {
            std::cout << "Saved to: " << output_file << "\n";
        } else {
            std::cerr << "Error: Failed to save result\n";
            return 1;
        }
        return 0;
#else
        std::cerr << "Error: ControlNet preprocessors require OpenCV. Please install OpenCV.\n";
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
        std::cerr << "Error: --model or --diffusion-model is required\n";
        return 1;
    }
    // 如果使用 diffusion-model 模式，需要 vae 和 llm (skip for batch mode)
    if (!batch_mode && !opts.diffusion_model.empty()) {
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
        std::cout << "\n";
    }
    
    // 构建生成参数
    myimg::GenerationParams params;
    if (!opts.model.empty()) {
        params.model_path = opts.model;
    }
    params.diffusion_model_path = opts.diffusion_model;
    params.vae_path = opts.vae;
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
            std::cerr << "Warning: Unknown hires upscaler '" << opts.hires_upscaler << "', using 'latent'\n";
            params.hires_upscaler = myimg::HiresUpscaler::Latent;
        }
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
    
    // Load preset if specified
    if (!opts.load_preset_path.empty()) {
        if (!load_preset(opts, opts.load_preset_path)) {
            return 1;
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
        if (opts.threads > 0) {
            std::cout << "Threads: " << opts.threads << "\n";
        }
        std::cout << "========================================\n\n";
        
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
            std::cout << "No image files found in input directory.\n";
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
                std::cout << "\n";
            }
            
            auto img_data = myimg::load_image_from_file(task.input_file);
            if (img_data.empty()) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "  Failed to load\n";
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
                size_t colon = opts.crop_ratio.find(':');
                if (colon != std::string::npos) {
                    float ratio_w = std::stof(opts.crop_ratio.substr(0, colon));
                    float ratio_h = std::stof(opts.crop_ratio.substr(colon + 1));
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
                }
            }
            
            // Apply photo adjustments
            if (opts.temperature != 0.0f || opts.brightness != 0.0f ||
                opts.contrast != 0.0f || opts.saturation != 0.0f ||
                opts.exposure != 0.0f || opts.highlights != 0.0f ||
                opts.shadows != 0.0f || opts.auto_enhance ||
                !opts.curves.empty() ||
                opts.sharpen_amount > 0.0f || opts.denoise_strength > 0.0f ||
                opts.luminance_denoise_strength > 0.0f || opts.color_denoise_strength > 0.0f ||
                opts.whiten_strength > 0.0f || opts.skin_smooth_strength > 0.0f ||
                !opts.skin_tone.empty() || opts.skin_even_strength > 0.0f ||
                !opts.preset.empty() ||
                opts.vignette_strength > 0.0f ||
                !opts.radial_filter.empty() ||
                !opts.graduated_filter.empty() ||
                !opts.lut_path.empty() ||
                opts.dehaze_strength > 0.0f ||
                opts.vibrance != 0.0f || opts.clarity > 0.0f ||
                opts.split_tone_strength > 0.0f ||
                opts.tint != 0.0f || opts.auto_white_balance ||
                opts.blacks != 0.0f || opts.whites != 0.0f ||
                !opts.brightness_curves.empty() ||
                !opts.r_curves.empty() || !opts.g_curves.empty() || !opts.b_curves.empty()) {
                
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
                if (opts.luminance_denoise_strength > 0.0f) {
                    tensor = myimg::luminance_denoise(tensor, opts.luminance_denoise_strength);
                }
                if (opts.color_denoise_strength > 0.0f) {
                    tensor = myimg::color_denoise(tensor, opts.color_denoise_strength);
                }
                if (opts.sharpen_amount > 0.0f) {
                    tensor = myimg::usm_sharpen(tensor, opts.sharpen_amount, opts.sharpen_radius, opts.sharpen_threshold);
                }
                
                // RGB curves
                if (!opts.curves.empty()) {
                    tensor = myimg::apply_curves(tensor, opts.curves);
                }
                
                // Brightness curves (luma only)
                if (!opts.brightness_curves.empty()) {
                    tensor = myimg::apply_brightness_curves(tensor, opts.brightness_curves);
                }
                
                // Per-channel RGB curves
                if (!opts.r_curves.empty() || !opts.g_curves.empty() || !opts.b_curves.empty()) {
                    tensor = myimg::apply_channel_curves(tensor, opts.r_curves, opts.g_curves, opts.b_curves);
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
                
                // 去雾
                if (opts.dehaze_strength > 0.0f) {
                    tensor = myimg::dehaze(tensor, opts.dehaze_strength);
                }
                
                // Vibrance
                if (opts.vibrance != 0.0f) {
                    tensor = myimg::adjust_vibrance(tensor, opts.vibrance);
                }
                
                // Clarity
                if (opts.clarity > 0.0f) {
                    tensor = myimg::enhance_clarity(tensor, opts.clarity);
                }
                
                // Split toning
                if (opts.split_tone_strength > 0.0f) {
                    tensor = myimg::split_tone(tensor, opts.split_tone_highlights, opts.split_tone_shadows, opts.split_tone_strength);
                }
                
                // Tint
                if (opts.tint != 0.0f) {
                    tensor = myimg::adjust_tint(tensor, opts.tint);
                }
                
                // Auto white balance
                if (opts.auto_white_balance) {
                    tensor = myimg::auto_white_balance(tensor);
                }
                
                // Black/White levels
                if (opts.blacks != 0.0f || opts.whites != 0.0f) {
                    tensor = myimg::adjust_levels(tensor, opts.blacks, opts.whites);
                }
                
                // Portrait retouching
                if (opts.whiten_strength > 0.0f) {
                    tensor = myimg::whiten(tensor, opts.whiten_strength);
                }
                if (opts.skin_smooth_strength > 0.0f) {
                    tensor = myimg::skin_smooth(tensor, opts.skin_smooth_strength);
                }
                
                // Skin tone matching
                if (!opts.skin_tone.empty()) {
                    tensor = myimg::skin_tone_match(tensor, opts.skin_tone, opts.skin_tone_strength);
                }
                if (opts.skin_even_strength > 0.0f) {
                    tensor = myimg::skin_tone_even(tensor, opts.skin_even_strength);
                }
                
                img_data = myimg::tensor_to_image_data(tensor);
            }
            
            myimg::Image image;
            image.width = img_data.width;
            image.height = img_data.height;
            image.channels = img_data.channels;
            image.data = std::move(img_data.data);
            
            if (!image.save_to_file(output_file)) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "  Failed to save\n";
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
        
        std::cout << "\n========================================\n";
        std::cout << "Batch processing complete!\n";
        std::cout << "Processed: " << processed.load() << "\n";
        if (failed.load() > 0) std::cout << "Failed: " << failed.load() << "\n";
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
    
    // 设置 JPEG 质量
    myimg::set_jpeg_quality(opts.jpeg_quality);
    
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
                               opts.vibrance != 0.0f || opts.clarity > 0.0f ||
                               opts.split_tone_strength > 0.0f ||
                               opts.tint != 0.0f || opts.auto_white_balance ||
                               opts.blacks != 0.0f || opts.whites != 0.0f ||
                               !opts.brightness_curves.empty() ||
                               !opts.r_curves.empty() || !opts.g_curves.empty() || !opts.b_curves.empty() ||
                               !opts.curves.empty() ||
                opts.sharpen_amount > 0.0f || opts.denoise_strength > 0.0f ||
                opts.luminance_denoise_strength > 0.0f || opts.color_denoise_strength > 0.0f ||
                               opts.whiten_strength > 0.0f || opts.skin_smooth_strength > 0.0f ||
                               !opts.skin_tone.empty() || opts.skin_even_strength > 0.0f ||
                               !opts.preset.empty() ||
                               opts.vignette_strength > 0.0f ||
                               !opts.radial_filter.empty() ||
                               !opts.graduated_filter.empty() ||
                               !opts.lut_path.empty() ||
                               opts.dehaze_strength > 0.0f;
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
            if (opts.luminance_denoise_strength > 0.0f) {
                std::cout << "Applying luminance denoise: " << opts.luminance_denoise_strength << "\n";
                tensor = myimg::luminance_denoise(tensor, opts.luminance_denoise_strength);
            }
            if (opts.color_denoise_strength > 0.0f) {
                std::cout << "Applying color denoise: " << opts.color_denoise_strength << "\n";
                tensor = myimg::color_denoise(tensor, opts.color_denoise_strength);
            }
            /* Smart denoise (disabled - conv2d crash under investigation)
            if (opts.smart_denoise_flag) {
                std::cout << "Applying smart denoise...\n";
                tensor = myimg::smart_denoise(tensor, 0.5f);
            }
            */
            
            // USM 锐化
            if (opts.sharpen_amount > 0.0f) {
                std::cout << "Applying USM sharpen...\n";
                tensor = myimg::usm_sharpen(tensor, opts.sharpen_amount, opts.sharpen_radius, opts.sharpen_threshold);
            }
            
            /* Smart sharpen (disabled - conv2d crash under investigation)
            if (opts.smart_sharpen_strength > 0.0f) {
                std::cout << "Applying smart sharpen...\n";
                tensor = myimg::smart_sharpen(tensor, opts.smart_sharpen_strength, opts.smart_sharpen_radius);
            }
            */
            
            /* Edge-mask sharpening (disabled - conv2d crash under investigation)
            if (opts.edge_sharpen_amount > 0.0f) {
                std::cout << "Applying edge-mask sharpen: amount=" << opts.edge_sharpen_amount
                          << " radius=" << opts.edge_sharpen_radius
                          << " threshold=" << opts.edge_sharpen_threshold << "\n";
                tensor = myimg::edge_mask_sharpen(tensor, opts.edge_sharpen_amount, opts.edge_sharpen_radius, opts.edge_sharpen_threshold);
            }
            */
            
            // RGB 曲线
            if (!opts.curves.empty()) {
                std::cout << "Applying curves: " << opts.curves << "\n";
                tensor = myimg::apply_curves(tensor, opts.curves);
            }
            
            // Brightness curves (luma only)
            if (!opts.brightness_curves.empty()) {
                std::cout << "Applying brightness curves: " << opts.brightness_curves << "\n";
                tensor = myimg::apply_brightness_curves(tensor, opts.brightness_curves);
            }
            
            // Per-channel RGB curves
            if (!opts.r_curves.empty() || !opts.g_curves.empty() || !opts.b_curves.empty()) {
                std::cout << "Applying channel curves...\n";
                tensor = myimg::apply_channel_curves(tensor, opts.r_curves, opts.g_curves, opts.b_curves);
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
            
            // 去雾
            if (opts.dehaze_strength > 0.0f) {
                std::cout << "Applying dehaze...\n";
                tensor = myimg::dehaze(tensor, opts.dehaze_strength);
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
            
            // Skin tone matching
            if (!opts.skin_tone.empty()) {
                std::cout << "Applying skin tone: " << opts.skin_tone << " (strength: " << opts.skin_tone_strength << ")\n";
                tensor = myimg::skin_tone_match(tensor, opts.skin_tone, opts.skin_tone_strength);
            }
            if (opts.skin_even_strength > 0.0f) {
                std::cout << "Applying skin tone evening: " << opts.skin_even_strength << "\n";
                tensor = myimg::skin_tone_even(tensor, opts.skin_even_strength);
            }
            
            img_data = myimg::tensor_to_image_data(tensor);
            image.width = img_data.width;
            image.height = img_data.height;
            image.channels = img_data.channels;
            image.data = std::move(img_data.data);
        }
        
        // 后处理变换
        bool has_transform = opts.resize_width > 0 || opts.resize_height > 0 ||
                            opts.flip_h || opts.flip_v || opts.rotate != 0 ||
                            !opts.crop.empty() || !opts.crop_center.empty() || !opts.crop_ratio.empty();
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
                size_t colon = opts.crop_ratio.find(':');
                if (colon != std::string::npos) {
                    float ratio_w = std::stof(opts.crop_ratio.substr(0, colon));
                    float ratio_h = std::stof(opts.crop_ratio.substr(colon + 1));
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
                }
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
