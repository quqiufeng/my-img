#pragma once

#include <string>
#include <vector>

namespace myimg {

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
    bool diffusion_fa = true;
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
    
    // FreeU
    bool freeu = false;
    float freeu_b1 = 1.3f;
    float freeu_b2 = 1.4f;
    float freeu_s1 = 0.9f;
    float freeu_s2 = 0.2f;

    // SAG
    bool sag = false;
    float sag_scale = 1.0f;

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
    bool embed_metadata = false;

    // 配置文件
    std::string config_file;

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
    
    // Prompt Schedule
    std::string prompt_schedule;  // 格式: "0-10:prompt1|11-20:prompt2"
    
    // Regional Prompting
    std::string regional_prompts; // 格式: "top:0.3,prompt1|bottom:0.3,prompt2"
    
    // Face Restoration
    bool face_restoration = false;
    std::string face_restore_model;
    float face_restore_fidelity = 0.5f;
    
    // IPAdapter
    bool ipadapter = false;
    std::string ipadapter_model;
    std::string ipadapter_clip_vision;
    std::string ipadapter_image;
    float ipadapter_weight = 1.0f;
    float ipadapter_start_at = 0.0f;
    float ipadapter_end_at = 1.0f;
    
    // T2I-Adapter
    bool t2i_adapter = false;
    std::string t2i_adapter_model;
    std::string t2i_adapter_image;
    float t2i_adapter_strength = 1.0f;
    
    // Face Swap
    bool face_swap = false;
    std::string face_swap_source;
    std::string face_swap_detection_model;
    std::string face_swap_model;
    
    // PhotoMaker
    bool photo_maker = false;
    std::string photo_maker_model;
    std::vector<std::string> photo_maker_id_images;
    float photo_maker_id_weight = 1.0f;
    
    // 系统
    int threads = -1;
    bool verbose = false;
    float max_vram = 0.0f;  // 最大显存限制 (GB, 0 = 禁用, -1 = 自动)

    // VAE 格式
    std::string vae_format = "auto";  // auto, flux, sd3, flux2

    // 后端选择
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
};

} // namespace myimg
