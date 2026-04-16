// ============================================================================
// sd-engine/core/workflow_builder.h
// ============================================================================
//
// 工作流构建器 - 用代码方式构建 ComfyUI 工作流
// 用于命令行工具快速生成工作流
// ============================================================================

#pragma once

#include "nlohmann/json.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace sdengine {

using json = nlohmann::json;

// 工作流构建器
class WorkflowBuilder {
  public:
    WorkflowBuilder() = default;

    // 添加节点，返回节点 ID
    std::string add_node(const std::string& class_type, const std::unordered_map<std::string, json>& inputs);

    // 添加 CheckpointLoaderSimple 节点
    std::string add_checkpoint_loader(const std::string& ckpt_path, const std::string& vae_path = "",
                                      const std::string& clip_path = "");

    // 添加 CLIPTextEncode 节点
    std::string add_clip_encode(const std::string& text, const std::string& clip_node_id, int clip_skip = -1);

    // 添加 EmptyLatentImage 节点
    std::string add_empty_latent(int width, int height, int batch_size = 1);

    // 添加 LoRALoader 节点
    std::string add_lora_loader(const std::string& lora_path, float strength_model = 1.0f, float strength_clip = 1.0f);

    // 添加 LoRAStack 节点
    std::string add_lora_stack(const std::vector<std::string>& lora_node_ids);

    // 添加 KSampler 节点
    std::string add_ksampler(const std::string& model_node_id, const std::string& positive_node_id,
                             const std::string& negative_node_id, const std::string& latent_node_id, int seed = 0,
                             int steps = 20, float cfg = 7.5f, const std::string& sampler = "euler",
                             float denoise = 1.0f, const std::string& lora_stack_node_id = "");

    // 添加 VAEDecode 节点
    std::string add_vae_decode(const std::string& latent_node_id, const std::string& vae_node_id);

    // 添加 SaveImage 节点
    std::string add_save_image(const std::string& image_node_id, const std::string& filename_prefix = "output");

    // 添加 LoadImage 节点
    std::string add_load_image(const std::string& image_path);

    // 添加 VAEEncode 节点（img2img）
    std::string add_vae_encode(const std::string& image_node_id, const std::string& vae_node_id);

    // 添加 ImageScale 节点
    std::string add_image_scale(const std::string& image_node_id, int width, int height,
                                const std::string& method = "bilinear");

    // 添加 UpscaleModelLoader 节点
    std::string add_upscale_model_loader(const std::string& model_path, bool use_gpu = true, int tile_size = 512);

    // 添加 ImageUpscaleWithModel 节点
    std::string add_image_upscale_with_model(const std::string& image_node_id, const std::string& upscaler_node_id);

    // 添加 ControlNetLoader 节点
    std::string add_controlnet_loader(const std::string& model_path);

    // 添加 ControlNetApply 节点
    std::string add_controlnet_apply(const std::string& conditioning_node_id, const std::string& controlnet_node_id,
                                     const std::string& image_node_id, float strength = 1.0f);

    // 添加 CannyEdgePreprocessor 节点
    std::string add_canny_preprocessor(const std::string& image_node_id, int low_threshold = 100,
                                       int high_threshold = 200);

    // 添加 LineArtLoader 节点
    std::string add_lineart_loader(const std::string& model_path);

    // 添加 LineArtPreprocessor 节点
    std::string add_lineart_preprocessor(const std::string& image_node_id, const std::string& lineart_model_node_id);

    // 添加 LoadImageMask 节点
    std::string add_load_image_mask(const std::string& image_path, const std::string& channel = "alpha");

    // 添加 IPAdapterLoader 节点
    std::string add_ipadapter_loader(const std::string& model_path, int cross_attention_dim = 768, int num_tokens = 4,
                                     int clip_embeddings_dim = 1024);

    // 添加 IPAdapterApply 节点
    std::string add_ipadapter_apply(const std::string& conditioning_node_id, const std::string& ipadapter_node_id,
                                    const std::string& image_node_id, float strength = 1.0f);

    // 添加 ConditioningCombine 节点
    std::string add_conditioning_combine(const std::string& cond1_node_id, const std::string& cond2_node_id);

    // 添加 ConditioningConcat 节点
    std::string add_conditioning_concat(const std::string& cond_to_node_id, const std::string& cond_from_node_id);

    // 添加 ConditioningAverage 节点
    std::string add_conditioning_average(const std::string& cond_to_node_id, const std::string& cond_from_node_id,
                                         float strength = 1.0f);

    // 添加 CLIPSetLastLayer 节点
    std::string add_clip_set_last_layer(const std::string& clip_node_id, int stop_at_clip_layer = -1);

    // 添加 CLIPVisionEncode 节点
    std::string add_clip_vision_encode(const std::string& clip_node_id, const std::string& image_node_id);

    // 添加 RemBGModelLoader 节点
    std::string add_rembg_model_loader(const std::string& model_path);

    // 添加 ImageRemoveBackground 节点
    std::string add_image_remove_background(const std::string& image_node_id, const std::string& model_node_id);

    // 添加图像预处理节点
    std::string add_image_invert(const std::string& image_node_id);
    std::string add_image_color_adjust(const std::string& image_node_id, float brightness = 1.0f, float contrast = 1.0f,
                                       float saturation = 1.0f);
    std::string add_image_blur(const std::string& image_node_id, int radius = 3);
    std::string add_image_grayscale(const std::string& image_node_id);
    std::string add_image_threshold(const std::string& image_node_id, int threshold = 128);

    // 添加 KSamplerAdvanced 节点
    std::string add_ksampler_advanced(const std::string& model_node_id, const std::string& positive_node_id,
                                      const std::string& negative_node_id, const std::string& latent_node_id,
                                      int seed = 0, int steps = 20, float cfg = 7.5f,
                                      const std::string& sampler = "euler", float denoise = 1.0f, int start_at_step = 0,
                                      int end_at_step = 10000, bool add_noise = true,
                                      const std::string& lora_stack_node_id = "");

    // 生成 JSON 字符串
    std::string to_json_string() const;

    // 保存到文件
    bool save_to_file(const std::string& path) const;

    // 清空
    void clear();

  private:
    json workflow_;
    int next_id_ = 1;

    std::string get_next_id();

  public:
    json make_link(const std::string& node_id, int slot = 0);
};

// 快速构建 txt2img 工作流
class Txt2ImgBuilder {
  public:
    static std::string build(const std::string& ckpt_path, const std::string& prompt,
                             const std::string& negative_prompt, int width = 512, int height = 512, int seed = 0,
                             int steps = 20, float cfg = 7.5f, const std::string& output_prefix = "txt2img_output");
};

// 快速构建 img2img 工作流
class Img2ImgBuilder {
  public:
    static std::string build(const std::string& ckpt_path, const std::string& input_image, const std::string& prompt,
                             const std::string& negative_prompt, float denoise = 0.75f, int seed = 0, int steps = 20,
                             float cfg = 7.5f, const std::string& output_prefix = "img2img_output");
};

// 快速构建图像处理工作流
class ImageProcessBuilder {
  public:
    static std::string build(const std::string& input_image, int target_width = 0, int target_height = 0,
                             int crop_x = -1, int crop_y = -1, int crop_w = -1, int crop_h = -1,
                             const std::string& output_prefix = "processed");
};

// 快速构建 Deep HighRes Fix 工作流
class DeepHiresBuilder {
  public:
    static std::string build(const std::string& ckpt_path, const std::string& prompt,
                             const std::string& negative_prompt, int target_width = 1024, int target_height = 1024,
                             int seed = 0, int steps = 30, float cfg = 7.0f, const std::string& input_image = "",
                             float strength = 1.0f, bool vae_tiling = false,
                             const std::string& output_prefix = "deep_hires_output");
};

// 快速构建 IPAdapter txt2img 工作流
class IPAdapterTxt2ImgBuilder {
  public:
    static std::string build(const std::string& ckpt_path, const std::string& prompt,
                             const std::string& negative_prompt, const std::string& ipadapter_path,
                             const std::string& reference_image, float ipadapter_strength = 1.0f,
                             int cross_attention_dim = 768, int num_tokens = 4, int clip_embeddings_dim = 1024,
                             int width = 512, int height = 512, int seed = 0, int steps = 20, float cfg = 7.5f,
                             const std::string& output_prefix = "ipadapter_output");
};

} // namespace sdengine
