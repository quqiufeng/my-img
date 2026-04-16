// ============================================================================
// sd-engine/core/workflow_builder.cpp
// ============================================================================

#include "workflow_builder.h"
#include <fstream>

namespace sdengine {

std::string WorkflowBuilder::get_next_id() {
    return std::to_string(next_id_++);
}

json WorkflowBuilder::make_link(const std::string& node_id, int slot) {
    return json::array({node_id, slot});
}

std::string WorkflowBuilder::add_node(const std::string& class_type, const std::map<std::string, json>& inputs) {
    std::string id = get_next_id();
    json node;
    node["class_type"] = class_type;
    node["inputs"] = json::object();
    for (const auto& [key, value] : inputs) {
        node["inputs"][key] = value;
    }
    workflow_[id] = node;
    return id;
}

std::string WorkflowBuilder::add_checkpoint_loader(const std::string& ckpt_path, const std::string& vae_path,
                                                   const std::string& clip_path) {
    std::map<std::string, json> inputs;
    inputs["ckpt_name"] = ckpt_path;
    if (!vae_path.empty())
        inputs["vae_name"] = vae_path;
    if (!clip_path.empty())
        inputs["clip_name"] = clip_path;
    return add_node("CheckpointLoaderSimple", inputs);
}

std::string WorkflowBuilder::add_clip_encode(const std::string& text, const std::string& clip_node_id, int clip_skip) {
    std::map<std::string, json> inputs;
    inputs["text"] = text;
    inputs["clip"] = make_link(clip_node_id, 1); // CLIP output slot
    if (clip_skip != -1)
        inputs["clip_skip"] = clip_skip;
    return add_node("CLIPTextEncode", inputs);
}

std::string WorkflowBuilder::add_empty_latent(int width, int height, int batch_size) {
    std::map<std::string, json> inputs;
    inputs["width"] = width;
    inputs["height"] = height;
    inputs["batch_size"] = batch_size;
    return add_node("EmptyLatentImage", inputs);
}

std::string WorkflowBuilder::add_lora_loader(const std::string& lora_path, float strength_model, float strength_clip) {
    std::map<std::string, json> inputs;
    inputs["lora_name"] = lora_path;
    inputs["strength_model"] = strength_model;
    inputs["strength_clip"] = strength_clip;
    return add_node("LoRALoader", inputs);
}

std::string WorkflowBuilder::add_lora_stack(const std::vector<std::string>& lora_node_ids) {
    std::map<std::string, json> inputs;
    for (size_t i = 0; i < lora_node_ids.size() && i < 4; i++) {
        inputs["lora_" + std::to_string(i + 1)] = make_link(lora_node_ids[i], 0);
    }
    return add_node("LoRAStack", inputs);
}

std::string WorkflowBuilder::add_ksampler(const std::string& model_node_id, const std::string& positive_node_id,
                                          const std::string& negative_node_id, const std::string& latent_node_id,
                                          int seed, int steps, float cfg, const std::string& sampler, float denoise,
                                          const std::string& lora_stack_node_id) {
    std::map<std::string, json> inputs;
    inputs["model"] = make_link(model_node_id, 0); // MODEL slot
    inputs["positive"] = make_link(positive_node_id, 0);
    inputs["negative"] = make_link(negative_node_id, 0);
    inputs["latent_image"] = make_link(latent_node_id, 0);
    inputs["seed"] = seed;
    inputs["steps"] = steps;
    inputs["cfg"] = cfg;
    inputs["sampler_name"] = sampler;
    inputs["denoise"] = denoise;
    if (!lora_stack_node_id.empty()) {
        inputs["lora_stack"] = make_link(lora_stack_node_id, 0);
    }
    return add_node("KSampler", inputs);
}

std::string WorkflowBuilder::add_vae_decode(const std::string& latent_node_id, const std::string& vae_node_id) {
    std::map<std::string, json> inputs;
    inputs["samples"] = make_link(latent_node_id, 0);
    inputs["vae"] = make_link(vae_node_id, 2); // VAE slot
    return add_node("VAEDecode", inputs);
}

std::string WorkflowBuilder::add_save_image(const std::string& image_node_id, const std::string& filename_prefix) {
    std::map<std::string, json> inputs;
    inputs["images"] = make_link(image_node_id, 0);
    inputs["filename_prefix"] = filename_prefix;
    return add_node("SaveImage", inputs);
}

std::string WorkflowBuilder::add_load_image(const std::string& image_path) {
    std::map<std::string, json> inputs;
    inputs["image"] = image_path;
    return add_node("LoadImage", inputs);
}

std::string WorkflowBuilder::add_vae_encode(const std::string& image_node_id, const std::string& vae_node_id) {
    std::map<std::string, json> inputs;
    inputs["pixels"] = make_link(image_node_id, 0);
    inputs["vae"] = make_link(vae_node_id, 2); // VAE slot
    return add_node("VAEEncode", inputs);
}

std::string WorkflowBuilder::add_image_scale(const std::string& image_node_id, int width, int height,
                                             const std::string& method) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["width"] = width;
    inputs["height"] = height;
    inputs["method"] = method;
    return add_node("ImageScale", inputs);
}

std::string WorkflowBuilder::add_upscale_model_loader(const std::string& model_path, bool use_gpu, int tile_size) {
    std::map<std::string, json> inputs;
    inputs["model_name"] = model_path;
    inputs["use_gpu"] = use_gpu;
    inputs["tile_size"] = tile_size;
    return add_node("UpscaleModelLoader", inputs);
}

std::string WorkflowBuilder::add_image_upscale_with_model(const std::string& image_node_id,
                                                          const std::string& upscaler_node_id) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["upscale_model"] = make_link(upscaler_node_id, 0);
    return add_node("ImageUpscaleWithModel", inputs);
}

std::string WorkflowBuilder::add_controlnet_loader(const std::string& model_path) {
    std::map<std::string, json> inputs;
    inputs["control_net_name"] = model_path;
    return add_node("ControlNetLoader", inputs);
}

std::string WorkflowBuilder::add_controlnet_apply(const std::string& conditioning_node_id,
                                                  const std::string& controlnet_node_id,
                                                  const std::string& image_node_id, float strength) {
    std::map<std::string, json> inputs;
    inputs["conditioning"] = make_link(conditioning_node_id, 0);
    inputs["control_net"] = make_link(controlnet_node_id, 0);
    inputs["image"] = make_link(image_node_id, 0);
    inputs["strength"] = strength;
    return add_node("ControlNetApply", inputs);
}

std::string WorkflowBuilder::add_canny_preprocessor(const std::string& image_node_id, int low_threshold,
                                                    int high_threshold) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["low_threshold"] = low_threshold;
    inputs["high_threshold"] = high_threshold;
    return add_node("CannyEdgePreprocessor", inputs);
}

std::string WorkflowBuilder::add_lineart_loader(const std::string& model_path) {
    std::map<std::string, json> inputs;
    inputs["model_name"] = model_path;
    return add_node("LineArtLoader", inputs);
}

std::string WorkflowBuilder::add_lineart_preprocessor(const std::string& image_node_id,
                                                      const std::string& lineart_model_node_id) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["lineart_model"] = make_link(lineart_model_node_id, 0);
    return add_node("LineArtPreprocessor", inputs);
}

std::string WorkflowBuilder::add_load_image_mask(const std::string& image_path, const std::string& channel) {
    std::map<std::string, json> inputs;
    inputs["image"] = image_path;
    inputs["channel"] = channel;
    return add_node("LoadImageMask", inputs);
}

std::string WorkflowBuilder::add_ipadapter_loader(const std::string& model_path, int cross_attention_dim,
                                                  int num_tokens, int clip_embeddings_dim) {
    std::map<std::string, json> inputs;
    inputs["ipadapter_file"] = model_path;
    inputs["cross_attention_dim"] = cross_attention_dim;
    inputs["num_tokens"] = num_tokens;
    inputs["clip_embeddings_dim"] = clip_embeddings_dim;
    return add_node("IPAdapterLoader", inputs);
}

std::string WorkflowBuilder::add_ipadapter_apply(const std::string& conditioning_node_id,
                                                 const std::string& ipadapter_node_id, const std::string& image_node_id,
                                                 float strength) {
    std::map<std::string, json> inputs;
    inputs["conditioning"] = make_link(conditioning_node_id, 0);
    inputs["ipadapter"] = make_link(ipadapter_node_id, 0);
    inputs["image"] = make_link(image_node_id, 0);
    inputs["strength"] = strength;
    return add_node("IPAdapterApply", inputs);
}

std::string WorkflowBuilder::add_conditioning_combine(const std::string& cond1_node_id,
                                                      const std::string& cond2_node_id) {
    std::map<std::string, json> inputs;
    inputs["conditioning_1"] = make_link(cond1_node_id, 0);
    inputs["conditioning_2"] = make_link(cond2_node_id, 0);
    return add_node("ConditioningCombine", inputs);
}

std::string WorkflowBuilder::add_conditioning_concat(const std::string& cond_to_node_id,
                                                     const std::string& cond_from_node_id) {
    std::map<std::string, json> inputs;
    inputs["conditioning_to"] = make_link(cond_to_node_id, 0);
    inputs["conditioning_from"] = make_link(cond_from_node_id, 0);
    return add_node("ConditioningConcat", inputs);
}

std::string WorkflowBuilder::add_conditioning_average(const std::string& cond_to_node_id,
                                                      const std::string& cond_from_node_id, float strength) {
    std::map<std::string, json> inputs;
    inputs["conditioning_to"] = make_link(cond_to_node_id, 0);
    inputs["conditioning_from"] = make_link(cond_from_node_id, 0);
    inputs["conditioning_to_strength"] = strength;
    return add_node("ConditioningAverage", inputs);
}

std::string WorkflowBuilder::add_clip_set_last_layer(const std::string& clip_node_id, int stop_at_clip_layer) {
    std::map<std::string, json> inputs;
    inputs["clip"] = make_link(clip_node_id, 0);
    inputs["stop_at_clip_layer"] = stop_at_clip_layer;
    return add_node("CLIPSetLastLayer", inputs);
}

std::string WorkflowBuilder::add_clip_vision_encode(const std::string& clip_node_id, const std::string& image_node_id) {
    std::map<std::string, json> inputs;
    inputs["clip"] = make_link(clip_node_id, 0);
    inputs["image"] = make_link(image_node_id, 0);
    return add_node("CLIPVisionEncode", inputs);
}

std::string WorkflowBuilder::add_rembg_model_loader(const std::string& model_path) {
    std::map<std::string, json> inputs;
    inputs["model_path"] = model_path;
    return add_node("RemBGModelLoader", inputs);
}

std::string WorkflowBuilder::add_image_remove_background(const std::string& image_node_id,
                                                         const std::string& model_node_id) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["model"] = make_link(model_node_id, 0);
    return add_node("ImageRemoveBackground", inputs);
}

std::string WorkflowBuilder::add_image_invert(const std::string& image_node_id) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    return add_node("ImageInvert", inputs);
}

std::string WorkflowBuilder::add_image_color_adjust(const std::string& image_node_id, float brightness, float contrast,
                                                    float saturation) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["brightness"] = brightness;
    inputs["contrast"] = contrast;
    inputs["saturation"] = saturation;
    return add_node("ImageColorAdjust", inputs);
}

std::string WorkflowBuilder::add_image_blur(const std::string& image_node_id, int radius) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["radius"] = radius;
    return add_node("ImageBlur", inputs);
}

std::string WorkflowBuilder::add_image_grayscale(const std::string& image_node_id) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    return add_node("ImageGrayscale", inputs);
}

std::string WorkflowBuilder::add_image_threshold(const std::string& image_node_id, int threshold) {
    std::map<std::string, json> inputs;
    inputs["image"] = make_link(image_node_id, 0);
    inputs["threshold"] = threshold;
    return add_node("ImageThreshold", inputs);
}

std::string WorkflowBuilder::add_ksampler_advanced(
    const std::string& model_node_id, const std::string& positive_node_id, const std::string& negative_node_id,
    const std::string& latent_node_id, int seed, int steps, float cfg, const std::string& sampler, float denoise,
    int start_at_step, int end_at_step, bool add_noise, const std::string& lora_stack_node_id) {
    std::map<std::string, json> inputs;
    inputs["model"] = make_link(model_node_id, 0);
    inputs["positive"] = make_link(positive_node_id, 0);
    inputs["negative"] = make_link(negative_node_id, 0);
    inputs["latent_image"] = make_link(latent_node_id, 0);
    inputs["seed"] = seed;
    inputs["steps"] = steps;
    inputs["cfg"] = cfg;
    inputs["sampler_name"] = sampler;
    inputs["denoise"] = denoise;
    inputs["start_at_step"] = start_at_step;
    inputs["end_at_step"] = end_at_step;
    inputs["add_noise"] = add_noise;
    if (!lora_stack_node_id.empty()) {
        inputs["lora_stack"] = make_link(lora_stack_node_id, 0);
    }
    return add_node("KSamplerAdvanced", inputs);
}

std::string WorkflowBuilder::to_json_string() const {
    return workflow_.dump(4);
}

bool WorkflowBuilder::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open())
        return false;
    file << workflow_.dump(4);
    return true;
}

void WorkflowBuilder::clear() {
    workflow_.clear();
    next_id_ = 1;
}

// ============================================================================
// Txt2ImgBuilder
// ============================================================================
std::string Txt2ImgBuilder::build(const std::string& ckpt_path, const std::string& prompt,
                                  const std::string& negative_prompt, int width, int height, int seed, int steps,
                                  float cfg, const std::string& output_prefix) {
    WorkflowBuilder builder;

    std::string loader = builder.add_checkpoint_loader(ckpt_path);
    std::string positive = builder.add_clip_encode(prompt, loader);
    std::string negative = builder.add_clip_encode(negative_prompt, loader);
    std::string latent = builder.add_empty_latent(width, height);
    std::string sampler = builder.add_ksampler(loader, positive, negative, latent, seed, steps, cfg, "euler", 1.0f);
    std::string decoded = builder.add_vae_decode(sampler, loader);
    builder.add_save_image(decoded, output_prefix);

    return builder.to_json_string();
}

// ============================================================================
// Img2ImgBuilder
// ============================================================================
std::string Img2ImgBuilder::build(const std::string& ckpt_path, const std::string& input_image,
                                  const std::string& prompt, const std::string& negative_prompt, float denoise,
                                  int seed, int steps, float cfg, const std::string& output_prefix) {
    WorkflowBuilder builder;

    std::string loader = builder.add_checkpoint_loader(ckpt_path);
    std::string positive = builder.add_clip_encode(prompt, loader);
    std::string negative = builder.add_clip_encode(negative_prompt, loader);
    std::string image = builder.add_load_image(input_image);
    std::string encoded = builder.add_vae_encode(image, loader);
    std::string sampler = builder.add_ksampler(loader, positive, negative, encoded, seed, steps, cfg, "euler", denoise);
    std::string decoded = builder.add_vae_decode(sampler, loader);
    builder.add_save_image(decoded, output_prefix);

    return builder.to_json_string();
}

// ============================================================================
// ImageProcessBuilder
// ============================================================================
std::string ImageProcessBuilder::build(const std::string& input_image, int target_width, int target_height, int crop_x,
                                       int crop_y, int crop_w, int crop_h, const std::string& output_prefix) {
    WorkflowBuilder builder;

    std::string image = builder.add_load_image(input_image);
    std::string current = image;

    if (target_width > 0 && target_height > 0) {
        current = builder.add_image_scale(current, target_width, target_height, "bilinear");
    }

    if (crop_x >= 0 && crop_y >= 0 && crop_w > 0 && crop_h > 0) {
        std::map<std::string, json> inputs;
        inputs["image"] = builder.make_link(current, 0);
        inputs["x"] = crop_x;
        inputs["y"] = crop_y;
        inputs["width"] = crop_w;
        inputs["height"] = crop_h;
        current = builder.add_node("ImageCrop", inputs);
    }

    builder.add_save_image(current, output_prefix);
    return builder.to_json_string();
}

// ============================================================================
// DeepHiresBuilder
// ============================================================================
std::string DeepHiresBuilder::build(const std::string& ckpt_path, const std::string& prompt,
                                    const std::string& negative_prompt, int target_width, int target_height, int seed,
                                    int steps, float cfg, const std::string& input_image, float strength,
                                    bool vae_tiling, const std::string& output_prefix) {
    WorkflowBuilder builder;

    std::string loader = builder.add_checkpoint_loader(ckpt_path);
    std::string positive = builder.add_clip_encode(prompt, loader);
    std::string negative = builder.add_clip_encode(negative_prompt, loader);

    std::map<std::string, json> inputs;
    inputs["model"] = builder.make_link(loader, 0);
    inputs["positive"] = builder.make_link(positive, 0);
    inputs["negative"] = builder.make_link(negative, 0);
    inputs["positive_text"] = builder.make_link(positive, 1);
    inputs["negative_text"] = builder.make_link(negative, 1);
    inputs["seed"] = seed;
    inputs["steps"] = steps;
    inputs["cfg"] = cfg;
    inputs["target_width"] = target_width;
    inputs["target_height"] = target_height;
    inputs["strength"] = strength;
    inputs["vae_tiling"] = vae_tiling;

    if (!input_image.empty()) {
        std::string image = builder.add_load_image(input_image);
        inputs["init_image"] = builder.make_link(image, 0);
    }

    std::string deep_hires = builder.add_node("DeepHighResFix", inputs);
    builder.add_save_image(deep_hires, output_prefix);

    return builder.to_json_string();
}

// ============================================================================
// IPAdapterTxt2ImgBuilder
// ============================================================================
std::string IPAdapterTxt2ImgBuilder::build(const std::string& ckpt_path, const std::string& prompt,
                                           const std::string& negative_prompt, const std::string& ipadapter_path,
                                           const std::string& reference_image, float ipadapter_strength,
                                           int cross_attention_dim, int num_tokens, int clip_embeddings_dim, int width,
                                           int height, int seed, int steps, float cfg,
                                           const std::string& output_prefix) {
    WorkflowBuilder builder;

    std::string loader = builder.add_checkpoint_loader(ckpt_path);
    std::string positive = builder.add_clip_encode(prompt, loader);
    std::string negative = builder.add_clip_encode(negative_prompt, loader);
    std::string latent = builder.add_empty_latent(width, height);

    std::string ipadapter =
        builder.add_ipadapter_loader(ipadapter_path, cross_attention_dim, num_tokens, clip_embeddings_dim);
    std::string ref_image = builder.add_load_image(reference_image);
    std::string ipadapter_positive = builder.add_ipadapter_apply(positive, ipadapter, ref_image, ipadapter_strength);

    std::string sampler =
        builder.add_ksampler(loader, ipadapter_positive, negative, latent, seed, steps, cfg, "euler", 1.0f);
    std::string decoded = builder.add_vae_decode(sampler, loader);
    builder.add_save_image(decoded, output_prefix);

    return builder.to_json_string();
}

} // namespace sdengine
