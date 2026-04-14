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
#include <map>
#include <vector>

namespace sdengine {

using json = nlohmann::json;

// 工作流构建器
class WorkflowBuilder {
public:
    WorkflowBuilder() = default;
    
    // 添加节点，返回节点 ID
    std::string add_node(const std::string& class_type, 
                         const std::map<std::string, json>& inputs);
    
    // 添加 CheckpointLoaderSimple 节点
    std::string add_checkpoint_loader(const std::string& ckpt_path,
                                      const std::string& vae_path = "",
                                      const std::string& clip_path = "");
    
    // 添加 CLIPTextEncode 节点
    std::string add_clip_encode(const std::string& text,
                                const std::string& clip_node_id,
                                int clip_skip = -1);
    
    // 添加 EmptyLatentImage 节点
    std::string add_empty_latent(int width, int height, int batch_size = 1);
    
    // 添加 KSampler 节点
    std::string add_ksampler(const std::string& model_node_id,
                             const std::string& positive_node_id,
                             const std::string& negative_node_id,
                             const std::string& latent_node_id,
                             int seed = 0,
                             int steps = 20,
                             float cfg = 7.5f,
                             const std::string& sampler = "euler",
                             float denoise = 1.0f);
    
    // 添加 VAEDecode 节点
    std::string add_vae_decode(const std::string& latent_node_id,
                               const std::string& vae_node_id);
    
    // 添加 SaveImage 节点
    std::string add_save_image(const std::string& image_node_id,
                               const std::string& filename_prefix = "output");
    
    // 添加 LoadImage 节点
    std::string add_load_image(const std::string& image_path);
    
    // 添加 VAEEncode 节点（img2img）
    std::string add_vae_encode(const std::string& image_node_id,
                               const std::string& vae_node_id);
    
    // 添加 ImageScale 节点
    std::string add_image_scale(const std::string& image_node_id,
                                int width, int height,
                                const std::string& method = "bilinear");
    
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
    static std::string build(const std::string& ckpt_path,
                             const std::string& prompt,
                             const std::string& negative_prompt,
                             int width = 512,
                             int height = 512,
                             int seed = 0,
                             int steps = 20,
                             float cfg = 7.5f,
                             const std::string& output_prefix = "txt2img_output");
};

// 快速构建 img2img 工作流
class Img2ImgBuilder {
public:
    static std::string build(const std::string& ckpt_path,
                             const std::string& input_image,
                             const std::string& prompt,
                             const std::string& negative_prompt,
                             float denoise = 0.75f,
                             int seed = 0,
                             int steps = 20,
                             float cfg = 7.5f,
                             const std::string& output_prefix = "img2img_output");
};

// 快速构建图像处理工作流
class ImageProcessBuilder {
public:
    static std::string build(const std::string& input_image,
                             int target_width = 0,
                             int target_height = 0,
                             int crop_x = -1, int crop_y = -1,
                             int crop_w = -1, int crop_h = -1,
                             const std::string& output_prefix = "processed");
};

// 快速构建 Deep HighRes Fix 工作流
class DeepHiresBuilder {
public:
    static std::string build(const std::string& ckpt_path,
                             const std::string& prompt,
                             const std::string& negative_prompt,
                             int target_width = 1024,
                             int target_height = 1024,
                             int seed = 0,
                             int steps = 30,
                             float cfg = 7.0f,
                             const std::string& input_image = "",
                             float strength = 1.0f,
                             bool vae_tiling = false,
                             const std::string& output_prefix = "deep_hires_output");
};

} // namespace sdengine
