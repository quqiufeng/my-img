// ============================================================================
// sd-engine/nodes/node_utils.h
// ============================================================================
// 核心节点公共辅助函数和结构体定义
// ============================================================================

#pragma once

#include "core/log.h"
#include "core/node.h"
#include "core/sd_ptr.h"
#include "stable-diffusion-ext.h"
#include "stable-diffusion.h"
#include "tensor.hpp"
#include <any>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

// stb_image for LoadImage/SaveImage (declare only, implementation in core_nodes.cpp or main)
#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"

#ifdef HAS_ONNXRUNTIME
#include "face/face_align.hpp"
#include "face/face_detect.hpp"
#include "face/face_restore.hpp"
#include "face/face_swap.hpp"
#include "face/face_utils.hpp"
#include "preprocessors/lineart.hpp"
#include <onnxruntime_cxx_api.h>
#endif

namespace sdengine {

// ============================================================================
// 公共辅助函数
// ============================================================================

/// 将 RGBA 图像转换为 RGB
inline std::vector<uint8_t> convert_rgba_to_rgb(const uint8_t* src, int w, int h) {
    std::vector<uint8_t> dst(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        dst[i * 3 + 0] = src[i * 4 + 0];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
    }
    return dst;
}

/// 从已分配的 data 创建 ImagePtr（失败时自动释放 data 并返回错误）
inline ImagePtr create_image_ptr(int w, int h, int c, MallocBuffer&& buffer, sd_error_t* out_err = nullptr) {
    sd_image_t* img = acquire_image();
    if (!img) {
        if (out_err)
            *out_err = sd_error_t::ERROR_MEMORY_ALLOCATION;
        return nullptr;
    }
    img->width = w;
    img->height = h;
    img->channel = c;
    img->data = buffer.release();
    return make_image_ptr(img);
}

/// 从 vector 创建 ImagePtr（内部自动转换为 MallocBuffer）
inline ImagePtr create_image_ptr(int w, int h, int c, std::vector<uint8_t>&& buffer, sd_error_t* out_err = nullptr) {
    auto mb = make_malloc_buffer(buffer.size());
    if (!mb) {
        if (out_err)
            *out_err = sd_error_t::ERROR_MEMORY_ALLOCATION;
        return nullptr;
    }
    memcpy(mb.get(), buffer.data(), buffer.size());
    return create_image_ptr(w, h, c, std::move(mb), out_err);
}

/// 获取 sd_ctx_t*，兼容 SDContextPtr 和裸指针（无异常控制流）
inline sd_ctx_t* extract_sd_ctx(const NodeInputs& inputs, const std::string& key) {
    auto it = inputs.find(key);
    if (it == inputs.end())
        return nullptr;
    const auto& val = it->second;
    if (val.type() == typeid(SDContextPtr))
        return std::any_cast<SDContextPtr>(val).get();
    if (val.type() == typeid(sd_ctx_t*))
        return std::any_cast<sd_ctx_t*>(val);
    return nullptr;
}

// ============================================================================
// 类型安全的节点输入提取辅助函数
// ============================================================================

/// 获取必需输入，类型不匹配时返回错误（无异常）
template <typename T>
inline sd_error_t get_input(const NodeInputs& inputs, const std::string& key, T& out_val) {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    const auto& val = it->second;
    if (val.type() != typeid(T)) {
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    out_val = std::any_cast<T>(val);
    return sd_error_t::OK;
}

/// 获取可选输入，不存在或类型不匹配时返回默认值（无异常）
template <typename T>
inline T get_input_opt(const NodeInputs& inputs, const std::string& key, const T& default_val = T{}) {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
        return default_val;
    }
    const auto& val = it->second;
    if (val.type() != typeid(T)) {
        return default_val;
    }
    return std::any_cast<T>(val);
}

/// 获取可选输入（指针版本），存在且类型匹配时返回 true 并通过指针输出
template <typename T>
inline bool get_input_ptr(const NodeInputs& inputs, const std::string& key, T* out_val) {
    if (!out_val)
        return false;
    auto it = inputs.find(key);
    if (it == inputs.end()) {
        return false;
    }
    const auto& val = it->second;
    if (val.type() != typeid(T)) {
        return false;
    }
    *out_val = std::any_cast<T>(val);
    return true;
}

// ============================================================================
// 共享结构体
// ============================================================================

/// CLIP 包装器（供 CLIPSetLastLayer / CLIPTextEncode 共享）
struct CLIPWrapper {
    sd_ctx_t* sd_ctx = nullptr;
    SDContextPtr sd_ctx_ptr; // keep shared_ptr alive if needed
    int clip_skip = -1;
};

/// LoRA 信息结构体（供 LoRALoader / LoRAStack / KSampler 共享）
struct LoRAInfo {
    std::string path;
    float strength;
};

/// IPAdapter 信息结构体（供 IPAdapterLoader / IPAdapterApply / KSampler 共享）
struct IPAdapterInfo {
    std::string path;
    int cross_attention_dim;
    int num_tokens;
    int clip_embeddings_dim;
    float strength;
};

/// ControlNetApply 传递的信息
struct ControlNetApplyInfo {
    ImagePtr control_image;
    float strength;
};

#ifdef HAS_ONNXRUNTIME
/// RemBG 模型封装
struct RemBGModel {
    std::unique_ptr<Ort::Session> session;
    Ort::Env env;
    Ort::MemoryInfo memory_info;
    std::string path;

    RemBGModel()
        : env(ORT_LOGGING_LEVEL_WARNING, "rembg"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};
#endif

/// DeepHighResFix 节点状态
struct DeepHiresNodeState {
    int phase1_steps;
    int phase2_steps;
    int phase1_w;
    int phase1_h;
    int phase2_w;
    int phase2_h;
    int target_w;
    int target_h;
    float phase1_cfg_scale;
    float phase2_cfg_scale;
    float phase3_cfg_scale;
    bool phase1_done;
    bool phase2_done;
};

// ============================================================================
// DeepHighResFix 辅助函数
// ============================================================================

sd::Tensor<float> upscale_latent_bilinear_node(const sd::Tensor<float>& latent, int target_w, int target_h,
                                               int channels);

sd::Tensor<float> deep_hires_node_latent_hook(sd::Tensor<float>& latent, int step, int total_steps, void* user_data);

void deep_hires_node_guidance_hook(float* txt_cfg, float* img_cfg, float* distilled_guidance, int step, int total_steps,
                                   void* user_data);

// ============================================================================
// KSampler 通用执行逻辑
// ============================================================================

sd_error_t run_sampler_common(sd_ctx_t* sd_ctx, const NodeInputs& inputs, sd_node_sample_params_t& sample_params,
                              sd_latent_t** out_result);

// ============================================================================
// 节点初始化函数（确保各翻译单元被链接）
// ============================================================================

void init_loader_nodes();
void init_conditioning_nodes();
void init_latent_nodes();
void init_image_nodes();
void init_preprocessor_nodes();
void init_face_nodes();

// ============================================================================
// ONNX 占位符宏（用于 !HAS_ONNXRUNTIME 时生成占位节点）
// ============================================================================

#define DEFINE_ONNX_PLACEHOLDER_NODE(class_name, type_name, category, input_defs_fn, output_defs_fn, err_code)         \
    class class_name : public Node {                                                                                   \
      public:                                                                                                          \
        std::string get_class_type() const override {                                                                  \
            return type_name;                                                                                          \
        }                                                                                                              \
        std::string get_category() const override {                                                                    \
            return category;                                                                                           \
        }                                                                                                              \
        std::vector<PortDef> get_inputs() const override {                                                             \
            return input_defs_fn();                                                                                    \
        }                                                                                                              \
        std::vector<PortDef> get_outputs() const override {                                                            \
            return output_defs_fn();                                                                                   \
        }                                                                                                              \
        sd_error_t execute(const NodeInputs&, NodeOutputs&) override {                                                 \
            ::sdengine::Logger::instance().log(::sdengine::LogLevel::ERROR, __FILE__, __LINE__,                        \
                                               "[ERROR] " type_name                                                    \
                                               ": ONNX Runtime not available. Build with HAS_ONNXRUNTIME to enable."); \
            return err_code;                                                                                           \
        }                                                                                                              \
    };

} // namespace sdengine
