// ============================================================================
// sd-engine/nodes/node_utils.h
// ============================================================================
// 核心节点公共辅助函数和结构体定义
// 仅使用 upstream 官方 API（stable-diffusion.h）
// ============================================================================

#pragma once

#include "core/log.h"
#include "core/node.h"
#include "core/sd_ptr.h"
#include "stable-diffusion.h"
#include <any>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <vector>

// stb_image for LoadImage/SaveImage
#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"

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

/// 从已分配的 data 创建 ImagePtr
inline ImagePtr create_image_ptr(int w, int h, int c, MallocBuffer&& buffer, sd_error_t* out_err = nullptr) {
    sd_image_t* img = new sd_image_t();
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

/// 从 vector 创建 ImagePtr
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

// ============================================================================
// 异常安全的 any_cast 包装
// ============================================================================

/// 安全地执行 std::any_cast，类型不匹配或异常时返回 std::nullopt
template <typename T>
inline std::optional<T> any_cast_safe(const std::any& val) {
    try {
        if (val.type() == typeid(T)) {
            return std::any_cast<T>(val);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("[any_cast_safe] Exception: %s\n", e.what());
    }
    return std::nullopt;
}

/// 获取 sd_ctx_t*，兼容 SDContextPtr 和裸指针（异常安全）
inline sd_ctx_t* extract_sd_ctx(const NodeInputs& inputs, const std::string& key) {
    auto it = inputs.find(key);
    if (it == inputs.end())
        return nullptr;
    const auto& val = it->second;
    if (val.type() == typeid(SDContextPtr)) {
        auto opt = any_cast_safe<SDContextPtr>(val);
        return opt ? opt->get() : nullptr;
    }
    if (val.type() == typeid(sd_ctx_t*)) {
        auto opt = any_cast_safe<sd_ctx_t*>(val);
        return opt ? *opt : nullptr;
    }
    return nullptr;
}

// ============================================================================
// 类型安全的节点输入提取辅助函数
// ============================================================================

/// 获取必需输入，类型不匹配时返回错误
inline sd_error_t get_required_string(const NodeInputs& inputs, const std::string& key, std::string& out) {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
        return sd_error_t::ERROR_MISSING_INPUT;
    }
    auto opt = any_cast_safe<std::string>(it->second);
    if (!opt) {
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    out = *opt;
    return sd_error_t::OK;
}

template <typename T>
inline sd_error_t get_input(const NodeInputs& inputs, const std::string& key, T& out_val) {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
        return sd_error_t::ERROR_MISSING_INPUT;
    }
    auto opt = any_cast_safe<T>(it->second);
    if (!opt) {
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    out_val = *opt;
    return sd_error_t::OK;
}

/// 获取可选输入，不存在或类型不匹配时返回默认值
template <typename T>
inline T get_input_opt(const NodeInputs& inputs, const std::string& key, const T& default_val = T{}) {
    auto it = inputs.find(key);
    if (it == inputs.end()) {
        return default_val;
    }
    auto opt = any_cast_safe<T>(it->second);
    if (!opt) {
        return default_val;
    }
    return *opt;
}

// ============================================================================
// 共享结构体
// ============================================================================

/// CLIP 包装器
struct CLIPWrapper {
    sd_ctx_t* sd_ctx = nullptr;
    SDContextPtr sd_ctx_ptr;
    int clip_skip = -1;
};

/// ControlNetApply 传递的信息
struct ControlNetApplyInfo {
    ImagePtr control_image;
    float strength;
};

// ============================================================================
// 生成参数构建辅助函数
// ============================================================================

struct LoRAInfo;

/// 从节点输入中填充 LoRA、ControlNet、IPAdapter、VAE Tiling 等扩展参数
/// 消除 Txt2Img/Img2Img/HiResFix 节点中的重复代码
sd_error_t fill_gen_params_from_inputs(const NodeInputs& inputs, sd_img_gen_params_t& gen_params,
                                       const char* log_prefix = "");

// ============================================================================
// 输入参数验证
// ============================================================================

/// 验证生成参数合法性
/// @return OK 表示通过，否则返回相应错误码并记录日志
sd_error_t validate_generation_params(int width, int height, int steps, float cfg_scale, float strength = -1.0f);

// ============================================================================
// 路径安全校验
// ============================================================================

/// 检查路径是否包含目录遍历攻击（如 ../etc/passwd）
/// @return true 表示路径安全，false 表示包含非法遍历
bool is_path_safe(const std::string& path);

/// 安全地规范化路径，去除冗余的 . 和 ..，同时检测遍历攻击
/// @param out_normalized 输出规范化后的路径
/// @return true 成功且安全，false 检测到非法遍历
bool normalize_path_safe(const std::string& path, std::string& out_normalized);

// ============================================================================
// 图像 I/O 函数
// ============================================================================

ImagePtr load_image_from_file(const std::string& path, int* out_w = nullptr, int* out_h = nullptr, int* out_c = nullptr);
sd_error_t save_image_to_file(const sd_image_t* img, const std::string& path);
ImagePtr resize_image(const sd_image_t* src, int target_w, int target_h);

// ============================================================================
// 节点初始化函数声明（确保各翻译单元被链接）
// ============================================================================

void init_loader_nodes();
void init_image_nodes();

// ============================================================================
// 错误处理宏
// ============================================================================

#define SD_RETURN_IF_ERROR(expr)                                                                                       \
    do {                                                                                                               \
        if (sd_error_t err = (expr); is_error(err)) {                                                                  \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define SD_RETURN_IF_NULL(ptr, err_code)                                                                               \
    do {                                                                                                               \
        if (!(ptr)) {                                                                                                  \
            return (err_code);                                                                                         \
        }                                                                                                              \
    } while (0)

} // namespace sdengine
