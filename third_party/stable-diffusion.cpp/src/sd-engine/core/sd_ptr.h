// ============================================================================
// sd-engine/core/sd_ptr.h
// ============================================================================
/// @file sd_ptr.h
/// @brief stable-diffusion.cpp C API 的智能指针封装
///
/// 仅使用 upstream 官方 API（stable-diffusion.h），不依赖任何扩展头文件。
/// 支持的类型：sd_ctx_t、sd_image_t、upscaler_ctx_t
// ============================================================================

#pragma once

#include "stable-diffusion.h"
#include <cstdlib>
#include <memory>

namespace sdengine {

// ============================================================================
// RAII wrapper for malloc-allocated pixel buffers
// ============================================================================

using MallocBuffer = std::unique_ptr<uint8_t[], decltype(&std::free)>;

inline MallocBuffer make_malloc_buffer(size_t size) {
    auto* p = static_cast<uint8_t*>(std::malloc(size));
    return MallocBuffer{p, &std::free};
}

// ============================================================================
// Custom deleters for stable-diffusion.cpp C API objects
// ============================================================================

/// @brief sd_image_t 自定义删除器
struct ImageDeleter {
    void operator()(sd_image_t* p) const {
        if (p) {
            if (p->data) {
                std::free(p->data);
            }
            delete p;
        }
    }
};

/// @brief upscaler_ctx_t 自定义删除器
struct UpscalerDeleter {
    void operator()(upscaler_ctx_t* p) const {
        if (p) {
            free_upscaler_ctx(p);
        }
    }
};

/// @brief sd_ctx_t 自定义删除器
struct SDContextDeleter {
    void operator()(sd_ctx_t* p) const {
        if (p) {
            free_sd_ctx(p);
        }
    }
};

using ImagePtr = std::shared_ptr<sd_image_t>;
using UpscalerPtr = std::shared_ptr<upscaler_ctx_t>;
using SDContextPtr = std::shared_ptr<sd_ctx_t>;

inline SDContextPtr make_sd_context_ptr(sd_ctx_t* p) {
    return SDContextPtr(p, SDContextDeleter{});
}

inline ImagePtr make_image_ptr(sd_image_t* p) {
    return ImagePtr(p, ImageDeleter{});
}

inline UpscalerPtr make_upscaler_ptr(upscaler_ctx_t* p) {
    return UpscalerPtr(p, UpscalerDeleter{});
}

} // namespace sdengine
