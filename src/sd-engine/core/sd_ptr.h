// ============================================================================
// sd-engine/core/sd_ptr.h
// ============================================================================
/// @file sd_ptr.h
/// @brief stable-diffusion.cpp C API 的智能指针封装 + 对象池
///
/// 为 sd_latent_t、sd_conditioning_t、sd_image_t、upscaler_ctx_t 提供
/// std::shared_ptr 封装和自定义 deleter，同时提供 sd_image_t 的全局对象池。
// ============================================================================

#pragma once

#include "object_pool.h"
#include "stable-diffusion-ext.h"
#include "stable-diffusion.h"
#include <memory>

namespace sdengine {

// ============================================================================
// 全局对象池
// ============================================================================

/// @brief 获取 sd_image_t 的全局对象池
///
/// 注意：sd_latent_t 和 sd_conditioning_t 在 stable-diffusion-ext.h 中是前向声明
/// （不完整类型），无法直接池化。只有 sd_image_t 有完整定义，可以池化。
inline ObjectPool<sd_image_t>& get_image_pool() {
    static ObjectPool<sd_image_t> pool([]() { return new sd_image_t(); },
                                       [](sd_image_t* p) {
                                           if (p) {
                                               if (p->data) {
                                                   free(p->data);
                                                   p->data = nullptr;
                                               }
                                               p->width = 0;
                                               p->height = 0;
                                               p->channel = 0;
                                           }
                                       },
                                       16);
    return pool;
}

// ============================================================================
// Custom deleters for stable-diffusion.cpp C API objects
// ============================================================================

/// @brief sd_latent_t 自定义删除器
struct LatentDeleter {
    void operator()(sd_latent_t* p) const {
        if (p) {
            sd_free_latent(p);
            // 注意：sd_free_latent 执行 delete p，所以不需要再归还到池
        }
    }
};

/// @brief sd_conditioning_t 自定义删除器
struct ConditioningDeleter {
    void operator()(sd_conditioning_t* p) const {
        if (p) {
            sd_free_conditioning(p);
        }
    }
};

/// @brief sd_image_t 自定义删除器（优先归还对象池）
struct ImageDeleter {
    void operator()(sd_image_t* p) const;
};

/// @brief upscaler_ctx_t 自定义删除器
struct UpscalerDeleter {
    void operator()(upscaler_ctx_t* p) const {
        if (p)
            free_upscaler_ctx(p);
    }
};

/// @brief sd_ctx_t 自定义删除器
struct SDContextDeleter {
    void operator()(sd_ctx_t* p) const {
        if (p)
            free_sd_ctx(p);
    }
};

using LatentPtr = std::shared_ptr<sd_latent_t>;                       ///< latent 智能指针
using ConditioningPtr = std::shared_ptr<sd_conditioning_t>;           ///< conditioning 智能指针
using ImagePtr = std::shared_ptr<sd_image_t>;                         ///< image 智能指针
using UpscalerPtr = std::shared_ptr<upscaler_ctx_t>;                  ///< upscaler 智能指针
using CLIPVisionOutputPtr = std::shared_ptr<sd_clip_vision_output_t>; ///< CLIP Vision 输出智能指针
using SDContextPtr = std::shared_ptr<sd_ctx_t>;                       ///< SD 上下文智能指针

/// @brief 创建 sd_ctx_t 智能指针
inline SDContextPtr make_sd_context_ptr(sd_ctx_t* p) {
    return SDContextPtr(p, SDContextDeleter{});
}

/// @brief 创建 latent 智能指针
inline LatentPtr make_latent_ptr(sd_latent_t* p) {
    return LatentPtr(p, LatentDeleter{});
}

/// @brief 创建 conditioning 智能指针
inline ConditioningPtr make_conditioning_ptr(sd_conditioning_t* p) {
    return ConditioningPtr(p, ConditioningDeleter{});
}

/// @brief 创建 CLIP Vision 输出智能指针
inline CLIPVisionOutputPtr make_clip_vision_output_ptr(sd_clip_vision_output_t* p) {
    return CLIPVisionOutputPtr(p, [](sd_clip_vision_output_t* ptr) { sd_free_clip_vision_output(ptr); });
}

/// @brief 创建 image 智能指针
inline ImagePtr make_image_ptr(sd_image_t* p) {
    return ImagePtr(p, ImageDeleter{});
}

/// @brief 创建 upscaler 智能指针
inline UpscalerPtr make_upscaler_ptr(upscaler_ctx_t* p) {
    return UpscalerPtr(p, UpscalerDeleter{});
}

// ============================================================================
// 池化版本（用于节点内部创建新对象时减少分配开销）
// ============================================================================

/// @brief 从对象池中获取一个 sd_image_t 实例
inline sd_image_t* acquire_image() {
    return get_image_pool().acquire();
}

/// @brief 将 sd_image_t 实例归还到对象池
inline void release_image(sd_image_t* p) {
    get_image_pool().release(p);
}

/// @name Latent / Conditioning Pooling Limitation
/// @{
///
/// sd_latent_t 和 sd_conditioning_t 在 stable-diffusion-ext.h 中是不完整类型
/// （opaque handle），其完整定义仅在 stable-diffusion.cpp 内部可见：
///   - sd_latent_t       { sd::Tensor<float> tensor; }
///   - sd_conditioning_t { SDCondition cond; }
///
/// 因此无法在当前头文件级别直接使用 ObjectPool<T> 对它们进行池化。
/// 若需要池化这些对象，需在 stable-diffusion.cpp 补丁中增加以下 C API：
///   - sd_latent_t*      sd_latent_pool_acquire();
///   - void              sd_latent_pool_release(sd_latent_t*);
///   - sd_conditioning_t* sd_conditioning_pool_acquire();
///   - void              sd_conditioning_pool_release(sd_conditioning_t*);
///
/// 在 sd_image_t 已经池化的前提下，latent/conditioning 的分配开销相对较小，
/// 此优化可作为后续进阶工程任务处理。
/// @}

// ImageDeleter 的 operator() 定义（必须在 release_image 声明之后）
inline void ImageDeleter::operator()(sd_image_t* p) const {
    if (p) {
        release_image(p);
    }
}

} // namespace sdengine
