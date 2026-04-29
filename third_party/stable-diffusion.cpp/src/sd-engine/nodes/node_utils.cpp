// ============================================================================
// sd-engine/nodes/node_utils.cpp
// ============================================================================
// 核心节点公共辅助函数实现
// 仅使用 upstream 官方 API（stable-diffusion.h）
// ============================================================================

#include "nodes/node_utils.h"
#include "adapter/sd_adapter.h"
#include "core/log.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"

namespace sdengine {

// ============================================================================
// 基础图像加载/保存函数
// ============================================================================

/// 从文件加载图像为 ImagePtr
ImagePtr load_image_from_file(const std::string& path, int* out_w, int* out_h, int* out_c) {
    int w, h, c;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!data) {
        LOG_ERROR("[LoadImage] Failed to load: %s\n", path.c_str());
        return nullptr;
    }
    
    auto img = create_image_ptr(w, h, c, MallocBuffer(data, &std::free));
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    if (out_c) *out_c = c;
    return img;
}

/// 保存图像到文件
sd_error_t save_image_to_file(const sd_image_t* img, const std::string& path) {
    if (!img || !img->data) {
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    
    int success = 0;
    if (img->channel == 4) {
        success = stbi_write_png(path.c_str(), img->width, img->height, 4, img->data, img->width * 4);
    } else if (img->channel == 3) {
        success = stbi_write_png(path.c_str(), img->width, img->height, 3, img->data, img->width * 3);
    } else {
        LOG_ERROR("[SaveImage] Unsupported channel count: %d\n", img->channel);
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    
    if (!success) {
        return sd_error_t::ERROR_FILE_IO;
    }
    return sd_error_t::OK;
}

/// 调整图像大小
ImagePtr resize_image(const sd_image_t* src, int target_w, int target_h) {
    if (!src || !src->data) return nullptr;
    
    auto dst = make_malloc_buffer(target_w * target_h * src->channel);
    if (!dst) return nullptr;
    
    int result = stbir_resize_uint8(
        src->data, src->width, src->height, 0,
        dst.get(), target_w, target_h, 0,
        src->channel
    );
    
    if (!result) return nullptr;
    return create_image_ptr(target_w, target_h, src->channel, std::move(dst));
}

// ============================================================================
// 生成参数扩展填充（消除 Txt2Img/Img2Img/HiResFix 重复代码）
// ============================================================================

sd_error_t fill_gen_params_from_inputs(const NodeInputs& inputs, sd_img_gen_params_t& gen_params,
                                       const char* log_prefix) {
    // 1. LoRA
    std::vector<sd_lora_t> lora_array;
    auto lora_it = inputs.find("lora_stack");
    if (lora_it != inputs.end()) {
        auto opt_loras = any_cast_safe<std::vector<LoRAInfo>>(lora_it->second);
        if (!opt_loras) {
            LOG_ERROR("%s Invalid lora_stack type\n", log_prefix);
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        auto& loras = *opt_loras;
        if (!loras.empty()) {
            lora_array.reserve(loras.size());
            for (const auto& lora : loras) {
                lora_array.push_back({false, lora.strength, lora.path.c_str()});
            }
            gen_params.loras = lora_array.data();
            gen_params.lora_count = (uint32_t)lora_array.size();
            LOG_INFO("%s Applying %zu LoRA(s)\n", log_prefix, loras.size());
        }
    }

    // 2. ControlNet
    auto ctrl_it = inputs.find("control_image");
    if (ctrl_it != inputs.end()) {
        auto opt_ctrl_img = any_cast_safe<ImagePtr>(ctrl_it->second);
        if (!opt_ctrl_img) {
            LOG_ERROR("%s Invalid control_image type\n", log_prefix);
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        auto ctrl_img = *opt_ctrl_img;
        if (ctrl_img && ctrl_img->data) {
            gen_params.control_image = *ctrl_img;
            gen_params.control_strength = get_input_opt<float>(inputs, "control_strength", 1.0f);
            LOG_INFO("%s ControlNet enabled, strength=%.2f\n", log_prefix, gen_params.control_strength);
        }
    }

    // 3. IPAdapter (ref_images)
    auto ref_it = inputs.find("ref_images");
    if (ref_it != inputs.end()) {
        auto opt_ref_list = any_cast_safe<std::vector<ImagePtr>>(ref_it->second);
        if (!opt_ref_list) {
            LOG_ERROR("%s Invalid ref_images type\n", log_prefix);
            return sd_error_t::ERROR_INVALID_INPUT;
        }
        auto& ref_list = *opt_ref_list;
        if (!ref_list.empty()) {
            std::vector<sd_image_t> ref_sd_images;
            ref_sd_images.reserve(ref_list.size());
            for (auto& img : ref_list) {
                if (img && img->data) {
                    ref_sd_images.push_back(*img);
                }
            }
            if (!ref_sd_images.empty()) {
                gen_params.ref_images = ref_sd_images.data();
                gen_params.ref_images_count = (int)ref_sd_images.size();
                LOG_INFO("%s IPAdapter enabled with %d ref image(s)\n", log_prefix, gen_params.ref_images_count);
            }
        }
    }

    // 4. VAE Tiling
    bool vae_tiling = get_input_opt<bool>(inputs, "vae_tiling", false);
    if (vae_tiling) {
        gen_params.vae_tiling_params.enabled = true;
        gen_params.vae_tiling_params.tile_size_x = 64;
        gen_params.vae_tiling_params.tile_size_y = 64;
        gen_params.vae_tiling_params.target_overlap = 0.5f;
        LOG_INFO("%s VAE tiling enabled\n", log_prefix);
    }

    return sd_error_t::OK;
}

// ============================================================================
// 输入参数验证
// ============================================================================

sd_error_t validate_generation_params(int width, int height, int steps, float cfg_scale, float strength) {
    if (width <= 0 || height <= 0) {
        LOG_ERROR("[Validate] Invalid dimensions: %dx%d (must be > 0)\n", width, height);
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    if (width % 64 != 0 || height % 64 != 0) {
        LOG_WARN("[Validate] Dimensions not aligned to 64: %dx%d, upstream may reject\n", width, height);
    }
    if (steps <= 0) {
        LOG_ERROR("[Validate] Invalid steps: %d (must be > 0)\n", steps);
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    if (cfg_scale <= 0.0f) {
        LOG_ERROR("[Validate] Invalid cfg_scale: %.2f (must be > 0)\n", cfg_scale);
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    if (strength >= 0.0f && (strength < 0.0f || strength > 1.0f)) {
        LOG_ERROR("[Validate] Invalid strength: %.2f (must be in [0, 1])\n", strength);
        return sd_error_t::ERROR_INVALID_INPUT;
    }
    return sd_error_t::OK;
}

// ============================================================================
// 路径安全校验
// ============================================================================

bool is_path_safe(const std::string& path) {
    // 检查是否包含目录遍历序列
    if (path.find("..") != std::string::npos) {
        return false;
    }
    // 检查是否包含空字节（某些系统上的路径截断攻击）
    if (path.find('\0') != std::string::npos) {
        return false;
    }
    // 检查是否以 / 开头（绝对路径，在某些场景下可能不安全）
    // 这里允许绝对路径，因为用户可能确实需要使用系统路径
    // 但禁止指向敏感系统目录
    if (path.find("/etc/") != std::string::npos || path.find("/proc/") != std::string::npos ||
        path.find("/sys/") != std::string::npos || path.find("/dev/") != std::string::npos ||
        path.find("/home/") != std::string::npos) {
        return false;
    }
    return true;
}

bool normalize_path_safe(const std::string& path, std::string& out_normalized) {
    out_normalized.clear();
    std::vector<std::string> parts;
    std::string current;
    
    for (char c : path) {
        if (c == '/' || c == '\\') {
            if (!current.empty()) {
                if (current == "..") {
                    // 目录遍历攻击
                    return false;
                }
                if (current != ".") {
                    parts.push_back(current);
                }
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        if (current == "..") {
            return false;
        }
        if (current != ".") {
            parts.push_back(current);
        }
    }
    
    for (size_t i = 0; i < parts.size(); i++) {
        if (i > 0) out_normalized += '/';
        out_normalized += parts[i];
    }
    
    return true;
}

} // namespace sdengine
