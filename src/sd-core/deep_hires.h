// =============================================================================
// sd-core/deep_hires.h
// =============================================================================
// 
// Deep HighRes Fix 公共头文件
// 
// 注意：sd_deep_hires_params_t 结构体定义在 stable-diffusion-ext.h 中
// =============================================================================

#pragma once

#include "stable-diffusion-ext.h"

#ifdef __cplusplus
extern "C" {
#endif

// 参数初始化函数声明（实现在 deep_hires.cpp 中）
void sd_deep_hires_params_init_impl(sd_deep_hires_params_t* params);

#ifdef __cplusplus
}
#endif
