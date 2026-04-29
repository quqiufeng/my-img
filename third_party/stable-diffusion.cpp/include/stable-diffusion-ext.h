#ifndef __STABLE_DIFFUSION_EXT_H__
#define __STABLE_DIFFUSION_EXT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "stable-diffusion.h"

/** 不透明的 tensor 类型（内部实现对外部隐藏） */
typedef struct sd_tensor_t sd_tensor_t;

/* ============================================================
 * stable-diffusion.cpp Extension API
 * 
 * 此文件提供原子操作 API，允许外部代码（如 libTorch）
 * 接管 HiRes Fix 等复杂流程。
 * 
 * 使用方式：
 *   1. sd_ext_generate_latent() 生成基础 latent
 *   2. 用 libTorch 对 latent 进行上采样、加噪等操作
 *   3. sd_ext_sample_latent() 从处理后的 latent 继续采样
 *   4. sd_ext_vae_decode() 解码为图像
 * 
 * 注意：所有返回 sd_tensor_t* 的 API，调用方负责释放。
 * ============================================================ */

/* ========== Tensor 扩展 API ========== */

/** 获取 tensor 维度数 */
SD_API int sd_ext_tensor_ndim(sd_tensor_t* tensor);

/** 获取指定维度的尺寸 */
SD_API int64_t sd_ext_tensor_shape(sd_tensor_t* tensor, int dim);

/** 获取 tensor 总元素数 */
SD_API int64_t sd_ext_tensor_nelements(sd_tensor_t* tensor);

/** 获取 tensor 数据类型：0=f32, 1=f16 */
SD_API int sd_ext_tensor_dtype(sd_tensor_t* tensor);

/** 
 * 获取 tensor 数据指针
 * 
 * 注意：sd::Tensor 目前基于 CPU 内存（std::vector），
 * 返回的是 CPU 内存指针。如需 GPU 处理，需要外部代码
 * 自行拷贝到 CUDA 设备。
 */
SD_API void* sd_ext_tensor_data_ptr(sd_tensor_t* tensor);

/** 释放 tensor */
SD_API void sd_ext_tensor_free(sd_tensor_t* tensor);

/** 
 * 从外部数据创建 tensor（数据会被拷贝）
 * 
 * @param data   数据指针（CPU 内存）
 * @param shape  维度数组 [dim0, dim1, ...]
 * @param ndim   维度数
 * @param dtype  0=f32, 1=f16
 */
SD_API sd_tensor_t* sd_ext_tensor_from_data(const void* data,
                                             const int64_t* shape,
                                             int ndim,
                                             int dtype);

/* ========== 原子操作 API ========== */

/** 
 * 生成基础 latent（完整采样流程，但不 VAE 解码）
 * 
 * 此函数等价于 generate_image() 的前半段：
 *   - 文本编码
 *   - 准备噪声 latent
 *   - 采样去噪
 *   - 返回采样后的 latent
 * 
 * 如果 params->hires.enabled == true，此函数会忽略 hires 设置，
 * 只生成基础分辨率的 latent。HiRes 处理应由调用方自行实现。
 * 
 * @param ctx    sd.cpp 上下文
 * @param params 生成参数（与 generate_image 相同）
 * @return       采样后的 latent tensor，失败返回 nullptr
 */
SD_API sd_tensor_t* sd_ext_generate_latent(sd_ctx_t* ctx,
                                            const sd_img_gen_params_t* params);

/** 
 * 从已有 latent 继续采样（HiRes refine）
 * 
 * 此函数等价于 generate_image() 的 HiRes 阶段：
 *   - 文本编码（使用传入的 prompt）
 *   - 接收上采样+加噪后的 latent
 *   - 执行扩散采样
 *   - 返回采样后的 latent
 * 
 * @param ctx            sd.cpp 上下文
 * @param init_latent    初始 latent（上采样后的）
 * @param noise          噪声 tensor（可空，为空则内部生成）
 * @param prompt         正提示词
 * @param negative_prompt 负提示词（可空）
 * @param sample_params  采样参数（sample_method, scheduler, steps, guidance 等）
 * @param width          目标宽度
 * @param height         目标高度
 * @return               采样后的 latent tensor，失败返回 nullptr
 */
SD_API sd_tensor_t* sd_ext_sample_latent(sd_ctx_t* ctx,
                                           sd_tensor_t* init_latent,
                                           sd_tensor_t* noise,
                                           const char* prompt,
                                           const char* negative_prompt,
                                           sd_sample_params_t* sample_params,
                                           int width,
                                           int height,
                                           float strength);

/** VAE 编码：图像 → latent */
SD_API sd_tensor_t* sd_ext_vae_encode(sd_ctx_t* ctx, sd_image_t image);

/** VAE 解码：latent → 图像 */
SD_API sd_image_t sd_ext_vae_decode(sd_ctx_t* ctx, sd_tensor_t* latent);

/** 
 * 创建随机噪声 latent
 * 
 * @param width   图像宽度
 * @param height  图像高度
 * @param seed    随机种子
 * @return        噪声 tensor
 */
SD_API sd_tensor_t* sd_ext_create_noise(sd_ctx_t* ctx,
                                         int width,
                                         int height,
                                         int64_t seed);

#ifdef __cplusplus
}
#endif

#endif // __STABLE_DIFFUSION_EXT_H__
