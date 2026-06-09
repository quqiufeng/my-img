#include "utils/image_postproc.h"
#include "utils/log.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace myimg {

// ============================================================
// 内部辅助: Image ↔ cv::Mat 互转
// ============================================================

/**
 * @brief 将 RGBA uint8 Image 转为 cv::Mat BGR
 */
static cv::Mat image_to_mat_bgr(const Image& img) {
    if (img.empty()) {
        LOG_WARN("Image to Mat: empty image");
        return cv::Mat();
    }
    const int ch = img.channels;
    if (ch == 4) {
        // RGBA → BGR
        cv::Mat rgba(img.height, img.width, CV_8UC4, const_cast<uint8_t*>(img.data.data()));
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        return bgr;
    } else if (ch == 3) {
        // RGB → BGR
        cv::Mat rgb(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return bgr;
    } else {
        LOG_WARN("Image to Mat: unsupported channels=%d (expected 3 or 4)", ch);
        return cv::Mat();
    }
}

/**
 * @brief 将 cv::Mat BGR 写回 RGBA Image
 */
static void mat_bgr_to_image(const cv::Mat& bgr, Image& out) {
    if (out.channels == 4) {
        cv::Mat rgba;
        cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
        out.data.assign(rgba.data, rgba.data + rgba.total() * rgba.channels());
    } else {
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        out.data.assign(rgb.data, rgb.data + rgb.total() * rgb.channels());
    }
}

// ============================================================
// Clarity: 局部对比度增强（大半径 USM）
// ============================================================
static void apply_clarity(cv::Mat& img, float amount) {
    if (amount <= 0.0f) return;

    // Clarity = 大半径 USM (radius ~30-50)
    // 增强中频细节而不影响边缘
    int radius = std::max(3, static_cast<int>(amount * 50.0f));
    if (radius % 2 == 0) radius++; // 高斯核必须为奇数

    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(radius, radius), 0);

    cv::Mat detail = img - blurred;
    cv::addWeighted(img, 1.0, detail, amount * 0.5, 0, img);
}

// ============================================================
// USM Sharpen (Unsharp Mask)
// ============================================================
static void apply_sharpen(cv::Mat& img, float amount, int radius, float threshold) {
    if (amount <= 0.0f) return;

    radius = std::max(1, std::min(radius, 10));
    if (radius % 2 == 0) radius++;

    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(radius, radius), 0);

    // USM: dst = src + amount * (src - blurred)
    // 只对差值超过 threshold 的像素生效
    cv::Mat diff = img - blurred;

    if (threshold > 0.0f) {
        // 创建掩模：仅增强幅度超过 threshold 的区域
        cv::Mat gray_diff;
        cv::cvtColor(diff, gray_diff, cv::COLOR_BGR2GRAY);
        cv::Mat mask = cv::abs(gray_diff) > threshold;

        cv::Mat enhanced;
        cv::addWeighted(img, 1.0, diff, amount, 0, enhanced);
        enhanced.copyTo(img, mask);
    } else {
        cv::addWeighted(img, 1.0, diff, amount, 0, img);
    }
}

// ============================================================
// Smart Sharpen: 边缘感知锐化（双边滤波 + USM）
// ============================================================
static void apply_smart_sharpen(cv::Mat& img, float strength, int radius) {
    if (strength <= 0.0f) return;

    radius = std::max(1, std::min(radius, 10));

    // 双边滤波保留边缘，平滑平坦区域
    cv::Mat bilateral;
    cv::bilateralFilter(img, bilateral, radius * 2, strength * 50.0, strength * 50.0);

    // 提取细节层（原图 - 双边滤波）
    cv::Mat detail = img - bilateral;

    // 计算局部方差掩模：平坦区域不锐化
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat variance;
    cv::GaussianBlur(gray.mul(gray), variance, cv::Size(radius * 2 + 1, radius * 2 + 1), 0);
    cv::Mat mean;
    cv::GaussianBlur(gray, mean, cv::Size(radius * 2 + 1, radius * 2 + 1), 0);
    cv::Mat localVar = variance - mean.mul(mean);

    // 归一化到 0-1
    double maxVar;
    cv::minMaxLoc(localVar, nullptr, &maxVar);
    cv::Mat weight;
    if (maxVar > 1.0) {
        localVar.convertTo(weight, CV_32F, 1.0 / maxVar);
    } else {
        weight = cv::Mat::ones(localVar.size(), CV_32F);
    }

    // 应用加权细节增强
    std::vector<cv::Mat> channels;
    cv::split(detail, channels);
    for (auto& ch : channels) {
        cv::Mat ch_float;
        ch.convertTo(ch_float, CV_32F);
        ch_float = ch_float.mul(weight) * strength;
        ch_float.convertTo(ch, CV_8U);
    }
    cv::merge(channels, detail);

    img = img + detail;
}

// ============================================================
// Edge Sharpen: 边缘掩模锐化（避免光晕）
// ============================================================
static void apply_edge_sharpen(cv::Mat& img, float amount, int radius, float threshold) {
    if (amount <= 0.0f) return;

    radius = std::max(1, std::min(radius, 10));
    if (radius % 2 == 0) radius++;

    // 1. 提取边缘
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray, edges, std::max(1.0, threshold * 255.0), std::max(1.0, threshold * 255.0 * 3.0));

    // 2. 膨胀边缘得到边缘区域掩模
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(radius, radius));
    cv::Mat edgeMask;
    cv::dilate(edges, edgeMask, kernel);

    // 3. USM 锐化全局
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(radius, radius), 0);
    cv::Mat sharpened;
    cv::addWeighted(img, 1.0 + amount, blurred, -amount, 0, sharpened);

    // 4. 只对边缘区域应用锐化
    sharpened.copyTo(img, edgeMask);
}

// ============================================================
// 主入口
// ============================================================
bool apply_image_postprocessing(Image& img, const PostProcessParams& params) {
    if (img.empty()) {
        LOG_WARN("Post-processing: empty image, skipping");
        return false;
    }

    // 检查是否有任何处理需要
    bool has_work = (params.clarity > 0.0f ||
                     params.sharpen_amount > 0.0f ||
                     params.smart_sharpen_strength > 0.0f ||
                     params.edge_sharpen_amount > 0.0f);
    if (!has_work) {
        return true; // 无操作，不算失败
    }

    LOG_INFO("Post-processing: clarity=%.2f, sharpen=%.2f(r=%d,t=%.0f), "
             "smart=%.2f(r=%d), edge=%.2f(r=%d,t=%.2f)",
             params.clarity,
             params.sharpen_amount, params.sharpen_radius, params.sharpen_threshold,
             params.smart_sharpen_strength, params.smart_sharpen_radius,
             params.edge_sharpen_amount, params.edge_sharpen_radius, params.edge_sharpen_threshold);

    try {
        cv::Mat bgr = image_to_mat_bgr(img);
        if (bgr.empty()) {
            LOG_ERROR("Post-processing: failed to convert image to OpenCV format");
            return false;
        }

        // 按顺序应用
        if (params.clarity > 0.0f) {
            apply_clarity(bgr, params.clarity);
            LOG_INFO("  Clarity: done (amount=%.2f)", params.clarity);
        }

        if (params.sharpen_amount > 0.0f) {
            apply_sharpen(bgr, params.sharpen_amount, params.sharpen_radius, params.sharpen_threshold);
            LOG_INFO("  USM Sharpen: done (amount=%.2f, radius=%d, threshold=%.0f)",
                     params.sharpen_amount, params.sharpen_radius, params.sharpen_threshold);
        }

        if (params.smart_sharpen_strength > 0.0f) {
            apply_smart_sharpen(bgr, params.smart_sharpen_strength, params.smart_sharpen_radius);
            LOG_INFO("  Smart Sharpen: done (strength=%.2f, radius=%d)",
                     params.smart_sharpen_strength, params.smart_sharpen_radius);
        }

        if (params.edge_sharpen_amount > 0.0f) {
            apply_edge_sharpen(bgr, params.edge_sharpen_amount, params.edge_sharpen_radius, params.edge_sharpen_threshold);
            LOG_INFO("  Edge Sharpen: done (amount=%.2f, radius=%d, threshold=%.2f)",
                     params.edge_sharpen_amount, params.edge_sharpen_radius, params.edge_sharpen_threshold);
        }

        // 写回 Image
        mat_bgr_to_image(bgr, img);
        LOG_INFO("Post-processing: completed successfully");
        return true;

    } catch (const cv::Exception& e) {
        LOG_ERROR("Post-processing OpenCV error: %s", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Post-processing error: %s", e.what());
        return false;
    }
}

} // namespace myimg
