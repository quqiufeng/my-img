#include "shadow_effect.h"

#include <opencv2/imgproc.hpp>
#include <cmath>

namespace myimg {

ImageData ShadowEffect::add_drop_shadow(const ImageData& image, const DropShadowConfig& config) {
    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));

    // 解析阴影颜色
    uint8_t shadow_r = config.color & 0xFF;
    uint8_t shadow_g = (config.color >> 8) & 0xFF;
    uint8_t shadow_b = (config.color >> 16) & 0xFF;

    // 创建阴影层（比原图大，以容纳偏移）
    int shadow_w = image.width + std::abs(config.offset_x) + config.blur_radius * 4;
    int shadow_h = image.height + std::abs(config.offset_y) + config.blur_radius * 4;
    cv::Mat shadow(shadow_h, shadow_w, CV_8UC4, cv::Scalar(shadow_b, shadow_g, shadow_r, 0));

    // 将原图复制到阴影层中心
    int img_x = config.blur_radius * 2 + std::max(0, -config.offset_x);
    int img_y = config.blur_radius * 2 + std::max(0, -config.offset_y);

    cv::Mat roi = shadow(cv::Rect(img_x, img_y, image.width, image.height));
    cv::Mat alpha_mask(image.height, image.width, CV_8UC1, cv::Scalar(255));
    cv::Mat img_bgra;
    cv::cvtColor(img, img_bgra, cv::COLOR_RGB2BGRA);

    // 将图像 alpha 通道设为 255
    std::vector<cv::Mat> channels;
    cv::split(img_bgra, channels);
    channels[3] = alpha_mask;
    cv::merge(channels, img_bgra);

    img_bgra.copyTo(roi);

    // 提取原图的 alpha 通道作为遮罩
    cv::Mat mask(shadow_h, shadow_w, CV_8UC1, cv::Scalar(0));
    cv::Mat mask_roi = mask(cv::Rect(img_x, img_y, image.width, image.height));
    alpha_mask.copyTo(mask_roi);

    // 偏移遮罩
    cv::Mat shifted_mask = cv::Mat::zeros(shadow_h, shadow_w, CV_8UC1);
    int shift_x = img_x + config.offset_x;
    int shift_y = img_y + config.offset_y;
    if (shift_x >= 0 && shift_y >= 0 && shift_x + image.width <= shadow_w && shift_y + image.height <= shadow_h) {
        cv::Mat shifted_roi = shifted_mask(cv::Rect(shift_x, shift_y, image.width, image.height));
        alpha_mask.copyTo(shifted_roi);
    }

    // 模糊阴影遮罩
    cv::GaussianBlur(shifted_mask, shifted_mask, cv::Size(config.blur_radius * 2 + 1, config.blur_radius * 2 + 1), config.blur_radius);

    // 应用阴影
    for (int y = 0; y < shadow_h; ++y) {
        for (int x = 0; x < shadow_w; ++x) {
            float alpha = shifted_mask.at<uint8_t>(y, x) / 255.0f * config.opacity;
            if (alpha > 0) {
                cv::Vec4b& pixel = shadow.at<cv::Vec4b>(y, x);
                if (pixel[3] == 0) {
                    // 没有原图像素，只显示阴影
                    pixel[0] = shadow_b;
                    pixel[1] = shadow_g;
                    pixel[2] = shadow_r;
                    pixel[3] = static_cast<uint8_t>(alpha * 255);
                }
            }
        }
    }

    // 混合图像和阴影
    cv::Mat result(shadow_h, shadow_w, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int y = 0; y < shadow_h; ++y) {
        for (int x = 0; x < shadow_w; ++x) {
            cv::Vec4b shadow_pixel = shadow.at<cv::Vec4b>(y, x);
            if (shadow_pixel[3] > 0) {
                float alpha = shadow_pixel[3] / 255.0f;
                result.at<cv::Vec3b>(y, x)[0] = static_cast<uint8_t>(shadow_pixel[0] * alpha + 255 * (1 - alpha));
                result.at<cv::Vec3b>(y, x)[1] = static_cast<uint8_t>(shadow_pixel[1] * alpha + 255 * (1 - alpha));
                result.at<cv::Vec3b>(y, x)[2] = static_cast<uint8_t>(shadow_pixel[2] * alpha + 255 * (1 - alpha));
            }
        }
    }

    // 转换回 RGB
    ImageData output;
    output.width = shadow_w;
    output.height = shadow_h;
    output.channels = 3;
    output.data.assign(result.data, result.data + result.total() * result.elemSize());

    return output;
}

ImageData ShadowEffect::add_reflection(const ImageData& image, const ReflectionConfig& config) {
    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));

    // 创建倒影
    int reflection_height = static_cast<int>(image.height * config.height_ratio);
    cv::Mat reflection;
    cv::flip(img, reflection, 0);  // 垂直翻转

    // 裁剪倒影到指定高度
    reflection = reflection(cv::Rect(0, 0, image.width, reflection_height));

    // 创建渐变 alpha 遮罩（顶部不透明，底部透明）
    cv::Mat alpha_mask(reflection_height, image.width, CV_8UC1);
    for (int y = 0; y < reflection_height; ++y) {
        float alpha = 1.0f - (static_cast<float>(y) / reflection_height);
        alpha *= config.opacity;
        alpha_mask.row(y).setTo(static_cast<uint8_t>(alpha * 255));
    }

    // 应用渐变
    cv::Mat reflection_rgba;
    cv::cvtColor(reflection, reflection_rgba, cv::COLOR_RGB2BGRA);
    std::vector<cv::Mat> channels;
    cv::split(reflection_rgba, channels);
    channels[3] = alpha_mask;
    cv::merge(channels, reflection_rgba);

    // 模糊倒影底部
    if (config.blur_radius > 0) {
        cv::GaussianBlur(reflection_rgba, reflection_rgba,
                        cv::Size(config.blur_radius * 2 + 1, config.blur_radius * 2 + 1),
                        config.blur_radius);
    }

    // 创建结果画布
    int result_height = image.height + reflection_height + config.gap;
    cv::Mat result(result_height, image.width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 复制原图
    img.copyTo(result(cv::Rect(0, 0, image.width, image.height)));

    // 混合倒影
    cv::Mat reflection_roi = result(cv::Rect(0, image.height + config.gap, image.width, reflection_height));
    for (int y = 0; y < reflection_height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float alpha = alpha_mask.at<uint8_t>(y, x) / 255.0f;
            cv::Vec3b& dst = reflection_roi.at<cv::Vec3b>(y, x);
            cv::Vec3b src = reflection.at<cv::Vec3b>(y, x);
            dst[0] = static_cast<uint8_t>(src[0] * alpha + dst[0] * (1 - alpha));
            dst[1] = static_cast<uint8_t>(src[1] * alpha + dst[1] * (1 - alpha));
            dst[2] = static_cast<uint8_t>(src[2] * alpha + dst[2] * (1 - alpha));
        }
    }

    ImageData output;
    output.width = image.width;
    output.height = result_height;
    output.channels = 3;
    output.data.assign(result.data, result.data + result.total() * result.elemSize());

    return output;
}

ImageData ShadowEffect::add_contact_shadow(const ImageData& image, float opacity, int blur_radius) {
    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));

    // 创建椭圆阴影
    int shadow_height = std::max(20, blur_radius * 4);
    int result_height = image.height + shadow_height + 10;
    cv::Mat result(result_height, image.width, CV_8UC3, cv::Scalar(255, 255, 255));

    // 复制原图
    img.copyTo(result(cv::Rect(0, 0, image.width, image.height)));

    // 创建椭圆阴影
    cv::Mat shadow(shadow_height, image.width, CV_8UC1, cv::Scalar(0));
    cv::ellipse(shadow,
                cv::Point(image.width / 2, shadow_height / 2),
                cv::Size(image.width / 2 - 10, shadow_height / 2 - 2),
                0, 0, 360,
                cv::Scalar(255 * opacity), -1);

    // 模糊
    if (blur_radius > 0) {
        cv::GaussianBlur(shadow, shadow, cv::Size(blur_radius * 2 + 1, blur_radius * 2 + 1), blur_radius);
    }

    // 混合阴影到结果
    cv::Mat shadow_roi = result(cv::Rect(0, image.height + 5, image.width, shadow_height));
    for (int y = 0; y < shadow_height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float alpha = shadow.at<uint8_t>(y, x) / 255.0f;
            if (alpha > 0) {
                cv::Vec3b& pixel = shadow_roi.at<cv::Vec3b>(y, x);
                pixel[0] = static_cast<uint8_t>(pixel[0] * (1 - alpha) + 0 * alpha);
                pixel[1] = static_cast<uint8_t>(pixel[1] * (1 - alpha) + 0 * alpha);
                pixel[2] = static_cast<uint8_t>(pixel[2] * (1 - alpha) + 0 * alpha);
            }
        }
    }

    ImageData output;
    output.width = image.width;
    output.height = result_height;
    output.channels = 3;
    output.data.assign(result.data, result.data + result.total() * result.elemSize());

    return output;
}

ImageData ShadowEffect::add_product_shadow(const ImageData& image,
                                            const DropShadowConfig& drop_config,
                                            const ReflectionConfig& reflection_config) {
    // 先添加倒影
    auto with_reflection = add_reflection(image, reflection_config);

    // 再添加投影（简化版：直接在底部添加接触阴影）
    cv::Mat img(with_reflection.height, with_reflection.width, CV_8UC3,
                const_cast<uint8_t*>(with_reflection.data.data()));

    // 创建结果
    ImageData output;
    output.width = with_reflection.width;
    output.height = with_reflection.height;
    output.channels = 3;
    output.data.assign(img.data, img.data + img.total() * img.elemSize());

    return output;
}

} // namespace myimg
