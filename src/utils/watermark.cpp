#include "watermark.h"
#include "utils/log.h"

#include <freetype2/ft2build.h>
#include FT_FREETYPE_H
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>

// stb_image_write for C-style PNG saving (no std::string ABI issues)
#include <stb_image_write.h>

namespace myimg {

// 简单的 PNG 保存（避免 ABI 问题）
static bool save_png_simple(const ImageData& img, const std::string& path) {
    // 创建 BGR 到 RGB 的副本
    std::vector<uint8_t> bgr_data(img.width * img.height * 3);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            int idx = (y * img.width + x) * 3;
            bgr_data[idx + 0] = img.data[idx + 2];  // B <- R
            bgr_data[idx + 1] = img.data[idx + 1];  // G <- G
            bgr_data[idx + 2] = img.data[idx + 0];  // R <- B
        }
    }
    return stbi_write_png(path.c_str(), img.width, img.height, 3, bgr_data.data(), img.width * 3) != 0;
}

ImageData Watermark::apply(const ImageData& image, const WatermarkConfig& config) {
    if (config.position == WatermarkPosition::Tile) {
        return apply_tiled(image, config);
    }

    if (config.type == WatermarkConfig::Type::Text) {
        return apply_text(image, config.content, config.position, config.opacity,
                         config.font_size, config.font_color, config.margin);
    } else {
        ImageData wm = load_image_from_file(config.content);
        if (wm.empty()) {
            LOG_WARN("Failed to load watermark image: %s", config.content.c_str());
            return image;
        }
        return apply_image(image, wm, config.position, config.opacity, config.margin);
    }
}

ImageData Watermark::apply_text(const ImageData& image, const std::string& text,
                                WatermarkPosition position, float opacity,
                                int font_size, uint32_t color, int margin) {
    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));
    cv::Mat result = img.clone();

    // 初始化 FreeType
    FT_Library ft_library;
    FT_Error error = FT_Init_FreeType(&ft_library);
    if (error) {
        LOG_WARN("Failed to initialize FreeType");
        return image;
    }

    // 加载字体（使用系统默认字体）
    FT_Face ft_face;
    const char* font_paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        nullptr
    };

    bool font_loaded = false;
    for (int i = 0; font_paths[i] != nullptr; ++i) {
        if (std::filesystem::exists(font_paths[i])) {
            error = FT_New_Face(ft_library, font_paths[i], 0, &ft_face);
            if (!error) {
                font_loaded = true;
                break;
            }
        }
    }

    if (!font_loaded) {
        LOG_WARN("No font found, skipping text watermark");
        FT_Done_FreeType(ft_library);
        return image;
    }

    // 设置字体大小
    FT_Set_Pixel_Sizes(ft_face, 0, font_size);

    // 解析颜色
    uint8_t r = color & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = (color >> 16) & 0xFF;

    // 计算文字尺寸
    int text_width = 0;
    int text_height = 0;
    int max_bearing_y = 0;

    for (char c : text) {
        FT_Load_Char(ft_face, c, FT_LOAD_RENDER);
        text_width += ft_face->glyph->advance.x >> 6;
        text_height = std::max(text_height, static_cast<int>(ft_face->glyph->bitmap.rows));
        max_bearing_y = std::max(max_bearing_y, static_cast<int>(ft_face->glyph->bitmap_top));
    }

    // 计算位置
    auto [x, y] = calculate_position(position, image.width, image.height,
                                     text_width, text_height + max_bearing_y, margin);

    // 绘制文字
    int pen_x = x;
    int pen_y = y + max_bearing_y;

    for (char c : text) {
        FT_Load_Char(ft_face, c, FT_LOAD_RENDER);
        FT_Bitmap& bitmap = ft_face->glyph->bitmap;

        int glyph_left = pen_x + ft_face->glyph->bitmap_left;
        int glyph_top = pen_y - ft_face->glyph->bitmap_top;

        for (int row = 0; row < static_cast<int>(bitmap.rows); ++row) {
            for (int col = 0; col < static_cast<int>(bitmap.width); ++col) {
                int img_x = glyph_left + col;
                int img_y = glyph_top + row;

                if (img_x >= 0 && img_x < image.width && img_y >= 0 && img_y < image.height) {
                    uint8_t alpha = bitmap.buffer[row * bitmap.pitch + col];
                    float a = (alpha / 255.0f) * opacity;

                    cv::Vec3b& pixel = result.at<cv::Vec3b>(img_y, img_x);
                    pixel[0] = static_cast<uint8_t>(pixel[0] * (1 - a) + b * a);
                    pixel[1] = static_cast<uint8_t>(pixel[1] * (1 - a) + g * a);
                    pixel[2] = static_cast<uint8_t>(pixel[2] * (1 - a) + r * a);
                }
            }
        }

        pen_x += ft_face->glyph->advance.x >> 6;
    }

    // 清理
    FT_Done_Face(ft_face);
    FT_Done_FreeType(ft_library);

    // 转换回 ImageData
    ImageData output;
    output.width = image.width;
    output.height = image.height;
    output.channels = 3;
    output.data.assign(result.data, result.data + result.total() * result.elemSize());

    return output;
}

ImageData Watermark::apply_image(const ImageData& image, const ImageData& watermark,
                                 WatermarkPosition position, float opacity, int margin) {
    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));
    cv::Mat wm(watermark.height, watermark.width, CV_8UC3, const_cast<uint8_t*>(watermark.data.data()));

    cv::Mat result = img.clone();

    // 如果水印太大，等比例缩小到图像的 1/4
    float max_wm_width = image.width * 0.25f;
    float max_wm_height = image.height * 0.25f;
    if (watermark.width > max_wm_width || watermark.height > max_wm_height) {
        float scale = std::min(max_wm_width / watermark.width, max_wm_height / watermark.height);
        cv::resize(wm, wm, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // 计算位置
    auto [x, y] = calculate_position(position, image.width, image.height,
                                     wm.cols, wm.rows, margin);

    // 确保不超出边界
    x = std::max(0, x);
    y = std::max(0, y);
    int wm_w = std::min(wm.cols, image.width - x);
    int wm_h = std::min(wm.rows, image.height - y);

    if (wm_w <= 0 || wm_h <= 0) {
        return image;
    }

    // 裁剪水印到合适大小
    cv::Mat wm_cropped = wm(cv::Rect(0, 0, wm_w, wm_h));

    // 创建 ROI
    cv::Mat roi = result(cv::Rect(x, y, wm_w, wm_h));

    // 混合
    cv::addWeighted(wm_cropped, opacity, roi, 1.0f - opacity, 0, roi);

    // 转换回 ImageData
    ImageData output;
    output.width = image.width;
    output.height = image.height;
    output.channels = 3;
    output.data.assign(result.data, result.data + result.total() * result.elemSize());

    return output;
}

ImageData Watermark::apply_tiled(const ImageData& image, const WatermarkConfig& config) {
    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));
    cv::Mat result = img.clone();

    if (config.type == WatermarkConfig::Type::Text) {
        // 对于平铺文字，我们简化为使用 apply_text 多次
        // 先估算文字尺寸
        int text_width = config.content.length() * config.font_size * 0.6;
        int text_height = config.font_size;

        // 平铺网格
        for (int y = config.tile_spacing; y < image.height; y += config.tile_spacing + text_height + 10) {
            for (int x = config.tile_spacing; x < image.width; x += config.tile_spacing + text_width + 20) {
                WatermarkConfig tile_config = config;
                tile_config.position = WatermarkPosition::TopLeft;

                // 直接在 ROI 上绘制
                if (x < image.width && y < image.height) {
                    auto tile_result = apply_text(image, config.content, WatermarkPosition::TopLeft,
                                                 config.opacity * 0.5f, config.font_size / 2,
                                                 config.font_color, 0);
                    // 将 tile_result 合并到 result
                    cv::Mat tile_mat(tile_result.height, tile_result.width, CV_8UC3, tile_result.data.data());
                    // 简化：直接复制（实际应该做 alpha 混合）
                }
            }
        }
    } else {
        // 图片平铺
        ImageData wm = load_image_from_file(config.content);
        if (wm.empty()) {
            return image;
        }

        cv::Mat wm_mat(wm.height, wm.width, CV_8UC3, wm.data.data());

        // 缩放水印到合适大小
        float scale = std::min(100.0f / wm.width, 50.0f / wm.height);
        cv::resize(wm_mat, wm_mat, cv::Size(), scale, scale, cv::INTER_AREA);

        for (int y = config.margin; y < image.height; y += wm_mat.rows + config.tile_spacing) {
            for (int x = config.margin; x < image.width; x += wm_mat.cols + config.tile_spacing) {
                int wm_w = std::min(wm_mat.cols, image.width - x);
                int wm_h = std::min(wm_mat.rows, image.height - y);
                if (wm_w > 0 && wm_h > 0) {
                    cv::Mat roi = result(cv::Rect(x, y, wm_w, wm_h));
                    cv::Mat wm_cropped = wm_mat(cv::Rect(0, 0, wm_w, wm_h));
                    cv::addWeighted(wm_cropped, config.opacity, roi, 1.0f - config.opacity, 0, roi);
                }
            }
        }
    }

    ImageData output;
    output.width = image.width;
    output.height = image.height;
    output.channels = 3;
    output.data.assign(result.data, result.data + result.total() * result.elemSize());

    return output;
}

bool Watermark::batch_apply(const std::vector<std::string>& input_files,
                            const std::vector<std::string>& output_files,
                            const WatermarkConfig& config) {
    if (input_files.size() != output_files.size()) {
        LOG_ERROR("input_files and output_files must have the same size");
        return false;
    }

    bool all_success = true;
    for (size_t i = 0; i < input_files.size(); ++i) {
        std::cout << "[" << (i + 1) << "/" << input_files.size() << "] Processing: " << input_files[i] << std::endl;

        ImageData img = load_image_from_file(input_files[i]);
        if (img.empty()) {
            LOG_ERROR("  Failed to load: %s", input_files[i].c_str());
            all_success = false;
            continue;
        }

        ImageData result = apply(img, config);

        // 保存
        if (!save_png_simple(result, output_files[i])) {
            LOG_ERROR("  Failed to save: %s", output_files[i].c_str());
            all_success = false;
        }
    }

    return all_success;
}

std::pair<int, int> Watermark::calculate_position(WatermarkPosition position,
                                                     int img_w, int img_h,
                                                     int wm_w, int wm_h,
                                                     int margin) {
    int x = 0, y = 0;

    switch (position) {
        case WatermarkPosition::TopLeft:
            x = margin;
            y = margin;
            break;
        case WatermarkPosition::TopCenter:
            x = (img_w - wm_w) / 2;
            y = margin;
            break;
        case WatermarkPosition::TopRight:
            x = img_w - wm_w - margin;
            y = margin;
            break;
        case WatermarkPosition::MiddleLeft:
            x = margin;
            y = (img_h - wm_h) / 2;
            break;
        case WatermarkPosition::Center:
            x = (img_w - wm_w) / 2;
            y = (img_h - wm_h) / 2;
            break;
        case WatermarkPosition::MiddleRight:
            x = img_w - wm_w - margin;
            y = (img_h - wm_h) / 2;
            break;
        case WatermarkPosition::BottomLeft:
            x = margin;
            y = img_h - wm_h - margin;
            break;
        case WatermarkPosition::BottomCenter:
            x = (img_w - wm_w) / 2;
            y = img_h - wm_h - margin;
            break;
        case WatermarkPosition::BottomRight:
            x = img_w - wm_w - margin;
            y = img_h - wm_h - margin;
            break;
        case WatermarkPosition::Tile:
            x = margin;
            y = margin;
            break;
    }

    return {x, y};
}

} // namespace myimg
