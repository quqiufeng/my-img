#pragma once

#include <string>
#include <vector>

#include "image_utils.h"

namespace myimg {

enum class WatermarkPosition {
    TopLeft,
    TopCenter,
    TopRight,
    MiddleLeft,
    Center,
    MiddleRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    Tile  // 平铺
};

struct WatermarkConfig {
    enum class Type { Text, Image } type = Type::Text;
    std::string content;           // 文字内容或图片路径
    WatermarkPosition position = WatermarkPosition::BottomRight;
    float opacity = 0.5f;          // 透明度 0.0-1.0
    int font_size = 24;            // 字体大小（仅文字）
    uint32_t font_color = 0xFFFFFFFF;  // RGBA (默认白色)
    float rotation = 0.0f;         // 旋转角度
    int margin = 20;               // 边距（像素）
    int tile_spacing = 100;        // 平铺间距（像素）
};

class Watermark {
public:
    // 应用水印到图像
    static ImageData apply(const ImageData& image, const WatermarkConfig& config);

    // 文字水印
    static ImageData apply_text(const ImageData& image, const std::string& text,
                                WatermarkPosition position, float opacity,
                                int font_size, uint32_t color, int margin);

    // 图片水印
    static ImageData apply_image(const ImageData& image, const ImageData& watermark,
                                 WatermarkPosition position, float opacity, int margin);

    // 平铺水印
    static ImageData apply_tiled(const ImageData& image, const WatermarkConfig& config);

    // 批量处理
    static bool batch_apply(const std::vector<std::string>& input_files,
                            const std::vector<std::string>& output_files,
                            const WatermarkConfig& config);

private:
    static std::pair<int, int> calculate_position(WatermarkPosition position,
                                                     int img_w, int img_h,
                                                     int wm_w, int wm_h,
                                                     int margin);
};

} // namespace myimg
