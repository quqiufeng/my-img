#include "utils/watermark.h"
#include "utils/image_utils.h"
#include <filesystem>
#include <iostream>

using namespace myimg;

// 创建测试图像
ImageData create_test_image_wm(int width, int height, uint8_t r, uint8_t g, uint8_t b) {
    ImageData img;
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.data.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            img.data[(y * width + x) * 3 + 0] = r;
            img.data[(y * width + x) * 3 + 1] = g;
            img.data[(y * width + x) * 3 + 2] = b;
        }
    }
    return img;
}

int main(int argc, char** argv) {
    std::cout << "=== Watermark Test Suite ===" << std::endl;

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: 文字水印 - 右下角
    std::cout << "\n[Test 1] Text watermark - BottomRight..." << std::endl;
    {
        auto img = create_test_image_wm(200, 200, 100, 100, 100);
        WatermarkConfig config;
        config.type = WatermarkConfig::Type::Text;
        config.content = "Test WM";
        config.position = WatermarkPosition::BottomRight;
        config.opacity = 0.7f;
        config.font_size = 20;
        config.font_color = 0xFFFFFFFF;  // 白色
        config.margin = 10;

        auto result = Watermark::apply(img, config);

        if (!result.empty() && result.width == 200 && result.height == 200) {
            std::cout << "  PASSED (watermark applied)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 2: 文字水印 - 多个位置
    std::cout << "\n[Test 2] Text watermark - multiple positions..." << std::endl;
    {
        auto img = create_test_image_wm(300, 300, 50, 50, 50);
        bool all_ok = true;

        std::vector<WatermarkPosition> positions = {
            WatermarkPosition::TopLeft, WatermarkPosition::TopCenter, WatermarkPosition::TopRight,
            WatermarkPosition::MiddleLeft, WatermarkPosition::Center, WatermarkPosition::MiddleRight,
            WatermarkPosition::BottomLeft, WatermarkPosition::BottomCenter, WatermarkPosition::BottomRight
        };

        for (auto pos : positions) {
            WatermarkConfig config;
            config.type = WatermarkConfig::Type::Text;
            config.content = "WM";
            config.position = pos;
            config.opacity = 0.5f;
            config.font_size = 16;
            config.margin = 5;

            auto result = Watermark::apply(img, config);
            if (result.empty()) {
                all_ok = false;
                break;
            }
        }

        if (all_ok) {
            std::cout << "  PASSED (all 9 positions)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 3: 平铺水印
    std::cout << "\n[Test 3] Tiled watermark..." << std::endl;
    {
        auto img = create_test_image_wm(400, 400, 80, 80, 80);
        WatermarkConfig config;
        config.type = WatermarkConfig::Type::Text;
        config.content = "TILE";
        config.position = WatermarkPosition::Tile;
        config.opacity = 0.3f;
        config.font_size = 16;
        config.tile_spacing = 80;

        auto result = Watermark::apply(img, config);

        if (!result.empty() && result.width == 400 && result.height == 400) {
            std::cout << "  PASSED (tiled watermark applied)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 4: 图片水印
    std::cout << "\n[Test 4] Image watermark..." << std::endl;
    {
        auto img = create_test_image_wm(400, 400, 60, 60, 60);
        auto wm = create_test_image_wm(50, 30, 255, 0, 0);  // 红色小图作为水印

        auto result = Watermark::apply_image(img, wm, WatermarkPosition::TopLeft, 0.6f, 10);

        if (!result.empty() && result.width == 400 && result.height == 400) {
            std::cout << "  PASSED (image watermark applied)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 5: 透明度测试
    std::cout << "\n[Test 5] Opacity test..." << std::endl;
    {
        auto img = create_test_image_wm(200, 200, 100, 100, 100);
        WatermarkConfig config;
        config.type = WatermarkConfig::Type::Text;
        config.content = "OPAQUE";
        config.position = WatermarkPosition::Center;
        config.opacity = 1.0f;  // 完全不透明
        config.font_size = 24;

        auto result = Watermark::apply(img, config);

        if (!result.empty()) {
            std::cout << "  PASSED (opaque watermark)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 6: 大图像水印缩放
    std::cout << "\n[Test 6] Large watermark scaling..." << std::endl;
    {
        auto img = create_test_image_wm(200, 200, 50, 50, 50);
        auto wm = create_test_image_wm(300, 300, 255, 0, 0);  // 比原图还大的水印

        auto result = Watermark::apply_image(img, wm, WatermarkPosition::Center, 0.5f, 10);

        if (!result.empty() && result.width == 200 && result.height == 200) {
            std::cout << "  PASSED (large watermark scaled down)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 7: 边界条件 - 边距为0
    std::cout << "\n[Test 7] Zero margin..." << std::endl;
    {
        auto img = create_test_image_wm(100, 100, 80, 80, 80);
        WatermarkConfig config;
        config.type = WatermarkConfig::Type::Text;
        config.content = "M";
        config.position = WatermarkPosition::TopLeft;
        config.opacity = 0.5f;
        config.margin = 0;

        auto result = Watermark::apply(img, config);

        if (!result.empty()) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
