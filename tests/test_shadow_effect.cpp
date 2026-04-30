#include "utils/shadow_effect.h"
#include <iostream>

using namespace myimg;

// 创建测试图像
ImageData create_test_image_shadow(int width, int height, uint8_t r, uint8_t g, uint8_t b) {
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
    std::cout << "=== Shadow Effect Test Suite ===" << std::endl;

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: Drop Shadow
    std::cout << "\n[Test 1] Drop shadow..." << std::endl;
    {
        auto img = create_test_image_shadow(100, 100, 200, 100, 50);
        DropShadowConfig config;
        config.offset_x = 15;
        config.offset_y = 15;
        config.blur_radius = 8;
        config.opacity = 0.6f;

        auto result = ShadowEffect::add_drop_shadow(img, config);

        if (!result.empty() && result.width > 100 && result.height > 100) {
            std::cout << "  PASSED (size=" << result.width << "x" << result.height << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 2: Reflection
    std::cout << "\n[Test 2] Reflection..." << std::endl;
    {
        auto img = create_test_image_shadow(100, 100, 100, 150, 200);
        ReflectionConfig config;
        config.height_ratio = 0.3f;
        config.opacity = 0.5f;
        config.gap = 5;

        auto result = ShadowEffect::add_reflection(img, config);

        int expected_height = 100 + static_cast<int>(100 * 0.3f) + 5;
        if (!result.empty() && result.width == 100 && result.height == expected_height) {
            std::cout << "  PASSED (size=" << result.width << "x" << result.height << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected 100x" << expected_height << ", got " << result.width << "x" << result.height << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 3: Contact Shadow
    std::cout << "\n[Test 3] Contact shadow..." << std::endl;
    {
        auto img = create_test_image_shadow(150, 150, 50, 100, 50);

        auto result = ShadowEffect::add_contact_shadow(img, 0.5f, 8);

        if (!result.empty() && result.width == 150 && result.height > 150) {
            std::cout << "  PASSED (size=" << result.width << "x" << result.height << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 4: Drop shadow with different offsets
    std::cout << "\n[Test 4] Drop shadow - negative offset..." << std::endl;
    {
        auto img = create_test_image_shadow(80, 80, 255, 0, 0);
        DropShadowConfig config;
        config.offset_x = -10;
        config.offset_y = -10;
        config.blur_radius = 5;
        config.opacity = 0.4f;

        auto result = ShadowEffect::add_drop_shadow(img, config);

        if (!result.empty() && result.width >= 80 && result.height >= 80) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 5: Reflection with small height ratio
    std::cout << "\n[Test 5] Reflection - small ratio..." << std::endl;
    {
        auto img = create_test_image_shadow(200, 100, 128, 128, 128);
        ReflectionConfig config;
        config.height_ratio = 0.1f;
        config.opacity = 0.3f;
        config.gap = 2;

        auto result = ShadowEffect::add_reflection(img, config);

        int expected_height = 100 + static_cast<int>(100 * 0.1f) + 2;
        if (!result.empty() && result.height == expected_height) {
            std::cout << "  PASSED" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected height " << expected_height << ", got " << result.height << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 6: Product shadow (combined)
    std::cout << "\n[Test 6] Product shadow (combined)..." << std::endl;
    {
        auto img = create_test_image_shadow(120, 120, 200, 200, 200);
        DropShadowConfig drop_config;
        drop_config.offset_x = 8;
        drop_config.offset_y = 8;
        drop_config.opacity = 0.5f;

        ReflectionConfig refl_config;
        refl_config.height_ratio = 0.25f;
        refl_config.opacity = 0.3f;

        auto result = ShadowEffect::add_product_shadow(img, drop_config, refl_config);

        if (!result.empty() && result.width == 120) {
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
