#include "utils/image_postproc.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

#define TEST_ASSERT(condition, msg) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << msg << " at line " << __LINE__ << std::endl; \
            return 1; \
        } \
    } while(0)

// 创建一个测试图像：渐变背景 + 清晰边缘
static myimg::Image create_test_image(int w, int h) {
    myimg::Image img;
    img.width = w;
    img.height = h;
    img.channels = 3;  // RGB
    img.data.resize(w * h * 3);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 3;
            // 渐变背景
            img.data[idx + 0] = static_cast<uint8_t>(128 + 64 * std::sin(x * 0.1) * std::cos(y * 0.1)); // R
            img.data[idx + 1] = static_cast<uint8_t>(100 + 50 * std::cos(x * 0.15 + y * 0.1));          // G
            img.data[idx + 2] = static_cast<uint8_t>(150 + 80 * std::sin(y * 0.12));                    // B
            // 在中心添加锐利方块边缘
            if (x > w/4 && x < 3*w/4 && y > h/4 && y < 3*h/4) {
                img.data[idx + 0] = 255;
                img.data[idx + 1] = 200;
                img.data[idx + 2] = 100;
            }
            // 添加细线
            if (x == w/2 || y == h/2) {
                img.data[idx + 0] = 0;
                img.data[idx + 1] = 0;
                img.data[idx + 2] = 0;
            }
        }
    }
    return img;
}

// 保存为 PPM（简单调试用）
static void save_ppm(const std::string& path, const myimg::Image& img) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    std::cout << "  saved: " << path << " (" << img.width << "x" << img.height << ")" << std::endl;
}

int main() {
    std::cout << "=== Image Post-Processing Tests ===" << std::endl;

    // Test 1: Empty image handling
    {
        std::cout << "Test 1: Empty image... ";
        myimg::Image empty;
        myimg::PostProcessParams pp;
        pp.clarity = 0.5f;
        bool ok = myimg::apply_image_postprocessing(empty, pp);
        TEST_ASSERT(!ok, "Should return false for empty image");
        std::cout << "PASSED" << std::endl;
    }

    // Test 2: No-op (all params = 0)
    {
        std::cout << "Test 2: No-op processing... ";
        auto img = create_test_image(64, 64);
        auto original_data = img.data;
        myimg::PostProcessParams pp;  // all defaults = 0
        bool ok = myimg::apply_image_postprocessing(img, pp);
        TEST_ASSERT(ok, "Should return true for no-op");
        TEST_ASSERT(img.data == original_data, "Image should be unchanged");
        std::cout << "PASSED" << std::endl;
    }

    // Test 3: Clarity
    {
        std::cout << "Test 3: Clarity (0.5)... ";
        auto img = create_test_image(64, 64);
        auto original_data = img.data;
        myimg::PostProcessParams pp;
        pp.clarity = 0.5f;
        bool ok = myimg::apply_image_postprocessing(img, pp);
        TEST_ASSERT(ok, "Should succeed");
        TEST_ASSERT(img.data != original_data, "Image should change after clarity");
        TEST_ASSERT(img.width == 64 && img.height == 64, "Dimensions unchanged");
        TEST_ASSERT(img.channels == 3, "Channels unchanged");
        std::cout << "PASSED" << std::endl;
    }

    // Test 4: USM Sharpen
    {
        std::cout << "Test 4: USM Sharpen (1.0)... ";
        auto img = create_test_image(64, 64);
        auto original_data = img.data;
        myimg::PostProcessParams pp;
        pp.sharpen_amount = 1.0f;
        pp.sharpen_radius = 2;
        bool ok = myimg::apply_image_postprocessing(img, pp);
        TEST_ASSERT(ok, "Should succeed");
        TEST_ASSERT(img.data != original_data, "Image should change after sharpen");
        std::cout << "PASSED" << std::endl;
    }

    // Test 5: Smart Sharpen
    {
        std::cout << "Test 5: Smart Sharpen (0.5)... ";
        auto img = create_test_image(64, 64);
        myimg::PostProcessParams pp;
        pp.smart_sharpen_strength = 0.5f;
        bool ok = myimg::apply_image_postprocessing(img, pp);
        TEST_ASSERT(ok, "Should succeed");
        std::cout << "PASSED" << std::endl;
    }

    // Test 6: Edge Sharpen
    {
        std::cout << "Test 6: Edge Sharpen (1.0)... ";
        auto img = create_test_image(64, 64);
        myimg::PostProcessParams pp;
        pp.edge_sharpen_amount = 1.0f;
        pp.edge_sharpen_radius = 2;
        pp.edge_sharpen_threshold = 0.3f;
        bool ok = myimg::apply_image_postprocessing(img, pp);
        TEST_ASSERT(ok, "Should succeed");
        std::cout << "PASSED" << std::endl;
    }

    // Test 7: Full pipeline (all effects)
    {
        std::cout << "Test 7: Full pipeline (all effects)... ";
        auto img = create_test_image(64, 64);
        myimg::PostProcessParams pp;
        pp.clarity = 0.3f;
        pp.sharpen_amount = 0.5f;
        pp.sharpen_radius = 2;
        pp.smart_sharpen_strength = 0.3f;
        pp.edge_sharpen_amount = 0.5f;
        pp.edge_sharpen_radius = 2;
        pp.edge_sharpen_threshold = 0.3f;
        bool ok = myimg::apply_image_postprocessing(img, pp);
        TEST_ASSERT(ok, "Should succeed");
        std::cout << "PASSED" << std::endl;
    }

    // Test 8: RGBA 4-channel support
    {
        std::cout << "Test 8: RGBA 4-channel support... ";
        auto img = create_test_image(32, 32);
        img.channels = 4;
        // Add alpha channel
        std::vector<uint8_t> rgba(img.width * img.height * 4);
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                int src_idx = (y * img.width + x) * 3;
                int dst_idx = (y * img.width + x) * 4;
                rgba[dst_idx + 0] = img.data[src_idx + 0];  // R
                rgba[dst_idx + 1] = img.data[src_idx + 1];  // G
                rgba[dst_idx + 2] = img.data[src_idx + 2];  // B
                rgba[dst_idx + 3] = 255;                     // A
            }
        }
        img.data = std::move(rgba);

        myimg::PostProcessParams pp;
        pp.sharpen_amount = 1.0f;
        bool ok = myimg::apply_image_postprocessing(img, pp);
        TEST_ASSERT(ok, "Should succeed with RGBA");
        TEST_ASSERT(img.channels == 4, "Should keep 4 channels");
        std::cout << "PASSED" << std::endl;
    }

    // Generate visual test output for inspection
    {
        std::cout << std::endl << "Generating visual test outputs..." << std::endl;
        auto img = create_test_image(200, 200);
        save_ppm("/tmp/postproc_original.ppm", img);

        myimg::PostProcessParams pp;
        pp.clarity = 0.4f;
        auto clarity_img = img;
        myimg::apply_image_postprocessing(clarity_img, pp);
        save_ppm("/tmp/postproc_clarity.ppm", clarity_img);

        pp = myimg::PostProcessParams();
        pp.sharpen_amount = 1.0f;
        pp.sharpen_radius = 2;
        auto sharpen_img = img;
        myimg::apply_image_postprocessing(sharpen_img, pp);
        save_ppm("/tmp/postproc_sharpen.ppm", sharpen_img);

        pp = myimg::PostProcessParams();
        pp.edge_sharpen_amount = 1.5f;
        pp.edge_sharpen_radius = 2;
        pp.edge_sharpen_threshold = 0.3f;
        auto edge_img = img;
        myimg::apply_image_postprocessing(edge_img, pp);
        save_ppm("/tmp/postproc_edge_sharpen.ppm", edge_img);

        pp = myimg::PostProcessParams();
        pp.clarity = 0.3f;
        pp.sharpen_amount = 0.5f;
        pp.sharpen_radius = 2;
        pp.smart_sharpen_strength = 0.3f;
        pp.edge_sharpen_amount = 0.5f;
        pp.edge_sharpen_radius = 2;
        auto full_img = img;
        myimg::apply_image_postprocessing(full_img, pp);
        save_ppm("/tmp/postproc_full.ppm", full_img);
    }

    std::cout << std::endl << "✅ All tests passed!" << std::endl;
    return 0;
}
