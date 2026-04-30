#include "utils/image_qa.h"
#include "utils/image_utils.h"
#include <filesystem>
#include <iostream>
#include <string>

using namespace myimg;

// 创建测试图像
ImageData create_test_image(int width, int height, uint8_t r, uint8_t g, uint8_t b) {
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

// 创建模糊图像（纯色，没有边缘）
ImageData create_blurry_image(int width, int height) {
    return create_test_image(width, height, 128, 128, 128);
}

// 创建清晰图像（有边缘）
ImageData create_sharp_image(int width, int height) {
    ImageData img;
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.data.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // 创建棋盘格图案（清晰边缘）
            uint8_t val = ((x / 10) % 2 == (y / 10) % 2) ? 255 : 0;
            img.data[(y * width + x) * 3 + 0] = val;
            img.data[(y * width + x) * 3 + 1] = val;
            img.data[(y * width + x) * 3 + 2] = val;
        }
    }
    return img;
}

// 创建过曝图像
ImageData create_overexposed_image(int width, int height) {
    return create_test_image(width, height, 255, 255, 255);
}

// 创建欠曝图像
ImageData create_underexposed_image(int width, int height) {
    return create_test_image(width, height, 2, 2, 2);
}

// 创建色偏图像
ImageData create_color_cast_image(int width, int height) {
    return create_test_image(width, height, 200, 100, 100);  // 红色偏色
}

// 创建正常色彩图像
ImageData create_normal_color_image(int width, int height) {
    return create_test_image(width, height, 128, 128, 128);  // 灰色
}

int main(int argc, char** argv) {
    std::cout << "=== Image QA Test Suite ===" << std::endl;

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: 模糊检测 - 模糊图像
    std::cout << "\n[Test 1] Blur detection - blurry image..." << std::endl;
    {
        auto img = create_blurry_image(100, 100);
        auto issue = ImageQA::check_blur(img, 100.0f);

        if (issue.severity == QASeverity::Error && issue.rule == QARuleType::Blur) {
            std::cout << "  PASSED (detected blur, value=" << issue.value << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Error, got value=" << issue.value << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 2: 模糊检测 - 清晰图像
    std::cout << "\n[Test 2] Blur detection - sharp image..." << std::endl;
    {
        auto img = create_sharp_image(100, 100);
        auto issue = ImageQA::check_blur(img, 100.0f);

        if (issue.severity == QASeverity::Pass) {
            std::cout << "  PASSED (sharp, value=" << issue.value << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Pass, got value=" << issue.value << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 3: 过曝检测
    std::cout << "\n[Test 3] Overexposure detection..." << std::endl;
    {
        auto img = create_overexposed_image(100, 100);
        auto issue = ImageQA::check_exposure(img, 0.95f);

        if (issue.severity == QASeverity::Error) {
            std::cout << "  PASSED (detected overexposure)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Error)" << std::endl;
            tests_failed++;
        }
    }

    // Test 4: 欠曝检测
    std::cout << "\n[Test 4] Underexposure detection..." << std::endl;
    {
        auto img = create_underexposed_image(100, 100);
        auto issue = ImageQA::check_exposure(img, 0.95f);

        if (issue.severity == QASeverity::Error) {
            std::cout << "  PASSED (detected underexposure)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Error)" << std::endl;
            tests_failed++;
        }
    }

    // Test 5: 色偏检测 - 有色偏
    std::cout << "\n[Test 5] Color cast detection - casted image..." << std::endl;
    {
        auto img = create_color_cast_image(100, 100);
        auto issue = ImageQA::check_color_cast(img, 15.0f);

        if (issue.severity == QASeverity::Error || issue.severity == QASeverity::Warning) {
            std::cout << "  PASSED (detected color cast, value=" << issue.value << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Error/Warning, got value=" << issue.value << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 6: 色偏检测 - 正常
    std::cout << "\n[Test 6] Color cast detection - normal image..." << std::endl;
    {
        auto img = create_normal_color_image(100, 100);
        auto issue = ImageQA::check_color_cast(img, 15.0f);

        if (issue.severity == QASeverity::Pass) {
            std::cout << "  PASSED (no color cast)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Pass)" << std::endl;
            tests_failed++;
        }
    }

    // Test 7: 分辨率检测 - 低分辨率
    std::cout << "\n[Test 7] Resolution detection - low res..." << std::endl;
    {
        auto img = create_test_image(400, 400, 128, 128, 128);
        auto issue = ImageQA::check_resolution(img, 800, 800);

        if (issue.severity == QASeverity::Error) {
            std::cout << "  PASSED (detected low resolution)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Error)" << std::endl;
            tests_failed++;
        }
    }

    // Test 8: 分辨率检测 - 高分辨率
    std::cout << "\n[Test 8] Resolution detection - high res..." << std::endl;
    {
        auto img = create_test_image(1200, 1200, 128, 128, 128);
        auto issue = ImageQA::check_resolution(img, 800, 800);

        if (issue.severity == QASeverity::Pass) {
            std::cout << "  PASSED (resolution OK)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected Pass)" << std::endl;
            tests_failed++;
        }
    }

    // Test 9: 完整质检流程
    std::cout << "\n[Test 9] Full QA pipeline..." << std::endl;
    {
        auto img = create_test_image(400, 400, 2, 2, 2);  // 欠曝 + 低分辨率
        auto rules = ImageQA::get_ecommerce_rules();
        auto result = ImageQA::check_image(img, rules);

        if (!result.pass && result.issues.size() >= 2) {
            std::cout << "  PASSED (detected " << result.issues.size() << " issues)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (expected fail with multiple issues, got " << result.issues.size() << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 10: JSON 导出
    std::cout << "\n[Test 10] JSON export..." << std::endl;
    {
        auto img = create_test_image(400, 400, 2, 2, 2);
        auto rules = ImageQA::get_ecommerce_rules();
        auto result = ImageQA::check_image(img, rules);
        auto json = ImageQA::export_json(result);

        bool has_pass = json.find("\"pass\"") != std::string::npos;
        bool has_issues = json.find("\"issues\"") != std::string::npos;
        bool has_recommendation = json.find("\"recommendation\"") != std::string::npos;

        if (has_pass && has_issues && has_recommendation) {
            std::cout << "  PASSED (valid JSON structure)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (missing JSON fields)" << std::endl;
            tests_failed++;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
