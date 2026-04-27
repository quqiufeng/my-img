#include "utils/image_utils.h"
#include <iostream>
#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

// 简单的测试宏
#define TEST_ASSERT(condition, msg) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << msg << " at line " << __LINE__ << std::endl; \
            return 1; \
        } \
    } while(0)

int main() {
    std::cout << "=== Image Utils Tests ===" << std::endl;
    
    // Test 1: Load non-existent image
    {
        std::cout << "Test 1: Load non-existent image... ";
        auto img = myimg::load_image_from_file("/nonexistent/path/image.png");
        TEST_ASSERT(img.empty(), "Should return empty for non-existent file");
        TEST_ASSERT(img.width == 0, "Width should be 0");
        TEST_ASSERT(img.height == 0, "Height should be 0");
        std::cout << "PASSED" << std::endl;
    }
    
    // Test 2: ImageData structure
    {
        std::cout << "Test 2: ImageData structure... ";
        myimg::ImageData img;
        TEST_ASSERT(img.empty(), "Should be empty initially");
        TEST_ASSERT(img.width == 0, "Width should be 0");
        TEST_ASSERT(img.height == 0, "Height should be 0");
        TEST_ASSERT(img.channels == 3, "Channels should be 3");
        
        img.width = 100;
        img.height = 100;
        img.data.resize(100 * 100 * 3);
        TEST_ASSERT(!img.empty(), "Should not be empty after resize");
        TEST_ASSERT(img.size() == 100 * 100 * 3, "Size should be 100*100*3");
        std::cout << "PASSED" << std::endl;
    }
    
    // Test 3: Load real image (if available)
    {
        std::cout << "Test 3: Load real image... ";
        fs::path test_img = "/mnt/e/app/portrait_2560x1440.png";
        if (fs::exists(test_img)) {
            auto img = myimg::load_image_from_file(test_img.string());
            TEST_ASSERT(!img.empty(), "Should load existing image");
            TEST_ASSERT(img.width == 2560, "Width should be 2560");
            TEST_ASSERT(img.height == 1440, "Height should be 1440");
            TEST_ASSERT(img.channels == 3, "Channels should be 3");
            TEST_ASSERT(img.data.size() == 2560 * 1440 * 3, "Data size should match");
            std::cout << "PASSED (loaded " << img.width << "x" << img.height << ")" << std::endl;
        } else {
            std::cout << "SKIPPED (no test image found)" << std::endl;
        }
    }
    
    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
