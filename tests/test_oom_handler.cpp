#include "utils/oom_handler.h"
#include <iostream>

using namespace myimg;

int main(int argc, char** argv) {
    std::cout << "=== OOM Handler Test Suite ===" << std::endl;

    int tests_passed = 0;
    int tests_failed = 0;

    // Test 1: VRAM 估算 - 基础分辨率
    std::cout << "\n[Test 1] VRAM estimation - 1280x720..." << std::endl;
    {
        int vram = OOMHandler::estimate_vram(1280, 720, 25, false, 0);

        if (vram > 3000 && vram < 8000) {
            std::cout << "  PASSED (estimated " << vram << " MB)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (unexpected: " << vram << " MB)" << std::endl;
            tests_failed++;
        }
    }

    // Test 2: VRAM 估算 - 带 HiRes
    std::cout << "\n[Test 2] VRAM estimation - 1280x720 with HiRes 45..." << std::endl;
    {
        int vram_no_hires = OOMHandler::estimate_vram(1280, 720, 25, false, 0);
        int vram_hires = OOMHandler::estimate_vram(1280, 720, 25, true, 45);

        if (vram_hires > vram_no_hires) {
            std::cout << "  PASSED (no_hires=" << vram_no_hires << "MB, hires=" << vram_hires << "MB)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (HiRes should require more VRAM)" << std::endl;
            tests_failed++;
        }
    }

    // Test 3: VRAM 估算 - 分辨率越高显存越大
    std::cout << "\n[Test 3] VRAM estimation - larger resolution needs more VRAM..." << std::endl;
    {
        int vram_small = OOMHandler::estimate_vram(512, 512, 25, false, 0);
        int vram_large = OOMHandler::estimate_vram(2048, 2048, 25, false, 0);

        if (vram_large > vram_small) {
            std::cout << "  PASSED (512x512=" << vram_small << "MB, 2048x2048=" << vram_large << "MB)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 4: OOM 预测
    std::cout << "\n[Test 4] OOM prediction..." << std::endl;
    {
        bool will_oom_4k = OOMHandler::will_oom(3840, 2160, 25, true, 50, 10240);
        bool will_oom_small = OOMHandler::will_oom(512, 512, 25, false, 0, 10240);

        if (will_oom_4k && !will_oom_small) {
            std::cout << "  PASSED (4K=OOM, 512x512=OK)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (4K=" << will_oom_4k << ", 512x512=" << will_oom_small << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 5: 降级策略 - 第一次重试
    std::cout << "\n[Test 5] Fallback - first retry..." << std::endl;
    {
        int width = 2560, height = 1440, steps = 25, hires_steps = 45;
        bool hires = true, freeu = true, sag = true;
        OOMFallbackConfig config;

        bool success = OOMHandler::apply_fallback(width, height, steps, hires, hires_steps,
                                                   freeu, sag, config, 0);

        if (success && hires_steps < 45) {
            std::cout << "  PASSED (hires_steps reduced to " << hires_steps << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 6: 降级策略 - 多次重试降低分辨率
    std::cout << "\n[Test 6] Fallback - reduce resolution after multiple retries..." << std::endl;
    {
        int width = 2560, height = 1440, steps = 25, hires_steps = 45;
        bool hires = true, freeu = true, sag = true;
        OOMFallbackConfig config;

        bool success1 = OOMHandler::apply_fallback(width, height, steps, hires, hires_steps,
                                                    freeu, sag, config, 0);
        bool success2 = OOMHandler::apply_fallback(width, height, steps, hires, hires_steps,
                                                    freeu, sag, config, 1);

        if (success1 && success2 && (width < 2560 || height < 1440)) {
            std::cout << "  PASSED (resolution reduced to " << width << "x" << height << ")" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (width=" << width << ", height=" << height << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 7: 降级策略 - 禁用增强功能
    std::cout << "\n[Test 7] Fallback - disable enhancements..." << std::endl;
    {
        int width = 1280, height = 720, steps = 25, hires_steps = 45;
        bool hires = true, freeu = true, sag = true;
        OOMFallbackConfig config;
        config.reduce_resolution = false;  // 不改变分辨率
        config.reduce_hires_steps = false; // 不改变 HiRes 步数

        bool success = OOMHandler::apply_fallback(width, height, steps, hires, hires_steps,
                                                   freeu, sag, config, 0);

        if (success && !freeu && !sag) {
            std::cout << "  PASSED (FreeU and SAG disabled)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (freeu=" << freeu << ", sag=" << sag << ")" << std::endl;
            tests_failed++;
        }
    }

    // Test 8: 降级策略 - 最大重试次数
    std::cout << "\n[Test 8] Fallback - max retries..." << std::endl;
    {
        int width = 512, height = 512, steps = 20, hires_steps = 20;
        bool hires = false, freeu = false, sag = false;
        OOMFallbackConfig config;
        config.max_retries = 2;

        bool success = OOMHandler::apply_fallback(width, height, steps, hires, hires_steps,
                                                   freeu, sag, config, 2);

        if (!success) {
            std::cout << "  PASSED (correctly stopped after max retries)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED (should have stopped)" << std::endl;
            tests_failed++;
        }
    }

    // Test 9: 友好建议
    std::cout << "\n[Test 9] Friendly suggestion..." << std::endl;
    {
        auto suggestion = OOMHandler::get_friendly_suggestion(12000, 10240);

        if (suggestion.find("显存不足") != std::string::npos &&
            suggestion.find("建议") != std::string::npos) {
            std::cout << "  PASSED (contains friendly message)" << std::endl;
            tests_passed++;
        } else {
            std::cout << "  FAILED" << std::endl;
            tests_failed++;
        }
    }

    // Test 10: 批量处理 OOM
    std::cout << "\n[Test 10] Batch OOM handler..." << std::endl;
    {
        int width = 1920, height = 1080, steps = 25, hires_steps = 45;
        bool hires = true, freeu = true, sag = true;
        OOMFallbackConfig config;

        bool success = OOMHandler::handle_batch_oom(width, height, steps, hires, hires_steps,
                                                     freeu, sag, config);

        if (success) {
            std::cout << "  PASSED (batch handler applied fallback)" << std::endl;
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
