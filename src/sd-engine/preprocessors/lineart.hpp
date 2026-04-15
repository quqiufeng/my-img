// ============================================================================
// lineart.hpp
// ============================================================================
// LineArt 线稿提取 ONNX 推理封装
// ============================================================================

#pragma once

#ifdef HAS_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace sdengine {

struct LineArtResult {
    std::vector<uint8_t> data;  // RGB data
    int width = 0;
    int height = 0;
    bool success = false;
};

class LineArtPreprocessor {
public:
    bool load(const std::string& model_path);
    LineArtResult process(const uint8_t* rgb_data, int width, int height);

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "lineart"};
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    int input_height_ = 512;
    int input_width_ = 512;

    // Resize image using nearest neighbor
    std::vector<uint8_t> resize_rgb(const uint8_t* src, int src_w, int src_h, int dst_w, int dst_h);
};

} // namespace sdengine

#endif // HAS_ONNXRUNTIME
