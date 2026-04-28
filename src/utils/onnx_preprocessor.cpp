#include "controlnet_preprocessors.h"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#endif

namespace myimg {

#ifdef HAVE_ONNXRUNTIME

// Global ONNX Runtime sessions (lazy loaded)
static std::unique_ptr<Ort::Session> g_openpose_onnx_session;
static bool g_openpose_onnx_loaded = false;

static bool load_openpose_onnx(const std::string& model_path) {
    if (g_openpose_onnx_loaded) return true;
    try {
        std::cout << "[INFO] Loading OpenPose ONNX model: " << model_path << "\n";
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "openpose");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        g_openpose_onnx_session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        g_openpose_onnx_loaded = true;
        std::cout << "[INFO] OpenPose ONNX model loaded successfully\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading OpenPose ONNX model: " << e.what() << "\n";
        return false;
    }
}

// ONNX-based depth estimation using MiDaS/DPT model
ImageData depth_map_onnx(const ImageData& img, const std::string& model_path) {
    std::cerr << "ONNX MiDaS model not available. Using TorchScript version instead.\n";
    return ImageData();
}

// ONNX-based OpenPose pose detection
ImageData openpose_onnx(const ImageData& img, const std::string& model_path) {
    if (img.empty()) return ImageData();
    
    std::string path = model_path;
    if (path.empty()) {
        path = "/opt/image/model/openpose_body.onnx";
    }
    
    if (!load_openpose_onnx(path)) {
        return ImageData();
    }
    
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Prepare input
        // OpenPose expects BGR, 368x368
        std::vector<float> input_tensor_values(1 * 3 * 368 * 368);
        
        // Convert RGB ImageData to BGR normalized float
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                int src_idx = (y * img.width + x) * 3;
                int dst_y = y * 368 / img.height;
                int dst_x = x * 368 / img.width;
                int dst_idx = dst_y * 368 + dst_x;
                
                if (dst_y < 368 && dst_x < 368) {
                    // BGR order, normalized
                    input_tensor_values[0 * 368 * 368 + dst_idx] = (img.data[src_idx + 2] / 255.0f - 0.485f) / 0.229f;
                    input_tensor_values[1 * 368 * 368 + dst_idx] = (img.data[src_idx + 1] / 255.0f - 0.456f) / 0.224f;
                    input_tensor_values[2 * 368 * 368 + dst_idx] = (img.data[src_idx + 0] / 255.0f - 0.406f) / 0.225f;
                }
            }
        }
        
        // Create input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, 3, 368, 368};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
        );
        
        // Run inference
        const char* input_names[] = {"input"};
        const char* output_names[] = {"paf", "heatmap"};
        
        auto output_tensors = g_openpose_onnx_session->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 2
        );
        
        // Get heatmap output
        float* heatmap_data = output_tensors[1].GetTensorMutableData<float>();
        auto heatmap_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        int heatmap_h = heatmap_shape[2];
        int heatmap_w = heatmap_shape[3];
        int num_keypoints = heatmap_shape[1];
        
        // Create visualization from first heatmap channel
        cv::Mat heatmap_img(heatmap_h, heatmap_w, CV_32FC1);
        for (int y = 0; y < heatmap_h; ++y) {
            for (int x = 0; x < heatmap_w; ++x) {
                heatmap_img.at<float>(y, x) = heatmap_data[y * heatmap_w + x];
            }
        }
        
        // Normalize to 0-255
        cv::Mat heatmap_norm;
        cv::normalize(heatmap_img, heatmap_norm, 0, 255, cv::NORM_MINMAX);
        heatmap_norm.convertTo(heatmap_norm, CV_8UC1);
        
        // Resize to original size
        cv::Mat result_gray;
        cv::resize(heatmap_norm, result_gray, cv::Size(img.width, img.height));
        
        // Convert to RGB heatmap visualization
        cv::Mat result_rgb;
        cv::applyColorMap(result_gray, result_rgb, cv::COLORMAP_JET);
        cv::cvtColor(result_rgb, result_rgb, cv::COLOR_BGR2RGB);
        
        ImageData result;
        result.width = result_rgb.cols;
        result.height = result_rgb.rows;
        result.channels = 3;
        result.data.assign(result_rgb.data, result_rgb.data + result_rgb.total() * result_rgb.channels());
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in openpose_onnx: " << e.what() << "\n";
        return ImageData();
    }
}

#else

ImageData depth_map_onnx(const ImageData& img, const std::string& model_path) {
    std::cerr << "ONNX Runtime not available. Depth estimation disabled.\n";
    return ImageData();
}

ImageData openpose_onnx(const ImageData& img, const std::string& model_path) {
    std::cerr << "ONNX Runtime not available. OpenPose disabled.\n";
    return ImageData();
}

#endif

} // namespace myimg
