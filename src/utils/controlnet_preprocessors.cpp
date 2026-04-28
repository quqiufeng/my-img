#include "controlnet_preprocessors.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>

namespace myimg {

// Global TorchScript models (lazy loaded)
static torch::jit::script::Module g_midas_model;
static torch::jit::script::Module g_openpose_model;
static bool g_midas_loaded = false;
static bool g_openpose_loaded = false;

static bool load_midas_model(const std::string& model_path) {
    if (g_midas_loaded) return true;
    try {
        std::cout << "[INFO] Loading MiDaS model: " << model_path << "\n";
        g_midas_model = torch::jit::load(model_path);
        g_midas_model.eval();
        g_midas_loaded = true;
        std::cout << "[INFO] MiDaS model loaded successfully\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading MiDaS model: " << e.what() << "\n";
        return false;
    }
}

static bool load_openpose_model(const std::string& model_path) {
    if (g_openpose_loaded) return true;
    try {
        std::cout << "[INFO] Loading OpenPose model: " << model_path << "\n";
        g_openpose_model = torch::jit::load(model_path);
        g_openpose_model.eval();
        g_openpose_loaded = true;
        std::cout << "[INFO] OpenPose model loaded successfully\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading OpenPose model: " << e.what() << "\n";
        return false;
    }
}

ImageData canny_edges(const ImageData& img, int low_threshold, int high_threshold) {
    if (img.empty()) return ImageData();
    
    try {
        cv::Mat input(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
        cv::Mat edges;
        cv::Canny(gray, edges, low_threshold, high_threshold);
        cv::Mat edges_rgb;
        cv::cvtColor(edges, edges_rgb, cv::COLOR_GRAY2RGB);
        
        ImageData result;
        result.width = edges_rgb.cols;
        result.height = edges_rgb.rows;
        result.channels = 3;
        result.data.assign(edges_rgb.data, edges_rgb.data + edges_rgb.total() * edges_rgb.channels());
        
        return result;
    } catch (const cv::Exception& e) {
        std::cerr << "Error in canny_edges: " << e.what() << "\n";
        return ImageData();
    }
}

ImageData lineart(const ImageData& img, int threshold) {
    if (img.empty()) return ImageData();
    
    try {
        cv::Mat input(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
        cv::Mat edges;
        cv::Canny(gray, edges, threshold * 0.5, threshold);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
        cv::Mat dilated;
        cv::dilate(edges, dilated, kernel);
        cv::Mat edges_rgb;
        cv::cvtColor(dilated, edges_rgb, cv::COLOR_GRAY2RGB);
        
        ImageData result;
        result.width = edges_rgb.cols;
        result.height = edges_rgb.rows;
        result.channels = 3;
        result.data.assign(edges_rgb.data, edges_rgb.data + edges_rgb.total() * edges_rgb.channels());
        
        return result;
    } catch (const cv::Exception& e) {
        std::cerr << "Error in lineart: " << e.what() << "\n";
        return ImageData();
    }
}

ImageData normal_map(const ImageData& img) {
    if (img.empty()) return ImageData();
    
    try {
        cv::Mat input(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
        cv::normalize(grad_x, grad_x, -1.0, 1.0, cv::NORM_MINMAX);
        cv::normalize(grad_y, grad_y, -1.0, 1.0, cv::NORM_MINMAX);
        
        cv::Mat normal(img.height, img.width, CV_32FC3);
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                float nx = grad_x.at<float>(y, x);
                float ny = grad_y.at<float>(y, x);
                float nz = 1.0f;
                float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 0) {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                }
                normal.at<cv::Vec3f>(y, x)[0] = (nx + 1.0f) * 0.5f;
                normal.at<cv::Vec3f>(y, x)[1] = (ny + 1.0f) * 0.5f;
                normal.at<cv::Vec3f>(y, x)[2] = nz * 0.5f + 0.5f;
            }
        }
        
        cv::Mat normal_8bit;
        normal.convertTo(normal_8bit, CV_8UC3, 255.0);
        
        ImageData result;
        result.width = normal_8bit.cols;
        result.height = normal_8bit.rows;
        result.channels = 3;
        result.data.assign(normal_8bit.data, normal_8bit.data + normal_8bit.total() * normal_8bit.channels());
        
        return result;
    } catch (const cv::Exception& e) {
        std::cerr << "Error in normal_map: " << e.what() << "\n";
        return ImageData();
    }
}

ImageData scribble(const ImageData& img, int threshold) {
    if (img.empty()) return ImageData();
    
    try {
        cv::Mat input(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
        cv::Mat edges;
        cv::Canny(gray, edges, threshold * 0.5, threshold);
        cv::Mat inverted = 255 - edges;
        cv::Mat scribble_rgb;
        cv::cvtColor(inverted, scribble_rgb, cv::COLOR_GRAY2RGB);
        
        ImageData result;
        result.width = scribble_rgb.cols;
        result.height = scribble_rgb.rows;
        result.channels = 3;
        result.data.assign(scribble_rgb.data, scribble_rgb.data + scribble_rgb.total() * scribble_rgb.channels());
        
        return result;
    } catch (const cv::Exception& e) {
        std::cerr << "Error in scribble: " << e.what() << "\n";
        return ImageData();
    }
}

ImageData depth_map(const ImageData& img, const std::string& model_path) {
    if (img.empty()) return ImageData();
    
    std::string path = model_path;
    if (path.empty()) {
        path = "/opt/image/model/midas_dpt_hybrid.pt";
    }
    
    if (!load_midas_model(path)) {
        std::cerr << "Failed to load MiDaS model. Use --depth-model PATH.\n";
        return ImageData();
    }
    
    try {
        // Convert ImageData to OpenCV Mat (RGB)
        cv::Mat input(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        
        // Resize to 384x384 (MiDaS input size)
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(384, 384));
        
        // Convert to float tensor [0, 1]
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
        
        // Convert BGR (OpenCV default) to RGB
        cv::Mat rgb_img;
        cv::cvtColor(float_img, rgb_img, cv::COLOR_BGR2RGB);
        
        // Create tensor [1, 3, H, W]
        auto tensor = torch::from_blob(rgb_img.data, {1, 384, 384, 3}, torch::kFloat32);
        tensor = tensor.permute({0, 3, 1, 2}); // [1, 3, 384, 384]
        
        // Normalize with ImageNet stats
        tensor = tensor.clone(); // Make a copy since from_blob doesn't own memory
        tensor[0][0] = (tensor[0][0] - 0.485f) / 0.229f;
        tensor[0][1] = (tensor[0][1] - 0.456f) / 0.224f;
        tensor[0][2] = (tensor[0][2] - 0.406f) / 0.225f;
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        at::Tensor output;
        {
            torch::NoGradGuard no_grad;
            output = g_midas_model.forward(inputs).toTensor();
        }
        
        // output is [1, H, W] - inverse depth
        // Invert and normalize to [0, 255]
        output = 1.0f / (output + 1e-6f); // Convert to depth
        output = output - output.min();
        output = output / (output.max() + 1e-6f) * 255.0f;
        
        // Convert to OpenCV Mat
        cv::Mat depth_mat(384, 384, CV_32FC1, output[0].data_ptr<float>());
        
        // Resize back to original size
        cv::Mat depth_resized;
        cv::resize(depth_mat, depth_resized, cv::Size(img.width, img.height));
        
        // Convert to 8-bit grayscale
        cv::Mat depth_8bit;
        depth_resized.convertTo(depth_8bit, CV_8UC1);
        
        // Convert to RGB
        cv::Mat depth_rgb;
        cv::cvtColor(depth_8bit, depth_rgb, cv::COLOR_GRAY2RGB);
        
        ImageData result;
        result.width = depth_rgb.cols;
        result.height = depth_rgb.rows;
        result.channels = 3;
        result.data.assign(depth_rgb.data, depth_rgb.data + depth_rgb.total() * depth_rgb.channels());
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in depth_map: " << e.what() << "\n";
        return ImageData();
    }
}

ImageData openpose(const ImageData& img, const std::string& model_path) {
    if (img.empty()) return ImageData();
    
    std::string path = model_path;
    if (path.empty()) {
        path = "/opt/image/model/openpose_body.pt";
    }
    
    if (!load_openpose_model(path)) {
        std::cerr << "Failed to load OpenPose model. Use --openpose-model PATH.\n";
        return ImageData();
    }
    
    try {
        // Convert ImageData to OpenCV Mat (RGB)
        cv::Mat input(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        
        // Resize to 368x368 (OpenPose input size)
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(368, 368));
        
        // Convert to float tensor [0, 1]
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
        
        // Create tensor [1, 3, H, W]
        auto tensor = torch::from_blob(float_img.data, {1, 368, 368, 3}, torch::kFloat32);
        tensor = tensor.permute({0, 3, 1, 2}); // [1, 3, 368, 368]
        tensor = tensor.clone();
        
        // Normalize
        tensor[0][0] = (tensor[0][0] - 0.485f) / 0.229f;
        tensor[0][1] = (tensor[0][1] - 0.456f) / 0.224f;
        tensor[0][2] = (tensor[0][2] - 0.406f) / 0.225f;
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        at::Tensor paf, heatmap;
        {
            torch::NoGradGuard no_grad;
            auto output = g_openpose_model.forward(inputs).toTuple();
            paf = output->elements()[0].toTensor();
            heatmap = output->elements()[1].toTensor();
        }
        
        // For now, create a simplified visualization
        // Extract body keypoints from heatmap (simplified)
        cv::Mat heatmap_mat(heatmap.size(2), heatmap.size(3), CV_32FC1, heatmap[0][0].data_ptr<float>());
        
        // Normalize heatmap to 0-255
        cv::Mat heatmap_norm;
        cv::normalize(heatmap_mat, heatmap_norm, 0, 255, cv::NORM_MINMAX);
        heatmap_norm.convertTo(heatmap_norm, CV_8UC1);
        
        // Resize to original size
        cv::Mat result_gray;
        cv::resize(heatmap_norm, result_gray, cv::Size(img.width, img.height));
        
        // Convert to RGB (heatmap as red channel)
        cv::Mat result_rgb;
        cv::cvtColor(result_gray, result_rgb, cv::COLOR_GRAY2RGB);
        
        // Enhance red channel for visualization
        std::vector<cv::Mat> channels(3);
        cv::split(result_rgb, channels);
        channels[2] = channels[0]; // Red channel
        channels[0] = cv::Mat::zeros(img.height, img.width, CV_8UC1); // Blue = 0
        channels[1] = cv::Mat::zeros(img.height, img.width, CV_8UC1); // Green = 0
        cv::merge(channels, result_rgb);
        
        ImageData result;
        result.width = result_rgb.cols;
        result.height = result_rgb.rows;
        result.channels = 3;
        result.data.assign(result_rgb.data, result_rgb.data + result_rgb.total() * result_rgb.channels());
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in openpose: " << e.what() << "\n";
        return ImageData();
    }
}

ImageData apply_preprocessor(const ImageData& img, const std::string& name, int param1, int param2, const std::string& model_path) {
    if (name == "canny" || name == "Canny") {
        return canny_edges(img, param1 > 0 ? param1 : 100, param2 > 0 ? param2 : 200);
    } else if (name == "lineart" || name == "Lineart") {
        return lineart(img, param1 > 0 ? param1 : 100);
    } else if (name == "normal" || name == "normal_map" || name == "Normal") {
        return normal_map(img);
    } else if (name == "scribble" || name == "Scribble") {
        return scribble(img, param1 > 0 ? param1 : 100);
    } else if (name == "depth" || name == "Depth") {
        return depth_map(img, model_path);
    } else if (name == "openpose" || name == "OpenPose") {
        return openpose(img, model_path);
    }
    
    std::cerr << "Unknown preprocessor: " << name << "\n";
    return ImageData();
}

} // namespace myimg
