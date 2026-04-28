#include "controlnet_preprocessors.h"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace myimg {

ImageData canny_edges(const ImageData& img, int low_threshold, int high_threshold) {
    if (img.empty()) return ImageData();
    
    try {
        // Convert to OpenCV Mat (RGB)
        cv::Mat input(img.height, img.width, CV_8UC3, const_cast<uint8_t*>(img.data.data()));
        
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
        
        // Apply Canny edge detection
        cv::Mat edges;
        cv::Canny(gray, edges, low_threshold, high_threshold);
        
        // Convert back to RGB (3 channels)
        cv::Mat edges_rgb;
        cv::cvtColor(edges, edges_rgb, cv::COLOR_GRAY2RGB);
        
        // Convert back to ImageData
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
        
        // Apply Canny with lower thresholds for more lines
        cv::Mat edges;
        cv::Canny(gray, edges, threshold * 0.5, threshold);
        
        // Dilate to thicken lines
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
        cv::Mat dilated;
        cv::dilate(edges, dilated, kernel);
        
        // Convert back to RGB
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
        
        // Compute gradients using Sobel
        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
        
        // Normalize gradients
        cv::normalize(grad_x, grad_x, -1.0, 1.0, cv::NORM_MINMAX);
        cv::normalize(grad_y, grad_y, -1.0, 1.0, cv::NORM_MINMAX);
        
        // Create normal map (R = X, G = Y, B = Z)
        cv::Mat normal(img.height, img.width, CV_32FC3);
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                float nx = grad_x.at<float>(y, x);
                float ny = grad_y.at<float>(y, x);
                float nz = 1.0f; // Approximate Z component
                
                // Normalize
                float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 0) {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                }
                
                // Map to 0-255
                normal.at<cv::Vec3f>(y, x)[0] = (nx + 1.0f) * 0.5f;
                normal.at<cv::Vec3f>(y, x)[1] = (ny + 1.0f) * 0.5f;
                normal.at<cv::Vec3f>(y, x)[2] = nz * 0.5f + 0.5f;
            }
        }
        
        // Convert to 8-bit
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
        
        // Apply Canny
        cv::Mat edges;
        cv::Canny(gray, edges, threshold * 0.5, threshold);
        
        // Invert (white lines on black background -> black lines on white background)
        cv::Mat inverted = 255 - edges;
        
        // Convert back to RGB
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

ImageData depth_map(const ImageData& img) {
    std::cerr << "Depth estimation requires MiDaS/DPT ONNX model. Use --depth-model PATH.\n";
    return ImageData();
}

ImageData openpose(const ImageData& img) {
    std::cerr << "OpenPose requires OpenPose ONNX model. Use --openpose-model PATH.\n";
    return ImageData();
}

ImageData apply_preprocessor(const ImageData& img, const std::string& name, int param1, int param2) {
    if (name == "canny" || name == "Canny") {
        return canny_edges(img, param1 > 0 ? param1 : 100, param2 > 0 ? param2 : 200);
    } else if (name == "lineart" || name == "Lineart") {
        return lineart(img, param1 > 0 ? param1 : 100);
    } else if (name == "normal" || name == "normal_map" || name == "Normal") {
        return normal_map(img);
    } else if (name == "scribble" || name == "Scribble") {
        return scribble(img, param1 > 0 ? param1 : 100);
    } else if (name == "depth" || name == "Depth") {
        return depth_map(img);
    } else if (name == "openpose" || name == "OpenPose") {
        return openpose(img);
    }
    
    std::cerr << "Unknown preprocessor: " << name << "\n";
    return ImageData();
}

} // namespace myimg
