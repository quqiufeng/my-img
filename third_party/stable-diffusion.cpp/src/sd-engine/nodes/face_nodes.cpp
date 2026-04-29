// ============================================================================
// sd-engine/nodes/face_nodes.cpp
// ============================================================================
// 人脸处理节点（GFPGAN / FaceSwap）
// 依赖 ONNX Runtime
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"
#include <fstream>
#include <algorithm>

#ifdef ONNXRUNTIME_AVAILABLE
#include "onnxruntime_cxx_api.h"
#endif

namespace sdengine {

// ============================================================================
// 简单的面部检测（基于肤色分割）
// ============================================================================
struct FaceBox {
    int x, y, w, h;
    float confidence;
};

static std::vector<FaceBox> detect_faces_simple(const sd_image_t* image) {
    std::vector<FaceBox> faces;
    int w = image->width;
    int h = image->height;
    int c = image->channel;
    
    // 基于肤色的简单检测
    std::vector<uint8_t> skin_mask(w * h, 0);
    int skin_pixels = 0;
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * c;
            uint8_t r = image->data[idx];
            uint8_t g = c > 1 ? image->data[idx + 1] : r;
            uint8_t b = c > 2 ? image->data[idx + 2] : r;
            
            // HSV 肤色范围（简化版）
            float max_val = std::max({r, g, b});
            float min_val = std::min({r, g, b});
            float diff = max_val - min_val;
            
            if (diff > 15 && r > 95 && g > 40 && b > 20 &&
                r > g && r > b && std::abs(r - g) > 15) {
                skin_mask[y * w + x] = 255;
                skin_pixels++;
            }
        }
    }
    
    // 如果检测到足够的肤色像素，假设中心区域是脸
    if (skin_pixels > w * h * 0.01) {
        int face_w = w / 3;
        int face_h = h / 3;
        int face_x = (w - face_w) / 2;
        int face_y = (h - face_h) / 3;
        
        faces.push_back({face_x, face_y, face_w, face_h, 0.8f});
    }
    
    return faces;
}

// ============================================================================
// FaceRestore - GFPGAN 人脸修复
// ============================================================================
class FaceRestoreNode : public Node {
  public:
    std::string get_class_type() const override { return "FaceRestore"; }
    std::string get_category() const override { return "face"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"image", "IMAGE", true, nullptr},
            {"model_path", "STRING", false, std::string("models/GFPGANv1.4.onnx")},
            {"fidelity", "FLOAT", false, 0.5f},
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr image;
        SD_RETURN_IF_ERROR(get_input(inputs, "image", image));
        SD_RETURN_IF_NULL(image.get(), sd_error_t::ERROR_INVALID_INPUT);

        std::string model_path = get_input_opt<std::string>(inputs, "model_path", "models/GFPGANv1.4.onnx");
        float fidelity = get_input_opt<float>(inputs, "fidelity", 0.5f);
        (void)fidelity;

        LOG_INFO("[FaceRestore] Processing image %dx%d\n", image->width, image->height);

#ifdef ONNXRUNTIME_AVAILABLE
        // 检查模型文件
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            LOG_WARN("[FaceRestore] Model not found: %s, using fallback enhancement\n", model_path.c_str());
            return fallback_restore(image, outputs);
        }

        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceRestore");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            
            Ort::Session session(env, model_path.c_str(), session_options);
            
            // 获取输入输出信息
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = session.GetInputNameAllocated(0, allocator);
            auto output_name = session.GetOutputNameAllocated(0, allocator);
            
            LOG_INFO("[FaceRestore] ONNX model loaded: %s -> %s\n", input_name.get(), output_name.get());
            
            // 简化的预处理：缩放到模型输入尺寸
            // GFPGAN 通常使用 512x512
            const int model_size = 512;
            auto resized = resize_image(image.get(), model_size, model_size);
            if (!resized) {
                return fallback_restore(image, outputs);
            }
            
            // 转换为 float 并归一化 [-1, 1]
            std::vector<float> input_tensor(model_size * model_size * 3);
            for (int i = 0; i < model_size * model_size; i++) {
                input_tensor[i * 3 + 0] = (resized->data[i * 3 + 0] / 255.0f - 0.5f) * 2.0f;
                input_tensor[i * 3 + 1] = (resized->data[i * 3 + 1] / 255.0f - 0.5f) * 2.0f;
                input_tensor[i * 3 + 2] = (resized->data[i * 3 + 2] / 255.0f - 0.5f) * 2.0f;
            }
            
            // 创建输入 tensor
            std::vector<int64_t> input_shape = {1, 3, model_size, model_size};
            Ort::Value input_tensor_value = Ort::Value::CreateTensor<float>(
                allocator.GetInfo(), input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());
            
            // 运行推理
            const char* input_names[] = {input_name.get()};
            const char* output_names[] = {output_name.get()};
            
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor_value, 1,
                output_names, 1
            );
            
            // 获取输出
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            
            // 创建输出图像
            auto result = make_malloc_buffer(image->width * image->height * 3);
            if (!result) return sd_error_t::ERROR_MEMORY_ALLOCATION;
            
            // 将模型输出转换回图像格式并缩放回原始尺寸
            std::vector<uint8_t> temp_output(model_size * model_size * 3);
            for (int i = 0; i < model_size * model_size; i++) {
                for (int c = 0; c < 3; c++) {
                    float val = output_data[i * 3 + c];
                    val = (val / 2.0f + 0.5f) * 255.0f;
                    val = std::max(0.0f, std::min(255.0f, val));
                    temp_output[i * 3 + c] = static_cast<uint8_t>(val);
                }
            }
            
            sd_image_t temp_img;
            temp_img.width = model_size;
            temp_img.height = model_size;
            temp_img.channel = 3;
            temp_img.data = temp_output.data();
            
            auto final_img = resize_image(&temp_img, image->width, image->height);
            if (!final_img) return fallback_restore(image, outputs);
            
            outputs["IMAGE"] = final_img;
            LOG_INFO("[FaceRestore] Face restoration completed with ONNX model\n");
            return sd_error_t::OK;
            
        } catch (const Ort::Exception& e) {
            LOG_ERROR("[FaceRestore] ONNX Runtime error: %s\n", e.what());
            return fallback_restore(image, outputs);
        }
#else
        LOG_WARN("[FaceRestore] ONNX Runtime not available, using fallback enhancement\n");
        return fallback_restore(image, outputs);
#endif
    }

  private:
    sd_error_t fallback_restore(const ImagePtr& image, NodeOutputs& outputs) {
        // 简单的图像增强作为 fallback
        int w = image->width;
        int h = image->height;
        int c = image->channel;
        
        auto faces = detect_faces_simple(image.get());
        if (faces.empty()) {
            LOG_WARN("[FaceRestore] No face detected, returning original image\n");
            outputs["IMAGE"] = image;
            return sd_error_t::OK;
        }
        
        // 复制原图
        auto result = make_malloc_buffer(w * h * c);
        if (!result) return sd_error_t::ERROR_MEMORY_ALLOCATION;
        std::memcpy(result.get(), image->data, w * h * c);
        
        // 对检测到的人脸区域进行锐化
        for (const auto& face : faces) {
            LOG_INFO("[FaceRestore] Enhancing face at (%d, %d, %d, %d)\n", face.x, face.y, face.w, face.h);
            
            // 简单的锐化核
            for (int y = face.y + 1; y < face.y + face.h - 1 && y < h - 1; y++) {
                for (int x = face.x + 1; x < face.x + face.w - 1 && x < w - 1; x++) {
                    for (int ch = 0; ch < c && ch < 3; ch++) {
                        int idx = (y * w + x) * c + ch;
                        int center = image->data[idx];
                        int top = image->data[((y - 1) * w + x) * c + ch];
                        int bottom = image->data[((y + 1) * w + x) * c + ch];
                        int left = image->data[(y * w + (x - 1)) * c + ch];
                        int right = image->data[(y * w + (x + 1)) * c + ch];
                        
                        int sharpened = 5 * center - top - bottom - left - right;
                        sharpened = std::max(0, std::min(255, sharpened));
                        result.get()[idx] = static_cast<uint8_t>(sharpened);
                    }
                }
            }
        }
        
        outputs["IMAGE"] = create_image_ptr(w, h, c, std::move(result));
        LOG_INFO("[FaceRestore] Fallback enhancement completed\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceRestore", FaceRestoreNode);

// ============================================================================
// FaceSwap - inswapper 人脸替换
// ============================================================================
class FaceSwapNode : public Node {
  public:
    std::string get_class_type() const override { return "FaceSwap"; }
    std::string get_category() const override { return "face"; }

    std::vector<PortDef> get_inputs() const override {
        return {
            {"source_image", "IMAGE", true, nullptr},
            {"target_image", "IMAGE", true, nullptr},
            {"model_path", "STRING", false, std::string("models/inswapper_128.onnx")},
        };
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"IMAGE", "IMAGE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        ImagePtr source;
        ImagePtr target;
        SD_RETURN_IF_ERROR(get_input(inputs, "source_image", source));
        SD_RETURN_IF_ERROR(get_input(inputs, "target_image", target));
        SD_RETURN_IF_NULL(source.get(), sd_error_t::ERROR_INVALID_INPUT);
        SD_RETURN_IF_NULL(target.get(), sd_error_t::ERROR_INVALID_INPUT);

        std::string model_path = get_input_opt<std::string>(inputs, "model_path", "models/inswapper_128.onnx");

        LOG_INFO("[FaceSwap] Swapping face from %dx%d to %dx%d\n",
                 source->width, source->height, target->width, target->height);

#ifdef ONNXRUNTIME_AVAILABLE
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            LOG_WARN("[FaceSwap] Model not found: %s, using fallback blend\n", model_path.c_str());
            return fallback_swap(source, target, outputs);
        }

        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FaceSwap");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            
            Ort::Session session(env, model_path.c_str(), session_options);
            LOG_INFO("[FaceSwap] ONNX model loaded: %s\n", model_path.c_str());
            
            // 简化的 face swap：直接返回目标图像（模型推理需要更复杂的预处理）
            LOG_WARN("[FaceSwap] Full ONNX inference not yet implemented, using fallback\n");
            return fallback_swap(source, target, outputs);
            
        } catch (const Ort::Exception& e) {
            LOG_ERROR("[FaceSwap] ONNX Runtime error: %s\n", e.what());
            return fallback_swap(source, target, outputs);
        }
#else
        LOG_WARN("[FaceSwap] ONNX Runtime not available, using fallback blend\n");
        return fallback_swap(source, target, outputs);
#endif
    }

  private:
    sd_error_t fallback_swap(const ImagePtr& source, const ImagePtr& target, NodeOutputs& outputs) {
        int w = target->width;
        int h = target->height;
        int c = target->channel;
        
        auto target_faces = detect_faces_simple(target.get());
        auto source_faces = detect_faces_simple(source.get());
        
        if (target_faces.empty() || source_faces.empty()) {
            LOG_WARN("[FaceSwap] No face detected in one or both images, returning target\n");
            outputs["IMAGE"] = target;
            return sd_error_t::OK;
        }
        
        // 复制目标图像
        auto result = make_malloc_buffer(w * h * c);
        if (!result) return sd_error_t::ERROR_MEMORY_ALLOCATION;
        std::memcpy(result.get(), target->data, w * h * c);
        
        // 简单的混合：将源脸区域混合到目标脸区域
        const auto& target_face = target_faces[0];
        const auto& source_face = source_faces[0];
        
        LOG_INFO("[FaceSwap] Blending face: target(%d,%d,%d,%d) <- source(%d,%d,%d,%d)\n",
                 target_face.x, target_face.y, target_face.w, target_face.h,
                 source_face.x, source_face.y, source_face.w, source_face.h);
        
        // 计算缩放比例
        float scale_x = static_cast<float>(source_face.w) / target_face.w;
        float scale_y = static_cast<float>(source_face.h) / target_face.h;
        
        for (int y = 0; y < target_face.h && y < h; y++) {
            for (int x = 0; x < target_face.w && x < w; x++) {
                int target_idx = ((target_face.y + y) * w + (target_face.x + x)) * c;
                
                int src_x = static_cast<int>(source_face.x + x * scale_x);
                int src_y = static_cast<int>(source_face.y + y * scale_y);
                src_x = std::min(src_x, static_cast<int>(source->width) - 1);
                src_y = std::min(src_y, static_cast<int>(source->height) - 1);
                int source_idx = (src_y * source->width + src_x) * source->channel;
                
                // 简单的 alpha 混合
                float alpha = 0.7f;
                for (int ch = 0; ch < c && ch < 3; ch++) {
                    int blended = static_cast<int>(
                        alpha * source->data[source_idx + ch] +
                        (1.0f - alpha) * target->data[target_idx + ch]
                    );
                    result.get()[target_idx + ch] = static_cast<uint8_t>(std::max(0, std::min(255, blended)));
                }
            }
        }
        
        outputs["IMAGE"] = create_image_ptr(w, h, c, std::move(result));
        LOG_INFO("[FaceSwap] Fallback blend completed\n");
        return sd_error_t::OK;
    }
};
REGISTER_NODE("FaceSwap", FaceSwapNode);

void init_face_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
