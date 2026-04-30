#pragma once

#include <string>
#include <vector>

#include "image_utils.h"

namespace myimg {

enum class QARuleType {
    Blur,       // 模糊检测
    Exposure,   // 过曝/欠曝检测
    ColorCast,  // 色偏检测
    Resolution, // 分辨率检测
    Compression // 压缩瑕疵检测
};

enum class QASeverity {
    Pass,    // 通过
    Warning, // 警告
    Error    // 错误
};

struct QAIssue {
    QARuleType rule;
    QASeverity severity;
    std::string message;
    float value = 0.0f;      // 检测值
    float threshold = 0.0f;  // 阈值
};

struct QAResult {
    bool pass = true;
    std::vector<QAIssue> issues;
    std::string recommendation;
};

struct QARule {
    QARuleType type;
    float threshold;
    QASeverity severity;
};

class ImageQA {
public:
    // 运行所有质检规则
    static QAResult check_image(const ImageData& image, const std::vector<QARule>& rules);

    // 单独检查
    static QAIssue check_blur(const ImageData& image, float threshold);
    static QAIssue check_exposure(const ImageData& image, float threshold);
    static QAIssue check_color_cast(const ImageData& image, float threshold);
    static QAIssue check_resolution(const ImageData& image, int min_width, int min_height);

    // 导出报告为 JSON 字符串
    static std::string export_json(const QAResult& result);

    // 预设规则集
    static std::vector<QARule> get_ecommerce_rules();
    static std::vector<QARule> get_photography_rules();
};

} // namespace myimg
