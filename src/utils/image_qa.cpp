#include "image_qa.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace myimg {

QAResult ImageQA::check_image(const ImageData& image, const std::vector<QARule>& rules) {
    QAResult result;

    for (const auto& rule : rules) {
        QAIssue issue;
        switch (rule.type) {
            case QARuleType::Blur:
                issue = check_blur(image, rule.threshold);
                break;
            case QARuleType::Exposure:
                issue = check_exposure(image, rule.threshold);
                break;
            case QARuleType::ColorCast:
                issue = check_color_cast(image, rule.threshold);
                break;
            case QARuleType::Resolution:
                // threshold 作为最小短边像素数
                issue = check_resolution(image, static_cast<int>(rule.threshold), static_cast<int>(rule.threshold));
                break;
            case QARuleType::Compression:
                // 暂时跳过，需要更复杂的算法
                continue;
        }

        if (issue.severity != QASeverity::Pass) {
            result.issues.push_back(issue);
            if (issue.severity == QASeverity::Error) {
                result.pass = false;
            }
        }
    }

    // 生成建议
    if (!result.pass) {
        result.recommendation = "图像未通过质检，请根据以下问题修复后重新处理：\n";
        for (const auto& issue : result.issues) {
            result.recommendation += "- " + issue.message + "\n";
        }
    } else if (!result.issues.empty()) {
        result.recommendation = "图像通过质检，但存在以下警告：\n";
        for (const auto& issue : result.issues) {
            result.recommendation += "- " + issue.message + "\n";
        }
    } else {
        result.recommendation = "图像通过所有质检项";
    }

    return result;
}

QAIssue ImageQA::check_blur(const ImageData& image, float threshold) {
    QAIssue issue;
    issue.rule = QARuleType::Blur;
    issue.threshold = threshold;

    // 将 ImageData 转换为 OpenCV Mat
    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

    // 拉普拉斯方差（Laplacian Variance）
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double variance = stddev.val[0] * stddev.val[0];

    issue.value = static_cast<float>(variance);

    if (variance < threshold * 0.5) {
        issue.severity = QASeverity::Error;
        issue.message = "图像过于模糊（清晰度 " + std::to_string(static_cast<int>(variance)) +
                       "，要求 >= " + std::to_string(static_cast<int>(threshold)) + "）";
    } else if (variance < threshold) {
        issue.severity = QASeverity::Warning;
        issue.message = "图像可能偏模糊（清晰度 " + std::to_string(static_cast<int>(variance)) +
                       "，建议 >= " + std::to_string(static_cast<int>(threshold)) + "）";
    } else {
        issue.severity = QASeverity::Pass;
        issue.message = "清晰度正常（" + std::to_string(static_cast<int>(variance)) + "）";
    }

    return issue;
}

QAIssue ImageQA::check_exposure(const ImageData& image, float threshold) {
    QAIssue issue;
    issue.rule = QARuleType::Exposure;
    issue.threshold = threshold;

    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

    // 计算直方图
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // 计算高光和暗部占比
    float total_pixels = static_cast<float>(image.width * image.height);
    float highlight_pixels = 0;  // > 250
    float shadow_pixels = 0;     // < 5

    for (int i = 250; i < 256; ++i) {
        highlight_pixels += hist.at<float>(i);
    }
    for (int i = 0; i < 5; ++i) {
        shadow_pixels += hist.at<float>(i);
    }

    float highlight_ratio = highlight_pixels / total_pixels;
    float shadow_ratio = shadow_pixels / total_pixels;

    issue.value = std::max(highlight_ratio, shadow_ratio);

    if (highlight_ratio > threshold) {
        issue.severity = QASeverity::Error;
        issue.message = "图像过曝（高光占比 " + std::to_string(static_cast<int>(highlight_ratio * 100)) +
                       "%，要求 < " + std::to_string(static_cast<int>(threshold * 100)) + "%）";
    } else if (shadow_ratio > threshold) {
        issue.severity = QASeverity::Error;
        issue.message = "图像欠曝（暗部占比 " + std::to_string(static_cast<int>(shadow_ratio * 100)) +
                       "%，要求 < " + std::to_string(static_cast<int>(threshold * 100)) + "%）";
    } else if (highlight_ratio > threshold * 0.7f || shadow_ratio > threshold * 0.7f) {
        issue.severity = QASeverity::Warning;
        issue.message = "曝光可能有问题（高光/暗部占比较高）";
    } else {
        issue.severity = QASeverity::Pass;
        issue.message = "曝光正常";
    }

    return issue;
}

QAIssue ImageQA::check_color_cast(const ImageData& image, float threshold) {
    QAIssue issue;
    issue.rule = QARuleType::ColorCast;
    issue.threshold = threshold;

    cv::Mat img(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.data.data()));

    // 灰度世界算法（Gray World Algorithm）
    cv::Scalar mean = cv::mean(img);
    float avg_r = static_cast<float>(mean[0]);
    float avg_g = static_cast<float>(mean[1]);
    float avg_b = static_cast<float>(mean[2]);

    // 计算各通道与平均值的偏差
    float avg_all = (avg_r + avg_g + avg_b) / 3.0f;
    float max_deviation = std::max({std::abs(avg_r - avg_all), std::abs(avg_g - avg_all), std::abs(avg_b - avg_all)});

    issue.value = max_deviation;

    // 转换为色偏角度（简化版）
    float color_cast_angle = max_deviation / avg_all * 100.0f;

    if (color_cast_angle > threshold * 1.5f) {
        issue.severity = QASeverity::Error;
        issue.message = "严重色偏（偏差 " + std::to_string(static_cast<int>(color_cast_angle)) +
                       "，要求 < " + std::to_string(static_cast<int>(threshold)) + "）";
    } else if (color_cast_angle > threshold) {
        issue.severity = QASeverity::Warning;
        issue.message = "存在色偏（偏差 " + std::to_string(static_cast<int>(color_cast_angle)) +
                       "，建议 < " + std::to_string(static_cast<int>(threshold)) + "）";
    } else {
        issue.severity = QASeverity::Pass;
        issue.message = "色彩正常";
    }

    return issue;
}

QAIssue ImageQA::check_resolution(const ImageData& image, int min_width, int min_height) {
    QAIssue issue;
    issue.rule = QARuleType::Resolution;
    issue.threshold = static_cast<float>(std::max(min_width, min_height));

    int min_dimension = std::min(image.width, image.height);
    issue.value = static_cast<float>(min_dimension);

    if (min_dimension <= min_width * 0.5f) {
        issue.severity = QASeverity::Error;
        issue.message = "分辨率过低（" + std::to_string(image.width) + "x" + std::to_string(image.height) +
                       "，要求短边 >= " + std::to_string(min_width) + "）";
    } else if (min_dimension < min_width) {
        issue.severity = QASeverity::Warning;
        issue.message = "分辨率可能不足（" + std::to_string(image.width) + "x" + std::to_string(image.height) +
                       "，建议短边 >= " + std::to_string(min_width) + "）";
    } else {
        issue.severity = QASeverity::Pass;
        issue.message = "分辨率符合要求（" + std::to_string(image.width) + "x" + std::to_string(image.height) + "）";
    }

    return issue;
}

std::string ImageQA::export_json(const QAResult& result) {
    std::stringstream json;
    json << "{\n";
    json << "  \"pass\": " << (result.pass ? "true" : "false") << ",\n";
    json << "  \"issues\": [\n";

    for (size_t i = 0; i < result.issues.size(); ++i) {
        const auto& issue = result.issues[i];
        json << "    {\n";
        json << "      \"rule\": \"";
        switch (issue.rule) {
            case QARuleType::Blur: json << "blur"; break;
            case QARuleType::Exposure: json << "exposure"; break;
            case QARuleType::ColorCast: json << "color_cast"; break;
            case QARuleType::Resolution: json << "resolution"; break;
            case QARuleType::Compression: json << "compression"; break;
        }
        json << "\",\n";
        json << "      \"severity\": \"";
        switch (issue.severity) {
            case QASeverity::Pass: json << "pass"; break;
            case QASeverity::Warning: json << "warning"; break;
            case QASeverity::Error: json << "error"; break;
        }
        json << "\",\n";
        json << "      \"message\": \"" << issue.message << "\",\n";
        json << "      \"value\": " << std::fixed << std::setprecision(2) << issue.value << ",\n";
        json << "      \"threshold\": " << issue.threshold << "\n";
        json << "    }";
        if (i < result.issues.size() - 1) {
            json << ",";
        }
        json << "\n";
    }

    json << "  ],\n";
    json << "  \"recommendation\": \"" << result.recommendation << "\"\n";
    json << "}\n";

    return json.str();
}

std::vector<QARule> ImageQA::get_ecommerce_rules() {
    return {
        {QARuleType::Blur, 100.0f, QASeverity::Error},
        {QARuleType::Exposure, 0.95f, QASeverity::Warning},
        {QARuleType::ColorCast, 15.0f, QASeverity::Warning},
        {QARuleType::Resolution, 800.0f, QASeverity::Error},
    };
}

std::vector<QARule> ImageQA::get_photography_rules() {
    return {
        {QARuleType::Blur, 50.0f, QASeverity::Warning},
        {QARuleType::Exposure, 0.90f, QASeverity::Warning},
        {QARuleType::ColorCast, 20.0f, QASeverity::Warning},
        {QARuleType::Resolution, 1920.0f, QASeverity::Warning},
    };
}

} // namespace myimg
