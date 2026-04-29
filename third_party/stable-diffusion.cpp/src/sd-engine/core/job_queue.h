// ============================================================================
// sd-engine/core/job_queue.h
// ============================================================================
/// @file job_queue.h
/// @brief HTTP Server 任务队列系统
///
/// JobQueue 管理异步工作流任务的提交、调度、执行和状态追踪。
/// 支持并发控制、超时统计和线程安全操作。
// ============================================================================

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace sdengine {

/// @brief 任务状态枚举
enum class JobStatus { PENDING, RUNNING, COMPLETED, FAILED, CANCELLED };

/// @brief 单个任务的数据结构
struct Job {
    std::string id;                                    ///< 任务唯一 ID
    std::string workflow_json;                         ///< 工作流 JSON 字符串
    JobStatus status = JobStatus::PENDING;             ///< 当前状态
    std::string error_msg;                             ///< 错误信息（失败时）
    long long elapsed_ms = 0;                          ///< 执行耗时（毫秒）
    std::chrono::system_clock::time_point created_at;  ///< 创建时间
    std::chrono::system_clock::time_point started_at;  ///< 开始执行时间
    std::chrono::system_clock::time_point completed_at;///< 完成时间
    std::string result;                                ///< 结果 JSON 字符串
};

/// @brief 任务队列管理器
///
/// 线程安全的任务队列，支持：
/// - 任务提交（返回唯一 ID）
/// - 工作线程阻塞等待可用任务
/// - 并发控制（限制同时执行的任务数）
/// - 任务状态查询和枚举
class JobQueue {
  public:
    /// @brief 构造函数
    /// @param max_concurrent 最大并发执行数（默认 1）
    explicit JobQueue(size_t max_concurrent = 1);

    /// @brief 提交新任务
    /// @param workflow_json 工作流 JSON 字符串
    /// @return 任务唯一 ID
    std::string submit(const std::string& workflow_json);

    /// @brief 获取指定任务的状态
    /// @param id 任务 ID
    /// @return 任务对象（ID 为空表示未找到）
    Job get_status(const std::string& id);

    /// @brief 获取所有任务列表
    /// @return 任务列表（按提交顺序）
    std::vector<Job> get_all_jobs();

    /// @brief 工作线程调用：阻塞等待下一个可执行任务
    /// @return 待执行的任务
    Job wait_for_job();

    /// @brief 标记任务完成
    /// @param id 任务 ID
    /// @param success 是否成功
    /// @param error 错误信息（失败时）
    /// @param elapsed_ms 执行耗时
    /// @param result 结果 JSON 字符串
    void complete_job(const std::string& id, bool success, const std::string& error,
                      long long elapsed_ms, const std::string& result);

    /// @brief 获取当前队列长度（等待中的任务数）
    size_t queue_size();

    /// @brief 获取当前正在执行的任务数
    size_t running_count();

    /// @brief 获取已完成的任务数
    size_t completed_count();

    /// @brief 获取失败的任务数
    size_t failed_count();

  private:
    std::string generate_id();

    size_t max_concurrent_;
    size_t running_count_;
    size_t completed_count_;
    size_t failed_count_;
    std::queue<std::string> queue_;
    std::map<std::string, Job> jobs_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace sdengine
