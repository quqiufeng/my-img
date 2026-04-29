// ============================================================================
// sd-engine/core/job_queue.cpp
// ============================================================================
// 任务队列实现
// ============================================================================

#include "core/job_queue.h"
#include "core/log.h"
#include <sstream>

namespace sdengine {

JobQueue::JobQueue(size_t max_concurrent)
    : max_concurrent_(max_concurrent), running_count_(0), completed_count_(0), failed_count_(0) {}

std::string JobQueue::submit(const std::string& workflow_json) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string id = generate_id();
    Job job;
    job.id = id;
    job.workflow_json = workflow_json;
    job.status = JobStatus::PENDING;
    job.created_at = std::chrono::system_clock::now();
    jobs_[id] = job;
    queue_.push(id);
    cv_.notify_one();
    LOG_INFO("[Queue] Job %s submitted (queue size: %zu)\n", id.c_str(), queue_.size());
    return id;
}

Job JobQueue::get_status(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(id);
    if (it != jobs_.end()) return it->second;
    return Job{};
}

std::vector<Job> JobQueue::get_all_jobs() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Job> result;
    for (const auto& [id, job] : jobs_) {
        result.push_back(job);
    }
    return result;
}

Job JobQueue::wait_for_job() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() && running_count_ < max_concurrent_; });

    std::string id = queue_.front();
    queue_.pop();
    auto& job = jobs_[id];
    job.status = JobStatus::RUNNING;
    job.started_at = std::chrono::system_clock::now();
    running_count_++;
    return job;
}

void JobQueue::complete_job(const std::string& id, bool success, const std::string& error,
                            long long elapsed_ms, const std::string& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(id);
    if (it != jobs_.end()) {
        it->second.status = success ? JobStatus::COMPLETED : JobStatus::FAILED;
        it->second.error_msg = error;
        it->second.elapsed_ms = elapsed_ms;
        it->second.completed_at = std::chrono::system_clock::now();
        it->second.result = result;
        if (success) {
            completed_count_++;
        } else {
            failed_count_++;
        }
        if (running_count_ > 0) {
            running_count_--;
        }
    }
    cv_.notify_one();
}

size_t JobQueue::queue_size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

size_t JobQueue::running_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_count_;
}

size_t JobQueue::completed_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_count_;
}

size_t JobQueue::failed_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_count_;
}

std::string JobQueue::generate_id() {
    static std::atomic<int> counter{0};
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::ostringstream oss;
    oss << "job_" << ms << "_" << counter++;
    return oss.str();
}

} // namespace sdengine
