// ============================================================================
// sd-engine/tools/sd-server.cpp
// ============================================================================
// HTTP API 服务器：接收 JSON 工作流并执行
// 基于 cpp-httplib（single-header）
// 支持请求队列和并发控制
// ============================================================================

#include "adapter/sd_adapter.h"
#include "core/job_queue.h"
#include "core/log.h"
#include "core/node.h"
#include "core/workflow.h"
#include "core/executor.h"
#include "stable-diffusion.h"
#include "nlohmann/json.hpp"
#include <httplib.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <future>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace sdengine;
using namespace std::chrono;

// ============================================================================
// 速率限制器（令牌桶）
// ============================================================================

class RateLimiter {
  public:
    RateLimiter(int max_tokens = 10, double refill_rate = 1.0)
        : max_tokens_(max_tokens), refill_rate_(refill_rate) {}

    bool allow(const std::string& ip) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = steady_clock::now();
        auto& bucket = buckets_[ip];

        // 补充令牌
        auto elapsed = duration_cast<duration<double>>(now - bucket.last_refill).count();
        int tokens_to_add = static_cast<int>(elapsed * refill_rate_);
        if (tokens_to_add > 0) {
            bucket.tokens = std::min(max_tokens_, bucket.tokens + tokens_to_add);
            bucket.last_refill = now;
        }

        if (bucket.tokens > 0) {
            bucket.tokens--;
            return true;
        }
        return false;
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = steady_clock::now();
        for (auto it = buckets_.begin(); it != buckets_.end();) {
            auto elapsed = duration_cast<duration<double>>(now - it->second.last_refill).count();
            if (elapsed > 60.0) { // 60 秒无活动的 IP 清理
                it = buckets_.erase(it);
            } else {
                ++it;
            }
        }
    }

  private:
    struct Bucket {
        int tokens = 10;
        steady_clock::time_point last_refill = steady_clock::now();
    };

    int max_tokens_;
    double refill_rate_;
    std::unordered_map<std::string, Bucket> buckets_;
    std::mutex mutex_;
};

// ============================================================================
// 全局状态
// ============================================================================

struct ServerState {
    std::string models_dir = "./models";
    std::string output_dir = "./output";
    int port = 8080;
    int max_concurrent = 1;
    bool verbose = false;
    bool rate_limit_enabled = true;
    int rate_limit_max = 10;
    double rate_limit_refill = 1.0;
};

static ServerState g_state;
static std::unique_ptr<JobQueue> g_queue;
static std::unique_ptr<RateLimiter> g_rate_limiter;
static std::atomic<bool> g_running{true};

// ============================================================================
// 辅助函数
// ============================================================================

std::vector<std::string> scan_models(const std::string& dir) {
    std::vector<std::string> models;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return models;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            if (ext == ".gguf" || ext == ".safetensors" || ext == ".ckpt") {
                models.push_back(entry.path().filename().string());
            }
        }
    }
    return models;
}

json nodes_to_json() {
    json nodes = json::array();
    auto& registry = NodeRegistry::instance();
    for (const auto& type : registry.get_supported_nodes()) {
        auto node = registry.create(type);
        if (!node) continue;
        json j;
        j["type"] = type;
        j["category"] = node->get_category();
        
        json inputs = json::array();
        for (const auto& def : node->get_inputs()) {
            json inp;
            inp["name"] = def.name;
            inp["type"] = def.type;
            inp["required"] = def.required;
            inputs.push_back(inp);
        }
        j["inputs"] = inputs;
        
        json outputs = json::array();
        for (const auto& def : node->get_outputs()) {
            json out;
            out["name"] = def.name;
            out["type"] = def.type;
            outputs.push_back(out);
        }
        j["outputs"] = outputs;
        
        nodes.push_back(j);
    }
    return nodes;
}

std::string status_to_string(JobStatus s) {
    switch (s) {
        case JobStatus::PENDING: return "pending";
        case JobStatus::RUNNING: return "running";
        case JobStatus::COMPLETED: return "completed";
        case JobStatus::FAILED: return "failed";
        case JobStatus::CANCELLED: return "cancelled";
    }
    return "unknown";
}

json job_to_json(const Job& job) {
    json j;
    j["id"] = job.id;
    j["status"] = status_to_string(job.status);
    j["elapsed_ms"] = job.elapsed_ms;
    if (!job.error_msg.empty()) j["error"] = job.error_msg;
    if (!job.result.empty()) {
        try {
            j["result"] = json::parse(job.result);
        } catch (...) {
            j["result"] = job.result;
        }
    }
    return j;
}

// ============================================================================
// 工作线程
// ============================================================================

void worker_thread() {
    while (g_running) {
        Job job = g_queue->wait_for_job();
        if (!g_running) break;

        LOG_INFO("[Worker] Starting job %s\n", job.id.c_str());
        auto start = steady_clock::now();
        
        Workflow workflow;
        std::string error_msg;
        json result;
        bool success = false;
        
        if (!workflow.load_from_string(job.workflow_json)) {
            error_msg = "Failed to parse workflow JSON";
        } else if (!workflow.validate(error_msg)) {
            error_msg = "Workflow validation failed: " + error_msg;
        } else {
            ExecutionCache cache;
            DAGExecutor executor(&cache);
            ExecutionConfig exec_config;
            exec_config.verbose = g_state.verbose;
            
            // 使用异步执行 + 超时检测
            auto future = std::async(std::launch::async, [&]() {
                return executor.execute(&workflow, exec_config);
            });
            
            const int timeout_seconds = 300; // 5 分钟超时
            if (future.wait_for(seconds(timeout_seconds)) == std::future_status::timeout) {
                error_msg = "Execution timeout (" + std::to_string(timeout_seconds) + "s)";
                LOG_ERROR("[Worker] Job %s timed out after %d seconds\n", job.id.c_str(), timeout_seconds);
            } else {
                auto err = future.get();
                if (is_error(err)) {
                    error_msg = "Execution failed";
                } else {
                    success = true;
                    auto end = steady_clock::now();
                    auto elapsed = duration_cast<milliseconds>(end - start).count();
                    result["success"] = true;
                    result["elapsed_ms"] = elapsed;
                }
            }
        }
        
        auto end = steady_clock::now();
        auto elapsed = duration_cast<milliseconds>(end - start).count();
        g_queue->complete_job(job.id, success, error_msg, elapsed, result.dump());
        LOG_INFO("[Worker] Job %s completed (%s, %lld ms)\n", 
                 job.id.c_str(), success ? "success" : "failed", elapsed);
    }
}

// ============================================================================
// HTTP 处理函数
// ============================================================================

void handle_health(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    json j;
    j["status"] = "ok";
    j["version"] = "sd-engine 1.0";
    j["queue_size"] = g_queue->queue_size();
    j["running"] = g_queue->running_count();
    res.set_content(j.dump(), "application/json");
}

void handle_models(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    json j;
    j["models"] = scan_models(g_state.models_dir);
    res.set_content(j.dump(2), "application/json");
}

void handle_nodes(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    json j;
    j["nodes"] = nodes_to_json();
    res.set_content(j.dump(2), "application/json");
}

void handle_submit(const httplib::Request& req, httplib::Response& res) {
    // 速率限制检查
    if (g_state.rate_limit_enabled && g_rate_limiter) {
        if (!g_rate_limiter->allow(req.remote_addr)) {
            res.status = 429;
            json err;
            err["error"] = "Rate limit exceeded. Please try again later.";
            res.set_content(err.dump(), "application/json");
            LOG_WARN("[RateLimit] IP %s exceeded rate limit\n", req.remote_addr.c_str());
            return;
        }
    }

    try {
        json req_json = json::parse(req.body);
        
        if (!req_json.contains("workflow")) {
            res.status = 400;
            res.set_content(R"({"error": "Missing 'workflow' field"})", "application/json");
            return;
        }
        
        std::string workflow_str = req_json["workflow"].dump();
        std::string job_id = g_queue->submit(workflow_str);
        
        json result;
        result["job_id"] = job_id;
        result["status"] = "pending";
        result["queue_position"] = g_queue->queue_size();
        res.set_content(result.dump(2), "application/json");
        
    } catch (const std::exception& e) {
        res.status = 500;
        json err;
        err["error"] = std::string("Exception: ") + e.what();
        res.set_content(err.dump(), "application/json");
    }
}

void handle_status(const httplib::Request& req, httplib::Response& res) {
    auto job_id = req.path_params.at("id");
    auto job = g_queue->get_status(job_id);
    
    if (job.id.empty()) {
        res.status = 404;
        res.set_content(R"({"error": "Job not found"})", "application/json");
        return;
    }
    
    res.set_content(job_to_json(job).dump(2), "application/json");
}

void handle_queue(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    auto jobs = g_queue->get_all_jobs();
    json result = json::array();
    for (const auto& job : jobs) {
        result.push_back(job_to_json(job));
    }
    json j;
    j["jobs"] = result;
    j["queue_size"] = g_queue->queue_size();
    j["running"] = g_queue->running_count();
    res.set_content(j.dump(2), "application/json");
}

void handle_generate_sync(const httplib::Request& req, httplib::Response& res) {
    try {
        json req_json = json::parse(req.body);
        
        if (!req_json.contains("workflow")) {
            res.status = 400;
            res.set_content(R"({"error": "Missing 'workflow' field"})", "application/json");
            return;
        }
        
        Workflow workflow;
        std::string workflow_str = req_json["workflow"].dump();
        if (!workflow.load_from_string(workflow_str)) {
            res.status = 400;
            res.set_content(R"({"error": "Invalid workflow JSON"})", "application/json");
            return;
        }
        
        std::string error_msg;
        if (!workflow.validate(error_msg)) {
            res.status = 400;
            json err;
            err["error"] = "Workflow validation failed";
            err["detail"] = error_msg;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        ExecutionCache cache;
        DAGExecutor executor(&cache);
        ExecutionConfig exec_config;
        exec_config.verbose = g_state.verbose;
        
        LOG_INFO("[Server] Executing workflow (sync)...\n");
        auto start = steady_clock::now();
        auto err = executor.execute(&workflow, exec_config);
        auto end = steady_clock::now();
        auto elapsed = duration_cast<milliseconds>(end - start).count();
        
        if (is_error(err)) {
            res.status = 500;
            json err_json;
            err_json["error"] = "Workflow execution failed";
            err_json["elapsed_ms"] = elapsed;
            res.set_content(err_json.dump(), "application/json");
            return;
        }
        
        json result;
        result["success"] = true;
        result["elapsed_ms"] = elapsed;
        
        if (req_json.contains("output_path")) {
            result["output_path"] = req_json["output_path"];
        }
        
        res.set_content(result.dump(2), "application/json");
        
    } catch (const std::exception& e) {
        res.status = 500;
        json err;
        err["error"] = std::string("Exception: ") + e.what();
        res.set_content(err.dump(), "application/json");
    }
}

// ============================================================================
// 主函数
// ============================================================================

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --models-dir PATH    Model directory (default: ./models)\n"
              << "  --output-dir PATH    Output directory (default: ./output)\n"
              << "  --port N             HTTP port (default: 8080)\n"
              << "  --max-concurrent N   Max concurrent jobs (default: 1)\n"
              << "  --verbose            Verbose logging\n"
              << "  --help               Show this help\n"
              << "\n"
              << "Endpoints:\n"
              << "  GET  /health         Health check + queue status\n"
              << "  GET  /models         List available models\n"
              << "  GET  /nodes          List supported nodes\n"
              << "  GET  /queue          View all jobs\n"
              << "  POST /submit         Submit workflow to queue (async)\n"
              << "  GET  /status/:id     Query job status\n"
              << "  POST /generate       Execute workflow synchronously\n";
}

bool parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return false;
        } else if (arg == "--models-dir") {
            if (++i >= argc) return false;
            g_state.models_dir = argv[i];
        } else if (arg == "--output-dir") {
            if (++i >= argc) return false;
            g_state.output_dir = argv[i];
        } else if (arg == "--port") {
            if (++i >= argc) return false;
            g_state.port = std::stoi(argv[i]);
        } else if (arg == "--max-concurrent") {
            if (++i >= argc) return false;
            g_state.max_concurrent = std::stoi(argv[i]);
        } else if (arg == "--verbose") {
            g_state.verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
    }
    return true;
}

// 外部声明
namespace sdengine {
    void init_builtin_nodes();
}

int main(int argc, char** argv) {
    if (!parse_args(argc, argv)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // 初始化所有内置节点
    sdengine::init_builtin_nodes();
    
    // 创建任务队列
    g_queue = std::make_unique<JobQueue>(g_state.max_concurrent);
    
    // 创建速率限制器
    g_rate_limiter = std::make_unique<RateLimiter>(g_state.rate_limit_max, g_state.rate_limit_refill);
    
    // 启动工作线程
    std::vector<std::thread> workers;
    for (int i = 0; i < g_state.max_concurrent; i++) {
        workers.emplace_back(worker_thread);
    }
    
    LOG_INFO("[Server] Starting sd-engine HTTP server on port %d\n", g_state.port);
    LOG_INFO("[Server] Models dir: %s\n", g_state.models_dir.c_str());
    LOG_INFO("[Server] Max concurrent: %d\n", g_state.max_concurrent);
    
    httplib::Server svr;
    
    svr.Get("/health", handle_health);
    svr.Get("/models", handle_models);
    svr.Get("/nodes", handle_nodes);
    svr.Get("/queue", handle_queue);
    svr.Post("/submit", handle_submit);
    svr.Get("/status/:id", handle_status);
    svr.Post("/generate", handle_generate_sync);
    
    LOG_INFO("[Server] Ready. Listening on http://0.0.0.0:%d\n", g_state.port);
    
    bool ok = svr.listen("0.0.0.0", g_state.port);
    
    // 关闭
    g_running = false;
    g_queue->complete_job("", false, "", 0, {}); // wake up workers
    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
    
    if (!ok) {
        LOG_ERROR("[Server] Failed to start server on port %d\n", g_state.port);
        return 1;
    }
    
    return 0;
}
