// ============================================================================
// sd-engine/nodes/model_loader_nodes.cpp
// ============================================================================

#include "core/log.h"
#include "nodes/node_utils.h"
#include <list>
#include <mutex>
#include <unordered_map>

namespace sdengine {

// ============================================================================
// 模型缓存（热加载支持 + LRU 淘汰）
// ============================================================================
class ModelCache {
  public:
    static ModelCache& instance() {
        static ModelCache cache;
        return cache;
    }

    explicit ModelCache(size_t max_size = 3) : max_size_(max_size) {}

    SDContextPtr get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // 更新 LRU：移动到链表尾部
            touch(it->second.lru_iter);
            LOG_INFO("[ModelCache] Cache hit: %s (refs=%ld)\n", key.c_str(), it->second.ctx.use_count());
            return it->second.ctx;
        }
        return nullptr;
    }

    void put(const std::string& key, SDContextPtr ctx) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // 更新已有条目
            it->second.ctx = ctx;
            touch(it->second.lru_iter);
        } else {
            // 新条目：检查是否需要淘汰
            if (cache_.size() >= max_size_ && !lru_list_.empty()) {
                evict_one();
            }
            lru_list_.push_back(key);
            CacheItem item;
            item.ctx = ctx;
            item.lru_iter = std::prev(lru_list_.end());
            cache_[key] = std::move(item);
        }
        LOG_INFO("[ModelCache] Cached model: %s (count=%zu/%zu)\n", key.c_str(), cache_.size(), max_size_);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        lru_list_.clear();
        LOG_INFO("[ModelCache] Cache cleared\n");
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    void set_max_size(size_t max_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_size_ = max_size;
        while (cache_.size() > max_size_ && !lru_list_.empty()) {
            evict_one_unlocked();
        }
    }

  private:
    struct CacheItem {
        SDContextPtr ctx;
        std::list<std::string>::iterator lru_iter;
    };

    std::unordered_map<std::string, CacheItem> cache_;
    std::list<std::string> lru_list_;  // 队首最久未使用，队尾最近使用
    std::mutex mutex_;
    size_t max_size_;

    void touch(std::list<std::string>::iterator& it) {
        lru_list_.splice(lru_list_.end(), lru_list_, it);
    }

    void evict_one() {
        if (lru_list_.empty()) return;
        std::string key = lru_list_.front();
        lru_list_.pop_front();
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            LOG_INFO("[ModelCache] Evicted: %s (refs=%ld)\n", key.c_str(), it->second.ctx.use_count());
            cache_.erase(it);
        }
    }

    void evict_one_unlocked() {
        if (lru_list_.empty()) return;
        std::string key = lru_list_.front();
        lru_list_.pop_front();
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            cache_.erase(it);
        }
    }
};

static std::string make_cache_key(const std::string& ckpt, const std::string& vae, 
                                   const std::string& clip, const std::string& cn,
                                   bool gpu, bool flash) {
    return ckpt + "|" + vae + "|" + clip + "|" + cn + "|" + (gpu ? "1" : "0") + "|" + (flash ? "1" : "0");
}

// ============================================================================
// CheckpointLoaderSimple - 加载模型
// ============================================================================
class CheckpointLoaderSimpleNode : public Node {
  public:
    std::string get_class_type() const override {
        return "CheckpointLoaderSimple";
    }
    std::string get_category() const override {
        return "loaders";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"ckpt_name", "STRING", true, std::string("")},
                {"vae_name", "STRING", false, std::string("")},
                {"clip_name", "STRING", false, std::string("")},
                {"control_net_path", "STRING", false, std::string("")},
                {"n_threads", "INT", false, 4},
                {"use_gpu", "BOOLEAN", false, true},
                {"flash_attn", "BOOLEAN", false, false}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {{"MODEL", "MODEL"}, {"CLIP", "CLIP"}, {"VAE", "VAE"}};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        std::string ckpt_path;
        SD_RETURN_IF_ERROR(get_input(inputs, "ckpt_name", ckpt_path));
        std::string vae_path = get_input_opt<std::string>(inputs, "vae_name", "");
        std::string clip_path = get_input_opt<std::string>(inputs, "clip_name", "");
        std::string control_net_path = get_input_opt<std::string>(inputs, "control_net_path", "");
        int n_threads = get_input_opt<int>(inputs, "n_threads", 4);
        bool use_gpu = get_input_opt<bool>(inputs, "use_gpu", true);
        bool flash_attn = get_input_opt<bool>(inputs, "flash_attn", false);

        if (ckpt_path.empty()) {
            LOG_ERROR("[ERROR] CheckpointLoaderSimple: ckpt_name is required\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        // 路径安全检查
        if (!is_path_safe(ckpt_path) || (!vae_path.empty() && !is_path_safe(vae_path)) ||
            (!clip_path.empty() && !is_path_safe(clip_path)) ||
            (!control_net_path.empty() && !is_path_safe(control_net_path))) {
            LOG_ERROR("[ERROR] CheckpointLoaderSimple: Illegal path detected (directory traversal attempt)\n");
            return sd_error_t::ERROR_INVALID_INPUT;
        }

        // 检查缓存
        std::string cache_key = make_cache_key(ckpt_path, vae_path, clip_path, control_net_path, use_gpu, flash_attn);
        auto cached = ModelCache::instance().get(cache_key);
        if (cached) {
            LOG_INFO("[CheckpointLoaderSimple] Using cached model: %s\n", ckpt_path.c_str());
            outputs["MODEL"] = cached;
            outputs["CLIP"] = cached;
            outputs["VAE"] = cached;
            return sd_error_t::OK;
        }

        LOG_INFO("[CheckpointLoaderSimple] Loading model: %s\n", ckpt_path.c_str());

        sd_ctx_params_t ctx_params;
        sd_ctx_params_init(&ctx_params);
        ctx_params.diffusion_model_path = ckpt_path.c_str();
        if (!vae_path.empty()) {
            ctx_params.vae_path = vae_path.c_str();
        }
        if (!clip_path.empty()) {
            ctx_params.llm_path = clip_path.c_str();
        }
        if (!control_net_path.empty()) {
            ctx_params.control_net_path = control_net_path.c_str();
            ctx_params.keep_control_net_on_cpu = !use_gpu;
            LOG_INFO("[CheckpointLoaderSimple] Loading ControlNet: %s\n", control_net_path.c_str());
        }
        ctx_params.n_threads = n_threads;
        ctx_params.offload_params_to_cpu = !use_gpu;
        ctx_params.keep_vae_on_cpu = !use_gpu;
        ctx_params.keep_clip_on_cpu = !use_gpu;
        ctx_params.flash_attn = use_gpu && flash_attn;
        ctx_params.diffusion_flash_attn = use_gpu && flash_attn;
        ctx_params.vae_decode_only = false;

        sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
        if (!sd_ctx) {
            LOG_ERROR("[ERROR] Failed to load checkpoint\n");
            return sd_error_t::ERROR_MODEL_LOADING;
        }

        LOG_INFO("[CheckpointLoaderSimple] Model loaded successfully\n");

        auto sd_ctx_ptr = make_sd_context_ptr(sd_ctx);
        ModelCache::instance().put(cache_key, sd_ctx_ptr);
        outputs["MODEL"] = sd_ctx_ptr;
        outputs["CLIP"] = sd_ctx_ptr;
        outputs["VAE"] = sd_ctx_ptr;

        return sd_error_t::OK;
    }
};
REGISTER_NODE("CheckpointLoaderSimple", CheckpointLoaderSimpleNode);

// ============================================================================
// UnloadModel - 释放模型上下文
// ============================================================================
class UnloadModelNode : public Node {
  public:
    std::string get_class_type() const override {
        return "UnloadModel";
    }
    std::string get_category() const override {
        return "model_management";
    }

    std::vector<PortDef> get_inputs() const override {
        return {{"model", "MODEL", true, nullptr}};
    }

    std::vector<PortDef> get_outputs() const override {
        return {};
    }

    sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) override {
        (void)outputs;
        auto it = inputs.find("model");
        if (it != inputs.end()) {
            const auto& model_val = it->second;
            if (model_val.type() == typeid(SDContextPtr)) {
                auto opt = any_cast_safe<SDContextPtr>(model_val);
                if (opt && *opt) {
                    LOG_INFO("[UnloadModel] Releasing model context (ref_count=%ld)\n", opt->use_count());
                }
            } else if (model_val.type() == typeid(sd_ctx_t*)) {
                auto opt = any_cast_safe<sd_ctx_t*>(model_val);
                if (opt && *opt) {
                    LOG_INFO("[UnloadModel] Releasing raw model context\n");
                    free_sd_ctx(*opt);
                }
            }
        }
        return sd_error_t::OK;
    }
};
REGISTER_NODE("UnloadModel", UnloadModelNode);

void init_model_loader_nodes() {
    // 空函数，确保本翻译单元被链接
}

} // namespace sdengine
