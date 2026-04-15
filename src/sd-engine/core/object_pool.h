// ============================================================================
// sd-engine/core/object_pool.h
// ============================================================================
/// @file object_pool.h
/// @brief 通用线程安全对象池模板
///
/// 设计目标：
/// - 减少频繁分配/释放的开销
/// - 线程安全
/// - 可设置最大池大小，防止无限增长
// ============================================================================

#pragma once

#include <vector>
#include <mutex>
#include <memory>
#include <functional>
#include <algorithm>

namespace sdengine {

/// @brief 通用线程安全对象池模板
/// @tparam T 池化对象类型（必须是完整类型）
///
/// 通过复用已分配的对象实例，减少 new/delete 开销。
/// 支持自定义创建器和重置器，以及最大池容量限制。
template <typename T>
class ObjectPool {
public:
    using Creator = std::function<T*()>;   ///< 对象创建函数类型
    using Resetter = std::function<void(T*)>; ///< 对象重置函数类型

    /// @brief 构造函数
    /// @param creator  对象创建器（为空时使用默认 new T()）
    /// @param resetter 对象重置器（在归还池前调用）
    /// @param max_size 池的最大容量
    ObjectPool(Creator creator, Resetter resetter, size_t max_size = 16)
        : creator_(creator), resetter_(resetter), max_size_(max_size) {}

    /// @brief 析构函数，释放池中所有对象
    ~ObjectPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (T* obj : pool_) {
            delete obj;
        }
        pool_.clear();
    }

    /// @brief 获取一个对象（优先从池中取，否则新建）
    T* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            T* obj = pool_.back();
            pool_.pop_back();
            return obj;
        }
        return creator_ ? creator_() : new T();
    }

    /// @brief 归还一个对象到池中
    /// @param obj 要归还的对象指针
    void release(T* obj) {
        if (!obj) return;
        if (resetter_) {
            resetter_(obj);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.size() < max_size_) {
            pool_.push_back(obj);
        } else {
            delete obj;
        }
    }

    /// @brief 预分配一定数量的对象
    /// @param count 预分配数量
    void reserve(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        while (pool_.size() < count && pool_.size() < max_size_) {
            T* obj = creator_ ? creator_() : new T();
            pool_.push_back(obj);
        }
    }

    /// @brief 获取当前池中可用对象数量
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

    /// @brief 获取池的最大容量
    size_t max_size() const {
        return max_size_;
    }

    /// @brief 动态调整最大容量
    void set_max_size(size_t max_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_size_ = max_size;
        while (pool_.size() > max_size_) {
            delete pool_.back();
            pool_.pop_back();
        }
    }

    /// @brief 清空池中所有对象
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (T* obj : pool_) {
            delete obj;
        }
        pool_.clear();
    }

private:
    Creator creator_;
    Resetter resetter_;
    size_t max_size_;
    std::vector<T*> pool_;
    mutable std::mutex mutex_;
};

} // namespace sdengine
