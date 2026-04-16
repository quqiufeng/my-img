// ============================================================================
// sd-engine/core/node.h
// ============================================================================
/// @file node.h
/// @brief 节点基类定义和节点注册表
///
/// 提供 sd-engine 中所有工作流节点的抽象基类、端口定义、错误码以及
/// 节点工厂（NodeRegistry）机制。
// ============================================================================

#pragma once

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sdengine {

/// @brief 执行错误码
enum class sd_error_t {
    OK = 0,                      ///< 成功
    ERROR_INVALID_INPUT = 1,     ///< 输入参数无效
    ERROR_EXECUTION_FAILED = 2,  ///< 执行失败
    ERROR_MEMORY_ALLOCATION = 3, ///< 内存分配失败
    ERROR_MODEL_LOADING = 4,     ///< 模型加载失败
    ERROR_ENCODING_FAILED = 5,   ///< 文本编码失败
    ERROR_SAMPLING_FAILED = 6,   ///< 采样失败
    ERROR_DECODING_FAILED = 7,   ///< VAE 解码失败
    ERROR_FILE_IO = 8,           ///< 文件 I/O 错误
    ERROR_MISSING_INPUT = 9,     ///< 缺少必要输入
    ERROR_UNKNOWN = 99           ///< 未知错误
};

/// @brief 判断错误码是否表示成功
inline bool is_ok(sd_error_t err) {
    return err == sd_error_t::OK;
}

/// @brief 判断错误码是否表示失败
inline bool is_error(sd_error_t err) {
    return err != sd_error_t::OK;
}

/// @brief 端口定义（输入或输出）
struct PortDef {
    std::string name;       ///< 端口名称
    std::string type;       ///< 端口类型（如 LATENT, IMAGE, CONDITIONING）
    bool required = true;   ///< 是否为必填端口
    std::any default_value; ///< 默认值（可选）
};

/// @brief 节点输入数据集合
using NodeInputs = std::unordered_map<std::string, std::any>;

/// @brief 节点输出数据集合
using NodeOutputs = std::unordered_map<std::string, std::any>;

/// @brief 节点之间的连接关系
struct Link {
    std::string src_node_id; ///< 源节点 ID
    int src_slot = 0;        ///< 源节点输出槽位索引
    std::string dst_node_id; ///< 目标节点 ID
    int dst_slot = 0;        ///< 目标节点输入槽位索引
};

/// @brief 工作流节点基类
///
/// 所有 sd-engine 节点必须继承此类，并实现 get_class_type、get_inputs、
/// get_outputs 和 execute 方法。
class Node {
  public:
    virtual ~Node() = default;

    /// @brief 设置节点 ID
    void set_id(const std::string& id) {
        id_ = id;
    }

    /// @brief 获取节点 ID
    std::string get_id() const {
        return id_;
    }

    /// @brief 返回节点的类类型名称（如 "KSampler"）
    virtual std::string get_class_type() const = 0;

    /// @brief 返回节点所属分类
    virtual std::string get_category() const {
        return "general";
    }

    /// @brief 返回输入端口定义列表
    virtual std::vector<PortDef> get_inputs() const = 0;

    /// @brief 返回输出端口定义列表
    virtual std::vector<PortDef> get_outputs() const = 0;

    /// @brief 执行节点逻辑
    /// @param inputs  输入数据
    /// @param outputs 输出数据（由节点填充）
    /// @return 执行结果错误码
    virtual sd_error_t execute(const NodeInputs& inputs, NodeOutputs& outputs) = 0;

    /// @brief 根据输入计算缓存哈希
    /// @param inputs 当前输入数据
    /// @return 哈希字符串，用于 ExecutionCache
    virtual std::string compute_hash(const NodeInputs& inputs) const;

  protected:
    std::string id_; ///< 节点唯一标识
};

/// @brief 节点创建函数类型
using NodeCreator = std::function<std::unique_ptr<Node>()>;

/// @brief 节点注册表（单例工厂）
///
/// 维护所有已注册节点类型的创建器，支持运行时根据 class_type 创建节点实例。
class NodeRegistry {
  public:
    /// @brief 获取全局单例实例
    static NodeRegistry& instance();

    /// @brief 注册一个节点类型
    void register_node(const std::string& class_type, NodeCreator creator);

    /// @brief 根据类型名称创建节点实例
    std::unique_ptr<Node> create(const std::string& class_type) const;

    /// @brief 获取所有已注册的节点类型名称列表
    std::vector<std::string> get_supported_nodes() const;

    /// @brief 检查是否已注册指定类型
    bool has_node(const std::string& class_type) const;

  private:
    std::unordered_map<std::string, NodeCreator> creators_;
};

/// @brief 节点自动注册宏
///
/// 在节点实现文件中使用此宏将节点类注册到 NodeRegistry。
/// @code
/// REGISTER_NODE("KSampler", KSamplerNode)
/// @endcode
#define REGISTER_NODE(class_type, node_class)                                                                          \
    static bool _registered_##node_class = []() {                                                                      \
        ::sdengine::NodeRegistry::instance().register_node(class_type,                                                 \
                                                           []() { return std::make_unique<node_class>(); });           \
        return true;                                                                                                   \
    }();

} // namespace sdengine
