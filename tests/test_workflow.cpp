// ============================================================================
// tests/test_workflow.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/workflow.h"
#include "core/executor.h"
#include "core/workflow_builder.h"

using namespace sdengine;

TEST_CASE("Workflow loads from string", "[workflow]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 5}}
    })";
    
    bool ok = wf.load_from_string(json_str);
    REQUIRE(ok);
    
    auto nodes = wf.get_all_nodes();
    REQUIRE(nodes.size() == 1);
}

TEST_CASE("Workflow topological sort", "[workflow]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 5}},
        "2": {"class_type": "ConstantInt", "inputs": {"value": 3}},
        "3": {"class_type": "AddInt", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
        "4": {"class_type": "PrintInt", "inputs": {"value": ["3", 0]}}
    })";
    
    REQUIRE(wf.load_from_string(json_str));
    
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));
    
    auto order = wf.topological_sort();
    REQUIRE(order.size() == 4);
}

TEST_CASE("Workflow validation detects cycles", "[workflow]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "AddInt", "inputs": {"a": ["2", 0], "b": ["3", 0]}},
        "2": {"class_type": "AddInt", "inputs": {"a": ["1", 0], "b": ["3", 0]}},
        "3": {"class_type": "ConstantInt", "inputs": {"value": 1}}
    })";
    
    REQUIRE(wf.load_from_string(json_str));
    
    std::string error_msg;
    REQUIRE(!wf.validate(error_msg));
    REQUIRE(error_msg.find("cycles") != std::string::npos);
}

TEST_CASE("Executor runs simple workflow", "[executor]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 7}},
        "2": {"class_type": "ConstantInt", "inputs": {"value": 8}},
        "3": {"class_type": "AddInt", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
        "4": {"class_type": "PrintInt", "inputs": {"value": ["3", 0]}}
    })";
    
    REQUIRE(wf.load_from_string(json_str));
    
    DAGExecutor executor;
    ExecutionConfig config;
    config.use_cache = true;
    config.verbose = false;
    
    sd_error_t err = executor.execute(&wf, config);
    REQUIRE(is_ok(err));
}

TEST_CASE("Node registry contains expected nodes", "[nodes]") {
    auto nodes = NodeRegistry::instance().get_supported_nodes();
    REQUIRE(NodeRegistry::instance().has_node("ConstantInt"));
    REQUIRE(NodeRegistry::instance().has_node("AddInt"));
    REQUIRE(NodeRegistry::instance().has_node("KSampler"));
    REQUIRE(NodeRegistry::instance().has_node("CLIPTextEncode"));
    REQUIRE(NodeRegistry::instance().has_node("VAEDecode"));
    REQUIRE(NodeRegistry::instance().has_node("DeepHighResFix"));
    REQUIRE(NodeRegistry::instance().has_node("LoRAStack"));
    REQUIRE(NodeRegistry::instance().has_node("UpscaleModelLoader"));
    REQUIRE(NodeRegistry::instance().has_node("ImageUpscaleWithModel"));
    REQUIRE(NodeRegistry::instance().has_node("ControlNetLoader"));
    REQUIRE(NodeRegistry::instance().has_node("ControlNetApply"));
    REQUIRE(NodeRegistry::instance().has_node("CannyEdgePreprocessor"));
    REQUIRE(NodeRegistry::instance().has_node("LoadImageMask"));
    REQUIRE(NodeRegistry::instance().has_node("IPAdapterLoader"));
    REQUIRE(NodeRegistry::instance().has_node("IPAdapterApply"));
    REQUIRE(NodeRegistry::instance().has_node("ImageBlend"));
    REQUIRE(NodeRegistry::instance().has_node("ImageCompositeMasked"));
    REQUIRE(NodeRegistry::instance().has_node("ConditioningCombine"));
    REQUIRE(NodeRegistry::instance().has_node("ConditioningConcat"));
    REQUIRE(NodeRegistry::instance().has_node("ConditioningAverage"));
    REQUIRE(NodeRegistry::instance().has_node("CLIPSetLastLayer"));
    REQUIRE(NodeRegistry::instance().has_node("CLIPVisionEncode"));
    REQUIRE(NodeRegistry::instance().has_node("ImageInvert"));
    REQUIRE(NodeRegistry::instance().has_node("ImageColorAdjust"));
    REQUIRE(NodeRegistry::instance().has_node("ImageBlur"));
    REQUIRE(NodeRegistry::instance().has_node("ImageGrayscale"));
    REQUIRE(NodeRegistry::instance().has_node("ImageThreshold"));
    REQUIRE(NodeRegistry::instance().has_node("RemBGModelLoader"));
    REQUIRE(NodeRegistry::instance().has_node("ImageRemoveBackground"));
}

TEST_CASE("ExecutionCache stores and retrieves", "[cache]") {
    ExecutionCache cache(1024 * 1024);
    
    NodeOutputs outputs;
    outputs["value"] = 42;
    
    cache.put("node1", "hash1", outputs);
    REQUIRE(cache.has("node1", "hash1"));
    
    auto retrieved = cache.get("node1", "hash1");
    REQUIRE(retrieved.count("value"));
    REQUIRE(std::any_cast<int>(retrieved["value"]) == 42);
}

TEST_CASE("Executor runs workflow in parallel", "[executor]") {
    Workflow wf;
    std::string json_str = R"({
        "1": {"class_type": "ConstantInt", "inputs": {"value": 1}},
        "2": {"class_type": "ConstantInt", "inputs": {"value": 2}},
        "3": {"class_type": "ConstantInt", "inputs": {"value": 3}},
        "4": {"class_type": "MultiplyInt", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
        "5": {"class_type": "MultiplyInt", "inputs": {"a": ["2", 0], "b": ["3", 0]}},
        "6": {"class_type": "AddInt", "inputs": {"a": ["4", 0], "b": ["5", 0]}},
        "7": {"class_type": "PrintInt", "inputs": {"value": ["6", 0]}}
    })";

    REQUIRE(wf.load_from_string(json_str));

    DAGExecutor executor;
    ExecutionConfig config;
    config.use_cache = true;
    config.verbose = false;
    config.max_threads = 4;  // 启用并行执行

    sd_error_t err = executor.execute(&wf, config);
    REQUIRE(is_ok(err));
}

TEST_CASE("WorkflowBuilder creates valid JSON", "[workflow_builder]") {
    WorkflowBuilder builder;
    std::string loader = builder.add_checkpoint_loader("model.safetensors");
    std::string positive = builder.add_clip_encode("a cat", loader);
    std::string negative = builder.add_clip_encode("", loader);
    std::string latent = builder.add_empty_latent(512, 512);
    std::string sampler = builder.add_ksampler(loader, positive, negative, latent, 0, 20, 7.5f);
    std::string decoded = builder.add_vae_decode(sampler, loader);
    builder.add_save_image(decoded, "output");
    
    std::string json_str = builder.to_json_string();
    REQUIRE(!json_str.empty());
    REQUIRE(json_str.find("CheckpointLoaderSimple") != std::string::npos);
    REQUIRE(json_str.find("CLIPTextEncode") != std::string::npos);
    REQUIRE(json_str.find("KSampler") != std::string::npos);
}
