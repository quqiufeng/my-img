// ============================================================================
// tests/test_workflow.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/workflow.h"
#include "core/executor.h"
#include "core/workflow_builder.h"
#include "core/cache.h"
#include "core/sd_ptr.h"
#include "stable-diffusion.h"

#ifdef HAS_ONNXRUNTIME
#include "face/face_restore.hpp"
#include "face/face_swap.hpp"
#endif

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
    REQUIRE(NodeRegistry::instance().has_node("FaceDetectModelLoader"));
    REQUIRE(NodeRegistry::instance().has_node("FaceDetect"));
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

#ifdef HAS_ONNXRUNTIME
TEST_CASE("FaceDetect nodes are registered", "[face]") {
    REQUIRE(NodeRegistry::instance().has_node("FaceDetectModelLoader"));
    REQUIRE(NodeRegistry::instance().has_node("FaceDetect"));
}

TEST_CASE("FaceRestore nodes are registered", "[face]") {
    REQUIRE(NodeRegistry::instance().has_node("FaceRestoreModelLoader"));
    REQUIRE(NodeRegistry::instance().has_node("FaceRestoreWithModel"));
}

TEST_CASE("FaceDetectModelLoader rejects empty path", "[face]") {
    auto node = NodeRegistry::instance().create("FaceDetectModelLoader");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    inputs["model_path"] = std::string("");
    NodeOutputs outputs;

    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("FaceRestoreModelLoader rejects empty path", "[face]") {
    auto node = NodeRegistry::instance().create("FaceRestoreModelLoader");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    inputs["model_path"] = std::string("");
    inputs["model_type"] = std::string("gfpgan");
    NodeOutputs outputs;

    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("FaceRestoreWithModel end-to-end with GFPGAN", "[face]") {
    // 1. Load face detector
    auto detect_loader = NodeRegistry::instance().create("FaceDetectModelLoader");
    REQUIRE(detect_loader != nullptr);
    NodeInputs detect_loader_inputs;
    detect_loader_inputs["model_path"] = std::string("/home/dministrator/models/yunet_320_320.onnx");
    NodeOutputs detect_loader_outputs;
    sd_error_t err = detect_loader->execute(detect_loader_inputs, detect_loader_outputs);
    REQUIRE(is_ok(err));

    // 2. Load GFPGAN restorer
    auto restore_loader = NodeRegistry::instance().create("FaceRestoreModelLoader");
    REQUIRE(restore_loader != nullptr);
    NodeInputs restore_loader_inputs;
    restore_loader_inputs["model_path"] = std::string("/home/dministrator/models/GFPGANv1.4.onnx");
    restore_loader_inputs["model_type"] = std::string("gfpgan");
    NodeOutputs restore_loader_outputs;
    err = restore_loader->execute(restore_loader_inputs, restore_loader_outputs);
    REQUIRE(is_ok(err));

    // 3. Load test image
    auto load_image = NodeRegistry::instance().create("LoadImage");
    REQUIRE(load_image != nullptr);
    NodeInputs load_inputs;
    load_inputs["image"] = std::string("/home/dministrator/models/test_face_pil.png");
    NodeOutputs load_outputs;
    err = load_image->execute(load_inputs, load_outputs);
    REQUIRE(is_ok(err));

    // 4. Run FaceRestoreWithModel
    auto restore_node = NodeRegistry::instance().create("FaceRestoreWithModel");
    REQUIRE(restore_node != nullptr);
    NodeInputs restore_inputs;
    restore_inputs["image"] = load_outputs["IMAGE"];
    restore_inputs["face_restore_model"] = restore_loader_outputs["FACE_RESTORE_MODEL"];
    restore_inputs["face_detect_model"] = detect_loader_outputs["FACE_DETECT_MODEL"];
    restore_inputs["codeformer_fidelity"] = 0.5f;
    NodeOutputs restore_outputs;
    err = restore_node->execute(restore_inputs, restore_outputs);
    REQUIRE(is_ok(err));
    REQUIRE(restore_outputs.count("IMAGE") > 0);
}

TEST_CASE("FaceRestoreWithModel end-to-end with CodeFormer", "[face]") {
    // 1. Load face detector
    auto detect_loader = NodeRegistry::instance().create("FaceDetectModelLoader");
    REQUIRE(detect_loader != nullptr);
    NodeInputs detect_loader_inputs;
    detect_loader_inputs["model_path"] = std::string("/home/dministrator/models/yunet_320_320.onnx");
    NodeOutputs detect_loader_outputs;
    sd_error_t err = detect_loader->execute(detect_loader_inputs, detect_loader_outputs);
    REQUIRE(is_ok(err));

    // 2. Load CodeFormer restorer
    auto restore_loader = NodeRegistry::instance().create("FaceRestoreModelLoader");
    REQUIRE(restore_loader != nullptr);
    NodeInputs restore_loader_inputs;
    restore_loader_inputs["model_path"] = std::string("/home/dministrator/models/codeformer.onnx");
    restore_loader_inputs["model_type"] = std::string("codeformer");
    NodeOutputs restore_loader_outputs;
    err = restore_loader->execute(restore_loader_inputs, restore_loader_outputs);
    REQUIRE(is_ok(err));

    // 3. Load test image
    auto load_image = NodeRegistry::instance().create("LoadImage");
    REQUIRE(load_image != nullptr);
    NodeInputs load_inputs;
    load_inputs["image"] = std::string("/home/dministrator/models/test_face_pil.png");
    NodeOutputs load_outputs;
    err = load_image->execute(load_inputs, load_outputs);
    REQUIRE(is_ok(err));

    // 4. Run FaceRestoreWithModel
    auto restore_node = NodeRegistry::instance().create("FaceRestoreWithModel");
    REQUIRE(restore_node != nullptr);
    NodeInputs restore_inputs;
    restore_inputs["image"] = load_outputs["IMAGE"];
    restore_inputs["face_restore_model"] = restore_loader_outputs["FACE_RESTORE_MODEL"];
    restore_inputs["face_detect_model"] = detect_loader_outputs["FACE_DETECT_MODEL"];
    restore_inputs["codeformer_fidelity"] = 0.8f;
    NodeOutputs restore_outputs;
    err = restore_node->execute(restore_inputs, restore_outputs);
    REQUIRE(is_ok(err));
    REQUIRE(restore_outputs.count("IMAGE") > 0);
}

TEST_CASE("FaceRestorer direct inference with GFPGAN", "[face]") {
    sdengine::face::FaceRestorer restorer;
    REQUIRE(restorer.load("/home/dministrator/models/GFPGANv1.4.onnx", sdengine::face::RestoreModelType::GFPGAN));

    // Create a dummy 512x512 RGB image
    std::vector<uint8_t> dummy_img(512 * 512 * 3, 128);
    auto result = restorer.restore(dummy_img.data(), 0.5f);
    REQUIRE(result.success);
    REQUIRE(result.restored_rgb.size() == 512 * 512 * 3);
}

TEST_CASE("FaceRestorer direct inference with CodeFormer", "[face]") {
    sdengine::face::FaceRestorer restorer;
    REQUIRE(restorer.load("/home/dministrator/models/codeformer.onnx", sdengine::face::RestoreModelType::CODEFORMER));

    // Create a dummy 512x512 RGB image
    std::vector<uint8_t> dummy_img(512 * 512 * 3, 128);
    auto result = restorer.restore(dummy_img.data(), 0.7f);
    REQUIRE(result.success);
    REQUIRE(result.restored_rgb.size() == 512 * 512 * 3);
}

TEST_CASE("FaceSwap nodes are registered", "[face]") {
    REQUIRE(NodeRegistry::instance().has_node("FaceSwapModelLoader"));
    REQUIRE(NodeRegistry::instance().has_node("FaceSwap"));
}

TEST_CASE("FaceSwapModelLoader rejects empty path", "[face]") {
    auto node = NodeRegistry::instance().create("FaceSwapModelLoader");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    inputs["inswapper_path"] = std::string("");
    inputs["arcface_path"] = std::string("/home/dministrator/models/arcface.onnx");
    NodeOutputs outputs;

    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("FaceSwapper direct inference", "[face]") {
    sdengine::face::FaceSwapper swapper;
    REQUIRE(swapper.load("/home/dministrator/models/inswapper_128.onnx", "/home/dministrator/models/arcface.onnx"));

    // Create dummy 128x128 RGB images
    std::vector<uint8_t> target_img(128 * 128 * 3, 150);
    std::vector<uint8_t> source_img(128 * 128 * 3, 180);

    auto result = swapper.swap(target_img.data(), source_img.data());
    REQUIRE(result.success);
    REQUIRE(result.swapped_rgb.size() == 128 * 128 * 3);
}
#endif

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

TEST_CASE("ExecutionCache basic operations", "[cache]") {
    ExecutionCache cache(1024 * 1024);

    NodeOutputs outputs;
    outputs["value"] = 42;

    REQUIRE(!cache.has("node1", "hash1"));
    cache.put("node1", "hash1", outputs);
    REQUIRE(cache.has("node1", "hash1"));
    REQUIRE(cache.size() == 1);

    auto got = cache.get("node1", "hash1");
    REQUIRE(got.count("value"));
    REQUIRE(std::any_cast<int>(got.at("value")) == 42);
}

TEST_CASE("ExecutionCache LRU eviction", "[cache]") {
    // 限制很小的缓存，只能容纳约 2 个条目
    ExecutionCache cache(3000);

    NodeOutputs out1, out2, out3;
    out1["v"] = 1;
    out2["v"] = 2;
    out3["v"] = 3;

    cache.put("n1", "h1", out1);
    cache.put("n2", "h2", out2);
    // 访问 n1，使其变为最近使用
    (void)cache.get("n1", "h1");

    cache.put("n3", "h3", out3);

    // n2 应该被淘汰（最久未使用）
    REQUIRE(cache.has("n1", "h1"));
    REQUIRE(!cache.has("n2", "h2"));
    REQUIRE(cache.has("n3", "h3"));
}

TEST_CASE("ExecutionCache clear", "[cache]") {
    ExecutionCache cache;
    NodeOutputs out;
    out["v"] = 1;
    cache.put("n", "h", out);
    REQUIRE(cache.size() == 1);

    cache.clear();
    REQUIRE(cache.size() == 0);
    REQUIRE(!cache.has("n", "h"));
}

TEST_CASE("ImageScale node resizes correctly", "[image_nodes]") {
    auto loader = NodeRegistry::instance().create("LoadImage");
    REQUIRE(loader != nullptr);

    // 创建一个 4x4 RGB 测试图像
    uint8_t* data = (uint8_t*)malloc(4 * 4 * 3);
    REQUIRE(data != nullptr);
    for (int i = 0; i < 4 * 4 * 3; i++) {
        data[i] = (uint8_t)(i % 256);
    }
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 4;
    img->height = 4;
    img->channel = 3;
    img->data = data;

    NodeInputs load_inputs;
    load_inputs["image"] = std::string("dummy.png");
    NodeOutputs load_outputs;
    // 直接构造 ImagePtr 绕过文件加载
    load_outputs["IMAGE"] = make_image_ptr(img);

    auto scaler = NodeRegistry::instance().create("ImageScale");
    REQUIRE(scaler != nullptr);

    NodeInputs scale_inputs;
    scale_inputs["image"] = load_outputs["IMAGE"];
    scale_inputs["width"] = 8;
    scale_inputs["height"] = 8;
    scale_inputs["method"] = std::string("bilinear");

    NodeOutputs scale_outputs;
    sd_error_t err = scaler->execute(scale_inputs, scale_outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(scale_outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->width == 8);
    REQUIRE(result->height == 8);
    REQUIRE(result->channel == 3);
}

TEST_CASE("ImageCrop node crops correctly", "[image_nodes]") {
    uint8_t* data = (uint8_t*)malloc(8 * 8 * 3);
    REQUIRE(data != nullptr);
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            data[(y * 8 + x) * 3 + 0] = (uint8_t)x;
            data[(y * 8 + x) * 3 + 1] = (uint8_t)y;
            data[(y * 8 + x) * 3 + 2] = 0;
        }
    }
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 8;
    img->height = 8;
    img->channel = 3;
    img->data = data;

    auto cropper = NodeRegistry::instance().create("ImageCrop");
    REQUIRE(cropper != nullptr);

    NodeInputs inputs;
    inputs["image"] = make_image_ptr(img);
    inputs["x"] = 2;
    inputs["y"] = 3;
    inputs["width"] = 4;
    inputs["height"] = 4;

    NodeOutputs outputs;
    sd_error_t err = cropper->execute(inputs, outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->width == 4);
    REQUIRE(result->height == 4);
    REQUIRE(result->channel == 3);

    // 验证左上角像素对应原图 (2,3)
    REQUIRE(result->data[0] == 2);
    REQUIRE(result->data[1] == 3);
}

TEST_CASE("ImageCrop node rejects invalid region", "[image_nodes]") {
    uint8_t* data = (uint8_t*)malloc(4 * 4 * 3);
    REQUIRE(data != nullptr);
    memset(data, 0, 4 * 4 * 3);
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 4;
    img->height = 4;
    img->channel = 3;
    img->data = data;

    auto cropper = NodeRegistry::instance().create("ImageCrop");
    REQUIRE(cropper != nullptr);

    NodeInputs inputs;
    inputs["image"] = make_image_ptr(img);
    inputs["x"] = 2;
    inputs["y"] = 2;
    inputs["width"] = 10; // 超出边界
    inputs["height"] = 10;

    NodeOutputs outputs;
    sd_error_t err = cropper->execute(inputs, outputs);
    REQUIRE(is_error(err));
}
