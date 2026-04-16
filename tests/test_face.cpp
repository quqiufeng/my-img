// ============================================================================
// tests/test_face.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/node.h"
#include "core/sd_ptr.h"
#include "stable-diffusion.h"

#ifdef HAS_ONNXRUNTIME
#include "face/face_restore.hpp"
#include "face/face_swap.hpp"
#endif

using namespace sdengine;

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
    auto detect_loader = NodeRegistry::instance().create("FaceDetectModelLoader");
    REQUIRE(detect_loader != nullptr);
    NodeInputs detect_loader_inputs;
    detect_loader_inputs["model_path"] = std::string("/home/dministrator/models/yunet_320_320.onnx");
    NodeOutputs detect_loader_outputs;
    sd_error_t err = detect_loader->execute(detect_loader_inputs, detect_loader_outputs);
    REQUIRE(is_ok(err));

    auto restore_loader = NodeRegistry::instance().create("FaceRestoreModelLoader");
    REQUIRE(restore_loader != nullptr);
    NodeInputs restore_loader_inputs;
    restore_loader_inputs["model_path"] = std::string("/home/dministrator/models/GFPGANv1.4.onnx");
    restore_loader_inputs["model_type"] = std::string("gfpgan");
    NodeOutputs restore_loader_outputs;
    err = restore_loader->execute(restore_loader_inputs, restore_loader_outputs);
    REQUIRE(is_ok(err));

    auto load_image = NodeRegistry::instance().create("LoadImage");
    REQUIRE(load_image != nullptr);
    NodeInputs load_inputs;
    load_inputs["image"] = std::string("/home/dministrator/models/test_face_pil.png");
    NodeOutputs load_outputs;
    err = load_image->execute(load_inputs, load_outputs);
    REQUIRE(is_ok(err));

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
    auto detect_loader = NodeRegistry::instance().create("FaceDetectModelLoader");
    REQUIRE(detect_loader != nullptr);
    NodeInputs detect_loader_inputs;
    detect_loader_inputs["model_path"] = std::string("/home/dministrator/models/yunet_320_320.onnx");
    NodeOutputs detect_loader_outputs;
    sd_error_t err = detect_loader->execute(detect_loader_inputs, detect_loader_outputs);
    REQUIRE(is_ok(err));

    auto restore_loader = NodeRegistry::instance().create("FaceRestoreModelLoader");
    REQUIRE(restore_loader != nullptr);
    NodeInputs restore_loader_inputs;
    restore_loader_inputs["model_path"] = std::string("/home/dministrator/models/codeformer.onnx");
    restore_loader_inputs["model_type"] = std::string("codeformer");
    NodeOutputs restore_loader_outputs;
    err = restore_loader->execute(restore_loader_inputs, restore_loader_outputs);
    REQUIRE(is_ok(err));

    auto load_image = NodeRegistry::instance().create("LoadImage");
    REQUIRE(load_image != nullptr);
    NodeInputs load_inputs;
    load_inputs["image"] = std::string("/home/dministrator/models/test_face_pil.png");
    NodeOutputs load_outputs;
    err = load_image->execute(load_inputs, load_outputs);
    REQUIRE(is_ok(err));

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

    std::vector<uint8_t> dummy_img(512 * 512 * 3, 128);
    auto result = restorer.restore(dummy_img.data(), 0.5f);
    REQUIRE(result.success);
    REQUIRE(result.restored_rgb.size() == 512 * 512 * 3);
}

TEST_CASE("FaceRestorer direct inference with CodeFormer", "[face]") {
    sdengine::face::FaceRestorer restorer;
    REQUIRE(restorer.load("/home/dministrator/models/codeformer.onnx", sdengine::face::RestoreModelType::CODEFORMER));

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

    std::vector<uint8_t> target_img(128 * 128 * 3, 150);
    std::vector<uint8_t> source_img(128 * 128 * 3, 180);

    auto result = swapper.swap(target_img.data(), source_img.data());
    REQUIRE(result.success);
    REQUIRE(result.swapped_rgb.size() == 128 * 128 * 3);
}

#endif
