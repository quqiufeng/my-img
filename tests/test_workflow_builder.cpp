// ============================================================================
// tests/test_workflow_builder.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/workflow.h"
#include "core/workflow_builder.h"

using namespace sdengine;

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


TEST_CASE("WorkflowBuilder empty builder returns valid JSON", "[workflow_builder]") {
    WorkflowBuilder builder;
    std::string json_str = builder.to_json_string();
    REQUIRE(json_str == "{}");
}


TEST_CASE("WorkflowBuilder clear resets state", "[workflow_builder]") {
    WorkflowBuilder builder;
    std::string id1 = builder.add_checkpoint_loader("a.safetensors");
    REQUIRE(id1 == "1");

    builder.clear();
    std::string id2 = builder.add_checkpoint_loader("b.safetensors");
    REQUIRE(id2 == "1");

    std::string json_str = builder.to_json_string();
    REQUIRE(json_str.find("a.safetensors") == std::string::npos);
    REQUIRE(json_str.find("b.safetensors") != std::string::npos);
}


TEST_CASE("WorkflowBuilder sequential IDs", "[workflow_builder]") {
    WorkflowBuilder builder;
    REQUIRE(builder.add_checkpoint_loader("1.safetensors") == "1");
    REQUIRE(builder.add_empty_latent(64, 64) == "2");
    REQUIRE(builder.add_clip_encode("text", "1") == "3");
}


TEST_CASE("WorkflowBuilder make_link format", "[workflow_builder]") {
    WorkflowBuilder builder;
    json link = builder.make_link("42", 3);
    REQUIRE(link.is_array());
    REQUIRE(link.size() == 2);
    REQUIRE(link[0] == "42");
    REQUIRE(link[1] == 3);
}


TEST_CASE("WorkflowBuilder add_lora_stack with empty vector", "[workflow_builder]") {
    WorkflowBuilder builder;
    std::string id = builder.add_lora_stack({});
    REQUIRE(!id.empty());

    std::string json_str = builder.to_json_string();
    REQUIRE(json_str.find("LoRAStack") != std::string::npos);
}


TEST_CASE("WorkflowBuilder save_to_file rejects invalid path", "[workflow_builder]") {
    WorkflowBuilder builder;
    builder.add_checkpoint_loader("model.safetensors");
    bool ok = builder.save_to_file("/nonexistent_dir_xyz/subdir/workflow.json");
    REQUIRE(!ok);
}


TEST_CASE("Txt2ImgBuilder produces valid workflow", "[workflow_builder]") {
    std::string json_str = Txt2ImgBuilder::build("model.safetensors", "a cat", "blurry", 512, 512, 42, 25, 7.0f);
    REQUIRE(!json_str.empty());

    Workflow wf;
    REQUIRE(wf.load_from_string(json_str));
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));
}


TEST_CASE("Img2ImgBuilder produces valid workflow", "[workflow_builder]") {
    std::string json_str =
        Img2ImgBuilder::build("model.safetensors", "/tmp/input.png", "a cat", "blurry", 0.75f, 42, 25, 7.0f);
    REQUIRE(!json_str.empty());

    Workflow wf;
    REQUIRE(wf.load_from_string(json_str));
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));
}


TEST_CASE("ImageProcessBuilder with no transform", "[workflow_builder]") {
    std::string json_str = ImageProcessBuilder::build("/tmp/input.png", 0, 0, -1, -1, -1, -1, "out");
    REQUIRE(!json_str.empty());

    Workflow wf;
    REQUIRE(wf.load_from_string(json_str));
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));
}


TEST_CASE("ImageProcessBuilder with scale and crop", "[workflow_builder]") {
    std::string json_str = ImageProcessBuilder::build("/tmp/input.png", 1024, 1024, 100, 100, 512, 512, "cropped");
    REQUIRE(!json_str.empty());

    Workflow wf;
    REQUIRE(wf.load_from_string(json_str));
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));
}


TEST_CASE("DeepHiresBuilder produces valid workflow", "[workflow_builder]") {
    std::string json_str =
        DeepHiresBuilder::build("model.safetensors", "masterpiece", "lowres", 1024, 1024, 0, 30, 7.0f, "", 1.0f);
    REQUIRE(!json_str.empty());

    Workflow wf;
    REQUIRE(wf.load_from_string(json_str));
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));
}


TEST_CASE("IPAdapterTxt2ImgBuilder produces valid workflow", "[workflow_builder]") {
    std::string json_str = IPAdapterTxt2ImgBuilder::build("model.safetensors", "a cat", "blurry", "/tmp/ipadapter.bin",
                                                          "/tmp/ref.png", 0.8f);
    REQUIRE(!json_str.empty());

    Workflow wf;
    REQUIRE(wf.load_from_string(json_str));
    std::string error_msg;
    REQUIRE(wf.validate(error_msg));
}


