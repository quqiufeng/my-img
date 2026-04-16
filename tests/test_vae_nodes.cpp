// ============================================================================
// tests/test_vae_nodes.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/node.h"
#include "core/sd_ptr.h"
#include "stable-diffusion.h"

using namespace sdengine;

TEST_CASE("VAEEncode rejects missing inputs", "[vae_nodes]") {
    auto node = NodeRegistry::instance().create("VAEEncode");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("VAEEncode rejects null image data", "[vae_nodes]") {
    auto node = NodeRegistry::instance().create("VAEEncode");
    REQUIRE(node != nullptr);

    sd_image_t image{};
    image.width = 4;
    image.height = 4;
    image.channel = 3;
    image.data = nullptr;

    NodeInputs inputs;
    inputs["pixels"] = image;
    sd_ctx_t* null_vae = nullptr;
    inputs["vae"] = null_vae;

    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("VAEDecode rejects missing inputs", "[vae_nodes]") {
    auto node = NodeRegistry::instance().create("VAEDecode");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("VAEDecode rejects null latent", "[vae_nodes]") {
    auto node = NodeRegistry::instance().create("VAEDecode");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    inputs["samples"] = LatentPtr{}; // null
    sd_ctx_t* null_vae = nullptr;
    inputs["vae"] = null_vae;

    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}
