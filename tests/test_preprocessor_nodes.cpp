// ============================================================================
// tests/test_preprocessor_nodes.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/node.h"
#include "core/sd_ptr.h"
#include "nodes/node_utils.h"

using namespace sdengine;

TEST_CASE("ImageRemoveBackground rejects missing inputs", "[preprocessor_nodes]") {
    auto node = NodeRegistry::instance().create("ImageRemoveBackground");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("ImageRemoveBackground rejects null image", "[preprocessor_nodes]") {
    auto node = NodeRegistry::instance().create("ImageRemoveBackground");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    inputs["image"] = ImagePtr{};                    // null
    inputs["model"] = std::shared_ptr<RemBGModel>{}; // null

    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("LineArtPreprocessor rejects missing inputs", "[preprocessor_nodes]") {
    auto node = NodeRegistry::instance().create("LineArtPreprocessor");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("LineArtPreprocessor rejects null image", "[preprocessor_nodes]") {
    auto node = NodeRegistry::instance().create("LineArtPreprocessor");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    inputs["image"] = ImagePtr{};                                     // null
    inputs["lineart_model"] = std::shared_ptr<LineArtPreprocessor>{}; // null

    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}
