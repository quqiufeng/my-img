// ============================================================================
// tests/test_conditioning_nodes.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/node.h"
#include "core/sd_ptr.h"
#include "nodes/node_utils.h"

using namespace sdengine;

TEST_CASE("ConditioningCombine rejects missing inputs", "[conditioning_nodes]") {
    auto node = NodeRegistry::instance().create("ConditioningCombine");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("ConditioningConcat rejects missing inputs", "[conditioning_nodes]") {
    auto node = NodeRegistry::instance().create("ConditioningConcat");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("ConditioningAverage rejects missing inputs", "[conditioning_nodes]") {
    auto node = NodeRegistry::instance().create("ConditioningAverage");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("ControlNetApply rejects missing inputs", "[conditioning_nodes]") {
    auto node = NodeRegistry::instance().create("ControlNetApply");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("ControlNetApply rejects null conditioning", "[conditioning_nodes]") {
    auto node = NodeRegistry::instance().create("ControlNetApply");
    REQUIRE(node != nullptr);

    auto buffer = make_malloc_buffer(4 * 4 * 3);
    REQUIRE(buffer != nullptr);
    memset(buffer.get(), 128, 4 * 4 * 3);
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 4;
    img->height = 4;
    img->channel = 3;
    img->data = buffer.release();

    NodeInputs inputs;
    inputs["conditioning"] = ConditioningPtr{}; // null
    inputs["image"] = make_image_ptr(img);
    inputs["strength"] = 1.0f;

    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("IPAdapterApply rejects missing inputs", "[conditioning_nodes]") {
    auto node = NodeRegistry::instance().create("IPAdapterApply");
    REQUIRE(node != nullptr);

    NodeInputs inputs;
    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}

TEST_CASE("IPAdapterApply rejects null conditioning", "[conditioning_nodes]") {
    auto node = NodeRegistry::instance().create("IPAdapterApply");
    REQUIRE(node != nullptr);

    auto buffer = make_malloc_buffer(4 * 4 * 3);
    REQUIRE(buffer != nullptr);
    memset(buffer.get(), 128, 4 * 4 * 3);
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 4;
    img->height = 4;
    img->channel = 3;
    img->data = buffer.release();

    IPAdapterInfo info;
    info.path = "/tmp/dummy.bin";
    info.strength = 1.0f;

    NodeInputs inputs;
    inputs["conditioning"] = ConditioningPtr{}; // null
    inputs["ipadapter"] = info;
    inputs["image"] = make_image_ptr(img);
    inputs["strength"] = 1.0f;

    NodeOutputs outputs;
    sd_error_t err = node->execute(inputs, outputs);
    REQUIRE(is_error(err));
}
