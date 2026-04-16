// ============================================================================
// tests/test_hash.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/node.h"
#include "stable-diffusion.h"

using namespace sdengine;

class HashTestNode : public Node {
  public:
    std::string get_class_type() const override {
        return "HashTest";
    }
    std::string get_category() const override {
        return "test";
    }
    std::vector<PortDef> get_inputs() const override {
        return {};
    }
    std::vector<PortDef> get_outputs() const override {
        return {{"out", "INT"}};
    }
    sd_error_t execute(const NodeInputs&, NodeOutputs&) override {
        return sd_error_t::OK;
    }
};

TEST_CASE("Node::compute_hash is deterministic for basic types", "[hash]") {
    HashTestNode node;
    NodeInputs inputs;
    inputs["i"] = 42;
    inputs["f"] = 3.14f;
    inputs["d"] = 2.718;
    inputs["s"] = std::string("hello");
    inputs["b"] = true;

    std::string h1 = node.compute_hash(inputs);
    std::string h2 = node.compute_hash(inputs);
    REQUIRE(h1 == h2);
    REQUIRE(h1.find("HashTest") != std::string::npos);
}

TEST_CASE("Node::compute_hash distinguishes different float values", "[hash]") {
    HashTestNode node;
    NodeInputs a, b;
    a["v"] = 1.0f;
    b["v"] = 1.0f + std::numeric_limits<float>::epsilon();

    REQUIRE(node.compute_hash(a) != node.compute_hash(b));
}

TEST_CASE("Node::compute_hash unifies positive and negative zero", "[hash]") {
    HashTestNode node;
    NodeInputs a, b;
    float pzero = 0.0f;
    float nzero = -0.0f;
    a["v"] = pzero;
    b["v"] = nzero;

    REQUIRE(node.compute_hash(a) == node.compute_hash(b));
}

TEST_CASE("Node::compute_hash handles image checksum sampling", "[hash]") {
    HashTestNode node;
    NodeInputs inputs;

    // 构造 8x8 RGB 测试图像
    sd_image_t img = {};
    img.width = 8;
    img.height = 8;
    img.channel = 3;
    std::vector<uint8_t> data(8 * 8 * 3, 128);
    data[0] = 255; // 改变一个像素
    img.data = data.data();

    inputs["image"] = img;
    std::string h = node.compute_hash(inputs);
    REQUIRE(h.find("[sdimg:8x8x3:") != std::string::npos);

    // 改变未采样到的像素（假设 stride > 1）不应影响哈希
    data[1] = 0;
    std::string h2 = node.compute_hash(inputs);
    // 由于 stride = max(1, 192/64) = 3，改变 data[1] 不影响 checksum
    REQUIRE(h == h2);
}
