// ============================================================================
// tests/test_nodes_registry.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/node.h"
#include "stable-diffusion.h"

using namespace sdengine;

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
