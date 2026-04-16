// ============================================================================
// tests/test_image_nodes.cpp
// ============================================================================

#include "catch_amalgamated.hpp"
#include "core/node.h"
#include "core/sd_ptr.h"
#include "stable-diffusion.h"

using namespace sdengine;

TEST_CASE("ImageScale node resizes correctly", "[image_nodes]") {
    auto loader = NodeRegistry::instance().create("LoadImage");
    REQUIRE(loader != nullptr);

    // 创建一个 4x4 RGB 测试图像
    auto buffer = make_malloc_buffer(4 * 4 * 3);
    REQUIRE(buffer != nullptr);
    for (int i = 0; i < 4 * 4 * 3; i++) {
        buffer[i] = (uint8_t)(i % 256);
    }
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 4;
    img->height = 4;
    img->channel = 3;
    img->data = buffer.release();

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
    auto buffer = make_malloc_buffer(8 * 8 * 3);
    REQUIRE(buffer != nullptr);
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            buffer[(y * 8 + x) * 3 + 0] = (uint8_t)x;
            buffer[(y * 8 + x) * 3 + 1] = (uint8_t)y;
            buffer[(y * 8 + x) * 3 + 2] = 0;
        }
    }
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 8;
    img->height = 8;
    img->channel = 3;
    img->data = buffer.release();

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
    auto buffer = make_malloc_buffer(4 * 4 * 3);
    REQUIRE(buffer != nullptr);
    memset(buffer.get(), 0, 4 * 4 * 3);
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 4;
    img->height = 4;
    img->channel = 3;
    img->data = buffer.release();

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

TEST_CASE("ImageInvert node inverts colors", "[image_nodes]") {
    auto buffer = make_malloc_buffer(2 * 2 * 3);
    REQUIRE(buffer != nullptr);
    buffer[0] = 0;
    buffer[1] = 128;
    buffer[2] = 255;
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 2;
    img->height = 2;
    img->channel = 3;
    img->data = buffer.release();

    auto inverter = NodeRegistry::instance().create("ImageInvert");
    REQUIRE(inverter != nullptr);

    NodeInputs inputs;
    inputs["image"] = make_image_ptr(img);
    NodeOutputs outputs;
    sd_error_t err = inverter->execute(inputs, outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->data[0] == 255);
    REQUIRE(result->data[1] == 127);
    REQUIRE(result->data[2] == 0);
}

TEST_CASE("ImageGrayscale node converts to grayscale", "[image_nodes]") {
    auto buffer = make_malloc_buffer(2 * 2 * 3);
    REQUIRE(buffer != nullptr);
    buffer[0] = 255;
    buffer[1] = 255;
    buffer[2] = 255;
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 2;
    img->height = 2;
    img->channel = 3;
    img->data = buffer.release();

    auto gray = NodeRegistry::instance().create("ImageGrayscale");
    REQUIRE(gray != nullptr);

    NodeInputs inputs;
    inputs["image"] = make_image_ptr(img);
    NodeOutputs outputs;
    sd_error_t err = gray->execute(inputs, outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->channel == 1);
    REQUIRE(result->data[0] == 255);
}

TEST_CASE("ImageThreshold node applies threshold", "[image_nodes]") {
    auto buffer = make_malloc_buffer(2 * 2 * 3);
    REQUIRE(buffer != nullptr);
    buffer[0] = 100;
    buffer[1] = 150;
    buffer[2] = 200;
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 2;
    img->height = 2;
    img->channel = 3;
    img->data = buffer.release();

    auto thresh = NodeRegistry::instance().create("ImageThreshold");
    REQUIRE(thresh != nullptr);

    NodeInputs inputs;
    inputs["image"] = make_image_ptr(img);
    inputs["threshold"] = 150;
    NodeOutputs outputs;
    sd_error_t err = thresh->execute(inputs, outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->data[0] == 0);   // 100 < 150
    REQUIRE(result->data[1] == 255); // 150 >= 150
    REQUIRE(result->data[2] == 255); // 200 >= 150
}

TEST_CASE("ImageBlur node blurs image", "[image_nodes]") {
    auto buffer = make_malloc_buffer(4 * 4 * 3);
    REQUIRE(buffer != nullptr);
    memset(buffer.get(), 128, 4 * 4 * 3);
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 4;
    img->height = 4;
    img->channel = 3;
    img->data = buffer.release();

    auto blur = NodeRegistry::instance().create("ImageBlur");
    REQUIRE(blur != nullptr);

    NodeInputs inputs;
    inputs["image"] = make_image_ptr(img);
    inputs["radius"] = 1;
    NodeOutputs outputs;
    sd_error_t err = blur->execute(inputs, outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->width == 4);
    REQUIRE(result->height == 4);
    REQUIRE(result->channel == 3);
}

TEST_CASE("ImageColorAdjust node adjusts brightness", "[image_nodes]") {
    auto buffer = make_malloc_buffer(2 * 2 * 3);
    REQUIRE(buffer != nullptr);
    buffer[0] = 128;
    buffer[1] = 128;
    buffer[2] = 128;
    sd_image_t* img = acquire_image();
    REQUIRE(img != nullptr);
    img->width = 2;
    img->height = 2;
    img->channel = 3;
    img->data = buffer.release();

    auto adjust = NodeRegistry::instance().create("ImageColorAdjust");
    REQUIRE(adjust != nullptr);

    NodeInputs inputs;
    inputs["image"] = make_image_ptr(img);
    inputs["brightness"] = 2.0f;
    inputs["contrast"] = 1.0f;
    inputs["saturation"] = 1.0f;
    NodeOutputs outputs;
    sd_error_t err = adjust->execute(inputs, outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->data[0] == 255); // 128 * 2.0 clamped to 255
}

TEST_CASE("ImageBlend node blends two images", "[image_nodes]") {
    auto buf1 = make_malloc_buffer(2 * 2 * 3);
    auto buf2 = make_malloc_buffer(2 * 2 * 3);
    REQUIRE(buf1 != nullptr);
    REQUIRE(buf2 != nullptr);
    memset(buf1.get(), 255, 2 * 2 * 3);
    memset(buf2.get(), 0, 2 * 2 * 3);

    sd_image_t* img1 = acquire_image();
    sd_image_t* img2 = acquire_image();
    REQUIRE(img1 != nullptr);
    REQUIRE(img2 != nullptr);
    img1->width = 2;
    img1->height = 2;
    img1->channel = 3;
    img1->data = buf1.release();
    img2->width = 2;
    img2->height = 2;
    img2->channel = 3;
    img2->data = buf2.release();

    auto blender = NodeRegistry::instance().create("ImageBlend");
    REQUIRE(blender != nullptr);

    NodeInputs inputs;
    inputs["image1"] = make_image_ptr(img1);
    inputs["image2"] = make_image_ptr(img2);
    inputs["blend_factor"] = 0.5f;
    inputs["blend_mode"] = std::string("normal");
    NodeOutputs outputs;
    sd_error_t err = blender->execute(inputs, outputs);
    REQUIRE(is_ok(err));

    ImagePtr result = std::any_cast<ImagePtr>(outputs["IMAGE"]);
    REQUIRE(result != nullptr);
    REQUIRE(result->data[0] == 128); // 255 * 0.5 + 0 * 0.5
}
