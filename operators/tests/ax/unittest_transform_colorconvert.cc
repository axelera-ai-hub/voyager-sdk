// Copyright Axelera AI, 2025
#include "unittest_ax_common.h"

namespace
{
TEST(Conversions, rgb2bgr)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "bgra" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);
  auto in_buf = std::vector<uint8_t>{ 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10,
    11, 0, 12, 13, 14, 0, 15, 16, 17, 0, 18, 19, 20, 0, 21, 22, 23, 0, 24, 25,
    26, 0, 27, 28, 29, 0, 30, 31, 32, 0, 33, 34, 35, 0, 36, 37, 38, 0, 39, 40,
    41, 0, 42, 43, 44, 0, 45, 46, 47, 0 };

  auto out_buf = std::vector<uint8_t>(in_buf.size());
  auto expected = std::vector<uint8_t>{ 2, 1, 0, 0, 5, 4, 3, 0, 8, 7, 6, 0, 11,
    10, 9, 0, 14, 13, 12, 0, 17, 16, 15, 0, 20, 19, 18, 0, 23, 22, 21, 0, 26,
    25, 24, 0, 29, 28, 27, 0, 32, 31, 30, 0, 35, 34, 33, 0, 38, 37, 36, 0, 41,
    40, 39, 0, 44, 43, 42, 0, 47, 46, 45, 0 };

  std::vector<size_t> strides{ 8 * 4 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 8, 2, int(strides[0]), 0, AxVideoFormat::RGBA },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 8, 2, 8 * 4, 0, AxVideoFormat::BGRA },
    out_buf.data(), { 8 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversions, bgr2rgb)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);
  auto in_buf = std::vector<uint8_t>{ 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10,
    11, 0, 12, 13, 14, 0, 15, 16, 17, 0, 18, 19, 20, 0, 21, 22, 23, 0, 24, 25,
    26, 0, 27, 28, 29, 0, 30, 31, 32, 0, 33, 34, 35, 0, 36, 37, 38, 0, 39, 40,
    41, 0, 42, 43, 44, 0, 45, 46, 47, 0 };

  auto out_buf = std::vector<uint8_t>(in_buf.size());
  auto expected = std::vector<uint8_t>{ 2, 1, 0, 0, 5, 4, 3, 0, 8, 7, 6, 0, 11,
    10, 9, 0, 14, 13, 12, 0, 17, 16, 15, 0, 20, 19, 18, 0, 23, 22, 21, 0, 26,
    25, 24, 0, 29, 28, 27, 0, 32, 31, 30, 0, 35, 34, 33, 0, 38, 37, 36, 0, 41,
    40, 39, 0, 44, 43, 42, 0, 47, 46, 45, 0 };

  std::vector<size_t> strides{ 8 * 4 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 8, 2, int(strides[0]), 0, AxVideoFormat::BGRA },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 8, 2, 8 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 8 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, yuyv2rgb)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };

  std::vector<size_t> strides{ 12 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::YUY2 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgb)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0x3A, 0x3A, 0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() / 3 * 2 * 4, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 15 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, nv12torgb)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() / 3 * 2 * 4, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };

  std::vector<size_t> strides{ 6, 6 };
  std::vector<size_t> offsets{ 0, 12 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::NV12 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Test for RGB to GRAY conversion
TEST(Conversion, rgb2gray)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "gray" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);
  auto in_buf = std::vector<uint8_t>{
    255, 0, 0, // Red
    0, 255, 0, // Green
    0, 0, 255, // Blue
    255, 255, 255 // White
  };

  // Gray output should be 1 byte per pixel
  auto out_buf = std::vector<uint8_t>(4, 0);
  auto expected = std::vector<uint8_t>{
    76, // Red (0.299*255 = 76)
    150, // Green (0.587*255 = 149.7 -> 150, different to OpenCL)
    29, // Blue (0.114*255 = 29)
    255 // White (0.299*255 + 0.587*255 + 0.114*255 = 255)
  };

  auto in = AxVideoInterface{ { 1, 4, 3, 0, AxVideoFormat::RGB }, in_buf.data(),
    { 3 }, { 0 }, -1 };

  auto out = AxVideoInterface{ { 1, 4, 1, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 1 }, { 0 }, -1 };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Test for BGRA to GRAY conversion
TEST(Conversion, bgra2gray)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "gray" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);
  auto in_buf = std::vector<uint8_t>{ 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255,
    255, 255, 255, 255, 255 };

  // Gray output should be 1 byte per pixel
  auto out_buf = std::vector<uint8_t>(4, 0);
  auto expected = std::vector<uint8_t>{
    29, // Blue (0.114*255 = 29)
    150, // Green (0.587*255 = 149.7 -> 150, different to OpenCL)
    76, // Red (0.299*255 = 76)
    255 // White (0.114*255 + 0.587*255 + 0.299*255 = 255)
  };

  std::vector<size_t> strides{ 16 }; // 4 pixels * 4 bytes
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 4, 1, int(strides[0]), 0, AxVideoFormat::BGRA },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 4, 1, 4, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 4 }, { 0 }, -1 };

  Ax::MetaMap metadata;

  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Test for YUY2 to GRAY conversion
TEST(Conversion, yuyv2gray)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "gray" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);

  // YUYV format: Y0 U0 Y1 V0
  auto in_buf = std::vector<uint8_t>{
    // Y0  U0  Y1  V0
    100, 128, 150, 128, // First macropixel
    200, 128, 250, 128 // Second macropixel
  };

  // Gray output should extract just the Y values
  auto out_buf = std::vector<uint8_t>(4, 0);
  auto expected = std::vector<uint8_t>{
    100, 150, 200, 250 // The 4 Y values from the YUYV input
  };

  std::vector<size_t> strides{ 8 }; // 2 macropixels * 4 bytes
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 4, 1, int(strides[0]), 0, AxVideoFormat::YUY2 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 4, 1, 4, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 4 }, { 0 }, -1 };

  Ax::MetaMap metadata;

  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Test that NV16 input is rejected with clear error message
TEST(Conversion, nv16_rejected)
{
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert", input);

  auto in_buf = std::vector<uint8_t>(640 * 480 * 2); // NV16 size
  auto out_buf = std::vector<uint8_t>(640 * 480 * 4, 0);

  std::vector<size_t> strides{ 640, 640 };
  std::vector<size_t> offsets{ 0, 640 * 480 };

  auto in = AxVideoInterface{ { 640, 480, int(strides[0]), 0, AxVideoFormat::NV16 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 640 * 4 }, { 0 }, -1 };

  Ax::MetaMap metadata;

  // Should throw an exception indicating NV16 is not supported
  ASSERT_THROW(
      {
        try {
          xform->transform(in, out, 0, 1, metadata);
        } catch (const std::runtime_error &e) {
          // Verify the error message mentions NV16 and suggests OpenCL
          EXPECT_TRUE(std::string(e.what()).find("NV16") != std::string::npos);
          EXPECT_TRUE(std::string(e.what()).find("colorconvert_cl") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}
} // namespace
