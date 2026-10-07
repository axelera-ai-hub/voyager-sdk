// Copyright Axelera AI, 2024
#include "unittest_ax_common.h"

extern bool has_opencl_platform();

namespace
{
TEST(resize_cl, two2one)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "16" },
    { "height", "16" },
    { "interpolation", "1" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf(32 * 32 * 4);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(16 * 16 * 4);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    66, 67, 68, 69, 74, 75, 76, 77, 82, 83, 84, 85, 90, 91, 92, 93,
    98, 99, 100, 101, 106, 107, 108, 109, 114, 115, 116, 117, 122, 123, 124, 125,
    130, 131, 132, 133, 138, 139, 140, 141, 146, 147, 148, 149, 154, 155, 156, 157,
    162, 163, 164, 165, 170, 171, 172, 173, 178, 179, 180, 181, 186, 187, 188, 189,

    // clang-format on
  };

  auto in = AxVideoInterface{ { 32, 32, 128, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 128 }, { 0 } };
  auto out = AxVideoInterface{ { 16, 16, 64, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 64 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, two2one_pillow_bilinear)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  // Verifies pillow_bilinear antialiased downscale (2:1). Unlike the legacy
  // 2-tap bilinear, the pillow_bilinear filter uses 4 taps per dimension with
  // boundary clamping. For this linear gradient input, old bilinear gives
  // (10, 18, 42, 50) while pillow_bilinear gives (12, 19, 40, 47).
  std::unordered_map<std::string, std::string> input = {
    { "width", "2" },
    { "height", "2" },
    { "interpolation", "2" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  // R=G=B = (y*4+x)*4, A=255; linear gradient across 4x4 grid
  std::vector<uint8_t> in_buf = {
    // clang-format off
     0,  0,  0, 255,   4,  4,  4, 255,   8,  8,  8, 255,  12, 12, 12, 255,
    16, 16, 16, 255,  20, 20, 20, 255,  24, 24, 24, 255,  28, 28, 28, 255,
    32, 32, 32, 255,  36, 36, 36, 255,  40, 40, 40, 255,  44, 44, 44, 255,
    48, 48, 48, 255,  52, 52, 52, 255,  56, 56, 56, 255,  60, 60, 60, 255,
    // clang-format on
  };
  std::vector<uint8_t> out_buf(2 * 2 * 4, 0xaa);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    12, 12, 12, 255,  19, 19, 19, 255,
    40, 40, 40, 255,  47, 47, 47, 255,
    // clang-format on
  };

  auto in = AxVideoInterface{ { 4, 4, 16, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 16 }, { 0 } };
  auto out = AxVideoInterface{ { 2, 2, 8, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 8 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, four2one)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf = {
    // clang-format off
    0x00, 0x00, 0x00, 0x00, 0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20,
    0x30, 0x30, 0x30, 0x30, 0x40, 0x40, 0x40, 0x40, 0x50, 0x50, 0x50, 0x50, 0x60,
    0x60, 0x60, 0x60, 0x70, 0x70, 0x70, 0x70, 0x80, 0x80, 0x80, 0x80, 0x90, 0x90,
    0x90, 0x90, 0xa0, 0xa0, 0xa0, 0xa0, 0xb0, 0xb0, 0xb0, 0xb0, 0xc0, 0xc0, 0xc0,
    0xc0, 0xd0, 0xd0, 0xd0, 0xd0, 0xe0, 0xe0, 0xe0, 0xe0, 0xf0, 0xf0, 0xf0, 0xf0,
    // clang-format on
  };
  std::vector<uint8_t> out_buf(1 * 1 * 4);

  auto expected = std::vector<uint8_t>{ 0x78, 0x78, 0x78, 0x78 };
  auto in = AxVideoInterface{ { 4, 4, 16, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 16 }, { 0 } };
  auto out = AxVideoInterface{ { 1, 1, 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 4 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, four2one_rgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf = {
    // clang-format off
    0x00, 0x00, 0x00, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x30, 0x30, 0x30,
    0x40, 0x40, 0x40, 0x50, 0x50, 0x50, 0x60, 0x60, 0x60, 0x70, 0x70, 0x70,
    0x80, 0x80, 0x80, 0x90, 0x90, 0x90, 0xa0, 0xa0, 0xa0, 0xb0, 0xb0, 0xb0,
    0xc0, 0xc0, 0xc0, 0xd0, 0xd0, 0xd0, 0xe0, 0xe0, 0xe0, 0xf0, 0xf0, 0xf0,
    // clang-format on
  };
  std::vector<uint8_t> out_buf(1 * 1 * 4);

  auto expected = std::vector<uint8_t>{ 0x78, 0x78, 0x78, 0xFF };
  auto in = AxVideoInterface{ { 4, 4, 12, 0, AxVideoFormat::RGB },
    in_buf.data(), { 12 }, { 0 } };
  auto out = AxVideoInterface{ { 1, 1, 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 4 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}


TEST(resize_cl, scale_up)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
    { "scale_up", "1" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf(2 * 2 * 4, 255);
  std::vector<uint8_t> out_buf(4 * 4 * 4, 128);

  auto expected = std::vector<uint8_t>(4 * 4 * 4, 255);
  auto in = AxVideoInterface{ { 2, 2, 8, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 8 }, { 0 } };
  auto out = AxVideoInterface{ { 4, 4, 16, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 16 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, no_scale_up)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
    { "scale_up", "0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf(2 * 2 * 4, 255);
  std::vector<uint8_t> out_buf(4 * 4 * 4, 128);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255,
    114, 114, 114, 255, 255, 255, 255, 255, 255, 255, 255, 255, 114, 114, 114, 255,
    114, 114, 114, 255, 255, 255, 255, 255, 255, 255, 255, 255, 114, 114, 114, 255,
    114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255,
    // clang-format on
  };
  auto in = AxVideoInterface{ { 2, 2, 8, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 8 }, { 0 } };
  auto out = AxVideoInterface{ { 4, 4, 16, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 16 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, halfpixel_centres_upscale)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "12" },
    { "height", "2" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf = {
    // clang-format off
    0x00, 0x00, 0x00, 0x00, 0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20,
    0x30, 0x30, 0x30, 0x30, 0x40, 0x40, 0x40, 0x40, 0x50, 0x50, 0x50, 0x50,
    0x00, 0x00, 0x00, 0x00, 0x10, 0x10, 0x10, 0x10, 0x20, 0x20, 0x20, 0x20,
    0x30, 0x30, 0x30, 0x30, 0x40, 0x40, 0x40, 0x40, 0x50, 0x50, 0x50, 0x50,
    // clang-format on
  };
  std::vector<uint8_t> out_buf(2 * 12 * 4, 0xaa);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c,
    0x14, 0x14, 0x14, 0x14, 0x1c, 0x1c, 0x1c, 0x1c, 0x24, 0x24, 0x24, 0x24,
    0x2c, 0x2c, 0x2c, 0x2c, 0x34, 0x34, 0x34, 0x34, 0x3c, 0x3c, 0x3c, 0x3c,
    0x44, 0x44, 0x44, 0x44, 0x4c, 0x4c, 0x4c, 0x4c, 0x50, 0x50, 0x50, 0x50,
    0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c,
    0x14, 0x14, 0x14, 0x14, 0x1c, 0x1c, 0x1c, 0x1c, 0x24, 0x24, 0x24, 0x24,
    0x2c, 0x2c, 0x2c, 0x2c, 0x34, 0x34, 0x34, 0x34, 0x3c, 0x3c, 0x3c, 0x3c,
    0x44, 0x44, 0x44, 0x44, 0x4c, 0x4c, 0x4c, 0x4c, 0x50, 0x50, 0x50, 0x50,
    // clang-format on
  };

  auto in = AxVideoInterface{ { 6, 2, 24, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 24 }, { 0 } };
  auto out = AxVideoInterface{ { 12, 2, 48, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 48 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, no_resize_with_normalize)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf = {
    // clang-format off
    0x80, 0x90, 0x70, 0x00, 0x80, 0xFF, 0x00, 0x00, 0x80, 0x90, 0x70, 0x00,
    0x80, 0x90, 0x70, 0x00, 0x80, 0xFF, 0x00, 0x00, 0x80, 0x90, 0x70, 0x00,
    0x80, 0x90, 0x70, 0x00, 0x80, 0xFF, 0x00, 0x00, 0x80, 0x90, 0x70, 0x00,
    0x80, 0x90, 0x70, 0x00, 0x80, 0xFF, 0x00, 0x00, 0x80, 0x90, 0x70, 0x00,
    // clang-format on
  };
  std::vector<uint8_t> out_buf(6 * 2 * 4, 0xaa);
  std::vector<uint8_t> expected = {
    // clang-format off
    0x00, 0x10, 0xF0, 0x00, 0x00, 0x7F, 0x80, 0x00, 0x00, 0x10, 0xF0, 0x00,
    0x00, 0x10, 0xF0, 0x00, 0x00, 0x7F, 0x80, 0x00, 0x00, 0x10, 0xF0, 0x00,
    0x00, 0x10, 0xF0, 0x00, 0x00, 0x7F, 0x80, 0x00, 0x00, 0x10, 0xF0, 0x00,
    0x00, 0x10, 0xF0, 0x00, 0x00, 0x7F, 0x80, 0x00, 0x00, 0x10, 0xF0, 0x00,
    // clang-format on
  };
  auto in = AxVideoInterface{ { 6, 2, 24, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 24 }, { 0 } };
  auto out = AxVideoInterface{ { 6, 2, 24, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 24 }, { 0 } };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, yuyvrgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format on
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    // clang-format off
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2);
  auto expected = std::vector<uint8_t>{
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    };
  std::vector<size_t> strides{ 12};
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::YUY2 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, i4202rgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0x3A, 0x3A,
    0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(4 * in_buf.size() * 2 / 3);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
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

TEST(resize_cl, yuyvrgb_i420)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format on
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    // clang-format off
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2);
  auto expected = std::vector<uint8_t>{
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    };
  std::vector<size_t> strides{ 12};
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::YUY2 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);

  {
    auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0x3A, 0x3A,
    0xC9, 0xC9, 0xC9,
      // clang-format on
    };

    auto out_buf = std::vector<uint8_t>(4 * in_buf.size() * 2 / 3);
    auto expected = std::vector<uint8_t>{
      // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
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
}


TEST(resize_cl, nv12torgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(4 * in_buf.size() * 2 / 3);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
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

TEST(resize_cl, nv12torgb_2_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size());

  std::vector<size_t> strides{ 6, 6 };
  std::vector<size_t> offsets{ 0, 12 };

  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::NV12 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, i4202rgb_2_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    0x3A, 0x3A, 0x3A,
    0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size());

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 0, 3 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, i4202rgb_3_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    0x3A, 0x3A, 0x3A,
  };
  auto in_buf3 = std::vector<uint8_t>{
    // clang-format on
    // clang-format off
    0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size());

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 0, 3 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2, in_buf3 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, y4442rgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 6x2 = 12 bytes
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // U plane: 6x2 = 12 bytes (full resolution - no subsampling)
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    // V plane: 6x2 = 12 bytes (full resolution - no subsampling)
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(6 * 2 * 4);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };

  std::vector<size_t> strides{ 6, 6, 6 };
  std::vector<size_t> offsets{ 0, 12, 24 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::Y444 },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, norm_rgba)
{
  // Verifies that normalisation produces signed int8 stored as two's-complement
  // uint8: values below 128 in the fused output become > 127 when reinterpreted.
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);

  // All pixels: (R=128, G=64, B=192, A=255)
  std::vector<uint8_t> in_buf(6 * 2 * 4, 0);
  for (int i = 0; i < 6 * 2; ++i) {
    in_buf[i * 4 + 0] = 128;
    in_buf[i * 4 + 1] = 64;
    in_buf[i * 4 + 2] = 192;
    in_buf[i * 4 + 3] = 255;
  }

  // With mul=1.0, add=-128 fused: out = pixel - 128
  //   R=128 -> 0   -> 0x00
  //   G=64  -> -64 -> 0xC0 (two's complement)
  //   B=192 -> 64  -> 0x40
  //   A=255 preserved by color_convert_float then clamped to char max -> 0x7F
  std::vector<uint8_t> expected(6 * 2 * 4);
  for (int i = 0; i < 6 * 2; ++i) {
    expected[i * 4 + 0] = 0x00;
    expected[i * 4 + 1] = 0xC0;
    expected[i * 4 + 2] = 0x40;
    expected[i * 4 + 3] = 0x7F;
  }

  std::vector<uint8_t> out_buf(expected.size(), 0xAA);
  auto in = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 6 * 4 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, y4442rgb_3_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 6x2 = 12 bytes
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    // U plane: 6x2 = 12 bytes (full resolution)
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    // clang-format on
  };
  auto in_buf3 = std::vector<uint8_t>{
    // clang-format off
    // V plane: 6x2 = 12 bytes (full resolution)
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size());

  std::vector<size_t> strides{ 6, 6, 6 };
  std::vector<size_t> offsets{ 0, 12, 24 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2, in_buf3 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::Y444 },
    nullptr, strides, offsets, -1, &planes };
  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, norm_gray_positive)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0." },
    { "std", "1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);

  // All pixels = 200. out = pixel - 128 = 72 -> 0x48
  std::vector<uint8_t> in_buf(6 * 2, 200);
  std::vector<uint8_t> expected(6 * 2, 0x48);
  std::vector<uint8_t> out_buf(expected.size(), 0xAA);
  auto in = AxVideoInterface{ { 6, 2, 6, 0, AxVideoFormat::GRAY8 },
    in_buf.data(), { 6 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { 6, 2, 6, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 6 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, y4442rgb_2_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 6x2 = 12 bytes
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    // U plane (12 bytes at offset 0): full-resolution, no subsampling
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    // V plane (12 bytes at offset 12): full-resolution
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size());

  std::vector<size_t> strides{ 6, 6, 6 };
  std::vector<size_t> offsets{ 0, 0, 12 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::Y444 },
    nullptr, strides, offsets, -1, &planes };
  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, norm_gray_negative)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0." },
    { "std", "1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);

  // All pixels = 50. out = pixel - 128 = -78 -> 0xB2 (two's complement)
  std::vector<uint8_t> in_buf(6 * 2, 50);
  std::vector<uint8_t> expected(6 * 2, 0xB2);
  std::vector<uint8_t> out_buf(expected.size(), 0xAA);
  auto in = AxVideoInterface{ { 6, 2, 6, 0, AxVideoFormat::GRAY8 },
    in_buf.data(), { 6 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { 6, 2, 6, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 6 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, aipu_padding_output_shape)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "256" }, { "height", "128" }, { "to_tensor", "1" },
    { "aipu_padding", "1,2,3,4" }, // top,bottom,left,right
  };
  auto xform = Ax::LoadTransform("resize_cl", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  auto out_interface = xform->set_output_interface(video_info);
  auto info = std::get<AxTensorsInterface>(out_interface);
  ASSERT_EQ(info.size(), 1);
  ASSERT_EQ(info[0].sizes.size(), 4);
  EXPECT_EQ(info[0].sizes[1], 128 + 1 + 2); // height + top + bottom
  EXPECT_EQ(info[0].sizes[2], 256 + 3 + 4); // width + left + right
}

TEST(resize_cl, aipu_padding_rgba)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  // No scaling (input size == target size), so the AIPU border is the only
  // thing under test: the interior must be an exact, un-blended copy of the
  // input, shifted by (pad_left, pad_top), with aipu_fill (alpha forced to
  // 255) everywhere else.
  std::unordered_map<std::string, std::string> input = {
    { "width", "2" },
    { "height", "2" },
    { "aipu_padding", "1,2,3,4" }, // top,bottom,left,right
    { "aipu_fill", "99" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);
  std::vector<uint8_t> in_buf = {
    // clang-format off
    10, 20, 30, 255,    40, 50, 60, 255,
    70, 80, 90, 255,    100, 110, 120, 255,
    // clang-format on
  };
  const int out_w = 2 + 3 + 4;
  const int out_h = 2 + 1 + 2;
  std::vector<uint8_t> out_buf(out_w * out_h * 4, 0xAA);
  std::vector<uint8_t> expected(out_w * out_h * 4);
  for (int r = 0; r < out_h; ++r) {
    for (int c = 0; c < out_w; ++c) {
      int idx = (r * out_w + c) * 4;
      bool inside = r >= 1 && r < 1 + 2 && c >= 3 && c < 3 + 2;
      if (inside) {
        int src_idx = ((r - 1) * 2 + (c - 3)) * 4;
        for (int ch = 0; ch < 4; ++ch) {
          expected[idx + ch] = in_buf[src_idx + ch];
        }
      } else {
        expected[idx + 0] = expected[idx + 1] = expected[idx + 2] = 99;
        expected[idx + 3] = 255;
      }
    }
  }

  auto in = AxVideoInterface{ { 2, 2, 2 * 4, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 2 * 4 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { out_w, out_h, out_w * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { out_w * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, y42b2rgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "6" },
    { "height", "2" },
    { "mean", "0.,0.,0." },
    { "std", "1.,1.,1." },
    { "quant_scale", "0.003921568859368563" },
    { "quant_zeropoint", "-128.0" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 6x2 = 12 bytes
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // U plane: 3x2 = 6 bytes (half horizontal resolution, full vertical)
    0x3A, 0x3A, 0x3A,
    0x3A, 0x3A, 0x3A,
    // V plane: 3x2 = 6 bytes
    0xC9, 0xC9, 0xC9,
    0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(6 * 2 * 4);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::Y42B },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Regression test for a chroma-interpolation bug in the pillow_bilinear (interpolation=2)
// Y42B sampler: it used the luma fractional weight to blend U/V samples instead of a
// weight scaled into chroma space, producing wrong (and discontinuous) chroma values
// whenever an output pixel's source position fell between two chroma samples.
// Y and V are held constant so only the interpolated U value (and hence G/B) varies;
// U0=0x00 and U1=0x80 are chosen so the expected blend weights land on exact integers.
TEST(resize_cl, y42b2rgb_pillow_bilinear_chroma)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "8" },
    { "height", "1" },
    { "interpolation", "2" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 4x1 = 4 bytes, constant
    0x98, 0x98, 0x98, 0x98,
    // U plane: 2x1 = 2 bytes, two distinct samples
    0x00, 0x80,
    // V plane: 2x1 = 2 bytes, constant (neutral)
    0x80, 0x80,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(8 * 4, 0xAA);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x9E, 0xD0, 0x00, 0xFF,   0x9E, 0xCA, 0x00, 0xFF,
    0x9E, 0xBD, 0x00, 0xFF,   0x9E, 0xB1, 0x3D, 0xFF,
    0x9E, 0xA4, 0x7E, 0xFF,   0x9E, 0x9E, 0x9E, 0xFF,
    0x9E, 0x9E, 0x9E, 0xFF,   0x9E, 0x9E, 0x9E, 0xFF,
    // clang-format on
  };

  std::vector<size_t> strides{ 4, 2, 2 };
  std::vector<size_t> offsets{ 0, 4, 6 };

  auto in = AxVideoInterface{ { 4, 1, int(strides[0]), 0, AxVideoFormat::Y42B },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 8, 1, 8 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 8 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Regression test for the same chroma-interpolation bug (see
// y42b2rgb_pillow_bilinear_chroma above) in nv_semiplanar_sampler_pb_two_plane,
// which serves the NV12/NV16 pillow_bilinear samplers. Same setup: Y and V
// held constant, U varies between two samples chosen so the expected blend
// weights land on exact integers. Chroma height is 1 (single UV row), so this
// isolates the horizontal weight; NV12's horizontal chroma divisor is the same
// (2) as Y42B's, so the underlying math -- and therefore the expected output
// -- is identical to y42b2rgb_pillow_bilinear_chroma.
TEST(resize_cl, nv12torgb_pillow_bilinear_chroma)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "8" },
    { "height", "2" },
    { "interpolation", "2" },
  };

  auto xform = Ax::LoadTransform("resize_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 4x2 = 8 bytes, constant
    0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98,
    // UV plane: 2x1 = 4 bytes (interleaved U,V pairs), one chroma row
    0x00, 0x80, 0x80, 0x80,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(8 * 2 * 4, 0xAA);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x9E, 0xD0, 0x00, 0xFF,   0x9E, 0xCA, 0x00, 0xFF,
    0x9E, 0xBD, 0x00, 0xFF,   0x9E, 0xB1, 0x3D, 0xFF,
    0x9E, 0xA4, 0x7E, 0xFF,   0x9E, 0x9E, 0x9E, 0xFF,
    0x9E, 0x9E, 0x9E, 0xFF,   0x9E, 0x9E, 0x9E, 0xFF,

    0x9E, 0xD0, 0x00, 0xFF,   0x9E, 0xCA, 0x00, 0xFF,
    0x9E, 0xBD, 0x00, 0xFF,   0x9E, 0xB1, 0x3D, 0xFF,
    0x9E, 0xA4, 0x7E, 0xFF,   0x9E, 0x9E, 0x9E, 0xFF,
    0x9E, 0x9E, 0x9E, 0xFF,   0x9E, 0x9E, 0x9E, 0xFF,
    // clang-format on
  };

  std::vector<size_t> strides{ 4, 4 };
  std::vector<size_t> offsets{ 0, 8 };

  auto in = AxVideoInterface{ { 4, 2, int(strides[0]), 0, AxVideoFormat::NV12 },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 8, 2, 8 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 8 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Regression test: resizing a zero-copy cropped sub-frame (AxVideoInfo::cropped
// + x_offset/y_offset, as produced by roicrop) used to return a solid
// fill-colour image because `image_dims` (the sampler's source-bounds check)
// wasn't widened to account for the crop offset folded into `corrected`. The
// underlying buffer is 8x6; the "real" content lives in a 4x3 window at
// (x_offset=4, y_offset=2), the rest is a sentinel that must never appear in
// the output. Input/output are both 4x3 (1:1 scale) so bilinear sampling
// degenerates to an exact identity mapping of the cropped window.
TEST(resize_cl, gray_cropped_subframe)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "3" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);

  constexpr uint8_t kSentinel = 0xAA;
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel,
    kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel,
    kSentinel, kSentinel, kSentinel, kSentinel,         10,         20,         30,         40,
    kSentinel, kSentinel, kSentinel, kSentinel,         50,         60,         70,         80,
    kSentinel, kSentinel, kSentinel, kSentinel,         90,        100,        110,        120,
    kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel,
    // clang-format on
  };
  auto expected = std::vector<uint8_t>{
    // clang-format off
    10,  20,  30,  40,
    50,  60,  70,  80,
    90, 100, 110, 120,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0);

  auto in = AxVideoInterface{ { 4, 3, 8, 0, AxVideoFormat::GRAY8, true, 4, 2, 6 },
    in_buf.data(), { 8 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { 4, 3, 4, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Regression test: upsampling (2x) a zero-copy cropped sub-frame used to bleed
// sentinel data from outside the crop into the left/top edge of the output.
// Once crop is folded into the source coordinate (see fold_crop_into_transform),
// a bilinear sample near the crop's left/top edge can compute a source x/y
// that is slightly below the crop origin (e.g. xi = crop_x - 1) while still
// being >= 0 in the physical buffer -- the neighbour clamp then has to clamp
// to crop_x/crop_y (the crop's own edge), not to 0 (the buffer's edge), or it
// reads the sentinel region just outside the crop instead of replicating the
// crop's edge pixel. Same 8x6 buffer/4x3 crop-at-(4,2) as gray_cropped_subframe,
// but resized up to 8x6 (2x) so the left/top edge samples actually extrapolate
// past the crop origin and would pull in sentinel data if the clamp regressed.
TEST(resize_cl, gray_cropped_subframe_upscale)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "8" },
    { "height", "6" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);

  constexpr uint8_t kSentinel = 0xAA;
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel,
    kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel,
    kSentinel, kSentinel, kSentinel, kSentinel,         10,         20,         30,         40,
    kSentinel, kSentinel, kSentinel, kSentinel,         50,         60,         70,         80,
    kSentinel, kSentinel, kSentinel, kSentinel,         90,        100,        110,        120,
    kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel, kSentinel,
    // clang-format on
  };
  // clang-format off
  auto expected = std::vector<uint8_t>{
     10,  13,  18,  23,  28,  33,  38,  40,
     20,  23,  28,  33,  38,  43,  48,  50,
     40,  43,  48,  53,  58,  63,  68,  70,
     60,  63,  68,  73,  78,  83,  88,  90,
     80,  83,  88,  93,  98, 103, 108, 110,
     90,  93,  98, 103, 108, 113, 118, 120,
  };
  // clang-format on
  auto out_buf = std::vector<uint8_t>(expected.size(), 0);

  auto in = AxVideoInterface{ { 4, 3, 8, 0, AxVideoFormat::GRAY8, true, 4, 2, 6 },
    in_buf.data(), { 8 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { 8, 6, 8, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 8 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Regression test: letterbox padding must be quantized identically to real
// content pixels when normalization (to_tensor + mean/std/quant_scale/
// quant_zeropoint) is baked into the resize_cl kernel -- this is the fused
// path used for on-device quantized model preprocessing (e.g.
// OpenCLetterBoxColorConvertToTensorAndNormalize), where the letterbox pad
// value represents a "neutral" colour that must go through the *same*
// quantization affine transform as the image content, exactly as it would if
// letterbox and normalize were still two separate CPU/GPU passes.
//
// The move to a destination-space early-exit for the letterbox border (see
// gray_cropped_subframe_upscale's sibling optimization) wrote the raw uint8
// `fill` byte directly to the output buffer, bypassing color_matrix entirely.
// With a real int8 zero-point (quant_zeropoint=-128, matching a typical
// quantized detection model's manifest), that made the padding come out as
// literal 114 instead of the correctly quantized value (114 - 128 = -14,
// stored as the uchar bit pattern 242) -- i.e. the padding silently stopped
// matching what the model saw during training/calibration.
TEST(resize_cl, letterbox_padding_matches_quantization)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "8" },
    { "height", "8" },
    { "letterbox", "1" },
    { "padding", "114" },
    { "to_tensor", "1" },
    { "mean", "0" },
    { "std", "1" },
    { "quant_scale", "0.0039215689" }, // 1/255
    { "quant_zeropoint", "-128" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);

  // 8x4 solid mid-gray (128) RGBA input, letterboxed into an 8x8 canvas:
  // scale = min(8/8, 8/4) = 1, so content occupies the middle 4 rows
  // (rows 2..5), with a 2-row pad band above and below.
  auto in_buf = std::vector<uint8_t>(8 * 4 * 4, 128);
  for (size_t i = 3; i < in_buf.size(); i += 4)
    in_buf[i] = 255; // alpha
  auto out_buf = std::vector<uint8_t>(8 * 8 * 4, 0xFF);

  auto in = AxVideoInterface{ { 8, 4, 8 * 4, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 8 * 4 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { 8, 8, 8 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 8 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  // mul = 1/(255*quant_scale*std) ~= 1, add = (255*quant_zeropoint*std*
  // quant_scale - 255*mean)*mul ~= -128 for these parameters, so a raw pixel
  // value v quantizes to round(v - 128) (saturated to int8, stored as the
  // uchar bit pattern). content(128) -> 0; padding(114) -> 242 (== -14).
  constexpr uint8_t expected_content = 0;
  constexpr uint8_t expected_padding = 242;
  constexpr uint8_t unquantized_padding = 114; // the pre-fix (buggy) value

  int content_row = 3, padding_row = 0, col = 3;
  uint8_t content_r = out_buf[(content_row * 8 + col) * 4 + 0];
  uint8_t padding_r = out_buf[(padding_row * 8 + col) * 4 + 0];

  ASSERT_EQ(content_r, expected_content);
  ASSERT_EQ(padding_r, expected_padding);
  ASSERT_NE(padding_r, unquantized_padding);
}

// Regression test: letterbox padding for a YUV-family input used to be
// colour-converted as if it were (fill, fill, fill) -- i.e. as if `fill` were
// also the chroma value -- instead of the chroma-neutral (fill, 128, 128)
// every YUV sampler returns at its own out-of-bounds/letterbox border. That
// produced a visibly tinted padding border instead of a neutral gray one.
// With U=V=128 exactly cancelling the matrix's chroma bias terms, R, G and B
// must come out numerically identical for the padding pixels.
TEST(resize_cl, letterbox_padding_neutral_for_yuv_input)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "8" },
    { "letterbox", "1" },
    { "padding", "114" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);

  // 4x4 NV12 input, letterboxed into a 4x8 canvas: scale = 1, content
  // occupies the middle 4 rows (rows 2..5), with a 2-row pad band above
  // and below.
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    200, 200, 200, 200,
    200, 200, 200, 200,
    200, 200, 200, 200,
    200, 200, 200, 200,
     90, 200,  90, 200, // U, V pairs
     90, 200,  90, 200,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(4 * 8 * 4, 0xFF);

  std::vector<size_t> strides{ 4, 4 };
  std::vector<size_t> offsets{ 0, 16 };
  auto in = AxVideoInterface{ { 4, 4, int(strides[0]), 0, AxVideoFormat::NV12 },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 4, 8, 4 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 4 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  for (int row : { 0, 1, 6, 7 }) {
    for (int col = 0; col < 4; ++col) {
      auto base = (row * 4 + col) * 4;
      EXPECT_EQ(out_buf[base + 0], out_buf[base + 1]) << "row " << row << " col " << col;
      EXPECT_EQ(out_buf[base + 1], out_buf[base + 2]) << "row " << row << " col " << col;
    }
  }
}

// Verifies the two borders nest correctly: the letterbox fill (200) sits
// around the scaled content within the requested width/height, and the
// AIPU border (fill 50) sits around that as an outer layer, independent of
// the letterbox logic.
TEST(resize_cl, aipu_padding_with_letterbox)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
    { "letterbox", "1" },
    { "padding", "200" }, // letterbox fill
    { "aipu_padding", "1,1,0,0" }, // top,bottom,left,right
    { "aipu_fill", "50" },
  };
  auto xform = Ax::LoadTransform("resize_cl", input);
  // 2 wide x 4 high, each row a distinct constant colour so no interpolation
  // ambiguity: scale factor is exactly 1 (2x4 -> letterboxed content is 2x4
  // within the 4x4 request), so every pixel is an exact copy.
  std::vector<uint8_t> in_buf = {
    // clang-format off
    1, 1, 1, 255,   1, 1, 1, 255,
    2, 2, 2, 255,   2, 2, 2, 255,
    3, 3, 3, 255,   3, 3, 3, 255,
    4, 4, 4, 255,   4, 4, 4, 255,
    // clang-format on
  };
  std::vector<uint8_t> out_buf(4 * 6 * 4, 0xAA);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    50, 50, 50, 255,    50, 50, 50, 255,    50, 50, 50, 255,    50, 50, 50, 255,
    200, 200, 200, 255, 1, 1, 1, 255,       1, 1, 1, 255,        200, 200, 200, 255,
    200, 200, 200, 255, 2, 2, 2, 255,       2, 2, 2, 255,        200, 200, 200, 255,
    200, 200, 200, 255, 3, 3, 3, 255,       3, 3, 3, 255,        200, 200, 200, 255,
    200, 200, 200, 255, 4, 4, 4, 255,       4, 4, 4, 255,        200, 200, 200, 255,
    50, 50, 50, 255,    50, 50, 50, 255,    50, 50, 50, 255,    50, 50, 50, 255,
    // clang-format on
  };

  auto in = AxVideoInterface{ { 2, 4, 2 * 4, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 2 * 4 }, { 0 }, -1 };
  auto out = AxVideoInterface{ { 4, 6, 4 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 4 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

} // namespace
