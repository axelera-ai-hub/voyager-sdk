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

} // namespace
