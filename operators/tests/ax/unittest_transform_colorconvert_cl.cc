// Copyright Axelera AI, 2024
#include "unittest_ax_common.h"

extern bool has_opencl_platform();

namespace
{
struct format_params {
  AxVideoFormat in;
  std::string out;
};

struct passthrough_params {
  AxVideoFormat in;
  std::string out;
  std::string flip;
  bool passthrough;
};

class PassthroughFixture : public ::testing::TestWithParam<passthrough_params>
{
};

INSTANTIATE_TEST_SUITE_P(ColorConvertTestSuite, PassthroughFixture,
    ::testing::Values(passthrough_params{ AxVideoFormat::RGBA, "bgra", "none", false },
        passthrough_params{ AxVideoFormat::BGRA, "rgba", "none", false },
        passthrough_params{ AxVideoFormat::NV12, "rgba", "none", false },
        passthrough_params{ AxVideoFormat::NV12, "bgra", "none", false },
        passthrough_params{ AxVideoFormat::NV16, "rgba", "none", false },
        passthrough_params{ AxVideoFormat::NV16, "bgra", "none", false },
        passthrough_params{ AxVideoFormat::I420, "rgba", "none", false },
        passthrough_params{ AxVideoFormat::I420, "bgra", "none", false },
        passthrough_params{ AxVideoFormat::YUY2, "rgba", "none", false },
        passthrough_params{ AxVideoFormat::YUY2, "bgra", "none", false },
        passthrough_params{ AxVideoFormat::BGRA, "bgra", "none", true },
        passthrough_params{ AxVideoFormat::RGBA, "rgba", "none", true },
        passthrough_params{ AxVideoFormat::RGB, "rgb", "none", true },
        passthrough_params{ AxVideoFormat::BGR, "bgr", "none", true },
        // Test grayscale conversion passthrough cases
        passthrough_params{ AxVideoFormat::NV12, "gray", "none", true },
        passthrough_params{ AxVideoFormat::NV16, "gray", "none", true },
        passthrough_params{ AxVideoFormat::I420, "gray", "none", true },
        passthrough_params{ AxVideoFormat::GRAY8, "gray", "none", true },
        // Test grayscale conversion non-passthrough cases
        passthrough_params{ AxVideoFormat::RGBA, "gray", "none", false },
        passthrough_params{ AxVideoFormat::BGRA, "gray", "none", false },
        passthrough_params{ AxVideoFormat::RGB, "gray", "none", false },
        passthrough_params{ AxVideoFormat::BGR, "gray", "none", false },
        passthrough_params{ AxVideoFormat::YUY2, "gray", "none", false },
        passthrough_params{ AxVideoFormat::RGB, "rgb", "none", true },
        passthrough_params{ AxVideoFormat::RGB, "rgb", "clockwise", false },
        passthrough_params{ AxVideoFormat::BGR, "bgr", "none", true },
        passthrough_params{ AxVideoFormat::BGR, "bgr", "counterclockwise", false }));


TEST_P(PassthroughFixture, can_passthrough)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  auto format = GetParam();
  std::unordered_map<std::string, std::string> input = {
    { "format", format.out },
    { "flip_method", format.flip },
  };

  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  std::vector<int8_t> in_buf(640 * 480 * 4);
  std::vector<int8_t> out_buf(640 * 480 * 4);

  std::vector<size_t> strides;
  std::vector<size_t> offsets;
  if (format.in == AxVideoFormat::NV12) {
    strides = { 640, 640 };
    offsets = { 0, 640 * 480 };
  } else if (format.in == AxVideoFormat::NV16) {
    strides = { 640, 640 };
    offsets = { 0, 640 * 480 };
  } else if (format.in == AxVideoFormat::I420) {
    strides = { 640, 640 / 2, 640 / 2 };
    offsets = { 0, 640 * 480, 640 * 480 * 5 / 4 };
  } else if (format.in == AxVideoFormat::YUY2) {
    strides = { 640 * 2 };
    offsets = { 0 };
  } else {
    strides = { 640 * 4 };
    offsets = { 0 };
  }
  auto in = AxVideoInterface{ { 640, 480, int(strides[0]), 0, format.in },
    in_buf.data(), strides, offsets, -1 };

  AxVideoFormat out_format;
  int stride_multiplier = 4; // Default for RGBA/BGRA

  if (format.out == "rgba") {
    out_format = AxVideoFormat::RGBA;
  } else if (format.out == "bgra") {
    out_format = AxVideoFormat::BGRA;
  } else if (format.out == "gray") {
    out_format = AxVideoFormat::GRAY8;
    stride_multiplier = 1; // Only 1 byte per pixel for grayscale
  } else if (format.out == "rgb") {
    out_format = AxVideoFormat::RGB;
    stride_multiplier = 3; // 3 bytes per pixel for RGB
  } else if (format.out == "bgr") {
    out_format = AxVideoFormat::BGR;
    stride_multiplier = 3; // 3 bytes per pixel for BGR
  } else {
    // Default to RGBA if unknown format
    out_format = AxVideoFormat::RGBA;
  }

  auto out = AxVideoInterface{ { 640, 480, 640 * stride_multiplier, 0, out_format },
    out_buf.data(), { static_cast<size_t>(640 * stride_multiplier) }, { 0 }, -1 };

  ASSERT_EQ(xform->can_passthrough(in, out), format.passthrough);
}

TEST(Conversions1, rgb2bgr)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "bgr" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 };

  auto out_buf = std::vector<uint8_t>(in_buf.size());
  auto expected = std::vector<uint8_t>{ 2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9,
    14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21, 26, 25, 24, 29, 28, 27, 32,
    31, 30, 35, 34, 33, 38, 37, 36, 41, 40, 39, 44, 43, 42, 47, 46, 45 };

  std::vector<size_t> strides{ 8 * 3 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 8, 2, int(strides[0]), 0, AxVideoFormat::RGB },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 8, 2, 8 * 3, 0, AxVideoFormat::BGR },
    out_buf.data(), { 8 * 3 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversions1, rgba2bgra)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "bgra" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
     0,  1,  2,  0,  3,  4,  5,  0,  6,  7,  8, 0,  9, 10, 11,  0,
    12, 13, 14,  0, 15, 16, 17,  0, 18, 19, 20, 0, 21, 22, 23,  0,
    24, 25, 26,  0, 27, 28, 29,  0, 30, 31, 32, 0, 33, 34, 35,  0,
    36, 37, 38,  0, 39, 40, 41,  0, 42, 43, 44, 0, 45, 46, 47,  0
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size());
  auto expected = std::vector<uint8_t>{
    // clang-format off
     2,  1,  0,  0,  5,  4,  3,  0,  8,  7,  6,  0, 11, 10,  9,  0,
    14, 13, 12,  0, 17, 16, 15,  0, 20, 19, 18,  0, 23, 22, 21,  0,
    26, 25, 24,  0, 29, 28, 27,  0, 32, 31, 30,  0, 35, 34, 33,  0,
    38, 37, 36,  0, 41, 40, 39,  0, 44, 43, 42,  0, 47, 46, 45,  0
    // clang-format on
  };

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
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgb" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size());
  auto expected = std::vector<uint8_t>{
    // clang-format off
    2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21,
    26, 25, 24, 29, 28, 27, 32, 31, 30, 35, 34, 33, 38, 37, 36, 41, 40, 39, 44, 43, 42, 47, 46, 45
    // clang-format on
  };

  std::vector<size_t> strides{ 8 * 3 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 8, 2, int(strides[0]), 0, AxVideoFormat::BGR },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 8, 2, 8 * 3, 0, AxVideoFormat::RGB },
    out_buf.data(), { 8 * 3 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversions, bgra2rgba)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
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

TEST(Conversions, bgra2rgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgb" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
     0,  1,  2,  0,  3,  4,  5,  0,  6,  7,  8,  0,  9, 10, 11,  0,
    12, 13, 14,  0, 15, 16, 17,  0, 18, 19, 20,  0, 21, 22, 23,  0,
    24, 25, 26,  0, 27, 28, 29,  0, 30, 31, 32,  0, 33, 34, 35,  0,
    36, 37, 38,  0, 39, 40, 41,  0, 42, 43, 44,  0, 45, 46, 47,  0
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
      2,  1,  0,  5,  4,  3,  8,  7,  6, 11, 10,  9,
     14, 13, 12, 17, 16, 15, 20, 19, 18, 23, 22, 21,
     26, 25, 24, 29, 28, 27, 32, 31, 30, 35, 34, 33,
     38, 37, 36, 41, 40, 39, 44, 43, 42, 47, 46, 45
    // clang-format off
    };
  auto out_buf = std::vector<uint8_t>(expected.size());

  std::vector<size_t> strides{ 8 * 4 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 8, 2, int(strides[0]), 0, AxVideoFormat::BGRA },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 8, 2, 8 * 3, 0, AxVideoFormat::RGB },
    out_buf.data(), { 8 * 3 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversions, bgra2bgr)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "bgr" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
     0,  1,  2,  0,  3,  4,  5,  0,  6,  7,  8,  0,  9, 10, 11,  0,
    12, 13, 14,  0, 15, 16, 17,  0, 18, 19, 20,  0, 21, 22, 23,  0,
    24, 25, 26,  0, 27, 28, 29,  0, 30, 31, 32,  0, 33, 34, 35,  0,
    36, 37, 38,  0, 39, 40, 41,  0, 42, 43, 44,  0, 45, 46, 47,  0
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    // clang-format off
    };
  auto out_buf = std::vector<uint8_t>(expected.size());

  std::vector<size_t> strides{ 8 * 4 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 8, 2, int(strides[0]), 0, AxVideoFormat::BGRA },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 8, 2, 8 * 3, 0, AxVideoFormat::BGR },
    out_buf.data(), { 8 * 3 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}


TEST(Conversion, yuyv2rgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
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

TEST(Conversion, i4202rgba)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0x3A, 0x3A,
    0x3A, 0x3A, 0x3A,
    0xC9, 0xC9, 0xC9,
    0xC9, 0xC9, 0xC9,
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

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

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
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
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

TEST(Conversion, nv16torgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
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

  std::vector<size_t> strides{ 6, 6 };
  std::vector<size_t> offsets{ 0, 12 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::NV16 },
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
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "gray" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
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
    149, // Green (0.587*255 = 149)
    29, // Blue (0.114*255 = 29)
    255 // White (0.299*255 + 0.587*255 + 0.114*255 = 255)
  };

  std::vector<size_t> strides{ 12 }; // 4 pixels * 3 bytes (RGB)
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 4, 1, int(strides[0]), 0, AxVideoFormat::RGB },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 4, 1, 4, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 4 }, { 0 }, -1 };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Test for grayscale passthrough from YUV formats (NV12/NV16/I420)
TEST(Conversion, yuv2gray_passthrough)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "gray" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);

  // Test with NV12 input format
  {
    auto in_buf = std::vector<uint8_t>(640 * 480 * 3 / 2); // NV12 size
    std::vector<size_t> strides{ 640, 640 };
    std::vector<size_t> offsets{ 0, 640 * 480 };

    auto in = AxVideoInterface{ { 640, 480, 640, 0, AxVideoFormat::NV12 },
      in_buf.data(), strides, offsets, -1 };

    auto out_buf = std::vector<uint8_t>(640 * 480, 0); // Grayscale size
    auto out = AxVideoInterface{ { 640, 480, 640, 0, AxVideoFormat::GRAY8 },
      out_buf.data(), { 640 }, { 0 }, -1 };

    // First confirm this is a passthrough case
    ASSERT_TRUE(xform->can_passthrough(in, out));

    // Check that the first 10 pixels match the input luma plane
    for (int i = 0; i < 10; i++) {
      ASSERT_EQ(out_buf[i], in_buf[i]);
    }
  }

  // Test with NV16 input format
  {
    auto in_buf = std::vector<uint8_t>(640 * 480 * 2); // NV16 size
    std::vector<size_t> strides{ 640, 640 };
    std::vector<size_t> offsets{ 0, 640 * 480 };

    auto in = AxVideoInterface{ { 640, 480, 640, 0, AxVideoFormat::NV16 },
      in_buf.data(), strides, offsets, -1 };

    auto out_buf = std::vector<uint8_t>(640 * 480, 0); // Grayscale size
    auto out = AxVideoInterface{ { 640, 480, 640, 0, AxVideoFormat::GRAY8 },
      out_buf.data(), { 640 }, { 0 }, -1 };

    // First confirm this is a passthrough case
    ASSERT_TRUE(xform->can_passthrough(in, out));

    // Check that the first 10 pixels match the input luma plane
    for (int i = 0; i < 10; i++) {
      ASSERT_EQ(out_buf[i], in_buf[i]);
    }
  }

  // Test with I420 input format
  {
    auto in_buf = std::vector<uint8_t>(640 * 480 * 3 / 2); // I420 size

    std::vector<size_t> strides{ 640, 640 / 2, 640 / 2 };
    std::vector<size_t> offsets{ 0, 640 * 480, 640 * 480 * 5 / 4 };

    auto in = AxVideoInterface{ { 640, 480, 640, 0, AxVideoFormat::I420 },
      in_buf.data(), strides, offsets, -1 };

    auto out_buf = std::vector<uint8_t>(640 * 480, 0); // Grayscale size
    auto out = AxVideoInterface{ { 640, 480, 640, 0, AxVideoFormat::GRAY8 },
      out_buf.data(), { 640 }, { 0 }, -1 };

    // First confirm this is a passthrough case
    ASSERT_TRUE(xform->can_passthrough(in, out));
  }
}


TEST(Conversion, i4202rgb_clockwise)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "clockwise" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    // clang-format on
  };
  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 2, 6, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(ConversionsFlip, rgb_clockwise)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgb" },
    { "flip_method", "clockwise" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  auto out_buf = std::vector<uint8_t>(in_buf.size(), 0xff);
  auto expected = std::vector<uint8_t>{ 6, 7, 8, 0, 1, 2, 9, 10, 11, 3, 4, 5 };
  std::vector<size_t> strides{ 2 * 3 };
  std::vector<size_t> offsets{ 0 };
  auto in = AxVideoInterface{ { 2, 2, int(strides[0]), 0, AxVideoFormat::RGB },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 2, 2, 2 * 3, 0, AxVideoFormat::RGB },
    out_buf.data(), { 2 * 3 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(ConversionsFlip, rgba_clockwise)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "clockwise" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{ 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 0 };
  auto out_buf = std::vector<uint8_t>(in_buf.size(), 0xff);
  auto expected
      = std::vector<uint8_t>{ 6, 7, 8, 0, 0, 1, 2, 0, 9, 10, 11, 0, 3, 4, 5, 0 };
  std::vector<size_t> strides{ 2 * 4 };
  std::vector<size_t> offsets{ 0 };
  auto in = AxVideoInterface{ { 2, 2, int(strides[0]), 0, AxVideoFormat::RGBA },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 2, 2, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(ConversionsFlip, bgr_counterclockwise)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "bgr" },
    { "flip_method", "counterclockwise" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(in_buf.size(), 0xff);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    3, 4, 5, 9, 10, 11,
    0, 1, 2, 6, 7, 8
    // clang-format on
  };
  std::vector<size_t> strides{ 2 * 3 };
  std::vector<size_t> offsets{ 0 };
  auto in = AxVideoInterface{ { 2, 2, int(strides[0]), 0, AxVideoFormat::BGR },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 2, 2, 2 * 3, 0, AxVideoFormat::BGR },
    out_buf.data(), { 2 * 3 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(ConversionsFlip, bgra_counterclockwise)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "bgra" },
    { "flip_method", "counterclockwise" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0, 1, 2, 0, 3, 4, 5, 0,
    6, 7, 8, 0, 9, 10, 11, 0
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(in_buf.size(), 0xff);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    3, 4, 5, 0, 9, 10, 11, 0,
    0, 1, 2, 0, 6, 7, 8, 0
    // clang-format on
  };
  std::vector<size_t> strides{ 2 * 4 };
  std::vector<size_t> offsets{ 0 };
  auto in = AxVideoInterface{ { 2, 2, int(strides[0]), 0, AxVideoFormat::BGRA },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 2, 2, 2 * 4, 0, AxVideoFormat::BGRA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgb_counterclockwise)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "counterclockwise" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    // clang-format on
  };
  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 2, 6, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgb_upper_left_diagonal)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "upper-left-diagonal" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x49, 0xFF, 0x14, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    // clang-format on
  };
  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 2, 6, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgb_upper_right_diagonal)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "upper-right-diagonal" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    // clang-format on
  };
  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 2, 6, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}


TEST(Conversion, i4202rgb_horizontal_flip)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "horizontal-flip" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    // clang-format on
  };
  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgb_vertical_flip)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "vertical-flip" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    // clang-format on
  };
  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgb_rotate180)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "rotate-180" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
    // clang-format on
  };
  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}


TEST(Conversion, yuyv_rotate180)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "rotate-180" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
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

TEST(Conversion, nv122rgb_rotate180)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
    { "flip_method", "rotate-180" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    0x00, 0x00, 0x00,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(in_buf.size() * 8 / 3, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF, 0x00, 0x87, 0x00, 0xFF,
    0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF, 0x49, 0xFF, 0x14, 0xFF,
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
// Test for BGRA to GRAY conversion
TEST(Conversion, bgra2gray)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "gray" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 0, 255, 255,
    255, 255, 255, 255
    // clang-format on
  };

  // Gray output should be 1 byte per pixel
  auto out_buf = std::vector<uint8_t>(4, 0);
  auto expected = std::vector<uint8_t>{
    29, // Blue (0.114*255 = 29)
    149, // Green (0.587*255 = 149)
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
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "gray" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);

  // YUYV format: Y0 U0 Y1 V0
  auto in_buf = std::vector<uint8_t>{
    // Y0  U0  Y1  V0
    100, 128, 150, 128, // First macropixel
    200, 128, 250, 128 // Second macropixel
  };

  // Gray output should extract just the Y values
  auto out_buf = std::vector<uint8_t>(4, 0);
  // Due to differing interpretations of YUV to grayscale conversion,
  // we now do the conversion, rather than use the Y values
  auto expected = std::vector<uint8_t>{ 97, 156, 214, 255 };

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

TEST(Conversion, i4202rgba_cropped)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x3A, 0x80, 0x80,
    0x80, 0x80, 0x80,
    0xC9, 0x80, 0x80,
    0x80, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::I420,
                                  true, 0, 0, 2 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, nv122rgba_cropped)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x98, 0x98, 0x00, 0x00,
    0x80, 0x80, 0x3A, 0xc9, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 6 };
  std::vector<size_t> offsets{ 0, 12 };

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::NV12,
                                  true, 2, 1, 2 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, nv122rgba_cropped_plus_I420)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x98, 0x98, 0x00, 0x00,
    0x80, 0x80, 0x3A, 0xc9, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 6 };
  std::vector<size_t> offsets{ 0, 12 };

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::NV12,
                                  true, 2, 1, 2 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);

  {
    std::unordered_map<std::string, std::string> input = {
      { "format", "rgba" },
    };
    auto in_buf = std::vector<uint8_t>{
      // clang-format off
    0x98, 0x98, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x3A, 0x80, 0x80,
    0x80, 0x80, 0x80,
    0xC9, 0x80, 0x80,
    0x80, 0x80, 0x80,
      // clang-format on
    };

    auto expected = std::vector<uint8_t>{
      // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
      // clang-format on
    };
    auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

    std::vector<size_t> strides{ 6, 3, 3 };
    std::vector<size_t> offsets{ 0, 12, 18 };

    auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::I420,
                                    true, 0, 0, 2 },
      in_buf.data(), strides, offsets, -1 };

    auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
      out_buf.data(), { 2 * 4 }, { 0 }, -1 };
    Ax::MetaMap metadata;
    xform->transform(in, out, 0, 1, metadata);
    ASSERT_EQ(out_buf, expected);
  }
}

TEST(Conversion, nv122rgba_cropped_two_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x98, 0x98, 0x00, 0x00,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    0x80, 0x80, 0x3A, 0xc9, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 6 };
  std::vector<size_t> offsets{ 0, 0 };

  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::NV12,
                                  true, 2, 1, 2 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgba_cropped_3_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    0x3A, 0x80, 0x80,
    0x80, 0x80, 0x80,
    // clang-format on
  };
  auto in_buf3 = std::vector<uint8_t>{
    // clang-format off
    0xC9, 0x80, 0x80,
    0x80, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2, in_buf3 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::I420,
                                  true, 0, 0, 2 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgba_cropped_2_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    0x3A, 0x80, 0x80,
    0x80, 0x80, 0x80,
    0xC9, 0x80, 0x80,
    0x80, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 0, 6 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::I420,
                                  true, 0, 0, 2 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, y4442rgba)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 6×2 = 12 bytes (full resolution)
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    // U plane: 6×2 = 12 bytes (full resolution, no subsampling)
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    0x3A, 0x3A, 0x3A, 0x3A, 0x3A, 0x3A,
    // V plane: 6×2 = 12 bytes (full resolution, no subsampling)
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    0xC9, 0xC9, 0xC9, 0xC9, 0xC9, 0xC9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(6 * 2 * 4, 0x99);
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
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

TEST(Conversion, y4442rgba_cropped_3_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 6×2, crop reads top-left 2×1
    0x98, 0x98, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    // U plane: 6×2, full-resolution chroma — positions (0,0) and (1,0) are sampled
    0x3A, 0x3A, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    // clang-format on
  };
  auto in_buf3 = std::vector<uint8_t>{
    // clang-format off
    // V plane: 6×2
    0xC9, 0xC9, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 6, 6 };
  std::vector<size_t> offsets{ 0, 12, 24 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2, in_buf3 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::Y444,
                                  true, 0, 0, 2 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, y4442rgba_cropped_2_planes)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);
  auto in_buf1 = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 6×2
    0x98, 0x98, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // clang-format on
  };
  auto in_buf2 = std::vector<uint8_t>{
    // clang-format off
    // U plane (12 bytes at offset 0): full-resolution, positions (0,0) and (1,0) sampled
    0x3A, 0x3A, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    // V plane (12 bytes at offset 12): full-resolution
    0xC9, 0xC9, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    // clang-format on
  };

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0xFF, 0x7E, 0x11, 0xFF, 0xFF, 0x7E, 0x11, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(expected.size(), 0x99);

  std::vector<size_t> strides{ 6, 6, 6 };
  std::vector<size_t> offsets{ 0, 0, 12 };
  std::vector<std::vector<uint8_t>> mem_planes = { in_buf1, in_buf2 };
  Ax::buffer_planes cl_planes{ mem_planes };
  opencl_planes planes = cl_planes.get_planes();

  auto in = AxVideoInterface{ { 2, 1, int(strides[0]), 0, AxVideoFormat::Y444,
                                  true, 0, 0, 2 },
    nullptr, strides, offsets, -1, &planes };

  auto out = AxVideoInterface{ { 2, 1, 2 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 2 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

// Validates that the Y444 bilinear sampler reads chroma at FULL resolution (x, y),
// not halved coordinates (x/2, y/2) as the I420 sampler does.
// U=0x80 for columns 0-1, U=0x40 for columns 2-3. A halving bug would cause
// pixel (2,*) to read U from column 1 (0x80) instead of column 2 (0x40),
// producing G=0x82/B=0x82 instead of the correct G=0x9B/B=0x01.
TEST(Conversion, y4442rgba_chroma_not_halved)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  auto xform = Ax::LoadTransform("colorconvert_cl", input);

  // 4×2 Y444: Y=0x80 everywhere, U=0x80 for columns 0-1, U=0x40 for columns 2-3,
  // V=0x80 everywhere (neutral V isolates the U effect on G and B channels).
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    // Y plane: 4×2 = 8 bytes
    0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80,
    // U plane: 4×2 = 8 bytes (spatially varying — left half 0x80, right half 0x40)
    0x80, 0x80, 0x40, 0x40,
    0x80, 0x80, 0x40, 0x40,
    // V plane: 4×2 = 8 bytes (neutral)
    0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80,
    // clang-format on
  };

  // Expected RGBA output (BT.601 limited-range, convert_uchar4_sat truncates):
  //   cols 0-1: Y=128,U=128,V=128 → R=0x82, G=0x82, B=0x82, A=0xFF
  //   cols 2-3: Y=128,U=64, V=128 → R=0x82, G=0x9B, B=0x01, A=0xFF
  // If chroma were halved, cols 2-3 would read U from col 1 (=0x80) → G=0x82,B=0x82 (wrong).
  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x82, 0x82, 0x82, 0xFF,  0x82, 0x82, 0x82, 0xFF,  0x82, 0x9B, 0x01, 0xFF,  0x82, 0x9B, 0x01, 0xFF,
    0x82, 0x82, 0x82, 0xFF,  0x82, 0x82, 0x82, 0xFF,  0x82, 0x9B, 0x01, 0xFF,  0x82, 0x9B, 0x01, 0xFF,
    // clang-format on
  };
  auto out_buf = std::vector<uint8_t>(4 * 2 * 4, 0x99);

  std::vector<size_t> strides{ 4, 4, 4 };
  std::vector<size_t> offsets{ 0, 8, 16 };

  auto in = AxVideoInterface{ { 4, 2, int(strides[0]), 0, AxVideoFormat::Y444 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 4, 2, 4 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 4 * 4 }, { 0 }, -1 };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  ASSERT_EQ(out_buf, expected);
}

} // namespace
