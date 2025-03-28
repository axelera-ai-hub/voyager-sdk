#include "unittest_transform_common.h"

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace
{
bool has_opencl_platform = [] {
  cl_platform_id platformId;
  cl_uint numPlatforms;

  auto error = clGetPlatformIDs(1, &platformId, &numPlatforms);
  if (error == CL_SUCCESS) {
    cl_uint num_devices = 0;
    cl_device_id device_id;
    error = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
  }
  return error == CL_SUCCESS;
}();

struct format_params {
  AxVideoFormat in;
  std::string out;
};

class ColorConvertFixture : public ::testing::TestWithParam<format_params>
{
};

INSTANTIATE_TEST_SUITE_P(ColorConvertTestSuite, ColorConvertFixture,
    ::testing::Values(format_params{ AxVideoFormat::RGBA, "bgra" },
        format_params{ AxVideoFormat::BGRA, "rgba" },
        format_params{ AxVideoFormat::NV12, "rgba" },
        format_params{ AxVideoFormat::NV12, "bgra" },
        format_params{ AxVideoFormat::I420, "rgba" },
        format_params{ AxVideoFormat::I420, "bgra" },
        format_params{ AxVideoFormat::YUY2, "rgba" },
        format_params{ AxVideoFormat::YUY2, "bgra" }));


TEST_P(ColorConvertFixture, happy_path)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  auto format = GetParam();
  std::unordered_map<std::string, std::string> input = {
    { "format", format.out },
  };

  Transformer color_convert("libtransform_colorconvert.so", input);
  std::vector<int8_t> in_buf(640 * 480 * 4);
  std::vector<int8_t> out_buf(640 * 480 * 4);

  std::vector<size_t> strides;
  std::vector<size_t> offsets;
  if (format.in == AxVideoFormat::NV12) {
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

  auto out = AxVideoInterface{
    { 640, 480, 640 * 4, 0, format.out == "rgba" ? AxVideoFormat::RGBA : AxVideoFormat::BGRA },
    out_buf.data(), { 640 * 4 }, { 0 }, -1
  };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  EXPECT_NO_THROW({ color_convert.transform(in, out, metadata, 0, 1); });
}

struct passthrough_params {
  AxVideoFormat in;
  std::string out;
  bool passthrough;
};

class PassthroughFixture : public ::testing::TestWithParam<passthrough_params>
{
};

INSTANTIATE_TEST_SUITE_P(ColorConvertTestSuite, PassthroughFixture,
    ::testing::Values(passthrough_params{ AxVideoFormat::RGBA, "bgra", false },
        passthrough_params{ AxVideoFormat::BGRA, "rgba", false },
        passthrough_params{ AxVideoFormat::NV12, "rgba", false },
        passthrough_params{ AxVideoFormat::NV12, "bgra", false },
        passthrough_params{ AxVideoFormat::I420, "rgba", false },
        passthrough_params{ AxVideoFormat::I420, "bgra", false },
        passthrough_params{ AxVideoFormat::YUY2, "rgba", false },
        passthrough_params{ AxVideoFormat::YUY2, "bgra", false },
        passthrough_params{ AxVideoFormat::BGRA, "bgra", true },
        passthrough_params{ AxVideoFormat::RGBA, "rgba", true }));


TEST_P(PassthroughFixture, can_passthrough)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  auto format = GetParam();
  std::unordered_map<std::string, std::string> input = {
    { "format", format.out },
  };

  Transformer color_convert("libtransform_colorconvert.so", input);
  std::vector<int8_t> in_buf(640 * 480 * 4);
  std::vector<int8_t> out_buf(640 * 480 * 4);

  std::vector<size_t> strides;
  std::vector<size_t> offsets;
  if (format.in == AxVideoFormat::NV12) {
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

  auto out = AxVideoInterface{
    { 640, 480, 640 * 4, 0, format.out == "rgba" ? AxVideoFormat::RGBA : AxVideoFormat::BGRA },
    out_buf.data(), { 640 * 4 }, { 0 }, -1
  };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  ASSERT_EQ(color_convert.can_passthrough(in, out), format.passthrough);
}

TEST(Conversions, rgb2bgr)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "bgra" },
  };
  Transformer color_convert("libtransform_colorconvert.so", input);
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
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  color_convert.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversions, bgr2rgb)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  Transformer color_convert("libtransform_colorconvert.so", input);
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
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  color_convert.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, yuyv2rgb)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  Transformer color_convert("libtransform_colorconvert.so", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9, 0x98, 0x3a, 0x98, 0xc9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2);
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
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  color_convert.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, i4202rgb)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  Transformer color_convert("libtransform_colorconvert.so", input);
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

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2);
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
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  color_convert.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}

TEST(Conversion, nv12torgb)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "format", "rgba" },
  };
  Transformer color_convert("libtransform_colorconvert.so", input);
  auto in_buf = std::vector<uint8_t>{
    // clang-format off
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x98, 0x98, 0x98, 0x98, 0x98, 0x98,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
    0x3A, 0xc9, 0x3A, 0xc9, 0x3A, 0xc9,
    // clang-format on
  };

  auto out_buf = std::vector<uint8_t>(in_buf.size() * 2);
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
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  color_convert.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}


} // namespace
