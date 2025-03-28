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


TEST(resize_cl, two2one)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "16" },
    { "height", "16" },
  };

  Transformer resize("libtransform_resize_cl.so", input);
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

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out, metadata, 0, 1);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, four2one)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
  };

  Transformer resize("libtransform_resize_cl.so", input);
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

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out, metadata, 0, 1);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, scale_up)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
    { "scale_up", "1" },
  };

  Transformer resize("libtransform_resize_cl.so", input);
  std::vector<uint8_t> in_buf(2 * 2 * 4, 255);
  std::vector<uint8_t> out_buf(4 * 4 * 4, 128);

  auto expected = std::vector<uint8_t>(4 * 4 * 4, 255);
  auto in = AxVideoInterface{ { 2, 2, 8, 0, AxVideoFormat::RGBA },
    in_buf.data(), { 8 }, { 0 } };
  auto out = AxVideoInterface{ { 4, 4, 16, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 16 }, { 0 } };

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out, metadata, 0, 1);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, no_scale_up)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "4" },
    { "height", "4" },
    { "scale_up", "0" },
  };

  Transformer resize("libtransform_resize_cl.so", input);
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

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out, metadata, 0, 1);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, halfpixel_centres_upscale)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "width", "12" },
    { "height", "2" },
  };

  Transformer resize("libtransform_resize_cl.so", input);
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

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out, metadata, 0, 1);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, no_resize_with_normalize)
{
  if (!has_opencl_platform) {
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

  Transformer resize("libtransform_resize_cl.so", input);
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

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out, metadata, 0, 1);

  EXPECT_EQ(out_buf, expected);
}

TEST(resize_cl, yuyvrgb)
{
  if (!has_opencl_platform) {
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

  Transformer resize("libtransform_resize_cl.so", input);
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
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, i4202rgb)
{
  if (!has_opencl_platform) {
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

  Transformer resize("libtransform_resize_cl.so", input);
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
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F, 0x7F, 0xFE, 0x91, 0x7F,
    // clang-format on
  };

  std::vector<size_t> strides{ 6, 3, 3 };
  std::vector<size_t> offsets{ 0, 12, 18 };

  auto in = AxVideoInterface{ { 6, 2, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 6, 2, 6 * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { 6 * 4 }, { 0 }, -1 };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_cl, nv12torgb)
{
  if (!has_opencl_platform) {
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
  Transformer resize("libtransform_resize_cl.so", input);
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
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  resize.transform(in, out);
  ASSERT_EQ(out_buf, expected);
}


} // namespace
