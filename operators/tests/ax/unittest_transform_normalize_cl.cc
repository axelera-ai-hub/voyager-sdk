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


TEST(normalize_cl, no_mean_or_scale)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };

  Transformer normalizer("libtransform_normalize_cl.so", input);
  std::vector<uint8_t> in_buf(4 * 4);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x80, 0x81, 0x82, 0x03, 0x84, 0x85, 0x86, 0x07,
    0x88, 0x89, 0x8a, 0x0b, 0x8c, 0x8d, 0x8e, 0x0f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 1, 4, 4 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 1, 4, 4 }, 1, out_buf.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  normalizer.transform(in, out, metadata, 0, 1);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_4x4)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };

  Transformer normalizer("libtransform_normalize_cl.so", input);
  std::vector<uint8_t> in_buf(4 * 4 * 2);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4 * 2);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x80, 0x81, 0x82, 0x03, 0x84, 0x85, 0x86, 0x07,
    0x88, 0x89, 0x8a, 0x0b, 0x8c, 0x8d, 0x8e, 0x0f,
    0x90, 0x91, 0x92, 0x13, 0x94, 0x95, 0x96, 0x17,
    0x98, 0x99, 0x9a, 0x1b, 0x9c, 0x9d, 0x9e, 0x1f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, out_buf.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  normalizer.transform(in, out, metadata, 0, 1);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_4x4_x)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.0, 0.0 ,0.0" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "-127.5" },
    { "quant_scale", "0.003919653594493866" },
  };

  Transformer normalizer("libtransform_normalize_cl.so", input);
  std::vector<uint8_t> in_buf(4 * 4 * 2);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4 * 2);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x80, 0x82, 0x83, 0x03, 0x85, 0x86, 0x87, 0x07,
    0x89, 0x8a, 0x8b, 0x0b, 0x8d, 0x8e, 0x8f, 0x0f,
    0x91, 0x92, 0x93, 0x13, 0x95, 0x96, 0x97, 0x17,
    0x99, 0x9a, 0x9b, 0x1b, 0x9d, 0x9e, 0x9f, 0x1f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, out_buf.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  normalizer.transform(in, out, metadata, 0, 1);
  EXPECT_EQ(out_buf, expected);
}


} // namespace
