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

INSTANTIATE_TEST_SUITE_P(BarrelCorrectTestSuite, ColorFormatFixture,
    ::testing::Values(FormatParam{ AxVideoFormat::RGBA, 0 },
        FormatParam{ AxVideoFormat::RGBA, 1 }, FormatParam{ AxVideoFormat::BGRA, 1 },
        FormatParam{ AxVideoFormat::BGRA, 0 }, FormatParam{ AxVideoFormat::NV12, 0 },
        FormatParam{ AxVideoFormat::I420, 0 }, FormatParam{ AxVideoFormat::YUY2, 0 }));

TEST_P(ColorFormatFixture, happy_path)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00115868,-0.00067655,0" },
    { "bgra_out", std::to_string(format.bgra_out) },
  };

  Transformer barrelcorrect("libtransform_barrelcorrect_cl.so", input);
  std::vector<int8_t> in_buf(1920 * 1080 * 4);
  std::vector<int8_t> out_buf(1920 * 1080 * 4);

  std::vector<size_t> strides;
  std::vector<size_t> offsets;
  if (format.format == AxVideoFormat::NV12) {
    strides = { 1920, 1920 };
    offsets = { 0, 1920 * 1080 };
  } else if (format.format == AxVideoFormat::I420) {
    strides = { 1920, 1920 / 2, 1920 / 2 };
    offsets = { 0, 1920 * 1080, 1920 * 1080 * 5 / 4 };
  } else if (format.format == AxVideoFormat::YUY2) {
    strides = { 1920 * 2 };
    offsets = { 0 };
  } else {
    strides = { 1920 * 4 };
    offsets = { 0 };
  }
  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, format.format },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{
    { 1920, 1080, 1920 * 4, 0, format.bgra_out ? AxVideoFormat::BGRA : AxVideoFormat::RGBA },
    out_buf.data(), { 1920 * 4 }, { 0 }, -1
  };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  EXPECT_NO_THROW({ barrelcorrect.transform(in, out, metadata, 0, 1); });
}

TEST(barrel_correction, invalid_camera_prams)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00115868,-0.00067655,0" },
  };

  EXPECT_THROW(Transformer perspective("libtransform_barrelcorrect_cl.so", input),
      std::runtime_error);
}

TEST(barrel_correction, invalid_camera_coefs)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00115868,-0.00067655,0,7.6" },
  };

  EXPECT_THROW(Transformer perspective("libtransform_barrelcorrect_cl.so", input),
      std::runtime_error);
}


} // namespace
