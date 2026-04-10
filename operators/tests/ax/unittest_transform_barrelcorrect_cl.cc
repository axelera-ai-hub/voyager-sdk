// Copyright Axelera AI, 2024
#include "unittest_ax_common.h"

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

extern bool has_opencl_platform();
namespace
{
class BCColorFormatFixture : public ::testing::TestWithParam<FormatParam>
{
};

INSTANTIATE_TEST_SUITE_P(BarrelCorrectTestSuite, BCColorFormatFixture,
    ::testing::Values(FormatParam{ AxVideoFormat::RGB, 0 },
        FormatParam{ AxVideoFormat::RGB, 3 }, FormatParam{ AxVideoFormat::BGR, 4 },
        FormatParam{ AxVideoFormat::BGR, 4 }, FormatParam{ AxVideoFormat::NV12, 3 },
        FormatParam{ AxVideoFormat::I420, 3 }, FormatParam{ AxVideoFormat::YUY2, 3 },
        FormatParam{ AxVideoFormat::GRAY8, 5 })); // Added GRAY8 format with GRAY output

TEST_P(BCColorFormatFixture, happy_path)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00115868,-0.00067655,0" },
    { "format", std::to_string(format.out_format) },
  };

  auto xform = Ax::LoadTransform("barrelcorrect_cl", input);

  // Choose buffer size and stride based on format
  int pixel_size
      = (format.format == AxVideoFormat::GRAY8) ?
            1 :
            ((format.format == AxVideoFormat::RGB || format.format == AxVideoFormat::BGR) ?
                    3 :
                    (format.format == AxVideoFormat::YUY2 ? 2 : 3));

  std::vector<int8_t> in_buf(1920 * 1080 * pixel_size);

  // Choose output size based on output format
  int out_pixel_size = (format.out_format == 5) ? 1 : 3; // GRAY=1, RGB/BGR=3
  std::vector<int8_t> out_buf(1920 * 1080 * out_pixel_size);

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
  } else if (format.format == AxVideoFormat::GRAY8) {
    strides = { 1920 };
    offsets = { 0 };
  } else if (format.format == AxVideoFormat::RGB || format.format == AxVideoFormat::BGR) {
    strides = { 1920 * 3 };
    offsets = { 0 };
  } else {
    strides = { 1920 * 4 };
    offsets = { 0 };
  }
  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, format.format },
    in_buf.data(), strides, offsets, -1 };

  AxVideoFormat out_format;
  if (format.out_format == 3) {
    out_format = AxVideoFormat::RGB;
  } else if (format.out_format == 4) {
    out_format = AxVideoFormat::BGR;
  } else if (format.out_format == 5) {
    out_format = AxVideoFormat::GRAY8;
  } else {
    out_format = AxVideoFormat::RGB; // Default
  }

  auto out = AxVideoInterface{ { 1920, 1080, 1920 * out_pixel_size, 0, out_format },
    out_buf.data(), { static_cast<size_t>(1920 * out_pixel_size) }, { 0 }, -1 };
  Ax::MetaMap metadata;
  EXPECT_NO_THROW({ xform->transform(in, out, 0, 1, metadata); });
}

TEST(barrel_correction, invalid_camera_prams)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00115868,-0.00067655,0" },
  };

  EXPECT_THROW(Ax::LoadTransform("barrelcorrect_cl", input), std::runtime_error);
}

TEST(barrel_correction, invalid_camera_coefs)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00115868,-0.00067655,0,7.6" },
  };

  EXPECT_THROW(Ax::LoadTransform("barrelcorrect_cl", input), std::runtime_error);
}

// Test RGB to GRAY8 output format conversion
TEST(barrel_correction, rgb_to_gray8_conversion)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00067655,0.0,-0.00115868" },
    { "normalized_properties", "0" }, { "format", "5" }, // GRAY output
  };

  auto xform = Ax::LoadTransform("barrelcorrect_cl", input);

  auto in_buf = std::vector<uint8_t>(1920 * 1080 * 3); // RGB input with all pixels set to white
  std::iota(in_buf.begin(), in_buf.end(), 0); // Fill with increasing values for testing
  auto out_buf = std::vector<uint8_t>(1920 * 1080, 0); // Grayscale output

  std::vector<size_t> strides{ 1920 * 3 }; // 4 pixels * 3 bytes
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, AxVideoFormat::RGB },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 1920, 1080, 1920, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 1920 }, { 0 }, -1 };

  Ax::MetaMap metadata;
  ASSERT_NO_THROW(xform->transform(in, out, 0, 1, metadata));

  EXPECT_FALSE(std::all_of(
      out_buf.begin(), out_buf.end(), [](uint8_t value) { return value == 0; }));
}

TEST(barrel_correction, gray8_to_gray8_conversion)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00067655,0.0,-0.00115868" },
    { "normalized_properties", "0" }, { "format", "5" }, // GRAY output
  };

  auto xform = Ax::LoadTransform("barrelcorrect_cl", input);

  // Simple grayscale input with recognizable values
  auto in_buf = std::vector<uint8_t>(1920 * 1080); // All pixels set to 100
  std::iota(in_buf.begin(), in_buf.end(), 0); // Fill with increasing values for testing
  auto out_buf = std::vector<uint8_t>(1920 * 1080, 0); // Grayscale output

  std::vector<size_t> strides{ 1920 };
  std::vector<size_t> offsets{ 0 };

  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, AxVideoFormat::GRAY8 },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 1920, 1080, 1920, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 1920 }, { 0 }, -1 };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_FALSE(std::all_of(
      out_buf.begin(), out_buf.end(), [](uint8_t value) { return value == 0; }));
}

// Test YUV (NV12) to GRAY8 conversion
TEST(barrel_correction, nv12_to_gray8_conversion)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00067655,0.0,-0.00115868" },
    { "normalized_properties", "0" }, { "format", "5" }, // GRAY output
  };

  auto xform = Ax::LoadTransform("barrelcorrect_cl", input);

  auto in_buf = std::vector<uint8_t>(1920 * 1080 * 3 / 2); // All pixels set to 100
  std::iota(in_buf.begin(), in_buf.end(), 0); // Fill with increasing values for testing
  auto out_buf = std::vector<uint8_t>(1920 * 1080, 0); // Grayscale output

  std::vector<size_t> strides{ 1920, 1920 };
  std::vector<size_t> offsets{ 0, 1920 * 1080 };

  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, AxVideoFormat::NV12 },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 1920, 1080, 1920, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 1920 }, { 0 }, -1 };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_FALSE(std::all_of(
      out_buf.begin(), out_buf.end(), [](uint8_t value) { return value == 0; }));
}

TEST(barrel_correction, nv16_to_gray8_conversion)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00067655,0.0,-0.00115868" },
    { "normalized_properties", "0" }, { "format", "gray8" }, // GRAY output
  };

  auto xform = Ax::LoadTransform("barrelcorrect_cl", input);

  auto in_buf = std::vector<uint8_t>(1920 * 1080 * 2);
  std::iota(in_buf.begin(), in_buf.end(), 0); // Fill with increasing values for testing
  auto out_buf = std::vector<uint8_t>(1920 * 1080, 0); // Grayscale output

  std::vector<size_t> strides{ 1920, 1920 };
  std::vector<size_t> offsets{ 0, 1920 * 1080 };

  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, AxVideoFormat::NV16 },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 1920, 1080, 1920, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 1920 }, { 0 }, -1 };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_FALSE(std::all_of(
      out_buf.begin(), out_buf.end(), [](uint8_t value) { return value == 0; }));
}

TEST(barrel_correction, i420_to_gray8_conversion)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "camera_props", "1180.74606734,1179.14890352,938.45253964,527.68112542" },
    { "distort_coefs", "-0.37793616,0.11966818,-0.00067655,0.0,-0.00115868" },
    { "normalized_properties", "0" }, { "format", "gray8" }, // GRAY output
  };

  auto xform = Ax::LoadTransform("barrelcorrect_cl", input);

  auto in_buf = std::vector<uint8_t>(1920 * 1080 * 3 / 2); // All pixels set to 100
  std::iota(in_buf.begin(), in_buf.end(), 0); // Fill with increasing values for testing
  auto out_buf = std::vector<uint8_t>(1920 * 1080, 0); // Grayscale output

  std::vector<size_t> strides{ 1920, 960, 960 };
  std::vector<size_t> offsets{ 0, 1920 * 1080, 1920 * 1080 + 960 * 540 };

  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, AxVideoFormat::I420 },
    in_buf.data(), strides, offsets, -1 };
  auto out = AxVideoInterface{ { 1920, 1080, 1920, 0, AxVideoFormat::GRAY8 },
    out_buf.data(), { 1920 }, { 0 }, -1 };

  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);

  EXPECT_FALSE(std::all_of(
      out_buf.begin(), out_buf.end(), [](uint8_t value) { return value == 0; }));
}


} // namespace
