// Copyright Axelera AI, 2025
#include "unittest_ax_common.h"

extern bool has_opencl_platform();

namespace
{
class PolarTransformFormatFixture : public ::testing::TestWithParam<FormatParam>
{
};

INSTANTIATE_TEST_SUITE_P(PolarTransformTestSuite, PolarTransformFormatFixture,
    ::testing::Values(FormatParam{ AxVideoFormat::RGB }, FormatParam{ AxVideoFormat::BGR },
        FormatParam{ AxVideoFormat::GRAY8 }, FormatParam{ AxVideoFormat::NV12 },
        FormatParam{ AxVideoFormat::I420 }, FormatParam{ AxVideoFormat::YUY2 }));

std::vector<uint8_t>
make_buffer(int width, int height, AxVideoFormat format)
{
  auto size = 0;
  if (format == AxVideoFormat::GRAY8) {
    size = width * height;
  } else if (format == AxVideoFormat::NV12 || format == AxVideoFormat::I420) {
    size = width * height * 3 / 2;
  } else if (format == AxVideoFormat::YUY2) {
    size = width * height * 2;
  } else if (format == AxVideoFormat::RGB || format == AxVideoFormat::BGR) {
    size = width * height * 3;
  } else if (format == AxVideoFormat::RGBA || format == AxVideoFormat::BGRA) {
    size = width * height * 4;
  } else {
    throw std::runtime_error(
        "Unsupported format in make_buffer: " + AxVideoFormatToString(format));
  }
  return std::vector<uint8_t>(size, 0);
}

AxVideoInterface
make_video_interface(int width, int height, AxVideoFormat format, std::vector<uint8_t> &buffer)
{
  if (format == AxVideoFormat::GRAY8) {
    return AxVideoInterface{ { width, height, width, 0, format }, buffer.data(),
      { size_t(width) }, { 0 } };
  } else if (format == AxVideoFormat::NV12) {
    return AxVideoInterface{ { width, height, width, 0, format }, buffer.data(),
      { size_t(width), size_t(width) }, { 0, size_t(width * height) } };
  } else if (format == AxVideoFormat::I420) {
    return AxVideoInterface{ { width, height, width, 0, format }, buffer.data(),
      { size_t(width), size_t(width / 2), size_t(width / 2) },
      { 0, size_t(width * height), size_t(width * height * 5 / 4) } };
  } else if (format == AxVideoFormat::YUY2) {
    return AxVideoInterface{ { width, height, width * 2, 0, format },
      buffer.data(), { size_t(width * 2) }, { 0 } };
  } else if (format == AxVideoFormat::RGB || format == AxVideoFormat::BGR) {
    return AxVideoInterface{ { width, height, width * 3, 0, format },
      buffer.data(), { size_t(width * 3) }, { 0 } };
  } else if (format == AxVideoFormat::RGBA || format == AxVideoFormat::BGRA) {
    return AxVideoInterface{ { width, height, width * 4, 0, format },
      buffer.data(), { size_t(width * 4) }, { 0 } };
  } else {
    throw std::runtime_error("Unsupported format in make_video_interface: "
                             + AxVideoFormatToString(format));
  }
}

TEST_P(PolarTransformFormatFixture, basic_polar_transform_test)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  auto out_fmt = (format.format == AxVideoFormat::GRAY8) ? "gray8" : "rgb";

  std::unordered_map<std::string, std::string> input = {
    { "center_x", "0.5" },
    { "center_y", "0.5" },
    { "inverse", "0" },
    { "linear_polar", "1" },
    { "rotate180", "0" },
    { "start_angle", std::to_string(M_PI / 2.0) },
    { "format", out_fmt },
  };

  auto xform = Ax::LoadTransform("polar_cl", input);

  int input_width = 100;
  int input_height = 100;

  auto out_format = (format.format == AxVideoFormat::GRAY8) ? AxVideoFormat::GRAY8 :
                                                              AxVideoFormat::RGB;
  auto input_buffer = make_buffer(input_width, input_height, format.format);

  auto input_info
      = make_video_interface(input_width, input_height, format.format, input_buffer);
  auto output_interface = xform->set_output_interface(input_info);
  auto output_info = std::get<AxVideoInterface>(output_interface);

  // Update output buffer with actual output dimensions
  auto output_width_actual = output_info.info.width;
  auto output_height_actual = output_info.info.height;
  auto output_buffer = make_buffer(output_width_actual, output_height_actual, out_format);
  output_info = make_video_interface(
      output_width_actual, output_height_actual, out_format, output_buffer);

  Ax::MetaMap metadata;
  ASSERT_NO_THROW(xform->transform(input_info, output_info, 0, 1, metadata));
}


TEST_P(PolarTransformFormatFixture, inverse_polar_transform_test)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  auto out_fmt = (format.format == AxVideoFormat::GRAY8) ? "gray8" : "bgr";

  std::unordered_map<std::string, std::string> input = {
    { "center_x", "0.5" },
    { "center_y", "0.5" },
    { "inverse", "1" },
    { "linear_polar", "1" },
    { "rotate180", "0" },
    { "start_angle", std::to_string(M_PI / 2.0) },
    { "format", out_fmt },
  };

  auto xform = Ax::LoadTransform("polar_cl", input);

  int input_width = 100;
  int input_height = 100;
  auto out_format = (format.format == AxVideoFormat::GRAY8) ? AxVideoFormat::GRAY8 :
                                                              AxVideoFormat::RGB;
  auto input_buffer = make_buffer(input_width, input_height, format.format);

  auto input_info
      = make_video_interface(input_width, input_height, format.format, input_buffer);
  auto output_interface = xform->set_output_interface(input_info);
  auto output_info = std::get<AxVideoInterface>(output_interface);

  // Update output buffer with actual output dimensions
  auto output_width_actual = output_info.info.width;
  auto output_height_actual = output_info.info.height;
  auto output_buffer = make_buffer(output_width_actual, output_height_actual, out_format);
  output_info = make_video_interface(
      output_width_actual, output_height_actual, out_format, output_buffer);

  Ax::MetaMap metadata;
  ASSERT_NO_THROW(xform->transform(input_info, output_info, 0, 1, metadata));
}


TEST_P(PolarTransformFormatFixture, polar_transform_with_custom_center)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  auto out_fmt = (format.format == AxVideoFormat::GRAY8) ? "gray8" : "rgb";

  std::unordered_map<std::string, std::string> input = {
    { "center_x", "0.3" },
    { "center_y", "0.7" },
    { "inverse", "0" },
    { "linear_polar", "1" },
    { "rotate180", "1" },
    { "start_angle", std::to_string(M_PI / 4.0) },
    { "format", out_fmt },
  };
  auto xform = Ax::LoadTransform("polar_cl", input);

  int input_width = 80;
  int input_height = 60;

  auto out_format = (format.format == AxVideoFormat::GRAY8) ? AxVideoFormat::GRAY8 :
                                                              AxVideoFormat::RGB;
  auto input_buffer = make_buffer(input_width, input_height, format.format);

  auto input_info
      = make_video_interface(input_width, input_height, format.format, input_buffer);
  auto output_interface = xform->set_output_interface(input_info);
  auto output_info = std::get<AxVideoInterface>(output_interface);

  // Update output buffer with actual output dimensions
  auto output_width_actual = output_info.info.width;
  auto output_height_actual = output_info.info.height;
  auto output_buffer = make_buffer(output_width_actual, output_height_actual, out_format);
  output_info = make_video_interface(
      output_width_actual, output_height_actual, out_format, output_buffer);

  Ax::MetaMap metadata;
  ASSERT_NO_THROW(xform->transform(input_info, output_info, 0, 1, metadata));
}

TEST_P(PolarTransformFormatFixture, polar_transform_semilog_mode)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  auto out_fmt = (format.format == AxVideoFormat::GRAY8) ? "gray8" : "rgb";

  std::unordered_map<std::string, std::string> input = {
    { "center_x", "0.5" },
    { "center_y", "0.5" },
    { "inverse", "0" },
    { "linear_polar", "0" },
    { "rotate180", "0" },
    { "start_angle", std::to_string(0.0) },
    { "max_radius", "50.0" },
    { "format", out_fmt },
  };

  auto xform = Ax::LoadTransform("polar_cl", input);

  int input_width = 100;
  int input_height = 100;

  auto out_format = (format.format == AxVideoFormat::GRAY8) ? AxVideoFormat::GRAY8 :
                                                              AxVideoFormat::RGB;
  auto input_buffer = make_buffer(input_width, input_height, format.format);

  auto input_info
      = make_video_interface(input_width, input_height, format.format, input_buffer);
  auto output_interface = xform->set_output_interface(input_info);
  auto output_info = std::get<AxVideoInterface>(output_interface);

  // Update output buffer with actual output dimensions
  auto output_width_actual = output_info.info.width;
  auto output_height_actual = output_info.info.height;
  auto output_buffer = make_buffer(output_width_actual, output_height_actual, out_format);
  output_info = make_video_interface(
      output_width_actual, output_height_actual, out_format, output_buffer);

  Ax::MetaMap metadata;
  ASSERT_NO_THROW(xform->transform(input_info, output_info, 0, 1, metadata));
}


TEST_P(PolarTransformFormatFixture, polar_transform_with_size_property)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  auto out_fmt = (format.format == AxVideoFormat::GRAY8) ? "gray8" : "rgb";

  std::unordered_map<std::string, std::string> input = {
    { "center_x", "0.5" },
    { "center_y", "0.5" },
    { "size", "128" },
    { "inverse", "0" },
    { "linear_polar", "1" },
    { "format", out_fmt },
  };

  auto xform = Ax::LoadTransform("polar_cl", input);

  int input_width = 160;
  int input_height = 140;

  auto out_format = (format.format == AxVideoFormat::GRAY8) ? AxVideoFormat::GRAY8 :
                                                              AxVideoFormat::RGB;
  auto input_buffer = make_buffer(input_width, input_height, format.format);

  auto input_info
      = make_video_interface(input_width, input_height, format.format, input_buffer);
  auto output_interface = xform->set_output_interface(input_info);
  auto output_info = std::get<AxVideoInterface>(output_interface);

  // Update output buffer with actual output dimensions
  auto output_width_actual = output_info.info.width;
  auto output_height_actual = output_info.info.height;
  auto output_buffer = make_buffer(output_width_actual, output_height_actual, out_format);
  output_info = make_video_interface(
      output_width_actual, output_height_actual, out_format, output_buffer);

  Ax::MetaMap metadata;
  ASSERT_NO_THROW(xform->transform(input_info, output_info, 0, 1, metadata));
}

class PolarTransformExceptionTest : public ::testing::Test
{
};

TEST_F(PolarTransformExceptionTest, test_conflicting_size_and_width_height)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  std::unordered_map<std::string, std::string> input = {
    { "size", "128" },
    { "width", "100" },
    { "height", "100" },
  };

  EXPECT_THROW(Ax::LoadTransform("polar_cl", input), std::runtime_error);
}

TEST_F(PolarTransformExceptionTest, test_unsupported_input_format)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  std::unordered_map<std::string, std::string> input = {
    { "center_x", "0.5" },
    { "center_y", "0.5" },
    { "format", "rgb" },
  };

  auto xform = Ax::LoadTransform("polar_cl", input);

  std::vector<uint8_t> input_buffer(100 * 100 * 4);
  std::vector<uint8_t> output_buffer(100 * 100 * 3);

  auto input_info = AxVideoInterface{ { 100, 100, 100 * 4, 0, AxVideoFormat::UNDEFINED },
    input_buffer.data(), { size_t(100 * 4) }, { 0 } };
  auto output_interface = xform->set_output_interface(input_info);
  auto output_info = std::get<AxVideoInterface>(output_interface);
  output_info.data = output_buffer.data();

  Ax::MetaMap metadata;
  EXPECT_THROW(xform->transform(input_info, output_info, 0, 1, metadata), std::runtime_error);
}

} // namespace
