// Copyright Axelera AI, 2026
#include "unittest_ax_common.h"

extern bool has_opencl_platform();

namespace
{
const auto crop_cl_lib = "roicrop_cl";

// --- Property validation (same rules as the CPU roicrop) ---

TEST(roicrop_cl, meta_and_cropping_x_fails)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "meta_key", "anything" },
    { "left", "224" },
  };
  EXPECT_THROW(Ax::LoadTransform(crop_cl_lib, input), std::runtime_error);
}

TEST(roicrop_cl, meta_and_cropping_y_fails)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "meta_key", "anything" },
    { "top", "224" },
  };
  EXPECT_THROW(Ax::LoadTransform(crop_cl_lib, input), std::runtime_error);
}

TEST(roicrop_cl, meta_and_cropping_width_fails)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "meta_key", "anything" },
    { "width", "224" },
  };
  EXPECT_THROW(Ax::LoadTransform(crop_cl_lib, input), std::runtime_error);
}

TEST(roicrop_cl, meta_and_cropping_height_fails)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "meta_key", "anything" },
    { "height", "224" },
  };
  EXPECT_THROW(Ax::LoadTransform(crop_cl_lib, input), std::runtime_error);
}

TEST(roicrop_cl, no_meta_missing_coords_fails)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "left", "0" }, { "top", "0" },
    // missing width and height
  };
  EXPECT_THROW(Ax::LoadTransform(crop_cl_lib, input), std::runtime_error);
}

// --- set_output_interface_from_meta (geometry computation, no GPU) ---

TEST(roicrop_cl, crop_box)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "left", "200" },
    { "top", "100" },
    { "width", "224" },
    { "height", "224" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  auto out_interface = xform->set_output_interface_from_meta(video_info, 0, 1, metadata);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.width, 224);
  EXPECT_EQ(info.height, 224);
  EXPECT_EQ(info.x_offset, 200);
  EXPECT_EQ(info.y_offset, 100);
  EXPECT_EQ(info.cropped, true);
}

TEST(roicrop_cl, x_bounds)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "left", "650" },
    { "top", "100" },
    { "width", "224" },
    { "height", "224" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  EXPECT_THROW(xform->set_output_interface_from_meta(video_info, 0, 1, metadata),
      std::runtime_error);
}

TEST(roicrop_cl, y_bounds)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "left", "0" },
    { "top", "480" },
    { "width", "224" },
    { "height", "224" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  EXPECT_THROW(xform->set_output_interface_from_meta(video_info, 0, 1, metadata),
      std::runtime_error);
}

TEST(roicrop_cl, width_clipped)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "left", "460" },
    { "top", "0" },
    { "width", "224" },
    { "height", "224" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  auto out_interface = xform->set_output_interface_from_meta(video_info, 0, 1, metadata);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.width, 180);
  EXPECT_EQ(info.height, 224);
  EXPECT_EQ(info.x_offset, 460);
  EXPECT_EQ(info.y_offset, 0);
  EXPECT_EQ(info.cropped, true);
}

TEST(roicrop_cl, height_clipped)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "left", "400" },
    { "top", "300" },
    { "width", "224" },
    { "height", "224" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  auto out_interface = xform->set_output_interface_from_meta(video_info, 0, 1, metadata);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.width, 224);
  EXPECT_EQ(info.height, 180);
  EXPECT_EQ(info.x_offset, 400);
  EXPECT_EQ(info.y_offset, 300);
  EXPECT_EQ(info.cropped, true);
}

// --- GPU transform tests ---

// Helper: build an RGBA image with a repeating per-row gradient.
// Row r, column c (0-indexed): pixel = { r%256, c%256, (r+c)%256, 255 }
static std::vector<uint8_t>
make_rgba_image(int width, int height)
{
  std::vector<uint8_t> buf(width * height * 4);
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      int idx = (r * width + c) * 4;
      buf[idx + 0] = static_cast<uint8_t>(r % 256);
      buf[idx + 1] = static_cast<uint8_t>(c % 256);
      buf[idx + 2] = static_cast<uint8_t>((r + c) % 256);
      buf[idx + 3] = 255;
    }
  }
  return buf;
}

TEST(roicrop_cl, transform_crops_correct_pixels)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  const int in_w = 64, in_h = 64;
  const int crop_x = 10, crop_y = 8;
  const int out_w = 20, out_h = 16;

  std::unordered_map<std::string, std::string> props = {
    { "left", std::to_string(crop_x) },
    { "top", std::to_string(crop_y) },
    { "width", std::to_string(out_w) },
    { "height", std::to_string(out_h) },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, props);

  auto in_buf = make_rgba_image(in_w, in_h);
  std::vector<uint8_t> out_buf(out_w * out_h * 4, 0);

  auto in = AxVideoInterface{ { in_w, in_h, in_w * 4, 0, AxVideoFormat::RGBA },
    in_buf.data(), { static_cast<size_t>(in_w * 4) }, { 0 }, -1 };
  auto out = AxVideoInterface{ { out_w, out_h, out_w * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { static_cast<size_t>(out_w * 4) }, { 0 }, -1 };

  Ax::MetaMap metadata;
  ASSERT_NO_THROW({ xform->transform(in, out, 0, 1, metadata); });

  // Verify that every output pixel matches the expected crop from the input.
  auto expected = make_rgba_image(in_w, in_h);
  for (int r = 0; r < out_h; ++r) {
    for (int c = 0; c < out_w; ++c) {
      int out_idx = (r * out_w + c) * 4;
      int in_idx = ((r + crop_y) * in_w + (c + crop_x)) * 4;
      EXPECT_EQ(out_buf[out_idx + 0], expected[in_idx + 0])
          << "R mismatch at (" << c << "," << r << ")";
      EXPECT_EQ(out_buf[out_idx + 1], expected[in_idx + 1])
          << "G mismatch at (" << c << "," << r << ")";
      EXPECT_EQ(out_buf[out_idx + 2], expected[in_idx + 2])
          << "B mismatch at (" << c << "," << r << ")";
    }
  }
}

TEST(roicrop_cl, transform_out_of_bounds_fills_black)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }

  // Crop starts 4 pixels before the left edge: columns 0-3 of output are OOB.
  const int in_w = 32, in_h = 32;
  const int crop_x = -4, crop_y = 0;
  const int out_w = 8, out_h = 8;

  std::unordered_map<std::string, std::string> props = {
    { "left", std::to_string(crop_x) },
    { "top", std::to_string(crop_y) },
    { "width", std::to_string(out_w) },
    { "height", std::to_string(out_h) },
  };

  // Note: set_output_interface_from_meta won't throw here (margin case handled
  // by the kernel's black-fill), but we call transform directly with explicit
  // out dimensions matching the requested crop.
  auto xform = Ax::LoadTransform(crop_cl_lib, props);

  std::vector<uint8_t> in_buf(in_w * in_h * 4, 200); // fill with 200
  std::vector<uint8_t> out_buf(out_w * out_h * 4, 99);

  auto in = AxVideoInterface{ { in_w, in_h, in_w * 4, 0, AxVideoFormat::RGBA },
    in_buf.data(), { static_cast<size_t>(in_w * 4) }, { 0 }, -1 };
  auto out = AxVideoInterface{ { out_w, out_h, out_w * 4, 0, AxVideoFormat::RGBA },
    out_buf.data(), { static_cast<size_t>(out_w * 4) }, { 0 }, -1 };

  Ax::MetaMap metadata;
  ASSERT_NO_THROW({ xform->transform(in, out, 0, 1, metadata); });

  // Columns 0-3 are out of bounds → black (0,0,0,255).
  for (int r = 0; r < out_h; ++r) {
    for (int c = 0; c < 4; ++c) {
      int idx = (r * out_w + c) * 4;
      EXPECT_EQ(out_buf[idx + 0], 0) << "R at OOB col " << c << " row " << r;
      EXPECT_EQ(out_buf[idx + 1], 0) << "G at OOB col " << c << " row " << r;
      EXPECT_EQ(out_buf[idx + 2], 0) << "B at OOB col " << c << " row " << r;
    }
    // Columns 4-7 map to input columns 0-3 which have value 200.
    for (int c = 4; c < out_w; ++c) {
      int idx = (r * out_w + c) * 4;
      EXPECT_EQ(out_buf[idx + 0], 200) << "R at valid col " << c << " row " << r;
      EXPECT_EQ(out_buf[idx + 1], 200) << "G at valid col " << c << " row " << r;
      EXPECT_EQ(out_buf[idx + 2], 200) << "B at valid col " << c << " row " << r;
    }
  }
}

TEST(roicrop_cl, query_supports_opencl_buffers)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> props = {
    { "left", "0" },
    { "top", "0" },
    { "width", "16" },
    { "height", "16" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, props);
  EXPECT_TRUE(xform->query_supports(Ax::PluginFeature::opencl_buffers));
}

// --- Format conversion tests ---

// When the ROI extends out of bounds (no margin), the kernel must run.
// Without a format option, RGB input → RGBA output (add_alpha).
TEST(roicrop_cl, oob_crop_rgb_input_defaults_to_rgba)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  // left=-4 puts the ROI partially outside the left edge.
  std::unordered_map<std::string, std::string> props = {
    { "left", "-4" },
    { "top", "0" },
    { "width", "8" },
    { "height", "8" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, props);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  Ax::MetaMap metadata;

  auto out_interface = xform->set_output_interface_from_meta(video_info, 0, 1, metadata);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.format, AxVideoFormat::RGBA);
  EXPECT_FALSE(info.cropped);
}

// When the ROI extends out of bounds with an explicit format option, that
// format is applied to the output.
TEST(roicrop_cl, oob_crop_explicit_format_bgra)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> props = {
    { "left", "-4" },
    { "top", "0" },
    { "width", "8" },
    { "height", "8" },
    { "format", "bgra" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, props);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  auto out_interface = xform->set_output_interface_from_meta(video_info, 0, 1, metadata);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.format, AxVideoFormat::BGRA);
  EXPECT_FALSE(info.cropped);
}

// When the crop is fully in bounds, the format option is ignored and the
// output format matches the input format.
TEST(roicrop_cl, inbounds_crop_ignores_format_option)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> props = {
    { "left", "0" },
    { "top", "0" },
    { "width", "16" },
    { "height", "16" },
    { "format", "bgra" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, props);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  auto out_interface = xform->set_output_interface_from_meta(video_info, 0, 1, metadata);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.format, AxVideoFormat::RGBA);
  EXPECT_TRUE(info.cropped);
}

// An unsupported output format string is rejected at
// set_output_interface_from_meta time when the crop is OOB.
TEST(roicrop_cl, invalid_format_option_throws)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> props = {
    { "left", "-4" },
    { "top", "0" },
    { "width", "8" },
    { "height", "8" },
    { "format", "nv12" },
  };
  auto xform = Ax::LoadTransform(crop_cl_lib, props);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  Ax::MetaMap metadata;

  EXPECT_THROW(xform->set_output_interface_from_meta(video_info, 0, 1, metadata),
      std::runtime_error);
}

} // namespace
