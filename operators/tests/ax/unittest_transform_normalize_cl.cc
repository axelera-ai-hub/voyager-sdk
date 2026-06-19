// Copyright Axelera AI, 2024
#include "unittest_ax_common.h"

extern bool has_opencl_platform();

namespace
{
TEST(normalize_cl, no_mean_or_scale)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };

  auto xform = Ax::LoadTransform("normalize_cl", input);
  std::vector<uint8_t> in_buf(4 * 4);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x81, 0x82, 0x83, 0x03, 0x85, 0x86, 0x87, 0x07,
    0x89, 0x8a, 0x8b, 0x0b, 0x8d, 0x8e, 0x8f, 0x0f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 1, 4, 4 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 1, 4, 4 }, 1, out_buf.data() } };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_rgb)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };

  auto xform = Ax::LoadTransform("normalize_cl", input);
  std::vector<uint8_t> in_buf(4 * 3);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x81, 0x82, 0x83, 0x00, 0x84, 0x85, 0x86, 0x00,
    0x87, 0x88, 0x89, 0x00, 0x8a, 0x8b, 0x8c, 0x00,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 1, 4, 3 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 1, 4, 4 }, 1, out_buf.data() } };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_grey)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };

  auto xform = Ax::LoadTransform("normalize_cl", input);
  std::vector<uint8_t> in_buf(16);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(16);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 1, 16, 1 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 1, 16, 1 }, 1, out_buf.data() } };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_4x4)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };

  auto xform = Ax::LoadTransform("normalize_cl", input);
  std::vector<uint8_t> in_buf(4 * 4 * 2);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4 * 2);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x81, 0x82, 0x83, 0x03, 0x85, 0x86, 0x87, 0x07,
    0x89, 0x8a, 0x8b, 0x0b, 0x8d, 0x8e, 0x8f, 0x0f,
    0x91, 0x92, 0x93, 0x13, 0x95, 0x96, 0x97, 0x17,
    0x99, 0x9a, 0x9b, 0x1b, 0x9d, 0x9e, 0x9f, 0x1f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, out_buf.data() } };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_4x4_grey)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };

  auto xform = Ax::LoadTransform("normalize_cl", input);
  std::vector<uint8_t> in_buf(16);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(16);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 4, 4, 1 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 4, 4, 1 }, 1, out_buf.data() } };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_4x4_x)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.0, 0.0 ,0.0" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "-127.5" },
    { "quant_scale", "0.003919653594493866" },
  };

  auto xform = Ax::LoadTransform("normalize_cl", input);
  std::vector<uint8_t> in_buf(4 * 4 * 2);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4 * 2);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x81, 0x82, 0x83, 0x03, 0x85, 0x86, 0x87, 0x07,
    0x89, 0x8a, 0x8b, 0x0b, 0x8d, 0x8e, 0x8f, 0x0f,
    0x91, 0x92, 0x93, 0x13, 0x95, 0x96, 0x97, 0x17,
    0x99, 0x9a, 0x9b, 0x1b, 0x9d, 0x9e, 0x9f, 0x1f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, out_buf.data() } };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  EXPECT_EQ(out_buf, expected);
}

TEST(normalize_cl, no_mean_or_scale_4x4_x_grey)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.0, 0.0 ,0.0" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "-127.5" },
    { "quant_scale", "0.003919653594493866" },
  };

  auto xform = Ax::LoadTransform("normalize_cl", input);
  std::vector<uint8_t> in_buf(4 * 4);
  std::iota(in_buf.begin(), in_buf.end(), 0);
  std::vector<uint8_t> out_buf(4 * 4);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 4, 4, 1 }, 1, in_buf.data() } };
  auto out = AxTensorsInterface{ { { 1, 4, 4, 1 }, 1, out_buf.data() } };
  Ax::MetaMap metadata;
  xform->transform(in, out, 0, 1, metadata);
  EXPECT_EQ(out_buf, expected);
}

} // namespace
