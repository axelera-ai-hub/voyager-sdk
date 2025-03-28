// Copyright Axelera AI, 2025
#include <gmock/gmock.h>
#include "unittest_inplace_common.h"

void
test_no_mean_or_scale(std::string simd = std::string())
{
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };
  if (!simd.empty()) {
    input["simd"] = simd;
  }

  Inplacer normalizer("libinplace_normalize.so", input);
  std::vector<uint8_t> in_buf(4 * 4);
  std::iota(in_buf.begin(), in_buf.end(), 0);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x80, 0x81, 0x82, 0x03, 0x84, 0x85, 0x86, 0x07,
    0x88, 0x89, 0x8a, 0x0b, 0x8c, 0x8d, 0x8e, 0x0f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 1, 4, 4 }, 1, in_buf.data() } };
  normalizer.inplace(in);
  EXPECT_EQ(in_buf, expected);
}

TEST(normalize, no_mean_or_scale)
{
  test_no_mean_or_scale();
  test_no_mean_or_scale("avx2");
  test_no_mean_or_scale("avx512");
}

void
test_no_mean_or_scale_4x4(std::string simd = std::string())
{
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.5, 0.5 ,0.5" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "0" },
    { "quant_scale", "0.003919653594493866" },
  };
  if (!simd.empty()) {
    input["simd"] = simd;
  }

  Inplacer normalizer("libinplace_normalize.so", input);
  std::vector<uint8_t> in_buf(4 * 4 * 2);
  std::iota(in_buf.begin(), in_buf.end(), 0);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x80, 0x81, 0x82, 0x03, 0x84, 0x85, 0x86, 0x07,
    0x88, 0x89, 0x8a, 0x0b, 0x8c, 0x8d, 0x8e, 0x0f,
    0x90, 0x91, 0x92, 0x13, 0x94, 0x95, 0x96, 0x17,
    0x98, 0x99, 0x9a, 0x1b, 0x9c, 0x9d, 0x9e, 0x1f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, in_buf.data() } };
  normalizer.inplace(in);
  EXPECT_EQ(in_buf, expected);
}

TEST(normalize, no_mean_or_scale_4x4)
{
  test_no_mean_or_scale_4x4();
  test_no_mean_or_scale_4x4("avx2");
  test_no_mean_or_scale_4x4("avx512");
}

void
test_no_mean_or_scale_4x4_x(std::string simd = std::string())
{
  std::unordered_map<std::string, std::string> input = {
    { "mean", "0.0, 0.0 ,0.0" },
    { "std", "1.0, 1.0, 1.0" },
    { "quant_zeropoint", "-127.5" },
    { "quant_scale", "0.003919653594493866" },
  };
  if (!simd.empty()) {
    input["simd"] = simd;
  }

  Inplacer normalizer("libinplace_normalize.so", input);
  std::vector<uint8_t> in_buf(4 * 4 * 2);
  std::iota(in_buf.begin(), in_buf.end(), 0);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    0x80, 0x82, 0x83, 0x03, 0x85, 0x86, 0x87, 0x07,
    0x89, 0x8a, 0x8b, 0x0b, 0x8d, 0x8e, 0x8f, 0x0f,
    0x91, 0x92, 0x93, 0x13, 0x95, 0x96, 0x97, 0x17,
    0x99, 0x9a, 0x9b, 0x1b, 0x9d, 0x9e, 0x9f, 0x1f,
    // clang-format on
  };
  auto in = AxTensorsInterface{ { { 1, 2, 4, 4 }, 1, in_buf.data() } };
  normalizer.inplace(in);
  EXPECT_EQ(in_buf, expected);
}

TEST(normalize, no_mean_or_scale_4x4_x)
{
  test_no_mean_or_scale_4x4_x();
  test_no_mean_or_scale_4x4_x("avx2");
  test_no_mean_or_scale_4x4_x("avx512");
}
