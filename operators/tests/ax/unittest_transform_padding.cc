// Copyright Axelera AI, 2023
#include <algorithm>
#include <gmock/gmock.h>
#include "unittest_transform_common.h"

std::vector<std::uint8_t>
range(size_t n)
{
  std::vector<std::uint8_t> data(n);
  std::iota(data.begin(), data.end(), std::uint8_t{ 0 });
  return data;
}

TEST(transform_padding, non_tensor_input)
{
  Transformer transformer("libtransform_padding.so", {});
  AxDataInterface inp_empty;
  EXPECT_THROW(transformer.set_output_interface(inp_empty), std::runtime_error);
  AxVideoInterface inp_video{ {}, nullptr };
  EXPECT_THROW(transformer.set_output_interface(inp_video), std::runtime_error);
}

void
check_init_failures(const std::string &padding,
    std::vector<std::vector<int>> shapes, int bytes, const std::string &regex)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", padding },
    { "fill", "42" },
  };
  Transformer transformer("libtransform_padding.so", input);
  AxTensorsInterface inp;
  for (auto shape : shapes) {
    inp.push_back({ shape, bytes, nullptr });
  }
  try {
    transformer.set_output_interface(inp);
  } catch (const std::runtime_error &e) {
    auto s = std::string{ e.what() };
    EXPECT_THAT(s, testing::MatchesRegex(regex));
    return;
  }
  FAIL() << "Expected runtime_error with message:\n  " << regex << "\nBut no exception was thrown";
}

TEST(transform_padding, test_invalid_squeeze_or_padding)
{
  check_init_failures("0", {}, 1, ".* requires single int8 tensor.*");
  check_init_failures("0", { { 1024 } }, 2, ".* requires single int8 tensor.*");
  check_init_failures("0", { { 1024 }, { 1024 } }, 1, ".* requires single int8 tensor.*");
  check_init_failures("0", { { 1, 1, 1, 1024 } }, 1, ".* must be a multiple of 2.*");
  check_init_failures("0,24", { { 1, 1, 3, 1024 } }, 1, ".*too short for input shape.*");
  check_init_failures("0,0,0,24", { { 1, 3, 1, 1024 } }, 1, ".*too short for input shape.*");
  check_init_failures("0,0,0,24", { { 3, 1, 1, 1024 } }, 1, ".*too short for input shape.*");
  check_init_failures("0,0,0,24", { { 3, 3, 3, 1024 } }, 1, ".*too short for input shape.*");
  check_init_failures("0,-16,0,24", { { 1, 1, 1, 1024 } }, 1, ".*can remove or add padding.*");
  check_init_failures("0,-16,0,24", { { 1, 1, 1, 1024 } }, 1, ".*can remove or add padding.*");
  check_init_failures("0,0,0,-1024", { { 1, 1024 } }, 1,
      ".*negative padding \\(0,0,0,-1024\\) greater than.*");
  check_init_failures("0,0,0,-1025", { { 1, 1024 } }, 1,
      ".*negative padding \\(0,0,0,-1025\\) greater than.*");
  check_init_failures("0,0,-512,-512", { { 1, 1024 } }, 1,
      ".*negative padding \\(0,0,-512,-512\\) greater than.*");
  check_init_failures("0,0,-1024,0", { { 1, 1024 } }, 1,
      ".*negative padding \\(0,0,-1024,0\\) greater than.*");
}

class remove_padding_fixture : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(remove_padding_fixture, test_remove_padding_and_reshape)
{
  const auto left = std::get<0>(GetParam());
  const auto right = std::get<1>(GetParam());
  ASSERT_TRUE(left <= 0 && right <= 0) << "the test assumes that left/right are <=0";
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0," + std::to_string(left) + "," + std::to_string(right) },
    { "fill", "42" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto inp_data = range(1024);
  const auto out_size = int(inp_data.size()) + left + right;
  std::vector<std::uint8_t> out_data(out_size);
  std::vector<std::uint8_t> expected(
      inp_data.begin() - left, inp_data.begin() - left + out_size);
  AxTensorsInterface inp{ { { 1, 1, 1, 1024 }, 1, inp_data.data() } };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto out_iface = std::get<AxTensorsInterface>(transformer.set_output_interface(inp));
  ASSERT_EQ(out_iface.size(), 1);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size }));
  EXPECT_EQ(out_iface[0].bytes, 1);
  transformer.transform(inp, out);

  EXPECT_EQ(expected, out_data);
}

INSTANTIATE_TEST_CASE_P(test_remove_padding, remove_padding_fixture,
    ::testing::Values(std::make_tuple(0, -24), std::make_tuple(-24, 0),
        std::make_tuple(-12, -12)));


class add_padding_fixture : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(add_padding_fixture, test_add_padding_2d)
{
  const auto left = std::get<0>(GetParam());
  const auto right = std::get<1>(GetParam());
  ASSERT_TRUE(left >= 0 && right >= 0) << "the test assumes that left/right are >=0";
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0," + std::to_string(left) + "," + std::to_string(right) },
    { "fill", "42" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto inp_data = range(1000);
  const auto out_size = int(inp_data.size()) + left + right;
  std::vector<std::uint8_t> out_data(inp_data.size() + left + right);
  auto expected = range(inp_data.size());
  std::vector<std::uint8_t> extra(left, std::uint8_t{ 42 });
  expected.insert(expected.begin(), extra.begin(), extra.end());
  expected.resize(out_size, std::uint8_t{ 42 });
  AxTensorsInterface inp{ { { 1, int(inp_data.size()) }, 1, inp_data.data() } };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto out_iface = std::get<AxTensorsInterface>(transformer.set_output_interface(inp));
  ASSERT_EQ(out_iface.size(), 1);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size }));
  EXPECT_EQ(out_iface[0].bytes, 1);
  transformer.transform(inp, out);

  EXPECT_EQ(expected, out_data);
}

INSTANTIATE_TEST_CASE_P(test_add_padding, add_padding_fixture,
    ::testing::Values(std::make_tuple(0, 24), std::make_tuple(24, 0),
        std::make_tuple(12, 12)));

TEST(optional_padding, test_add_padding_2d)
{
  const auto left = 2;
  const auto right = 3;
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,2,3" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  const auto out_size = int(inp_data.size()) + left + right;
  std::vector<std::uint8_t> out_data(out_size, 0xcd);
  std::vector<std::uint8_t> expected{ 0xcd, 0xcd, 0, 1, 2, 3, 0xcd, 0xcd, 0xcd };
  AxTensorsInterface inp{ { { 1, 1, 1, in_size }, 1, inp_data.data() } };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto out_iface = std::get<AxTensorsInterface>(transformer.set_output_interface(inp));
  ASSERT_EQ(out_iface.size(), 1);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size }));
  EXPECT_EQ(out_iface[0].bytes, 1);
  transformer.transform(inp, out);

  EXPECT_EQ(expected, out_data);
}

TEST(input_shape, incompatible)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0" },
    { "input_shape", "1,1,4,1" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 1, 3 }, 1, inp_data.data() } };
  EXPECT_THROW(transformer.set_output_interface(inp), std::runtime_error);
}

TEST(output_shape, incompatible)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0" },
    { "output_shape", "1,1,4,1" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 1, 3 }, 1, inp_data.data() } };
  EXPECT_THROW(transformer.set_output_interface(inp), std::runtime_error);
}

TEST(input_shape, is_output_when_no_output_shape)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0,0,0,0,0" },
    { "input_shape", "4,2,6,1" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 4, 2, 2, 3 }, 1, inp_data.data() } };
  auto interface = transformer.set_output_interface(inp);
  auto out = std::get<AxTensorsInterface>(interface);
  ASSERT_EQ(out[0].sizes, std::vector<int>({ 4, 2, 6, 1 }));
}

TEST(input_shape, reshaped_padding)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0,2,2,0,0" },
    { "input_shape", "1,1,6,1" },
    { "fill", "99" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 2, 3 }, 1, inp_data.data() } };
  auto interface = transformer.set_output_interface(inp);
  auto out = std::get<AxTensorsInterface>(interface);
  ASSERT_EQ(out[0].sizes, std::vector<int>({ 1, 1, 10, 1 }));
}

TEST(input_shape, reshaped_padding_values)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0,1,3,0,0" },
    { "input_shape", "1,1,6,1" },
    { "fill", "99" },
  };
  Transformer transformer("libtransform_padding.so", input);
  auto in_size = 6;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 2, 3 }, 1, inp_data.data() } };
  auto out_size = 10;
  std::vector<std::uint8_t> out_data(out_size, 0xcd);
  std::vector<std::uint8_t> expected{ 99, 0, 1, 2, 3, 4, 5, 99, 99, 99 };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto interface = transformer.set_output_interface(inp);
  transformer.transform(inp, out);
  ASSERT_EQ(expected, out_data);
}
