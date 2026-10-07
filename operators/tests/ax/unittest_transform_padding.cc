// Copyright Axelera AI, 2023
#include <algorithm>
#include <gmock/gmock.h>
#include "unittest_ax_common.h"

std::vector<std::uint8_t>
range(size_t n)
{
  std::vector<std::uint8_t> data(n);
  std::iota(data.begin(), data.end(), std::uint8_t{ 0 });
  return data;
}

TEST(transform_padding, non_tensor_input)
{
  auto xform = Ax::LoadTransform("padding", {});
  AxDataInterface inp_empty;
  EXPECT_THROW(xform->set_output_interface(inp_empty), std::runtime_error);
  AxVideoInterface inp_video{ {}, nullptr };
  EXPECT_THROW(xform->set_output_interface(inp_video), std::runtime_error);
}

void
check_init_failures(const std::string &padding,
    std::vector<std::vector<int>> shapes, int bytes, const std::string &regex)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", padding },
    { "fill", "42" },
  };
  auto xform = Ax::LoadTransform("padding", input);
  AxTensorsInterface inp;
  for (auto shape : shapes) {
    inp.push_back({ shape, bytes, nullptr });
  }
  try {
    xform->set_output_interface(inp);
  } catch (const std::runtime_error &e) {
    auto s = std::string{ e.what() };
    EXPECT_THAT(s, testing::MatchesRegex(regex));
    return;
  }
  FAIL() << "Expected runtime_error with message:\n  " << regex << "\nBut no exception was thrown";
}

TEST(transform_padding, test_invalid_squeeze_or_padding)
{
  check_init_failures("0", {}, 1, ".* requires at least one int8 tensor.*");
  check_init_failures("0", { { 1024 } }, 2, ".* requires at least one int8 tensor.*");
  check_init_failures("", { { 1024 }, { 1024 } }, 1, ".*no padding configurations provided.*");
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
  check_init_failures("0,0,0,0", { { 1, 1024 }, { 1, 1024 }, { 1, 1024 } }, 1,
      ".*fewer padding configurations than tensors.*");
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
  auto xform = Ax::LoadTransform("padding", input);
  auto inp_data = range(1024);
  const auto out_size = int(inp_data.size()) + left + right;
  std::vector<std::uint8_t> out_data(out_size);
  std::vector<std::uint8_t> expected(
      inp_data.begin() - left, inp_data.begin() - left + out_size);
  AxTensorsInterface inp{ { { 1, 1, 1, 1024 }, 1, inp_data.data() } };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out_iface.size(), 1);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size }));
  EXPECT_EQ(out_iface[0].bytes, 1);
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

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
  auto xform = Ax::LoadTransform("padding", input);
  auto inp_data = range(1000);
  const auto out_size = int(inp_data.size()) + left + right;
  std::vector<std::uint8_t> out_data(inp_data.size() + left + right);
  auto expected = range(inp_data.size());
  std::vector<std::uint8_t> extra(left, std::uint8_t{ 42 });
  expected.insert(expected.begin(), extra.begin(), extra.end());
  expected.resize(out_size, std::uint8_t{ 42 });
  AxTensorsInterface inp{ { { 1, int(inp_data.size()) }, 1, inp_data.data() } };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out_iface.size(), 1);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size }));
  EXPECT_EQ(out_iface[0].bytes, 1);
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

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
  auto xform = Ax::LoadTransform("padding", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  const auto out_size = int(inp_data.size()) + left + right;
  std::vector<std::uint8_t> out_data(out_size, 0xcd);
  std::vector<std::uint8_t> expected{ 0xcd, 0xcd, 0, 1, 2, 3, 0xcd, 0xcd, 0xcd };
  AxTensorsInterface inp{ { { 1, 1, 1, in_size }, 1, inp_data.data() } };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
  ASSERT_EQ(out_iface.size(), 1);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size }));
  EXPECT_EQ(out_iface[0].bytes, 1);
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  EXPECT_EQ(expected, out_data);
}

TEST(input_shape, incompatible)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0" },
    { "input_shape", "1,1,4,1" },
  };
  auto xform = Ax::LoadTransform("padding", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 1, 3 }, 1, inp_data.data() } };
  EXPECT_THROW(xform->set_output_interface(inp), std::runtime_error);
}

TEST(output_shape, incompatible)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0" },
    { "output_shape", "1,1,4,1" },
  };
  auto xform = Ax::LoadTransform("padding", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 1, 3 }, 1, inp_data.data() } };
  EXPECT_THROW(xform->set_output_interface(inp), std::runtime_error);
}

TEST(input_shape, is_output_when_no_output_shape)
{
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,0,0,0,0,0" },
    { "input_shape", "4,2,6,1" },
  };
  auto xform = Ax::LoadTransform("padding", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 4, 2, 2, 3 }, 1, inp_data.data() } };
  auto interface = xform->set_output_interface(inp);
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
  auto xform = Ax::LoadTransform("padding", input);
  auto in_size = 4;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 2, 3 }, 1, inp_data.data() } };
  auto interface = xform->set_output_interface(inp);
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
  auto xform = Ax::LoadTransform("padding", input);
  auto in_size = 6;
  auto inp_data = range(in_size);
  AxTensorsInterface inp{ { { 1, 1, 2, 3 }, 1, inp_data.data() } };
  auto out_size = 10;
  std::vector<std::uint8_t> out_data(out_size, 0xcd);
  std::vector<std::uint8_t> expected{ 99, 0, 1, 2, 3, 4, 5, 99, 99, 99 };
  AxTensorsInterface out{ { { 1, out_size }, 1, out_data.data() } };

  auto interface = xform->set_output_interface(inp);
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);
  ASSERT_EQ(expected, out_data);
}

TEST(multi_tensor_padding, test_different_padding_for_each_tensor)
{
  // Test with different paddings for multiple tensors
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,2,2|0,0,1,3|0,0,0,4" },
    { "fill", "42" },
  };
  auto xform = Ax::LoadTransform("padding", input);

  // Create three input tensors with different sizes
  const int tensor1_size = 4;
  const int tensor2_size = 6;
  const int tensor3_size = 8;

  auto inp_data1 = range(tensor1_size);
  auto inp_data2 = range(tensor2_size);
  auto inp_data3 = range(tensor3_size);

  // Calculate expected output sizes
  const int out_size1 = tensor1_size + 2 + 2; // padding 2,2
  const int out_size2 = tensor2_size + 1 + 3; // padding 1,3
  const int out_size3 = tensor3_size + 0 + 4; // padding 0,4

  std::vector<std::uint8_t> out_data1(out_size1);
  std::vector<std::uint8_t> out_data2(out_size2);
  std::vector<std::uint8_t> out_data3(out_size3);

  // Create expected outputs
  std::vector<std::uint8_t> expected1(out_size1, 42);
  std::copy(inp_data1.begin(), inp_data1.end(), expected1.begin() + 2);

  std::vector<std::uint8_t> expected2(out_size2, 42);
  std::copy(inp_data2.begin(), inp_data2.end(), expected2.begin() + 1);

  std::vector<std::uint8_t> expected3(out_size3, 42);
  std::copy(inp_data3.begin(), inp_data3.end(), expected3.begin());

  // Set up input and output interfaces
  AxTensorsInterface inp{ { { 1, tensor1_size }, 1, inp_data1.data() },
    { { 1, tensor2_size }, 1, inp_data2.data() },
    { { 1, tensor3_size }, 1, inp_data3.data() } };

  AxTensorsInterface out{ { { 1, out_size1 }, 1, out_data1.data() },
    { { 1, out_size2 }, 1, out_data2.data() },
    { { 1, out_size3 }, 1, out_data3.data() } };

  auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));

  // Verify output interface has the correct sizes
  ASSERT_EQ(out_iface.size(), 3);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size1 }));
  EXPECT_EQ(out_iface[1].sizes, std::vector<int>({ 1, out_size2 }));
  EXPECT_EQ(out_iface[2].sizes, std::vector<int>({ 1, out_size3 }));

  // Apply the transformation
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Verify the results
  EXPECT_EQ(expected1, out_data1);
  EXPECT_EQ(expected2, out_data2);
  EXPECT_EQ(expected3, out_data3);
}

TEST(multi_tensor_padding, test_fallback_padding)
{
  // Test with fewer padding configurations than tensors
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,2,2|0,0,1,3" }, // Only 2 padding configs for 3 tensors
    { "fill", "42" },
  };
  auto xform = Ax::LoadTransform("padding", input);

  // Create three input tensors with different sizes
  const int tensor1_size = 4;
  const int tensor2_size = 6;
  const int tensor3_size = 8;

  auto inp_data1 = range(tensor1_size);
  auto inp_data2 = range(tensor2_size);
  auto inp_data3 = range(tensor3_size);

  // Set up input interface with 3 tensors
  AxTensorsInterface inp{ { { 1, tensor1_size }, 1, inp_data1.data() },
    { { 1, tensor2_size }, 1, inp_data2.data() },
    { { 1, tensor3_size }, 1, inp_data3.data() } };

  // Now we expect an error when trying to set output interface
  try {
    auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));
    FAIL() << "Expected runtime_error because there are fewer padding configurations than tensors";
  } catch (const std::runtime_error &e) {
    std::string error_message = e.what();
    EXPECT_THAT(error_message,
        testing::MatchesRegex(".*fewer padding configurations than tensors.*"));
    EXPECT_THAT(error_message, testing::HasSubstr("expected 3 but got 2"));
  }
}

TEST(multi_tensor_padding, test_depadding_multiple_tensors)
{
  // Test with negative paddings (depadding) for multiple tensors
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,0,-1|0,0,-2,0|0,0,-1,-1" },
  };
  auto xform = Ax::LoadTransform("padding", input);

  // Create three input tensors
  const int tensor1_size = 8;
  const int tensor2_size = 8;
  const int tensor3_size = 8;

  auto inp_data1 = range(tensor1_size);
  auto inp_data2 = range(tensor2_size);
  auto inp_data3 = range(tensor3_size);

  // Calculate expected output sizes after depadding
  const int out_size1 = tensor1_size - 1; // padding 0,-1
  const int out_size2 = tensor2_size - 2; // padding -2,0
  const int out_size3 = tensor3_size - 2; // padding -1,-1

  std::vector<std::uint8_t> out_data1(out_size1);
  std::vector<std::uint8_t> out_data2(out_size2);
  std::vector<std::uint8_t> out_data3(out_size3);

  // Create expected outputs
  std::vector<std::uint8_t> expected1(inp_data1.begin(), inp_data1.begin() + out_size1);
  std::vector<std::uint8_t> expected2(inp_data2.begin() + 2, inp_data2.begin() + 2 + out_size2);
  std::vector<std::uint8_t> expected3(inp_data3.begin() + 1, inp_data3.begin() + 1 + out_size3);

  // Set up input and output interfaces
  AxTensorsInterface inp{ { { 1, tensor1_size }, 1, inp_data1.data() },
    { { 1, tensor2_size }, 1, inp_data2.data() },
    { { 1, tensor3_size }, 1, inp_data3.data() } };

  AxTensorsInterface out{ { { 1, out_size1 }, 1, out_data1.data() },
    { { 1, out_size2 }, 1, out_data2.data() },
    { { 1, out_size3 }, 1, out_data3.data() } };

  auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));

  // Verify output interface has the correct sizes
  ASSERT_EQ(out_iface.size(), 3);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size1 }));
  EXPECT_EQ(out_iface[1].sizes, std::vector<int>({ 1, out_size2 }));
  EXPECT_EQ(out_iface[2].sizes, std::vector<int>({ 1, out_size3 }));

  // Apply the transformation
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Verify the results
  EXPECT_EQ(expected1, out_data1);
  EXPECT_EQ(expected2, out_data2);
  EXPECT_EQ(expected3, out_data3);
}

TEST(multi_tensor_padding, test_exact_padding_match)
{
  // Test with exactly matching number of paddings and tensors
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,2,2|0,0,1,3|0,0,0,4" }, // Exactly 3 padding configs for 3 tensors
    { "fill", "42" },
  };
  auto xform = Ax::LoadTransform("padding", input);

  // Create three input tensors with different sizes
  const int tensor1_size = 4;
  const int tensor2_size = 6;
  const int tensor3_size = 8;

  auto inp_data1 = range(tensor1_size);
  auto inp_data2 = range(tensor2_size);
  auto inp_data3 = range(tensor3_size);

  // Calculate expected output sizes
  const int out_size1 = tensor1_size + 2 + 2; // padding 2,2
  const int out_size2 = tensor2_size + 1 + 3; // padding 1,3
  const int out_size3 = tensor3_size + 0 + 4; // padding 0,4

  std::vector<std::uint8_t> out_data1(out_size1);
  std::vector<std::uint8_t> out_data2(out_size2);
  std::vector<std::uint8_t> out_data3(out_size3);

  // Create expected outputs
  std::vector<std::uint8_t> expected1(out_size1, 42);
  std::copy(inp_data1.begin(), inp_data1.end(), expected1.begin() + 2);

  std::vector<std::uint8_t> expected2(out_size2, 42);
  std::copy(inp_data2.begin(), inp_data2.end(), expected2.begin() + 1);

  std::vector<std::uint8_t> expected3(out_size3, 42);
  std::copy(inp_data3.begin(), inp_data3.end(), expected3.begin());

  // Set up input and output interfaces
  AxTensorsInterface inp{ { { 1, tensor1_size }, 1, inp_data1.data() },
    { { 1, tensor2_size }, 1, inp_data2.data() },
    { { 1, tensor3_size }, 1, inp_data3.data() } };

  AxTensorsInterface out{ { { 1, out_size1 }, 1, out_data1.data() },
    { { 1, out_size2 }, 1, out_data2.data() },
    { { 1, out_size3 }, 1, out_data3.data() } };

  // This should now work properly - no errors
  auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));

  // Verify output interface has the correct sizes
  ASSERT_EQ(out_iface.size(), 3);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size1 }));
  EXPECT_EQ(out_iface[1].sizes, std::vector<int>({ 1, out_size2 }));
  EXPECT_EQ(out_iface[2].sizes, std::vector<int>({ 1, out_size3 }));

  // Apply the transformation
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Verify the results
  EXPECT_EQ(expected1, out_data1);
  EXPECT_EQ(expected2, out_data2);
  EXPECT_EQ(expected3, out_data3);
}

TEST(multi_tensor_padding, test_more_paddings_than_tensors)
{
  // Test with more padding configurations than tensors
  std::unordered_map<std::string, std::string> input = {
    { "padding", "0,0,2,2|0,0,1,3|0,0,0,4|0,0,3,3" }, // 4 padding configs for 2 tensors
    { "fill", "42" },
  };
  auto xform = Ax::LoadTransform("padding", input);

  // Create only two input tensors
  const int tensor1_size = 4;
  const int tensor2_size = 6;

  auto inp_data1 = range(tensor1_size);
  auto inp_data2 = range(tensor2_size);

  // Calculate expected output sizes
  const int out_size1 = tensor1_size + 2 + 2; // padding 2,2
  const int out_size2 = tensor2_size + 1 + 3; // padding 1,3

  std::vector<std::uint8_t> out_data1(out_size1);
  std::vector<std::uint8_t> out_data2(out_size2);

  // Create expected outputs
  std::vector<std::uint8_t> expected1(out_size1, 42);
  std::copy(inp_data1.begin(), inp_data1.end(), expected1.begin() + 2);

  std::vector<std::uint8_t> expected2(out_size2, 42);
  std::copy(inp_data2.begin(), inp_data2.end(), expected2.begin() + 1);

  // Set up input and output interfaces
  AxTensorsInterface inp{ { { 1, tensor1_size }, 1, inp_data1.data() },
    { { 1, tensor2_size }, 1, inp_data2.data() } };

  AxTensorsInterface out{ { { 1, out_size1 }, 1, out_data1.data() },
    { { 1, out_size2 }, 1, out_data2.data() } };

  // This should work fine - it's okay to have more padding configs than needed
  auto out_iface = std::get<AxTensorsInterface>(xform->set_output_interface(inp));

  // Verify output interface has the correct sizes
  ASSERT_EQ(out_iface.size(), 2);
  EXPECT_EQ(out_iface[0].sizes, std::vector<int>({ 1, out_size1 }));
  EXPECT_EQ(out_iface[1].sizes, std::vector<int>({ 1, out_size2 }));

  // Apply the transformation
  Ax::MetaMap metadata;
  xform->transform(inp, out, 0, 1, metadata);

  // Verify the results
  EXPECT_EQ(expected1, out_data1);
  EXPECT_EQ(expected2, out_data2);
}
