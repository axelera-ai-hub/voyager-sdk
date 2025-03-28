#include <gmock/gmock.h>
#include "unittest_transform_common.h"

TEST(yolo_preproc, non_tensor_input)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  AxVideoInterface inp_video{ {}, nullptr };
  AxDataInterface input{ inp_video };
  EXPECT_THROW(transformer.set_output_interface(input), std::runtime_error);
}

TEST(yolo_preproc, test_single_tensor_input)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  AxTensorsInterface inp;
  inp.push_back({ { 1, 2, 2, 3 }, 1, nullptr });
  inp.push_back({ { 1, 2, 2, 3 }, 1, nullptr });
  AxDataInterface input{ inp };
  EXPECT_THROW(transformer.set_output_interface(input), std::runtime_error);
}

TEST(yolo_preproc, test_single_4D_tensor_input)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  AxTensorsInterface inp;
  inp.push_back({ { 1, 2, 3 }, 1, nullptr });
  AxDataInterface input{ inp };
  EXPECT_THROW(transformer.set_output_interface(input), std::runtime_error);
}

TEST(yolo_preproc, test_even_height_input)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  AxTensorsInterface inp;
  inp.push_back({ { 1, 1, 2, 3 }, 1, nullptr });
  AxDataInterface input{ inp };
  EXPECT_THROW(transformer.set_output_interface(input), std::runtime_error);
}

TEST(yolo_preproc, test_even_width_input)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  AxTensorsInterface inp;
  inp.push_back({ { 1, 2, 1, 3 }, 1, nullptr });
  AxDataInterface input{ inp };
  EXPECT_THROW(transformer.set_output_interface(input), std::runtime_error);
}

TEST(yolo_preproc, test_rgb_input)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  AxTensorsInterface inp;
  inp.push_back({ { 1, 2, 2, 2 }, 1, nullptr });
  AxDataInterface input{ inp };
  EXPECT_THROW(transformer.set_output_interface(input), std::runtime_error);
}

TEST(yolo_preproc, two_by_two)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  std::vector<int8_t> data(2 * 2 * 3);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 2, 2, 3 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(1 * 1 * 12);
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11,
    // clang-format on
  };

  AxTensorsInterface output{ { { 1, 1, 1, 12 }, 1, result.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};

  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, two_by_two_by_two)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  std::vector<int8_t> data(2 * 2 * 2 * 3);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 2, 2, 2, 3 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 1 * 1 * 12);
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11,
    12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23,
    // clang-format on
  };

  AxTensorsInterface output{ { { 2, 1, 1, 12 }, 1, result.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};

  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, two_by_two_rgba_in)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  std::vector<int8_t> data(2 * 2 * 4);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 2, 2, 4 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(1 * 1 * 12);
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 8, 9, 10, 4, 5, 6, 12, 13, 14,
    // clang-format on
  };

  AxTensorsInterface output{ { { 1, 1, 1, 12 }, 1, result.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};

  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}


TEST(yolo_preproc, four_by_four)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  std::vector<int8_t> data(4 * 4 * 3);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 4, 4, 3 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 2 * 12);
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17,
    6, 7, 8, 18, 19, 20, 9, 10, 11, 21, 22, 23,
    24, 25, 26, 36, 37, 38, 27, 28, 29, 39, 40, 41,
    30, 31, 32, 42, 43, 44, 33, 34, 35, 45, 46, 47,
    // clang-format on
  };

  AxTensorsInterface output{ { { 1, 2, 2, 12 }, 1, result.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};

  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, four_by_four_rgba_in)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  std::vector<int8_t> data(4 * 4 * 4);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 4, 4, 4 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 2 * 12);
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 16, 17, 18, 4, 5, 6, 20, 21, 22,
    8, 9, 10, 24, 25, 26, 12, 13, 14, 28, 29, 30,
    32, 33, 34, 48, 49, 50, 36, 37, 38, 52, 53, 54,
    40, 41, 42, 56, 57, 58, 44, 45, 46, 60, 61, 62,
    // clang-format on
  };

  AxTensorsInterface output{ { { 1, 2, 2, 12 }, 1, result.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};

  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, two_by_four_by_four_rgba_in)
{
  Transformer transformer("libtransform_yolopreproc.so", {});
  std::vector<int8_t> data(2 * 4 * 4 * 4);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 2, 4, 4, 4 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 2 * 2 * 12);
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 16, 17, 18, 4, 5, 6, 20, 21, 22,
    8, 9, 10, 24, 25, 26, 12, 13, 14, 28, 29, 30,
    32, 33, 34, 48, 49, 50, 36, 37, 38, 52, 53, 54,
    40, 41, 42, 56, 57, 58, 44, 45, 46, 60, 61, 62,
    64, 65, 66, 80, 81, 82, 68, 69, 70, 84, 85, 86,
    72, 73, 74, 88, 89, 90, 76, 77, 78, 92, 93, 94,
    96, 97, 98, 112, 113, 114, 100, 101, 102, 116, 117, 118,
    104, 105, 106, 120, 121, 122, 108, 109, 110, 124, 125, 126,
    // clang-format on
  };

  AxTensorsInterface output{ { { 2, 2, 2, 12 }, 1, result.data() } };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};

  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, four_by_four_rgba_in_with_end_pad)
{
  std::unordered_map<std::string, std::string> properties = {
    { "padding", "0, 0, 0, 0, 0, 0 , 0, 2" },
  };

  Transformer transformer("libtransform_yolopreproc.so", properties);
  std::vector<int8_t> data(4 * 4 * 4);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 4, 4, 4 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 2 * 14, -51);
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 16, 17, 18, 4, 5, 6, 20, 21, 22, -51, -51,
    8, 9, 10, 24, 25, 26, 12, 13, 14, 28, 29, 30, -51, -51,
    32, 33, 34, 48, 49, 50, 36, 37, 38, 52, 53, 54, -51, -51,
    40, 41, 42, 56, 57, 58, 44, 45, 46, 60, 61, 62, -51, -51,
    // clang-format on
  };
  auto outp = inp;
  outp[0].data = result.data();
  AxDataInterface output{ outp };
  output = transformer.set_output_interface(output);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};
  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, four_by_four_rgba_in_with_start_and_end_pad)
{
  std::unordered_map<std::string, std::string> properties = {
    { "padding", "0, 0, 0, 0, 0, 0 , 2, 2" },
  };

  Transformer transformer("libtransform_yolopreproc.so", properties);
  std::vector<int8_t> data(4 * 4 * 4);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 4, 4, 4 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 2 * 16, -51);
  std::vector<int8_t> expected{
    // clang-format off
    -51, -51, 0, 1, 2, 16, 17, 18, 4, 5, 6, 20, 21, 22, -51, -51,
    -51, -51, 8, 9, 10, 24, 25, 26, 12, 13, 14, 28, 29, 30, -51, -51,
    -51, -51, 32, 33, 34, 48, 49, 50, 36, 37, 38, 52, 53, 54, -51, -51,
    -51, -51, 40, 41, 42, 56, 57, 58, 44, 45, 46, 60, 61, 62, -51, -51,
    // clang-format on
  };
  auto outp = inp;
  outp[0].data = result.data();
  AxDataInterface output{ outp };
  output = transformer.set_output_interface(output);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};
  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, four_by_four_rgba_in_with_start_and_end_pad_with_x_pad)
{
  std::unordered_map<std::string, std::string> properties = {
    { "padding", "0, 0, 0, 0, 1, 1 , 2, 2" },
  };

  Transformer transformer("libtransform_yolopreproc.so", properties);
  std::vector<int8_t> data(4 * 4 * 4);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 4, 4, 4 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 4 * 16, -51);
  std::vector<int8_t> expected{
    // clang-format off
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, 0, 1, 2, 16, 17, 18, 4, 5, 6, 20, 21, 22, -51, -51,
    -51, -51, 8, 9, 10, 24, 25, 26, 12, 13, 14, 28, 29, 30, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, 32, 33, 34, 48, 49, 50, 36, 37, 38, 52, 53, 54, -51, -51,
    -51, -51, 40, 41, 42, 56, 57, 58, 44, 45, 46, 60, 61, 62, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    // clang-format on
  };
  auto outp = inp;
  outp[0].data = result.data();
  AxDataInterface output{ outp };
  output = transformer.set_output_interface(output);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};
  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}

TEST(yolo_preproc, four_by_four_rgba_in_with_start_and_end_pad_with_xy_pad)
{
  std::unordered_map<std::string, std::string> properties = {
    { "padding", "0, 0, 1, 1, 1, 1 , 2, 2" },
  };

  Transformer transformer("libtransform_yolopreproc.so", properties);
  std::vector<int8_t> data(4 * 4 * 4);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 4, 4, 4 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(4 * 4 * 16, -51);
  std::vector<int8_t> expected{
    // clang-format off
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, 0, 1, 2, 16, 17, 18, 4, 5, 6, 20, 21, 22, -51, -51,
    -51, -51, 8, 9, 10, 24, 25, 26, 12, 13, 14, 28, 29, 30, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, 32, 33, 34, 48, 49, 50, 36, 37, 38, 52, 53, 54, -51, -51,
    -51, -51, 40, 41, 42, 56, 57, 58, 44, 45, 46, 60, 61, 62, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    -51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,-51, -51, -51, -51,
    // clang-format on
  };
  auto outp = inp;
  outp[0].data = result.data();
  AxDataInterface output{ outp };
  output = transformer.set_output_interface(output);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};
  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}


TEST(yolo_preproc, four_by_four_rgb_in_with_end_pad_with_fill)
{
  std::unordered_map<std::string, std::string> properties = {
    { "padding", "0, 0, 0, 0, 0, 0 , 0, 3" },
    { "fill", "42" },
  };

  Transformer transformer("libtransform_yolopreproc.so", properties);
  std::vector<int8_t> data(4 * 4 * 3);
  std::iota(std::begin(data), std::end(data), 0);
  AxTensorsInterface inp{ { { 1, 4, 4, 3 }, 1, data.data() } };
  AxDataInterface input{ inp };
  std::vector<int8_t> result(2 * 2 * 15, std::int8_t{ -1 });
  std::vector<int8_t> expected{
    // clang-format off
    0, 1, 2, 12, 13, 14, 3, 4, 5, 15, 16, 17, 42, 42, 42,
    6, 7, 8, 18, 19, 20, 9, 10, 11, 21, 22, 23, 42, 42, 42,
    24, 25, 26, 36, 37, 38, 27, 28, 29, 39, 40, 41, 42, 42, 42,
    30, 31, 32, 42, 43, 44, 33, 34, 35, 45, 46, 47, 42, 42, 42,
    // clang-format on
  };
  auto outp = inp;
  outp[0].data = result.data();
  AxDataInterface output{ outp };
  output = transformer.set_output_interface(output);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map = {};
  transformer.transform(input, output, map, 0, 1);
  ASSERT_EQ(result, expected);
}
