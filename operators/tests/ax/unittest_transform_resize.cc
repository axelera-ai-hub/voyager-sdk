#include "unittest_transform_common.h"

namespace
{
const auto resize_lib = "libtransform_resize.so";

TEST(resize_letterbox, output_is_square_of_resize_size_if_letterbox)
{
  int resize_size = 256;
  std::unordered_map<std::string, std::string> input = {
    { "size", std::to_string(resize_size) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  auto out_interface = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.width, info.height);
  EXPECT_EQ(info.width, resize_size);
}

TEST(resize_letterbox, output_is_width_height_if_no_size_andletterbox)
{
  int out_width = 256;
  int out_height = 128;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(out_width) },
    { "height", std::to_string(out_height) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  AxVideoInterface video_info{ { 640, 480, 640 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  auto out_interface = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.width, out_width);
  EXPECT_EQ(info.height, out_height);
}


TEST(resize_letterbox, maintains_aspect_ratio_if_no_letterbox_and_only_size_provided)
{
  int resize_size = 256;
  std::unordered_map<std::string, std::string> input = {
    { "size", std::to_string(resize_size) },
    { "letterbox", "0" },
  };
  Transformer resizer(resize_lib, input);
  AxVideoInterface video_info{ { 1024, 512, 1024 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  auto out_interface = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_interface).info;
  EXPECT_EQ(info.height, 256);
  EXPECT_EQ(info.width, 512);
}

TEST(resize_letterbox, top_bottom_default_padding)
{
  int resize_size = 4;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 2 * 4, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 4, 255);
  AxVideoInterface in_info{ { 4, 2, 4 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
    114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255, 114, 114, 114, 255,

    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, top_bottom_default_padding_rgb)
{
  int resize_size = 4;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 2 * 3, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 3, 255);
  AxVideoInterface in_info{ { 4, 2, 4 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
    114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114,

    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, top_bottom_default_padding_gray8)
{
  int resize_size = 4;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 2 * 1, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 1, 255);
  AxVideoInterface in_info{ { 4, 2, 4 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    114, 114, 114, 114,
     33,  33,  33,  33,
     33,  33,  33,  33,
    114, 114, 114, 114,

    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}


TEST(resize_letterbox, left_right_default_padding)
{
  int resize_size = 4;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(2 * 4 * 4, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 4, 255);
  AxVideoInterface in_info{ { 2, 4, 2 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    114, 114, 114, 255, 33, 33, 33, 33, 33, 33, 33, 33, 114, 114, 114, 255,
    114, 114, 114, 255, 33, 33, 33, 33, 33, 33, 33, 33, 114, 114, 114, 255,
    114, 114, 114, 255, 33, 33, 33, 33, 33, 33, 33, 33, 114, 114, 114, 255,
    114, 114, 114, 255, 33, 33, 33, 33, 33, 33, 33, 33, 114, 114, 114, 255,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, left_right_default_padding_rgb)
{
  int resize_size = 4;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(2 * 4 * 3, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 3, 255);
  AxVideoInterface in_info{ { 2, 4, 2 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    114, 114, 114, 33, 33, 33, 33, 33, 33, 114, 114, 114,
    114, 114, 114, 33, 33, 33, 33, 33, 33, 114, 114, 114,
    114, 114, 114, 33, 33, 33, 33, 33, 33, 114, 114, 114,
    114, 114, 114, 33, 33, 33, 33, 33, 33, 114, 114, 114,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, left_right_default_padding_gray8)
{
  int resize_size = 4;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(2 * 4 * 1, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 1, 255);
  AxVideoInterface in_info{ { 2, 4, 2 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    114, 33, 33, 114,
    114, 33, 33, 114,
    114, 33, 33, 114,
    114, 33, 33, 114,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, top_bottom_overidden_padding)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 2 * 4, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 4, 255);
  AxVideoInterface in_info{ { 4, 2, 4 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,

    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, top_bottom_overidden_padding_rgb)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 2 * 3, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 3, 255);
  AxVideoInterface in_info{ { 4, 2, 4 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42,  42,  42, 42,  42,  42, 42,  42,  42, 42,  42,  42,
     33,  33,  33, 33,  33,  33, 33,  33,  33, 33,  33,  33,
     33,  33,  33, 33,  33,  33, 33,  33,  33, 33,  33,  33,
     42,  42,  42, 42,  42,  42, 42,  42,  42, 42,  42,  42,

    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, top_bottom_overidden_padding_gray8)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 2 * 1, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 1, 255);
  AxVideoInterface in_info{ { 4, 2, 4 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42, 42, 42, 42,
     33, 33, 33, 33,
     33, 33, 33, 33,
     42, 42, 42, 42,

    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, left_right_overidden_padding)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(2 * 4 * 4, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 4, 255);
  AxVideoInterface in_info{ { 2, 4, 2 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    // clang-format on
  };

  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, left_right_overidden_padding_rgb)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(2 * 4 * 3, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 3, 255);
  AxVideoInterface in_info{ { 2, 4, 2 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    // clang-format on
  };

  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, left_right_overidden_padding_gray8)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(2 * 4 * 1, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 1, 255);
  AxVideoInterface in_info{ { 2, 4, 2 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    42, 33, 33, 42,
    42, 33, 33, 42,
    42, 33, 33, 42,
    42, 33, 33, 42,
    // clang-format on
  };

  ASSERT_EQ(out_buf, expected);
}


TEST(resize_letterbox, left_right_overidden_padding_with_resize)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 8 * 4, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 4, 255);
  AxVideoInterface in_info{ { 4, 8, 4 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    42, 42, 42, 255, 33, 33, 33, 33, 33, 33, 33, 33, 42, 42, 42, 255,
    // clang-format on
  };

  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, left_right_overidden_padding_with_resize_rgb)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 8 * 3, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 3, 255);
  AxVideoInterface in_info{ { 4, 8, 4 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    42, 42, 42, 33, 33, 33, 33, 33, 33, 42, 42, 42,
    // clang-format on
  };

  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, left_right_overidden_padding_with_resize_gray8)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 8 * 1, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 1, 255);
  AxVideoInterface in_info{ { 4, 8, 4 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
    42, 33, 33, 42,
    42, 33, 33, 42,
    42, 33, 33, 42,
    42, 33, 33, 42,
    // clang-format on
  };

  ASSERT_EQ(out_buf, expected);
}


TEST(resize_letterbox, top_bottom_overidden_padding_with_resize)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(8 * 4 * 4, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 4, 255);
  AxVideoInterface in_info{ { 8, 4, 8 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,  33,
     42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,  42,  42,  42, 255,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, top_bottom_overidden_padding_with_resize_rgb)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(8 * 4 * 3, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 3, 255);
  AxVideoInterface in_info{ { 8, 4, 8 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42,  42,  42, 42,  42,  42,  42,  42,  42,  42,  42,  42,
     33,  33,  33, 33,  33,  33,  33,  33,  33,  33,  33,  33,
     33,  33,  33, 33,  33,  33,  33,  33,  33,  33,  33,  33,
     42,  42,  42, 42,  42,  42,  42,  42,  42,  42,  42,  42,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, top_bottom_overidden_padding_with_resize_GRAY8)
{
  int resize_size = 4;
  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(resize_size) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(8 * 4 * 1, 33);
  auto out_buf = std::vector<uint8_t>(4 * 4 * 1, 255);
  AxVideoInterface in_info{ { 8, 4, 8 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 4, 4 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42,  42,  42, 42,
     33,  33,  33, 33,
     33,  33,  33, 33,
     42,  42,  42, 42,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, input_taller_than_output_non_square)
{
  int width = 4;
  int height = 3;

  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(width) },
    { "height", std::to_string(height) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(4 * 6 * 1, 33);
  auto out_buf = std::vector<uint8_t>(4 * 3 * 1, 255);
  AxVideoInterface in_info{ { 4, 6, 4 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 4, 3, 4 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42,  33,  33, 42,
     42,  33,  33, 42,
     42,  33,  33, 42,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, input_shorter_than_output_non_square)
{
  int width = 3;
  int height = 4;

  int padding = 42;
  std::unordered_map<std::string, std::string> input = {
    { "width", std::to_string(width) },
    { "height", std::to_string(height) },
    { "padding", std::to_string(padding) },
    { "letterbox", "1" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(6 * 4 * 1, 33);
  auto out_buf = std::vector<uint8_t>(3 * 4 * 1, 255);
  AxVideoInterface in_info{ { 6, 4, 6 * 1, 0, AxVideoFormat::GRAY8 }, in_buf.data() };
  AxVideoInterface out_info{ { 3, 4, 3 * 1, 0, AxVideoFormat::GRAY8 }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>{
    // clang-format off
     42,  42,  42,
     33,  33,  33,
     33,  33,  33,
     42,  42,  42,
    // clang-format on
  };
  ASSERT_EQ(out_buf, expected);
}

TEST(resize_letterbox, test_size_maintains_aspect_ratio)
{
  int out_size = 2;

  std::unordered_map<std::string, std::string> input = {
    { "size", std::to_string(out_size) },
    { "letterbox", "0" },
  };
  Transformer resizer(resize_lib, input);
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;

  auto in_buf = std::vector<uint8_t>(6 * 4 * 4, 33);
  auto out_buf = std::vector<uint8_t>(3 * 2 * 4, 255);
  AxVideoInterface in_info{ { 6, 4, 6 * 4, 0, AxVideoFormat::RGBA }, in_buf.data() };
  AxVideoInterface out_info{ { 3, 2, 3 * 4, 0, AxVideoFormat::RGBA }, out_buf.data() };
  resizer.transform(in_info, out_info, metadata, 0, 1);

  auto expected = std::vector<uint8_t>(3 * 2 * 4, 33);
  ASSERT_EQ(out_buf, expected);
}


} // namespace
