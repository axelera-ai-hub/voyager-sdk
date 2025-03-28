// Copyright Axelera AI, 2023
#include "unittest_transform_common.h"


TEST(crop_meta, test_no_scalesize)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  EXPECT_THROW(resizer.set_output_interface(video_info), std::runtime_error);
}

TEST(crop_meta, test_no_cropsize)
{
  std::unordered_map<std::string, std::string> input = {
    { "scalesize", "256" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  EXPECT_THROW(resizer.set_output_interface(video_info), std::runtime_error);
}

TEST(crop_meta, test_cropsize_larger_than_scale_size)
{
  std::unordered_map<std::string, std::string> input = {
    { "scalesize", "256" },
    { "cropsize", "257" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  EXPECT_THROW(resizer.set_output_interface(video_info), std::runtime_error);
}

TEST(crop_meta, test_640_480_scale_256_crop_224)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "scalesize", "256" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 420;
  int expected_padding_left = 110;
  int expected_padding_top = 30;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_640_480_scale_512_crop_448)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "448" },
    { "scalesize", "512" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 420;
  int expected_padding_left = 110;
  int expected_padding_top = 30;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_1920_1080_scale_256_crop_224)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "scalesize", "256" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  AxVideoInterface video_info{ { 1920, 1080, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 945;
  int expected_padding_left = 487;
  int expected_padding_top = 67;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_720_1280_scale_256_crop_224)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "scalesize", "256" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  AxVideoInterface video_info{ { 720, 1280, 720 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 630;
  int expected_padding_left = 45;
  int expected_padding_top = 325;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_portrait_roi)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "scalesize", "256" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  int crop_x = 100;
  int crop_y = 160;
  AxVideoInterface video_info{
    { 720, 1280, 720 * 3, 0, AxVideoFormat::RGB, true, crop_x, crop_y }, nullptr
  };
  auto out_info = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 630;
  int expected_padding_left = 45 + crop_x;
  int expected_padding_top = 325 + crop_y;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_landscape_roi)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "scalesize", "256" },
  };
  Transformer resizer("libtransform_centrecropextra.so", input);
  int crop_x = 100;
  int crop_y = 160;
  AxVideoInterface video_info{
    { 1280, 720, 1280 * 3, 0, AxVideoFormat::RGB, true, crop_x, crop_y }, nullptr
  };
  auto out_info = resizer.set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 630;
  int expected_padding_left = 325 + crop_x;
  int expected_padding_top = 45 + crop_y;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}
