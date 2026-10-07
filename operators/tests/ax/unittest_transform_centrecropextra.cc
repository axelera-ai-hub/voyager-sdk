// Copyright Axelera AI, 2023
#include "unittest_ax_common.h"

TEST(crop_meta, test_cropsize_with_crop_width)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "crop_width", "224" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  EXPECT_THROW(xform->set_output_interface(video_info), std::runtime_error);
}

TEST(crop_meta, test_cropsize_with_crop_height)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "crop_width", "224" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  EXPECT_THROW(xform->set_output_interface(video_info), std::runtime_error);
}

TEST(crop_meta, test_no_cropsize)
{
  std::unordered_map<std::string, std::string> input = {
    { "scalesize", "256" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  EXPECT_THROW(xform->set_output_interface(video_info), std::runtime_error);
}

TEST(crop_meta, test_640_480_scale_256_crop_224)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "scalesize", "256" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
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
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
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
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 1920, 1080, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
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

TEST(crop_meta, test_500_375_scale_256_crop_224)
{
  std::unordered_map<std::string, std::string> input = {
    { "crop_width", "224" },
    { "crop_height", "224" },
    { "scalesize", "256" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 350, 220, 350 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 193;
  int expected_padding_left = 78;
  int expected_padding_top = 13;
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
  auto xform = Ax::LoadTransform("centrecropextra", input);
  int crop_x = 100;
  int crop_y = 160;
  AxVideoInterface video_info{
    { 720, 1280, 1024 * 3, 0, AxVideoFormat::RGB, true, crop_x, crop_y, 1440 }, nullptr
  };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 630;
  int expected_padding_left = 45 + crop_x;
  int expected_padding_top = 325 + crop_y;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
  EXPECT_EQ(info.actual_height, 1440);
}

TEST(crop_meta, test_landscape_roi)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "224" },
    { "scalesize", "256" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  int crop_x = 100;
  int crop_y = 160;
  AxVideoInterface video_info{
    { 1280, 720, 1920 * 3, 0, AxVideoFormat::RGB, true, crop_x, crop_y, 960 }, nullptr
  };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 630;
  int expected_padding_left = 325 + crop_x;
  int expected_padding_top = 45 + crop_y;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
  EXPECT_EQ(info.actual_height, 960);
}

TEST(crop_meta, test_no_scale)
{
  std::unordered_map<std::string, std::string> input = {
    { "cropsize", "448" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_size = 448;
  int expected_padding_left = 96;
  int expected_padding_top = 16;
  EXPECT_EQ(info.width, expected_output_size);
  EXPECT_EQ(info.height, expected_output_size);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_no_scale_width_and_height)
{
  std::unordered_map<std::string, std::string> input = {
    { "crop_width", "448" },
    { "crop_height", "220" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_width = 448;
  int expected_output_height = 220;
  int expected_padding_top = 130;
  int expected_padding_left = 96;
  EXPECT_EQ(info.width, expected_output_width);
  EXPECT_EQ(info.height, expected_output_height);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_scale_width_and_height_same)
{
  std::unordered_map<std::string, std::string> input = {
    { "scalesize", "240" },
    { "crop_width", "200" },
    { "crop_height", "200" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_width = 400;
  int expected_output_height = 400;
  int expected_padding_top = 40;
  int expected_padding_left = 120;
  EXPECT_EQ(info.width, expected_output_width);
  EXPECT_EQ(info.height, expected_output_height);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_crop_width_and_height_different)
{
  std::unordered_map<std::string, std::string> input = {
    { "scalesize", "240" },
    { "crop_width", "300" },
    { "crop_height", "200" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_width = 600;
  int expected_output_height = 400;
  int expected_padding_top = 40;
  int expected_padding_left = 20;
  EXPECT_EQ(info.width, expected_output_width);
  EXPECT_EQ(info.height, expected_output_height);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_scale_width_and_height_different)
{
  std::unordered_map<std::string, std::string> input = {
    { "scale_width", "320" },
    { "scale_height", "240" },
    { "crop_width", "300" },
    { "crop_height", "200" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_width = 600;
  int expected_output_height = 400;
  int expected_padding_top = 40;
  int expected_padding_left = 20;
  EXPECT_EQ(info.width, expected_output_width);
  EXPECT_EQ(info.height, expected_output_height);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_crop_width_larger_than_input)
{
  std::unordered_map<std::string, std::string> input = {
    { "crop_width", "660" },
    { "crop_height", "200" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_width = 640;
  int expected_output_height = 200;
  int expected_padding_top = 140;
  int expected_padding_left = 0;
  EXPECT_EQ(info.width, expected_output_width);
  EXPECT_EQ(info.height, expected_output_height);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}

TEST(crop_meta, test_crop_height_larger_than_input)
{
  std::unordered_map<std::string, std::string> input = {
    { "crop_width", "600" },
    { "crop_height", "500" },
  };
  auto xform = Ax::LoadTransform("centrecropextra", input);
  AxVideoInterface video_info{ { 640, 480, 640 * 3, 0, AxVideoFormat::RGB }, nullptr };
  auto out_info = xform->set_output_interface(video_info);
  auto info = std::get<AxVideoInterface>(out_info).info;
  int expected_output_width = 600;
  int expected_output_height = 480;
  int expected_padding_top = 0;
  int expected_padding_left = 20;
  EXPECT_EQ(info.width, expected_output_width);
  EXPECT_EQ(info.height, expected_output_height);
  EXPECT_EQ(info.stride, video_info.info.stride);
  EXPECT_EQ(info.x_offset, expected_padding_left);
  EXPECT_EQ(info.y_offset, expected_padding_top);
}
