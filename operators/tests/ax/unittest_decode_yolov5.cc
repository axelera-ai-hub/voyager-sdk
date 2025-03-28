// Copyright Axelera AI, 2023
#include "gtest/gtest.h"
#include <gmodule.h>
#include "gmock/gmock.h"
#include "unittest_decode_common.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <filesystem>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"

namespace fs = std::filesystem;

namespace
{
struct object_meta {
  std::vector<int32_t> boxes;
  std::vector<float> scores;
  std::vector<int32_t> classes;
};

object_meta
get_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);

  auto actual_metadata = meta->get_extern_meta();
  EXPECT_EQ(actual_metadata.size(), 3);

  auto p_boxes = reinterpret_cast<const int32_t *>(actual_metadata[0].meta);
  auto p_scores = reinterpret_cast<const float *>(actual_metadata[1].meta);
  auto p_classes = reinterpret_cast<const int32_t *>(actual_metadata[2].meta);
  auto actual_boxes = std::vector<int32_t>{ p_boxes,
    p_boxes + actual_metadata[0].meta_size / sizeof(int32_t) };
  auto actual_scores = std::vector<float>{ p_scores,
    p_scores + actual_metadata[1].meta_size / sizeof(float) };
  auto actual_classes = std::vector<int32_t>{ p_classes,
    p_classes + actual_metadata[2].meta_size / sizeof(int32_t) };

  return { actual_boxes, actual_scores, actual_classes };
}

template <typename T>
AxTensorsInterface
tensors_from_vector(std::vector<T> &tensors, std::vector<int> sizes)
{
  return {
    { sizes, sizeof tensors[0], tensors.data() },
  };
}

TEST(yolov5_errors, no_zero_points_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "scales", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  ;
  EXPECT_THROW(Decoder("libdecode_yolov5.so", properties), std::runtime_error);
}

TEST(yolov5_errors, no_scales_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov5.so", properties), std::runtime_error);
}

TEST(yolov5_errors, different_scale_and_zero_point_sizes_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov5.so", properties), std::runtime_error);
}


TEST(yolov5_errors, must_provide_anchors)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1, 2" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov5.so", properties), std::runtime_error);
}

TEST(yolov5_errors, must_provide_classes)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1, 2" },
    { "anchors", "10, 10" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov5.so", properties), std::runtime_error);
}

TEST(yolov5_decode, blank_lines_ignored_in_classlabels)
{
  std::string meta_identifier = "yolov5";
  auto labels_file = tempfile("\n     \nBeaver\n\nDog\nPony\nLion\nKettle\n");

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0" },
    { "scales", "1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "1.0" },
    { "classes", "5" },
    { "multiclass", "0" },
    { "sigmoid_in_postprocess", "1" },
    { "classlabels_file", labels_file.filename() },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  //  This should not throw (it used to)
  Decoder decoder("libdecode_yolov5.so", properties);
}

TEST(yolov5_decode, num_classes_needn_not_be_same_size_as_classlabels)
{
  std::string meta_identifier = "yolov5";
  auto labels_file = tempfile("Beaver\nDog\nPony\nLion\nKettle\n");

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0" },
    { "scales", "1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "1.0" },
    { "classes", "8" },
    { "multiclass", "0" },
    { "sigmoid_in_postprocess", "1" },
    { "classlabels_file", labels_file.filename() },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  //  This should not throw (it used to)
  Decoder decoder("libdecode_yolov5.so", properties);
}

TEST(yolov5_decode_scores, all_filtered_at_max_confidence)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0" },
    { "scales", "1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "1.0" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  EXPECT_EQ(actual_classes, std::vector<int32_t>{});
  EXPECT_EQ(actual_scores, std::vector<float>{});
}

TEST(yolov5_decode_scores, none_filtered_at_min_confidence_with_multiclass)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0" },
    { "scales", "1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.0" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
    { "letterbox", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 1, 2, 3 };
  auto expected_scores = std::vector<float>{ 0.25F, 0.25F, 0.25F, 0.25F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}

TEST(yolov5_decode_scores, all_but_first_highest_filtered_at_min_confidence_with_non_multiclass)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = { 0, 0, 0, 0, 0, -1, 0, 0, 0 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0" },
    { "scales", "1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.0" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1 };
  auto expected_scores = std::vector<float>{ 0.25F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}

TEST(yolov5_decode_scores, with_multiclass_all_below_threshold_are_filtered)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = { 0, 0, 0, 0, 0, -1, 0, -1, 0 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0" },
    { "scales", "1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.25F, 0.25F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}

TEST(yolov5_decode_scores, dequantize_with_sigmoid)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = { 14, 14, 14, 14, 14, 13, 15, 12, 1 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "2.0" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1 };
  auto expected_scores = std::vector<float>{ 0.44039851F };
  ASSERT_EQ(actual_classes, expected_classes);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
}

TEST(yolov5_decode_scores, pure_dequantize)
{
  //  Given the scale and zero point values of 12 -> 0.8, without applying sigmoid during dequantization,
  //  the direct mapping of score values to confidence levels is used.
  std::vector<int8_t> yolo = { 0, 0, 0, 0, 12, 11, 11, 10, 12 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "10" },
    { "scales", "0.4" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 1, 3 };
  auto expected_scores = std::vector<float>{ 0.32F, 0.32F, 0.64F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_THAT(actual_scores, testing::Pointwise(testing::FloatEq(), expected_scores));
}

TEST(yolov5_decode_scores, two_by_two)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = {
    14, 14, 14, 14, // x
    14, 14, 14, 14, // y
    14, 14, 14, 14, // w
    14, 14, 14, 14, // h
    14, 0, 1, 14, // score
    13, 13, 13, 13, // class 0 score
    14, 14, 10, 1, // class 1 score
    12, 12, 12, 12, // class 2 score
    1, 1, 1, 14, // class 3 score
  };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 2, 2 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.25F, 0.25F };

  auto expected_boxes = std::vector<int32_t>{ 13, 11, 19, 21, 45, 43, 51, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_THAT(actual_scores, testing::ElementsAreArray(expected_scores));
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, letterbox)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = {
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // x
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // y
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // w
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // h
    0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 1, 14, 0, 0, 0, 0, // score
    0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 0, 0, 0, 0, // class 0 score
    0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 10, 1, 0, 0, 0, 0, // class 1 score
    0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 0, 0, 0, 0, // class 2 score,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 14, 0, 0, 0, 0, // class 3 score
  };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 128, 64, 128, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 4, 4 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.25F, 0.25F };

  auto expected_boxes = std::vector<int32_t>{ 13, 43, 19, 53, 109, 43, 115, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_THAT(actual_scores, testing::ElementsAreArray(expected_scores));
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, letterbox_topk)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = {
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // x
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // y
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // w
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // h
    0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 1, 14, 0, 0, 0, 0, // score
    0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 13, 0, 0, 0, 0, // class 0 score
    0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 10, 1, 0, 0, 0, 0, // class 1 score
    0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 0, 0, 0, 0, // class 2 score,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 15, 0, 0, 0, 0, // class 3 score
  };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "topk", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 128, 64, 128, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 4, 4 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 3 };
  auto expected_scores = std::vector<float>{ 0.3655293F };

  auto expected_boxes = std::vector<int32_t>{ 109, 43, 115, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 1);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, two_by_two_transposed)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25

  std::vector<int8_t> yolo = {
    // clang-format off
    14, 14, 14, 14, 14, 13, 14, 12, 10,
    14, 14, 14, 14, 00, 13, 14, 12, 10,
    14, 14, 14, 14, 01, 13, 10, 12, 10,
    14, 14, 14, 14, 14, 13, 01, 12, 14,
    // clang-format on
  };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "transpose", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 2, 2, 9 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.25F, 0.25F };

  auto expected_boxes = std::vector<int32_t>{ 13, 11, 19, 21, 45, 43, 51, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_THAT(actual_scores, testing::ElementsAreArray(expected_scores));
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, two_by_two_transposed_pad)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25

  std::vector<int8_t> yolo = {
    // clang-format off
    14, 14, 14, 14, 14, 13, 14, 12, 10, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 13, 14, 12, 10, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 01, 13, 10, 12, 10, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 14, 13, 01, 12, 14, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    // clang-format on
  };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "transpose", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 2, 2, 64 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.25F, 0.25F };

  auto expected_boxes = std::vector<int32_t>{ 13, 11, 19, 21, 45, 43, 51, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_THAT(actual_scores, testing::ElementsAreArray(expected_scores));
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, letterbox_topk_transpose)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = {
    // clang-format off
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 14, 13, 14, 12, 01,
    14, 14, 14, 14, 00, 13, 14, 12, 01,
    14, 14, 14, 14, 01, 13, 10, 12, 01,
    14, 14, 14, 14, 14, 13, 01, 12, 15,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    14, 14, 14, 14, 00, 00, 00, 00, 00,
    // clang-format on
  };

  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "topk", "1" },
    { "transpose", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 128, 64, 128, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 4, 4, 9 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 3 };
  auto expected_scores = std::vector<float>{ 0.3655293F };

  auto expected_boxes = std::vector<int32_t>{ 109, 43, 115, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 1);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, letterbox_topk_transpose_float)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<float> yolo = {
    // clang-format off
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -1.0F, 0.0F, -2.0F, -3.0F,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -1.0F, 0.0F, -2.0F, -10.0F,
    0.0F, 0.0F, 0.0F, 0.0F, -6.0F, -1.0F, -3.0F, -2.0F, -10.0F,
    0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -1.0F, -9.0F, -2.0F, 0.1F,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    0.0F, 0.0F, 0.0F, 0.0F, -10, -10, -10, -10, -10,
    // clang-format on
  };

  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "topk", "1" },
    { "transpose", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 128, 64, 128, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 4, 4, 9 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 3 };
  auto expected_scores = std::vector<float>{ 0.3655293F };

  auto expected_boxes = std::vector<int32_t>{ 109, 43, 115, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 1);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, two_by_two_with_2_acnhors)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = {
    14, 14, 14, 14, // x
    14, 14, 14, 14, // y
    14, 14, 14, 14, // w
    14, 14, 14, 14, // h
    10, 0, 1, 14, // score
    13, 13, 13, 13, // class 0 score
    14, 14, 10, 1, // class 1 score
    12, 12, 12, 12, // class 2 score
    1, 1, 1, 14, // class 3 score
    14, 14, 14, 14, // x
    14, 14, 14, 14, // y
    14, 14, 14, 14, // w
    14, 14, 14, 14, // h
    14, 0, 1, 10, // score
    13, 13, 13, 13, // class 0 score
    14, 14, 10, 1, // class 1 score
    12, 12, 12, 12, // class 2 score
    1, 1, 1, 14, // class 3 score
  };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3, 0.3, 0.2" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 18, 2, 2 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.25F, 0.25F };

  auto expected_boxes = std::vector<int32_t>{ 11, 13, 21, 19, 45, 43, 51, 53 };

  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_THAT(actual_scores, testing::ElementsAreArray(expected_scores));
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, four_by_four_and_two_by_two)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo4x4 = {
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // x
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // y
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // w
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // h
    0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 1, 14, 0, 0, 0, 0, // score
    0, 0, 0, 0, 0, 0, 0, 0, 14, 13, 13, 13, 0, 0, 0, 0, // class 0 score
    0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 10, 1, 0, 0, 0, 0, // class 1 score
    0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 0, 0, 0, 0, // class 2 score,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 14, 0, 0, 0, 0, // class 3 score
  };

  std::vector<int8_t> yolo2x2 = {
    14, 14, 14, 14, // x
    14, 14, 14, 14, // y
    14, 14, 14, 14, // w
    14, 14, 14, 14, // h
    14, 0, 1, 14, // score
    13, 13, 13, 13, // class 0 score
    14, 14, 10, 1, // class 1 score
    12, 12, 12, 14, // class 2 score
    1, 1, 1, 12, // class 3 score
  };

  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14, 14" },
    { "scales", "1.0, 1.0" },
    { "anchors", "0.2, 0.3, 0.3, 0.2" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 640, 640, 640, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo4x4, { 1, 9, 4, 4 });
  auto tensors2x2 = tensors_from_vector(yolo2x2, { 1, 9, 2, 2 });
  tensors.push_back(std::move(tensors2x2[0]));

  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 3, 1, 2 };
  auto expected_scores = std::vector<float>{ 0.25, 0.25, 0.25, 0.25 };

  auto expected_boxes = std::vector<int32_t>{ 64, 376, 96, 424, 544, 376, 576,
    424, 112, 128, 208, 192, 432, 448, 528, 512 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 4);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_FLOAT_EQ(actual_scores[1], expected_scores[1]);
  EXPECT_FLOAT_EQ(actual_scores[2], expected_scores[2]);
  EXPECT_FLOAT_EQ(actual_scores[3], expected_scores[3]);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, two_by_two_and_four_by_four)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo4x4 = {
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // x
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // y
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // w
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // h
    0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 1, 14, 0, 0, 0, 0, // score
    0, 0, 0, 0, 0, 0, 0, 0, 14, 13, 13, 13, 0, 0, 0, 0, // class 0 score
    0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 10, 1, 0, 0, 0, 0, // class 1 score
    0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 0, 0, 0, 0, // class 2 score,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 14, 0, 0, 0, 0, // class 3 score
  };

  std::vector<int8_t> yolo2x2 = {
    14, 14, 14, 14, // x
    14, 14, 14, 14, // y
    14, 14, 14, 14, // w
    14, 14, 14, 14, // h
    14, 0, 1, 14, // score
    13, 13, 13, 13, // class 0 score
    14, 14, 10, 1, // class 1 score
    12, 12, 12, 14, // class 2 score
    1, 1, 1, 12, // class 3 score
  };

  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14, 14" },
    { "scales", "1.0, 1.0" },
    { "anchors", "0.2, 0.3, 0.3, 0.2" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 640, 640, 640, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo2x2, { 1, 9, 2, 2 });
  auto tensors4x4 = tensors_from_vector(yolo4x4, { 1, 9, 4, 4 });
  tensors.push_back(std::move(tensors4x4[0]));

  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 3, 1, 2 };
  auto expected_scores = std::vector<float>{ 0.25, 0.25, 0.25, 0.25 };

  auto expected_boxes = std::vector<int32_t>{ 64, 376, 96, 424, 544, 376, 576,
    424, 112, 128, 208, 192, 432, 448, 528, 512 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 4);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_FLOAT_EQ(actual_scores[1], expected_scores[1]);
  EXPECT_FLOAT_EQ(actual_scores[2], expected_scores[2]);
  EXPECT_FLOAT_EQ(actual_scores[3], expected_scores[3]);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, four_by_two)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo4x4 = {
    // clang-format off
    14, 14, 14, 14, 14, 14, 14, 14,  // x
    14, 14, 14, 14, 14, 14, 14, 14,  // y
    14, 14, 14, 14, 14, 14, 14, 14,  // w
    14, 14, 14, 14, 14, 14, 14, 14,  // h
     0, 14,  0,  0,  0,  1, 14,  0,  // score
     0, 14, 13,  0,  0, 13, 13,  0,  // class 0 score
     0, 13, 14,  0,  0, 10,  1,  0,  // class 1 score
     0, 12, 12,  0,  0, 12, 12,  0,  // class 2 score,
     0,  1,  1,  0,  0,  1, 14,  0,  // class 3 score
    // clang-format on
  };

  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "320" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 640, 640, 640, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo4x4, { 1, 9, 2, 4 });

  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 3 };
  auto expected_scores = std::vector<float>{ 0.25, 0.25 };

  auto expected_boxes = std::vector<int32_t>{ 128, 112, 192, 208, 448, 432, 512, 528 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 2);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_FLOAT_EQ(actual_scores[1], expected_scores[1]);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_scores, two_by_four)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo4x4 = {
    // clang-format off
    14, 14, 14, 14, 14, 14, 14, 14,  // x
    14, 14, 14, 14, 14, 14, 14, 14,  // y
    14, 14, 14, 14, 14, 14, 14, 14,  // w
    14, 14, 14, 14, 14, 14, 14, 14,  // h
     0,  0, 14,  0,  1, 14,  0,  0,  // score
     0,  0, 14, 13, 13, 13,  0,  0,  // class 0 score
     0,  0, 13, 14, 10,  1,  0,  0,  // class 1 score
     0,  0, 12, 12, 12, 12,  0,  0,  // class 2 score,
     0,  0,  1,  1,  1, 14,  0,  0,  // class 3 score
    // clang-format on
  };

  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "1.0" },
    { "anchors", "0.2, 0.3" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "320" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 640, 640, 640, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo4x4, { 1, 9, 4, 2 });

  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 3 };
  auto expected_scores = std::vector<float>{ 0.25, 0.25 };

  auto expected_boxes = std::vector<int32_t>{ 128, 112, 192, 208, 448, 432, 512, 528 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 2);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_FLOAT_EQ(actual_scores[1], expected_scores[1]);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_filter, two_by_two_and_four_by_four)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo4x4 = {
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // x
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // y
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // w
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, // h
    0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 1, 14, 0, 0, 0, 0, // score
    0, 0, 0, 0, 0, 0, 0, 0, 14, 13, 13, 13, 0, 0, 0, 0, // class 0 score
    0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 10, 1, 0, 0, 0, 0, // class 1 score
    0, 0, 0, 0, 0, 0, 0, 0, 12, 12, 12, 12, 0, 0, 0, 0, // class 2 score,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 14, 0, 0, 0, 0, // class 3 score
  };

  std::vector<int8_t> yolo2x2 = {
    14, 14, 14, 14, // x
    14, 14, 14, 14, // y
    14, 14, 14, 14, // w
    14, 14, 14, 14, // h
    14, 0, 1, 14, // score
    13, 13, 13, 13, // class 0 score
    14, 14, 10, 1, // class 1 score
    12, 12, 12, 14, // class 2 score
    1, 1, 1, 12, // class 3 score
  };

  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14, 14" },
    { "scales", "1.0, 1.0" },
    { "anchors", "0.2, 0.3, 0.3, 0.2" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "class_agnostic", "0" },
    { "label_filter", "2, 3" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 640, 640, 640, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo2x2, { 1, 9, 2, 2 });
  auto tensors4x4 = tensors_from_vector(yolo4x4, { 1, 9, 4, 4 });
  tensors.push_back(std::move(tensors4x4[0]));

  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 3, 2 };
  auto expected_scores = std::vector<float>{ 0.25, 0.25 };

  auto expected_boxes = std::vector<int32_t>{ 544, 376, 576, 424, 432, 448, 528, 512 };

  EXPECT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores.size(), 2);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
  EXPECT_FLOAT_EQ(actual_scores[1], expected_scores[1]);
  EXPECT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_remove_degnerate_boxes, remove_with_no_width)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = { 14, 14, 0, 14, 14, 13, 15, 12, 1 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "2.0" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{};
  auto expected_scores = std::vector<float>{};
  auto expected_boxes = std::vector<int32_t>{};
  ASSERT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores, expected_scores);
  ASSERT_EQ(actual_boxes, expected_boxes);
}

TEST(yolov5_decode_remove_degnerate_boxes, remove_with_no_height)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  std::vector<int8_t> yolo = { 14, 14, 14, 0, 14, 13, 15, 12, 1 };
  std::string meta_identifier = "yolov5";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14" },
    { "scales", "2.0" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov5.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{};
  auto expected_scores = std::vector<float>{};
  auto expected_boxes = std::vector<int32_t>{};
  ASSERT_EQ(actual_classes, expected_classes);
  ASSERT_EQ(actual_scores, expected_scores);
  ASSERT_EQ(actual_boxes, expected_boxes);
}

} // namespace
