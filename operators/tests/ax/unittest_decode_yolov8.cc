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
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"

#include "AxDataInterface.h"

namespace fs = std::filesystem;

namespace
{

struct object_meta {
  std::vector<int32_t> boxes;
  std::vector<float> scores;
  std::vector<int32_t> classes;
};

struct kpts_xyv {
  int32_t x;
  int32_t y;
  float v;
};

struct kpts_meta {
  std::vector<int32_t> boxes;
  std::vector<kpts_xyv> kpts;
  std::vector<float> scores;
  int kpts_per_box;
};

object_meta
get_object_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);
  EXPECT_EQ(typeid(*meta), typeid(AxMetaObjDetection));

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

kpts_meta
get_kpts_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);
  EXPECT_EQ(typeid(*meta), typeid(AxMetaKptsDetection));

  auto actual_metadata = meta->get_extern_meta();
  EXPECT_EQ(actual_metadata.size(), 4);

  auto p_boxes = reinterpret_cast<const int32_t *>(actual_metadata[0].meta);
  auto p_kpts = reinterpret_cast<const kpts_xyv *>(actual_metadata[1].meta);
  auto p_scores = reinterpret_cast<const float *>(actual_metadata[2].meta);
  auto p_kpts_per_box = reinterpret_cast<const int *>(actual_metadata[3].meta);

  auto actual_boxes = std::vector<int32_t>{ p_boxes,
    p_boxes + actual_metadata[0].meta_size / sizeof(int32_t) };
  auto actual_kpts = std::vector<kpts_xyv>{ p_kpts,
    p_kpts + actual_metadata[1].meta_size / sizeof(kpts_xyv) };
  auto actual_scores = std::vector<float>{ p_scores,
    p_scores + actual_metadata[2].meta_size / sizeof(float) };

  return { actual_boxes, actual_kpts, actual_scores, *p_kpts_per_box };
}
template <typename T>
AxTensorsInterface
tensors_from_vector(std::vector<T> &tensors, std::vector<int> sizes)
{
  return {
    { sizes, sizeof tensors[0], tensors.data() },
  };
}

TEST(yolov8_errors, no_zero_points_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov8.so", properties), std::runtime_error);
}


TEST(yolov8_errors, no_scales_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov8.so", properties), std::runtime_error);
}

TEST(yolov8_errors, different_scale_and_zero_point_sizes_throws)
{
  std::vector<int8_t> scores = { 0, 0, 0, 0, 0, 1, 1, 1 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov8.so", properties), std::runtime_error);
}

TEST(yolov8_decode_scores, all_filtered_at_max_confidence)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 0, 0, 0, 0 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "confidence_threshold", "1.0" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov8.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  score_tensors.push_back(box_tensors[0]);

  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_object_meta(map, meta_identifier);
  EXPECT_EQ(actual_classes, std::vector<int32_t>{});
  EXPECT_EQ(actual_scores, std::vector<float>{});
}
TEST(yolov8_decode_scores, none_filtered_at_min_confidence_with_multiclass)
{
  //  With the scale and zero point values 0 -> 0.50
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 0, 0, 0, 0 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "confidence_threshold", "0.0" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov8.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  score_tensors.push_back(box_tensors[0]);

  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_object_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 1, 2, 3 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F, 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}
TEST(yolov8_decode_scores, all_but_first_highest_filtered_at_min_confidence_with_non_multiclass)
{
  //  With the scale and zero point values 0 -> 0.5
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 0, 0, 0, 0 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "confidence_threshold", "0.0" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov8.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  score_tensors.push_back(box_tensors[0]);
  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_object_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0 };
  auto expected_scores = std::vector<float>{ 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}
TEST(yolov8_decode_scores, with_multiclass_all_below_threshold_are_filtered)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.250
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ -1, 0, -1, 0 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "confidence_threshold", "0.4" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov8.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  score_tensors.push_back(box_tensors[0]);
  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_object_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}

TEST(yolov8_decode_kpts, decode_kpts)
{
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 1 };
  auto kpts = std::vector<int8_t>{ 51 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0, 0" },
    { "scales", "1.0, 1.0, 0" },
    { "confidence_threshold", "0.20" },
    { "classes", "1" },
    { "multiclass", "0" },
    { "kpts_shape", "17,3" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov8.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 1 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  auto kpts_tensors = tensors_from_vector(boxes, { 1, 1, 1, 51 });
  score_tensors.push_back(box_tensors[0]);
  score_tensors.push_back(kpts_tensors[0]);

  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);
  auto [actual_boxes, actual_kpts, actual_scores, kpts_per_box]
      = get_kpts_meta(map, meta_identifier);
  EXPECT_EQ(kpts_per_box, 17);
  EXPECT_EQ(actual_kpts.size(), 17);
}
TEST(yolov8_decode_scores, dequantize_with_sigmoid)
{
  //  With the scale and zero point values 14 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 13, 15, 12, 1 };
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "14, 14" },
    { "scales", "2.0, 2.0" },
    { "confidence_threshold", "0.20" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "sigmoid_in_postprocess", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "scale_up", "1" },
  };
  Decoder decoder("libdecode_yolov8.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  score_tensors.push_back(box_tensors[0]);
  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_object_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1 };
  auto expected_scores = std::vector<float>{ 0.88079703 };
  ASSERT_EQ(actual_classes, expected_classes);
  EXPECT_FLOAT_EQ(actual_scores[0], expected_scores[0]);
}
} // namespace
