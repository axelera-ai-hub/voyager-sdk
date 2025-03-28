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

TEST(ssd_errors, no_zero_points_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_ssd2.so", properties), std::runtime_error);
}

TEST(ssd_errors, no_scales_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0" },
  };
  EXPECT_THROW(Decoder("libdecode_ssd2.so", properties), std::runtime_error);
}

TEST(ssd_errors, different_scale_and_zero_point_sizes_throws)
{
  std::vector<int8_t> scores = { 0, 0, 0, 0, 0, 1, 1, 1 };
  std::string meta_identifier = "yolov5";

  ;
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder decoder("libdecode_ssd2.so", properties), std::runtime_error);
}

TEST(ssd_errors, must_provide_classes)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1, 2" },
    { "anchors", "10, 10" },
  };
  EXPECT_THROW(Decoder("libdecode_ssd2.so", properties), std::runtime_error);
}


TEST(ssd_decode_scores, all_filtered_at_max_confidence)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  // clang-format off

  //  First level only has 3 priors
  std::vector<int8_t> ssd_box = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  std::vector<int8_t> ssd_scores = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };
  // clang-format on

  std::string meta_identifier = "ssd";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "confidence_threshold", "1.0" },
    { "classes", "5" },
    { "class_agnostic", "1" },
  };
  Decoder decoder("libdecode_ssd2.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(ssd_box, { 1, 1, 1, 3 * 4 });
  auto score_tensors = tensors_from_vector(ssd_scores, { 1, 1, 1, 3 * (5 + 1) });
  tensors.push_back(score_tensors[0]);
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  EXPECT_EQ(actual_classes, std::vector<int32_t>{});
  EXPECT_EQ(actual_scores, std::vector<float>{});
}

TEST(ssd_decode_scores, none_filtered_at_min_confidence_with_multiclass)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  // clang-format off

  //  First level only has 3 priors
  std::vector<int8_t> ssd_box = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  std::vector<int8_t> ssd_scores = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };
  // clang-format on

  std::string meta_identifier = "ssd";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.0" },
    { "classes", "4" },
    { "class_agnostic", "0" },
  };
  Decoder decoder("libdecode_ssd2.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(ssd_box, { 1, 1, 1, 3 * 4 });
  auto score_tensors = tensors_from_vector(ssd_scores, { 1, 1, 1, 3 * (5 + 1) });
  tensors.push_back(score_tensors[0]);
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F,
    0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}

TEST(ssd_decode_scores, all_but_first_highest_filtered_at_min_confidence_with_non_multiclass)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  // clang-format off

  //  First level only has 3 priors
  std::vector<int8_t> ssd_box = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  std::vector<int8_t> ssd_scores = {
    0, 0, -1, -1, -1, -1,
    0, -1, 0, -1, -1, -1,
    0, -1, -1, 0, -1, -1
  };
  // clang-format on

  std::string meta_identifier = "ssd";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.0" },
    { "classes", "5" },
    { "class_agnostic", "1" },
  };
  Decoder decoder("libdecode_ssd2.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(ssd_box, { 1, 1, 1, 3 * 4 });
  auto score_tensors = tensors_from_vector(ssd_scores, { 1, 1, 1, 3 * (5 + 1) });
  tensors.push_back(score_tensors[0]);
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 1, 2 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}


TEST(ssd_decode_scores, with_multiclass_all_below_threshold_are_filtered)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  // clang-format off

  //  First level only has 3 priors
  std::vector<int8_t> ssd_box = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  std::vector<int8_t> ssd_scores = {
    0, 0, 0, -1, -1, -1,
    0, -1, 0, -1, 0, -1,
    0, -1, -1, 0, -1, 0
  };
  // clang-format on

  std::string meta_identifier = "ssd";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0" },
    { "scales", "1, 1" },
    { "anchors", "10, 10" },
    { "confidence_threshold", "0.5" },
    { "classes", "5" },
    { "class_agnostic", "0" },
  };
  Decoder decoder("libdecode_ssd2.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(ssd_box, { 1, 1, 1, 3 * 4 });
  auto score_tensors = tensors_from_vector(ssd_scores, { 1, 1, 1, 3 * (5 + 1) });
  tensors.push_back(score_tensors[0]);
  decoder.decode_to_meta(tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0, 1, 1, 3, 2, 4 };
  auto expected_scores = std::vector<float>{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}

} // namespace
