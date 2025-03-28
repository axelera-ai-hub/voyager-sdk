
// Copyright Axelera AI, 2024
#include "gtest/gtest.h"
#include <gmodule.h>
#include "gmock/gmock.h"
#include "unittest_decode_common.h"

#include <filesystem>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "GstAxStreamerUtils.hpp"

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

TEST(rtmdet_errors, no_zero_points_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_rtmdet.so", properties), std::runtime_error);
}

TEST(rtmdet_errors, no_scales_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0" },
  };
  EXPECT_THROW(Decoder("libdecode_rtmdet.so", properties), std::runtime_error);
}


TEST(rtmdet_errors, different_scale_and_zero_point_sizes_throws)
{
  std::vector<int8_t> scores = { 0, 0, 0, 0, 0, 1, 1, 1 };

  ;
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_rtmdet.so", properties), std::runtime_error);
}

TEST(rtmdet_decode_scores, invalid_tensor_size)
{
  // Input vector with size of 9 is invalid
  std::vector<int8_t> yolo(9);
  std::string meta_identifier = "rtmdet";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0" },
    { "scales", "1" },
    { "confidence_threshold", "1.0" },
  };
  Decoder decoder("libdecode_rtmdet.so", properties);

  AxVideoInterface video_info{ { 64, 48, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(yolo, { 1, 9, 1, 1 });
  EXPECT_THROW(decoder.decode_to_meta(tensors, 0, 1, map, video_info), std::runtime_error);
}
TEST(rtmdet_decode_scores, all_filtered_at_max_confidence)
{
  std::vector<int8_t> rtmdet(268800);
  std::string meta_identifier = "rtmdet";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0,0,0,0,0,0" },
    { "scales", "1.0,1.0,1.0,1.0,1.0,1.0" },
    { "confidence_threshold", "0.5" },
  };
  Decoder decoder("libdecode_rtmdet.so", properties);

  AxVideoInterface video_info{ { 1920, 1080, 1920, 0, AxVideoFormat::BGRA }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(rtmdet, { 268800, 1, 1, 1 });

  decoder.decode_to_meta(tensors, 0, 1, map, video_info);
  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  EXPECT_EQ(actual_classes, std::vector<int32_t>{});
  EXPECT_EQ(actual_scores, std::vector<float>{});
}

TEST(rtmdet_decode_scores, one_detection)
{
  std::vector<int8_t> rtmdet(268800, 0);

  // Two dummy detections with confidance 1.0
  rtmdet[268800 / 2] = 127;
  rtmdet[(268800 / 2) + 64] = 127;

  std::string meta_identifier = "rtmdet";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0,0,0,0,0,0" },
    { "scales", "1.0,1.0,1.0,1.0,1.0,1.0" },
    { "confidence_threshold", "0.5" },
  };
  Decoder decoder("libdecode_rtmdet.so", properties);

  AxVideoInterface video_info{ { 1920, 1080, 1920, 0, AxVideoFormat::BGRA }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensors = tensors_from_vector(rtmdet, { 268800, 1, 1, 1 });

  decoder.decode_to_meta(tensors, 0, 1, map, video_info);
  auto [actual_boxes, actual_scores, actual_classes] = get_meta(map, meta_identifier);
  EXPECT_EQ(actual_classes.size(), 2);
  EXPECT_EQ(actual_scores.size(), 2);
  EXPECT_EQ(actual_boxes.size(), 8);
}
} // namespace
