// Copyright Axelera AI, 2024
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
#include "AxMetaSegmentsDetection.hpp"

#include "AxDataInterface.h"

namespace fs = std::filesystem;

namespace
{

struct segment_meta {
  std::vector<int32_t> boxes;
  std::vector<float> scores;
  std::vector<int32_t> classes;
  std::vector<size_t> mask_shape;
  std::vector<float> masks;
  ;
};

segment_meta
get_segment_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);
  EXPECT_EQ(typeid(*meta), typeid(AxMetaSegmentsDetection));

  auto actual_metadata = meta->get_extern_meta();
  EXPECT_TRUE(actual_metadata.size() == 7 || actual_metadata.size() == 6);

  auto p_boxes = reinterpret_cast<const int32_t *>(actual_metadata[0].meta);
  auto p_scores = reinterpret_cast<const float *>(actual_metadata[2].meta);
  auto p_shape = reinterpret_cast<const size_t *>(actual_metadata[3].meta);
  auto p_classes = reinterpret_cast<const int32_t *>(actual_metadata[4].meta);

  auto actual_boxes = std::vector<int32_t>{ p_boxes,
    p_boxes + actual_metadata[0].meta_size / sizeof(int32_t) };
  auto actual_scores = std::vector<float>{ p_scores,
    p_scores + actual_metadata[2].meta_size / sizeof(float) };
  auto actual_shape = std::vector<size_t>{ p_shape,
    p_shape + actual_metadata[3].meta_size / sizeof(size_t) };
  auto actual_classes = std::vector<int32_t>{ p_classes,
    p_classes + actual_metadata[4].meta_size / sizeof(int32_t) };

  std::vector<float> masks;
  auto *smeta = dynamic_cast<AxMetaSegmentsDetection *>(meta);
  for (auto i = 0; i < smeta->num_elements(); ++i) {
    std::vector<float> smap = smeta->get_segment_map(i);
    masks.insert(masks.end(), smap.begin(), smap.end());
  }

  return { actual_boxes, actual_scores, actual_classes, actual_shape, masks };
}

template <typename T>
AxTensorsInterface
tensors_from_vector(std::vector<T> &tensors, std::vector<int> sizes)
{
  return {
    { sizes, sizeof tensors[0], tensors.data() },
  };
}

TEST(yolov8seg_errors, no_zero_points_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov8seg.so", properties), std::runtime_error);
}


TEST(yolov8seg_errors, no_scales_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov8seg.so", properties), std::runtime_error);
}

TEST(yolov8seg_errors, different_scale_and_zero_point_sizes_throws)
{
  std::unordered_map<std::string, std::string> properties = {
    { "zero_points", "0, 0" },
    { "scales", "1" },
  };
  EXPECT_THROW(Decoder("libdecode_yolov8seg.so", properties), std::runtime_error);
}

TEST(yolov8seg_decode_scores, all_filtered_at_max_confidence)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.25
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 0, 0, 0, 0 };
  auto masks = std::vector<int8_t>(32);
  auto prototypes = std::vector<int8_t>(10 * 10 * 32);
  std::string meta_identifier = "yolov8seg";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0, 0, 0" },
    { "scales", "1, 1, 1, 1" },
    { "confidence_threshold", "1.0" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov8seg.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  auto prototype_tensor = tensors_from_vector(prototypes, { 1, 10, 10, 32 });
  auto masks_tensors = tensors_from_vector(masks, { 1, 1, 1, 32 });
  score_tensors.push_back(box_tensors[0]);
  score_tensors.push_back(masks_tensors[0]);
  score_tensors.push_back(prototype_tensor[0]);

  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  // auto [actual_boxes, actual_scores, actual_classes] = get_segment_meta(map, meta_identifier);
  auto [actual_boxes, actual_scores, actual_classes, actual_mask_shape, actual_mask]
      = get_segment_meta(map, meta_identifier);
  EXPECT_EQ(actual_classes, std::vector<int32_t>{});
  EXPECT_EQ(actual_scores, std::vector<float>{});
}
TEST(yolov8seg_decode_scores, none_filtered_at_min_confidence_with_multiclass)
{
  //  With the scale and zero point values 0 -> 0.50
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 0, 0, 0, 0 };
  auto masks = std::vector<int8_t>(32);
  auto prototypes = std::vector<int8_t>(10 * 10 * 32);
  std::string meta_identifier = "yolov8seg";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0, 0, 0" },
    { "scales", "1, 1, 1, 1" },
    { "confidence_threshold", "0.0" },
    { "classes", "4" },
    { "multiclass", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov8seg.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  auto prototype_tensor = tensors_from_vector(prototypes, { 1, 10, 10, 32 });
  auto masks_tensors = tensors_from_vector(masks, { 1, 1, 1, 32 });
  score_tensors.push_back(box_tensors[0]);
  score_tensors.push_back(masks_tensors[0]);
  score_tensors.push_back(prototype_tensor[0]);

  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes, actual_mask_shape, actual_mask]
      = get_segment_meta(map, meta_identifier);

  EXPECT_EQ(actual_mask_shape.size(), 3);
  EXPECT_EQ(actual_mask_shape[1], 10);
  EXPECT_EQ(actual_mask_shape[2], 10);
  EXPECT_EQ(actual_mask.size(), 4 * 10 * 10);

  auto expected_classes = std::vector<int32_t>{ 0, 1, 2, 3 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F, 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);
}
TEST(yolov8seg_decode_scores, all_but_first_highest_filtered_at_min_confidence_with_non_multiclass)
{
  //  With the scale and zero point values 0 -> 0.5
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>{ 0, 0, 0, 0 };
  auto masks = std::vector<int8_t>(32);
  auto prototypes = std::vector<int8_t>(10 * 10 * 32);
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0, 0, 0" },
    { "scales", "1, 1, 1, 1" },
    { "confidence_threshold", "0.0" },
    { "classes", "4" },
    { "multiclass", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov8seg.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 4 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  auto prototype_tensor = tensors_from_vector(prototypes, { 1, 10, 10, 32 });
  auto masks_tensors = tensors_from_vector(masks, { 1, 1, 1, 32 });
  score_tensors.push_back(box_tensors[0]);
  score_tensors.push_back(masks_tensors[0]);
  score_tensors.push_back(prototype_tensor[0]);
  decoder.decode_to_meta(score_tensors, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes, actual_mask_shape, actual_mask]
      = get_segment_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 0 };
  auto expected_scores = std::vector<float>{ 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);

  EXPECT_EQ(actual_mask_shape.size(), 3);
  EXPECT_EQ(actual_mask_shape[1], 10);
  EXPECT_EQ(actual_mask_shape[2], 10);
  EXPECT_EQ(actual_mask.size(), 1 * 10 * 10);
}
TEST(yolov8seg_decode_scores, with_multiclass_all_below_threshold_are_filtered)
{
  //  With the scale and zero point values 0 -> 0.5
  //  This means with an object score of 0.5, the confidence of a class with a
  //  score value of 0 becomes 0.250
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>(80, -1);
  auto score_vals = std::vector<int8_t>{ -1, 0, -1, 0 };
  std::copy(score_vals.begin(), score_vals.end(), scores.begin());
  auto masks = std::vector<int8_t>(32, 0);
  auto prototypes = std::vector<int8_t>(10 * 10 * 32);
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0, 0, 0" },
    { "scales", "1, 1, 1, 1" },
    { "confidence_threshold", "0.4" },
    { "classes", "80" },
    { "multiclass", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov8seg.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 80 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  auto prototype_tensor = tensors_from_vector(prototypes, { 1, 10, 10, 32 });
  auto masks_tensors = tensors_from_vector(masks, { 1, 1, 1, 32 });
  prototype_tensor.push_back(masks_tensors[0]);
  prototype_tensor.push_back(score_tensors[0]);
  prototype_tensor.push_back(box_tensors[0]);
  decoder.decode_to_meta(prototype_tensor, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes, actual_mask_shape, actual_mask]
      = get_segment_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);

  EXPECT_EQ(actual_mask_shape.size(), 3);
  EXPECT_EQ(actual_mask_shape[1], 10);
  EXPECT_EQ(actual_mask_shape[2], 10);
  EXPECT_EQ(actual_mask.size(), 2 * 10 * 10);
}

TEST(yolov8seg_decode_scores, without_scaling_segments)
{
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>(80, -1);
  auto score_vals = std::vector<int8_t>{ -1, 0, -1, 0 };
  std::copy(score_vals.begin(), score_vals.end(), scores.begin());
  auto masks = std::vector<int8_t>(32, 0);
  auto prototypes = std::vector<int8_t>(10 * 10 * 32);
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0, 0, 0" },
    { "scales", "1, 1, 1, 1" },
    { "confidence_threshold", "0.4" },
    { "classes", "80" },
    { "multiclass", "1" },
    { "scale_mask_to_input", "0" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov8seg.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 80 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  auto prototype_tensor = tensors_from_vector(prototypes, { 1, 10, 10, 32 });
  auto masks_tensors = tensors_from_vector(masks, { 1, 1, 1, 32 });
  prototype_tensor.push_back(masks_tensors[0]);
  prototype_tensor.push_back(score_tensors[0]);
  prototype_tensor.push_back(box_tensors[0]);
  decoder.decode_to_meta(prototype_tensor, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes, actual_mask_shape, actual_mask]
      = get_segment_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);

  EXPECT_EQ(actual_mask_shape.size(), 3);
  EXPECT_EQ(actual_mask_shape[1], 10); // 10x10 prototype tensor width and height
  EXPECT_EQ(actual_mask_shape[2], 10);
  EXPECT_EQ(actual_mask.size(), 2 * 10 * 10);
}

TEST(yolov8seg_decode_scores, with_heatmap)
{
  auto boxes = std::vector<int8_t>(64);
  auto scores = std::vector<int8_t>(80, -1);
  auto score_vals = std::vector<int8_t>{ -1, 0, -1, 0 };
  std::copy(score_vals.begin(), score_vals.end(), scores.begin());
  auto masks = std::vector<int8_t>(32, 0);
  auto prototypes = std::vector<int8_t>(10 * 10 * 32);
  std::string meta_identifier = "yolov8";

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "zero_points", "0, 0, 0, 0" },
    { "scales", "1, 1, 1, 1" },
    { "confidence_threshold", "0.4" },
    { "classes", "80" },
    { "multiclass", "1" },
    { "heatmap", "1" },
    { "model_width", "640" },
    { "model_height", "640" },
  };
  Decoder decoder("libdecode_yolov8seg.so", properties);

  AxVideoInterface video_info{ { 64, 64, 64, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto score_tensors = tensors_from_vector(scores, { 1, 1, 1, 80 });
  auto box_tensors = tensors_from_vector(boxes, { 1, 1, 1, 64 });
  auto prototype_tensor = tensors_from_vector(prototypes, { 1, 10, 10, 32 });
  auto masks_tensors = tensors_from_vector(masks, { 1, 1, 1, 32 });
  prototype_tensor.push_back(masks_tensors[0]);
  prototype_tensor.push_back(score_tensors[0]);
  prototype_tensor.push_back(box_tensors[0]);
  decoder.decode_to_meta(prototype_tensor, 0, 1, map, video_info);

  auto [actual_boxes, actual_scores, actual_classes, actual_mask_shape, actual_mask]
      = get_segment_meta(map, meta_identifier);
  auto expected_classes = std::vector<int32_t>{ 1, 3 };
  auto expected_scores = std::vector<float>{ 0.5F, 0.5F };
  EXPECT_EQ(actual_classes, expected_classes);
  EXPECT_EQ(actual_scores, expected_scores);

  EXPECT_EQ(actual_mask_shape.size(), 3);
  EXPECT_EQ(actual_mask_shape[1], 10);
  EXPECT_EQ(actual_mask_shape[2], 10);
  EXPECT_EQ(actual_mask.size(), 2 * 10 * 10);
}
} // namespace
