// Copyright Axelera AI, 2024
#include "gtest/gtest.h"
#include <gmodule.h>
#include "gmock/gmock.h"
#include "unittest_decode_common.h"

#include <string>
#include <unordered_map>

#include "AxMeta.hpp"
#include "AxMetaSemanticSegmentation.hpp"

#include "AxDataInterface.h"

using ::testing::ContainerEq;
using ::testing::ElementsAre;

namespace fs = std::filesystem;

namespace
{

struct sem_seg_meta {
  std::vector<int> class_ids;
  std::vector<float> probabilities;
  std::vector<int> shape;
};

sem_seg_meta
get_semantic_seg_meta(
    const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier, bool probs = false)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr) << " for id=" << meta_identifier;
  if (!meta) {
    return { {}, {} };
  }
  auto &x = *meta;
  const auto &tid = typeid(x);
  EXPECT_EQ(tid, typeid(AxMetaSemanticSegmentation));

  auto actual_metadata = meta->get_extern_meta();
  EXPECT_EQ(actual_metadata.size(), 2);

  auto p_shape = reinterpret_cast<const int *>(actual_metadata[0].meta);
  auto shape = std::vector<int>{ p_shape,
    p_shape + actual_metadata[0].meta_size / sizeof(int) };

  if (probs) {
    auto p_probabilities = reinterpret_cast<const float *>(actual_metadata[1].meta);
    auto probabilities = std::vector<float>{ p_probabilities,
      p_probabilities + actual_metadata[1].meta_size / sizeof(float) };
    return { {}, probabilities, shape };
  } else {
    auto p_class_ids = reinterpret_cast<const int *>(actual_metadata[1].meta);
    auto class_ids = std::vector<int>{ p_class_ids,
      p_class_ids + actual_metadata[1].meta_size / sizeof(int) };
    return { class_ids, {}, shape };
  }
}

template <typename T>
AxTensorsInterface
tensors_from_vector(std::vector<T> &tensors, std::vector<int> sizes)
{
  return {
    { sizes, sizeof tensors[0], tensors.data() },
  };
}

TEST(semantic_segmentation_decode, probability_out)
{
  std::string meta_identifier = "semantic_seg";

  std::vector<float> probs;
  probs.reserve(500);
  std::vector<float> unique_numbers = { 0.2, 0.4, 0.6, 0.8, 1.0 };

  for (int i = 0; i < 100; ++i) {
    probs.insert(probs.end(), unique_numbers.begin(), unique_numbers.end());
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "class_map_out", "0" },
  };
  Decoder decoder("libdecode_semantic_seg.so", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto probs_tensor = tensors_from_vector(probs, { 1, 10, 10, 5 });

  decoder.decode_to_meta(probs_tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier, true);

  EXPECT_EQ(0, actual_class_ids.size());
  EXPECT_EQ(500, actual_probs.size());
  EXPECT_EQ(3, actual_shape.size());

  EXPECT_THAT(actual_shape, ElementsAre(10, 10, 5));
  EXPECT_THAT(actual_probs, ContainerEq(probs));
}


TEST(semantic_segmentation_decode, happy_path)
{
  std::string meta_identifier = "semantic_seg";

  std::vector<float> probs;
  probs.reserve(500);
  std::vector<float> unique_numbers = { 0.2, 0.4, 0.6, 0.8, 1.0 };

  for (int i = 0; i < 100; ++i) {
    probs.insert(probs.end(), unique_numbers.begin(), unique_numbers.end());
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
  };
  Decoder decoder("libdecode_semantic_seg.so", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto probs_tensor = tensors_from_vector(probs, { 1, 10, 10, 5 });

  decoder.decode_to_meta(probs_tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  EXPECT_EQ(100, actual_class_ids.size());
  EXPECT_EQ(0, actual_probs.size());
  EXPECT_EQ(3, actual_shape.size());

  EXPECT_THAT(actual_shape, ElementsAre(10, 10, 5));

  EXPECT_TRUE(std::all_of(actual_class_ids.begin(), actual_class_ids.end(),
      [](int x) { return x == 4; }));
}


TEST(semantic_segmentation_decode, binary_path)
{
  std::string meta_identifier = "semantic_seg";

  std::vector<float> probs;
  probs.reserve(100);

  for (int i = 0; i < 100; ++i) {
    probs[i] = i % 2 ? -1.0f : 1.0f;
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "threshold", "0.5" },
  };
  Decoder decoder("libdecode_semantic_seg.so", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto probs_tensor = tensors_from_vector(probs, { 1, 10, 10, 1 });

  decoder.decode_to_meta(probs_tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  EXPECT_EQ(100, actual_class_ids.size());
  EXPECT_EQ(0, actual_probs.size());
  EXPECT_EQ(3, actual_shape.size());

  EXPECT_THAT(actual_shape, ElementsAre(10, 10, 1));

  EXPECT_EQ(50, std::count(actual_class_ids.begin(), actual_class_ids.end(), 1));
  EXPECT_EQ(50, std::count(actual_class_ids.begin(), actual_class_ids.end(), 0));
}

} // namespace
