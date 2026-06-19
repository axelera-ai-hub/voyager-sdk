// Copyright Axelera AI, 2025
#include "gtest/gtest.h"
#include <gmodule.h>
#include "gmock/gmock.h"
#include "unittest_ax_common.h"

#include <string>
#include <unordered_map>

#include "AxMeta.hpp"
#include "AxMetaImage.hpp"

#include "AxDataInterface.h"

using ::testing::ContainerEq;

namespace fs = std::filesystem;

namespace
{

using VectorType = std::variant<std::vector<float>, std::vector<uint8_t>>;
struct image_meta {
  VectorType depth;
  int channels;
  int width;
  int height;
  bool is_float;
};

image_meta
get_image_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return { {}, {} };
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);
  EXPECT_EQ(typeid(*meta), typeid(AxMetaImage));

  auto actual_metadata = meta->get_extern_meta();
  EXPECT_EQ(actual_metadata.size(), 5);
  auto p_is_float = reinterpret_cast<const bool *>(actual_metadata[1].meta);

  VectorType depth;
  if (*p_is_float) {
    auto p_depth = reinterpret_cast<const float *>(actual_metadata[0].meta);
    depth = std::vector<float>(
        p_depth, p_depth + actual_metadata[0].meta_size / sizeof(float));
  } else {
    auto p_depth = reinterpret_cast<const uint8_t *>(actual_metadata[0].meta);
    depth = std::vector<uint8_t>(
        p_depth, p_depth + actual_metadata[0].meta_size / sizeof(uint8_t));
  }
  auto width = *reinterpret_cast<const int *>(actual_metadata[2].meta);
  auto height = *reinterpret_cast<const int *>(actual_metadata[3].meta);
  auto channels = *reinterpret_cast<const int *>(actual_metadata[4].meta);
  return { depth, channels, width, height, *p_is_float };
}

template <typename T>
AxTensorsInterface
tensors_from_vector(std::vector<T> &tensors, std::vector<int> sizes)
{
  return {
    { sizes, sizeof tensors[0], tensors.data() },
  };
}

TEST(decode_image, passthru_test)
{
  std::string meta_identifier = "image_meta";

  std::vector<float> image(100);
  std::iota(image.begin(), image.end(), 1.0);

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "scale", "0" },
    { "output_datatype", "float32" },
  };
  auto decoder = Ax::LoadDecode("image", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto input_tensor = tensors_from_vector(image, { 1, 1, 10, 10 });

  decoder->decode_to_meta(input_tensor, 0, 1, map, video_info);

  auto [actual_depth, actual_channels, actual_width, actual_height, is_float]
      = get_image_meta(map, meta_identifier);
  EXPECT_TRUE(is_float);
  auto *input_depth = std::get_if<std::vector<float>>(&actual_depth);
  EXPECT_EQ(1, actual_channels);
  EXPECT_EQ(10, actual_width);
  EXPECT_EQ(10, actual_height);
  EXPECT_EQ(100, input_depth->size());
  EXPECT_THAT(*input_depth, ContainerEq(image));
}

TEST(decode_image, scale_test)
{
  std::string meta_identifier = "image_meta";

  std::vector<float> image(100, 0.5);

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "scale", "1" },
    { "output_datatype", "float32" },
  };
  auto decoder = Ax::LoadDecode("image", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto input_tensor = tensors_from_vector(image, { 1, 1, 10, 10 });

  decoder->decode_to_meta(input_tensor, 0, 1, map, video_info);

  auto [actual_depth, actual_channels, actual_width, actual_height, is_float]
      = get_image_meta(map, meta_identifier);
  EXPECT_TRUE(is_float);
  auto *input_depth = std::get_if<std::vector<float>>(&actual_depth);
  EXPECT_EQ(1, actual_channels);
  EXPECT_EQ(10, actual_width);
  EXPECT_EQ(10, actual_height);
  EXPECT_EQ(100, input_depth->size());
  for (auto i = 0; i < 100; ++i) {
    EXPECT_EQ(input_depth->at(i), 127.5);
  }
}

TEST(image_decode, uint8_test)
{
  std::string meta_identifier = "image_meta";

  std::vector<float> image(100, 0.5);

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "scale", "1" },
    { "output_datatype", "uint8" },
  };
  auto decoder = Ax::LoadDecode("image", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto input_tensor = tensors_from_vector(image, { 1, 1, 10, 10 });

  decoder->decode_to_meta(input_tensor, 0, 1, map, video_info);

  auto [actual_depth, actual_channels, actual_width, actual_height, is_float]
      = get_image_meta(map, meta_identifier);
  EXPECT_FALSE(is_float);
  auto *input_depth = std::get_if<std::vector<uint8_t>>(&actual_depth);
  EXPECT_EQ(1, actual_channels);
  EXPECT_EQ(10, actual_width);
  EXPECT_EQ(10, actual_height);
  EXPECT_EQ(100, input_depth->size());
  for (auto i = 0; i < 100; ++i) {
    EXPECT_EQ(input_depth->at(i), 127);
  }
}

// Regression test for NCHW dimension mapping with non-trivial C and H!=W
TEST(decode_image, nchw_dimension_test)
{
  std::string meta_identifier = "image_meta";

  // Create tensor with shape {1, 3, 8, 10} (N=1, C=3, H=8, W=10)
  std::vector<float> image(3 * 8 * 10);
  std::iota(image.begin(), image.end(), 1.0);

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "scale", "0" },
    { "output_datatype", "float32" },
  };
  auto decoder = Ax::LoadDecode("image", properties);

  AxVideoInterface video_info{ { 10, 8, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto input_tensor = tensors_from_vector(image, { 1, 3, 8, 10 });

  decoder->decode_to_meta(input_tensor, 0, 1, map, video_info);

  auto [actual_depth, actual_channels, actual_width, actual_height, is_float]
      = get_image_meta(map, meta_identifier);
  EXPECT_TRUE(is_float);
  auto *input_depth = std::get_if<std::vector<float>>(&actual_depth);

  // Verify NCHW dimensions are correctly mapped: C=3, H=8, W=10
  EXPECT_EQ(3, actual_channels);
  EXPECT_EQ(10, actual_width);
  EXPECT_EQ(8, actual_height);
  EXPECT_EQ(240, input_depth->size());
  EXPECT_THAT(*input_depth, ContainerEq(image));
}
} // namespace
