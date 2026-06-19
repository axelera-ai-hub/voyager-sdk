// Copyright Axelera AI, 2024
#include "gtest/gtest.h"
#include <gmodule.h>
#include "gmock/gmock.h"
#include "unittest_ax_common.h"

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
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto probs_tensor = tensors_from_vector(probs, { 1, 10, 10, 5 });

  decoder->decode_to_meta(probs_tensor, 0, 1, map, video_info);

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
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto probs_tensor = tensors_from_vector(probs, { 1, 10, 10, 5 });

  decoder->decode_to_meta(probs_tensor, 0, 1, map, video_info);

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
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  AxVideoInterface video_info{ { 10, 10, 3, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto probs_tensor = tensors_from_vector(probs, { 1, 10, 10, 1 });

  decoder->decode_to_meta(probs_tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  EXPECT_EQ(100, actual_class_ids.size());
  EXPECT_EQ(0, actual_probs.size());
  EXPECT_EQ(3, actual_shape.size());

  EXPECT_THAT(actual_shape, ElementsAre(10, 10, 1));

  EXPECT_EQ(50, std::count(actual_class_ids.begin(), actual_class_ids.end(), 1));
  EXPECT_EQ(50, std::count(actual_class_ids.begin(), actual_class_ids.end(), 0));
}


TEST(semantic_segmentation_decode, letterbox_cityscapes_aspect_ratio)
{
  // Test letterboxing for Cityscapes (2048x1024 letterboxed to 1024x1024)
  // Output from model is 128x128, but should be cropped to 64x128 (removing vertical padding)
  std::string meta_identifier = "semantic_seg";

  // Create 128x128x19 int8 tensor with known pattern
  std::vector<int8_t> data(128 * 128 * 64, 0); // 64 channels padded

  // Fill middle 64 rows (32-96) with class 1, top/bottom with class 0 (letterbox padding)
  for (int h = 0; h < 128; ++h) {
    for (int w = 0; w < 128; ++w) {
      for (int c = 0; c < 19; ++c) {
        int idx = h * 128 * 64 + w * 64 + c;
        // Middle region gets higher values for class 1
        if (h >= 32 && h < 96) {
          if (c == 1) {
            data[idx] = 100; // Class 1 wins in middle region
          } else {
            data[idx] = 0;
          }
        } else {
          // Padding region: class 0 wins
          if (c == 0) {
            data[idx] = 100;
          } else {
            data[idx] = 0;
          }
        }
      }
    }
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "padding", "0,0,0,0,0,0,0,45" }, // 45 channels of padding
    { "model_width", "1024" },
    { "model_height", "1024" },
    { "scale_up", "0" }, // LETTERBOX_CONTAIN
    { "letterbox", "1" },
  };
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  // Original image is 2048x1024 (Cityscapes)
  AxVideoInterface video_info{ { 2048, 1024, 0, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensor = tensors_from_vector(data, { 1, 128, 128, 64 });

  decoder->decode_to_meta(tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  // Should output 64x128 (height is halved due to letterbox)
  EXPECT_THAT(actual_shape, ElementsAre(64, 128, 19));
  EXPECT_EQ(64 * 128, actual_class_ids.size());

  // All pixels should be class 1 (middle region, padding removed)
  EXPECT_TRUE(std::all_of(actual_class_ids.begin(), actual_class_ids.end(),
      [](int x) { return x == 1; }));
}


TEST(semantic_segmentation_decode, letterbox_square_image_no_crop)
{
  // Test with square image (1024x1024) - no letterbox cropping needed
  std::string meta_identifier = "semantic_seg";

  std::vector<int8_t> data(128 * 128 * 64, 0);

  // All pixels should be class 2
  for (int h = 0; h < 128; ++h) {
    for (int w = 0; w < 128; ++w) {
      for (int c = 0; c < 19; ++c) {
        int idx = h * 128 * 64 + w * 64 + c;
        if (c == 2) {
          data[idx] = 100;
        } else {
          data[idx] = 0;
        }
      }
    }
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "padding", "0,0,0,0,0,0,0,45" },
    { "model_width", "1024" },
    { "model_height", "1024" },
    { "scale_up", "1" },
    { "letterbox", "1" },
  };
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  // Square image - no letterbox padding needed
  AxVideoInterface video_info{ { 1024, 1024, 0, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensor = tensors_from_vector(data, { 1, 128, 128, 64 });

  decoder->decode_to_meta(tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  // Full 128x128 output (no cropping)
  EXPECT_THAT(actual_shape, ElementsAre(128, 128, 19));
  EXPECT_EQ(128 * 128, actual_class_ids.size());

  EXPECT_TRUE(std::all_of(actual_class_ids.begin(), actual_class_ids.end(),
      [](int x) { return x == 2; }));
}


TEST(semantic_segmentation_decode, letterbox_disabled)
{
  // Test with letterbox disabled - should output full tensor
  std::string meta_identifier = "semantic_seg";

  std::vector<int8_t> data(128 * 128 * 64, 0);

  for (int h = 0; h < 128; ++h) {
    for (int w = 0; w < 128; ++w) {
      for (int c = 0; c < 19; ++c) {
        int idx = h * 128 * 64 + w * 64 + c;
        if (c == 3) {
          data[idx] = 100;
        } else {
          data[idx] = 0;
        }
      }
    }
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier }, { "padding", "0,0,0,0,0,0,0,45" },
    { "model_width", "1024" }, { "model_height", "1024" }, { "scale_up", "0" },
    { "letterbox", "0" }, // Disabled
  };
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  AxVideoInterface video_info{ { 2048, 1024, 0, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensor = tensors_from_vector(data, { 1, 128, 128, 64 });

  decoder->decode_to_meta(tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  // Full output since letterbox is disabled
  EXPECT_THAT(actual_shape, ElementsAre(128, 128, 19));
  EXPECT_EQ(128 * 128, actual_class_ids.size());

  EXPECT_TRUE(std::all_of(actual_class_ids.begin(), actual_class_ids.end(),
      [](int x) { return x == 3; }));
}


TEST(semantic_segmentation_decode, letterbox_portrait_image)
{
  // Test portrait image (512x1024 letterboxed to 1024x1024)
  // Should crop width, keep full height
  std::string meta_identifier = "semantic_seg";

  std::vector<int8_t> data(128 * 128 * 64, 0);

  // Fill middle 64 columns (32-96) with class 4, left/right with class 0
  for (int h = 0; h < 128; ++h) {
    for (int w = 0; w < 128; ++w) {
      for (int c = 0; c < 19; ++c) {
        int idx = h * 128 * 64 + w * 64 + c;
        if (w >= 32 && w < 96) {
          if (c == 4) {
            data[idx] = 100;
          } else {
            data[idx] = 0;
          }
        } else {
          if (c == 0) {
            data[idx] = 100;
          } else {
            data[idx] = 0;
          }
        }
      }
    }
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "padding", "0,0,0,0,0,0,0,45" },
    { "model_width", "1024" },
    { "model_height", "1024" },
    { "scale_up", "0" },
    { "letterbox", "1" },
  };
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  // Portrait: 512x1024
  AxVideoInterface video_info{ { 512, 1024, 0, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensor = tensors_from_vector(data, { 1, 128, 128, 64 });

  decoder->decode_to_meta(tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  // Should output 128x64 (width is halved)
  EXPECT_THAT(actual_shape, ElementsAre(128, 64, 19));
  EXPECT_EQ(128 * 64, actual_class_ids.size());

  // All should be class 4 (middle columns, padding removed)
  EXPECT_TRUE(std::all_of(actual_class_ids.begin(), actual_class_ids.end(),
      [](int x) { return x == 4; }));
}


TEST(semantic_segmentation_decode, letterbox_with_quantization)
{
  // Test letterbox with quantized int8 input and dequantization
  std::string meta_identifier = "semantic_seg";

  std::vector<int8_t> data(64 * 64 * 32, 0); // Smaller for test

  // Create pattern in middle region
  for (int h = 0; h < 64; ++h) {
    for (int w = 0; w < 64; ++w) {
      for (int c = 0; c < 19; ++c) {
        int idx = h * 64 * 32 + w * 32 + c;
        // Middle 32 rows: class 5
        if (h >= 16 && h < 48) {
          data[idx] = (c == 5) ? 50 : -50;
        } else {
          data[idx] = (c == 0) ? 50 : -50;
        }
      }
    }
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "padding", "0,0,0,0,0,0,0,13" },
    { "scales", "0.1" },
    { "zero_points", "0" },
    { "model_width", "512" },
    { "model_height", "512" },
    { "scale_up", "0" },
    { "letterbox", "1" },
  };
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  // 1024x512 image
  AxVideoInterface video_info{ { 1024, 512, 0, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensor = tensors_from_vector(data, { 1, 64, 64, 32 });

  decoder->decode_to_meta(tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  // Should crop to 32x64
  EXPECT_THAT(actual_shape, ElementsAre(32, 64, 19));
  EXPECT_EQ(32 * 64, actual_class_ids.size());

  // Should be class 5
  EXPECT_TRUE(std::all_of(actual_class_ids.begin(), actual_class_ids.end(),
      [](int x) { return x == 5; }));
}


TEST(semantic_segmentation_decode, interpolate_before_argmax)
{
  // Test interpolation of logits before argmax for improved accuracy
  // Model output is 64x64, interpolated to video resolution 256x256
  std::string meta_identifier = "semantic_seg";

  std::vector<int8_t> data(64 * 64 * 32, 0);

  // Create a simple pattern: top half is class 0, bottom half is class 1
  for (int h = 0; h < 64; ++h) {
    for (int w = 0; w < 64; ++w) {
      for (int c = 0; c < 19; ++c) {
        int idx = h * 64 * 32 + w * 32 + c;
        if (h < 32) {
          // Top half: class 0
          if (c == 0) {
            data[idx] = 100;
          } else {
            data[idx] = 0;
          }
        } else {
          // Bottom half: class 1
          if (c == 1) {
            data[idx] = 100;
          } else {
            data[idx] = 0;
          }
        }
      }
    }
  }

  std::unordered_map<std::string, std::string> properties = {
    { "meta_key", meta_identifier },
    { "padding", "0,0,0,0,0,0,0,13" },
    { "scales", "0.1" },
    { "zero_points", "0" },
    { "model_width", "256" },
    { "model_height", "256" },
    { "scale_up", "0" },
    { "letterbox", "0" },
    { "interpolate_before_argmax", "1" },
  };
  auto decoder = Ax::LoadDecode("semantic_seg", properties);

  // Original video is 256x256
  AxVideoInterface video_info{ { 256, 256, 0, 0, AxVideoFormat::RGB }, nullptr };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> map{};
  auto tensor = tensors_from_vector(data, { 1, 64, 64, 32 });

  decoder->decode_to_meta(tensor, 0, 1, map, video_info);

  auto [actual_class_ids, actual_probs, actual_shape]
      = get_semantic_seg_meta(map, meta_identifier);

  // Should output at video resolution (256x256) not model output resolution (64x64)
  EXPECT_THAT(actual_shape, ElementsAre(256, 256, 19));
  EXPECT_EQ(256 * 256, actual_class_ids.size());

  // Verify pattern is preserved after interpolation
  // Top half should be mostly class 0, bottom half mostly class 1
  int top_class_0_count = 0;
  int bottom_class_1_count = 0;
  for (int h = 0; h < 256; ++h) {
    for (int w = 0; w < 256; ++w) {
      int idx = h * 256 + w;
      if (h < 128 && actual_class_ids[idx] == 0) {
        top_class_0_count++;
      }
      if (h >= 128 && actual_class_ids[idx] == 1) {
        bottom_class_1_count++;
      }
    }
  }

  // Most pixels should match the expected pattern
  EXPECT_GT(top_class_0_count, 128 * 256 * 0.95); // At least 95% correct in top half
  EXPECT_GT(bottom_class_1_count, 128 * 256 * 0.95); // At least 95% correct in bottom half
}


} // namespace
