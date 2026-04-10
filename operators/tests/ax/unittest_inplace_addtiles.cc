// Copyright Axelera AI, 2025
#include "gmock/gmock.h"
#include "unittest_ax_common.h"

#include <string>
#include "AxMetaObjectDetection.hpp"

namespace
{
std::vector<uint32_t>
get_meta(const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    std::string meta_identifier)
{
  auto position = map.find(meta_identifier);
  if (position == map.end()) {
    return {};
  }
  auto *meta = position->second.get();
  EXPECT_NE(meta, nullptr);
  EXPECT_EQ(typeid(*meta), typeid(AxMetaObjDetectionTiles));

  auto actual_metadata = meta->get_extern_meta();
  EXPECT_EQ(actual_metadata.size(), 3);

  auto p_boxes = reinterpret_cast<const uint32_t *>(actual_metadata[0].meta);
  auto actual_boxes = std::vector<uint32_t>{ p_boxes,
    p_boxes + actual_metadata[0].meta_size / sizeof(int32_t) };

  return actual_boxes;
}


TEST(full_screen, should_return_all_tiles_plus)
{
  std::string meta_identifier = "tiles";

  std::unordered_map<std::string, std::string> input = {
    { "meta_key", meta_identifier },
    { "tile_size", "640" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "tile_overlap", "0" },
    { "tile_position", "none" },
  };
  auto tiler = Ax::LoadInPlace("addtiles", input);
  Ax::MetaMap metadata;
  AxVideoInterface video_info{ { 1280, 720, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  tiler->inplace(video_info, 0, 1, metadata);
  auto boxes = get_meta(metadata, meta_identifier);
  std::vector<uint32_t> expected = {
    // clang-format off
    0, 0, 1279, 719,
    0, 0, 639, 639,
    640, 0, 1279, 639,
    0, 80, 639, 719,
    640, 80, 1279, 719
    // clang-format on
  };
  EXPECT_EQ(expected, boxes);
}

TEST(left, should_return_left_tiles_plus)
{
  std::string meta_identifier = "tiles";

  std::unordered_map<std::string, std::string> input = {
    { "meta_key", meta_identifier },
    { "tile_size", "640" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "tile_overlap", "0" },
    { "tile_position", "left" },
  };
  auto tiler = Ax::LoadInPlace("addtiles", input);
  Ax::MetaMap metadata;
  AxVideoInterface video_info{ { 1280, 720, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  tiler->inplace(video_info, 0, 1, metadata);
  auto boxes = get_meta(metadata, meta_identifier);
  std::vector<uint32_t> expected = {
    // clang-format off
    0, 0, 1279, 719,
    0, 0, 639, 639,
    0, 80, 639, 719
    // clang-format on
  };
  EXPECT_EQ(expected, boxes);
}

TEST(right, should_return_right_tiles_plus)
{
  std::string meta_identifier = "tiles";

  std::unordered_map<std::string, std::string> input = {
    { "meta_key", meta_identifier },
    { "tile_size", "640" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "tile_overlap", "0" },
    { "tile_position", "right" },
  };
  auto tiler = Ax::LoadInPlace("addtiles", input);
  Ax::MetaMap metadata;
  AxVideoInterface video_info{ { 1280, 720, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  tiler->inplace(video_info, 0, 1, metadata);
  auto boxes = get_meta(metadata, meta_identifier);
  std::vector<uint32_t> expected = {
    // clang-format off
    0, 0, 1279, 719,
    640, 0, 1279, 639,
    640, 80, 1279, 719
    // clang-format on
  };
  EXPECT_EQ(expected, boxes);
}

TEST(top, should_return_top_tiles_plus)
{
  std::string meta_identifier = "tiles";

  std::unordered_map<std::string, std::string> input = {
    { "meta_key", meta_identifier },
    { "tile_size", "640" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "tile_overlap", "0" },
    { "tile_position", "top" },
  };
  auto tiler = Ax::LoadInPlace("addtiles", input);
  Ax::MetaMap metadata;
  AxVideoInterface video_info{ { 1280, 720, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  tiler->inplace(video_info, 0, 1, metadata);
  auto boxes = get_meta(metadata, meta_identifier);
  std::vector<uint32_t> expected = {
    // clang-format off
    0, 0, 1279, 719,
    0, 0, 639, 359,
    640, 0, 1279, 359 // clang-format on
  };
  EXPECT_EQ(expected, boxes);
}

TEST(bottom, should_return_bottom_tiles_plus)
{
  std::string meta_identifier = "tiles";

  std::unordered_map<std::string, std::string> input = {
    { "meta_key", meta_identifier },
    { "tile_size", "640" },
    { "model_width", "640" },
    { "model_height", "640" },
    { "tile_overlap", "0" },
    { "tile_position", "bottom" },
  };
  auto tiler = Ax::LoadInPlace("addtiles", input);
  Ax::MetaMap metadata;
  AxVideoInterface video_info{ { 1280, 720, 1280 * 4, 0, AxVideoFormat::RGBA }, nullptr };
  tiler->inplace(video_info, 0, 1, metadata);
  auto boxes = get_meta(metadata, meta_identifier);
  std::vector<uint32_t> expected = {
    // clang-format off
    0, 0, 1279, 719,
    0, 360, 639, 719,
    640, 360, 1279, 719
    // clang-format on
  };
  EXPECT_EQ(expected, boxes);
}

} // namespace
