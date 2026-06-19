// Copyright Axelera AI, 2025
#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaStreamId.hpp"
#include "AxOpUtils.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

using json = nlohmann::json;

struct addtiles_properties {
  std::string meta_key{ "" };
  std::string tile_size{};
  std::string tile_position{};
  size_t slice_size{ 1080 };
  size_t tile_overlap{ 0 };
  size_t model_width{ 0 };
  size_t model_height{ 0 };
  size_t tile_width{ 0 };
  size_t tile_height{ 0 };
  std::string json_file{ "" };
  mutable std::vector<std::array<int, 4>> tiles{};
  mutable time_t last_modified{ 0 };
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "tile_size",
    "tile_overlap",
    "tile_position",
    "model_width",
    "model_height",
    "tile_json",
  };
  return allowed_properties;
}

bool
validate_tiles(std::span<std::array<int, 4>> tiles, int image_width, int image_height)
{
  for (const auto &tile : tiles) {
    if (tile[0] < 0 || tile[1] < 0 || tile[2] < 0 || tile[3] < 0) {
      return false;
    }
    if (tile[2] < tile[0] || tile[3] < tile[1]) {
      return false;
    }
    if (tile[0] >= image_width || tile[1] >= image_height) {
      return false;
    }
    if (tile[2] >= image_width || tile[3] >= image_height) {
      return false;
    }
  }
  return true;
}

time_t
get_last_modified_time(const std::string &path)
{
  if (path.empty()) {
    return 0;
  }
  std::error_code ec;
  auto ftime = std::filesystem::last_write_time(path, ec);
  if (ec) {
    return 0;
  }
  auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
      ftime - std::filesystem::file_time_type::clock::now()
      + std::chrono::system_clock::now());
  return std::chrono::system_clock::to_time_t(sctp);
}

void
determine_slice_sizes(addtiles_properties &properties, Ax::Logger &logger)
{
  if (properties.tile_size == "default") {
    properties.tile_width = properties.model_width;
    properties.tile_height = properties.model_height;
    return;
  }
  auto parts = Ax::Internal::split(properties.tile_size, 'x');
  if (parts.size() == 1) {
    auto slice_size = std::stoi(std::string(parts[0]));

    auto longest = std::max(properties.model_width, properties.model_height);
    if (slice_size < longest) {
      logger(AX_WARN)
          << "inplace_addtiles: slice_size (" << slice_size
          << ") is less than the model's longest side (" << longest << ")."
          << " Using the model's longest side as slice_size." << std::endl;
      slice_size = longest;
    }
    auto model_ratio = static_cast<float>(properties.model_width) / properties.model_height;
    auto slice_width = model_ratio > 1 ? slice_size : slice_size * model_ratio;
    auto slice_height = model_ratio > 1 ? slice_size / model_ratio : slice_size;
    properties.tile_width = slice_width;
    properties.tile_height = slice_height;
    return;
  } else {
    properties.tile_width = std::stoi(std::string(parts[0]));
    properties.tile_height = std::stoi(std::string(parts[1]));
  }
}

std::vector<std::array<int, 4>>
load_tiles_from_json(const std::string &json_file, Ax::Logger &logger)
{
  try {
    std::ifstream new_tiles(json_file);
    if (!new_tiles.is_open()) {
      logger(AX_ERROR) << "Could not open tile JSON file: " << json_file << std::endl;
      return {};
    }
    json j;
    new_tiles >> j;
    new_tiles.close();
    std::vector<std::array<int, 4>> tiles = j.get<std::vector<std::array<int, 4>>>();
    for (auto &tile : tiles) {

      tile[2] = tile[0] + tile[2] - 1;
      tile[3] = tile[1] + tile[3] - 1;
    }
    return tiles;
  } catch (std::exception &e) {
    logger(AX_ERROR) << "Error loading tiles from JSON: " << e.what() << std::endl;
    return {};
  }
}

struct tiling_params {
  int tile_region_width;
  int tile_region_height;
  int col_start;
  int row_start;
};

tiling_params
determine_tile_params(int width, int height, const std::string &position)
{
  int col_start = 0;
  int row_start = 0;
  int tile_region_width = width;
  int tile_region_height = height;
  if (position == "none") {
    //  Nothing to do
  } else if (position == "left") {
    tile_region_width = width / 2;
  } else if (position == "right") {
    col_start = tile_region_width / 2;
    tile_region_width = width / 2;
  } else if (position == "top") {
    tile_region_height = height / 2;
  } else if (position == "bottom") {
    tile_region_height = height / 2;
    row_start = height / 2;
  } else {
    throw std::runtime_error("Invalid tile position");
  }
  return { tile_region_width, tile_region_height, col_start, row_start };
}

std::vector<std::array<int, 4>>
determine_tiles(const addtiles_properties &properties, int image_width, int image_height)
{
  if (!properties.tiles.empty()) {
    return properties.tiles;
  }

  auto [tile_region_width, tile_region_height, col_start, row_start]
      = determine_tile_params(image_width, image_height, properties.tile_position);

  int slice_width = properties.tile_width;
  int slice_height = properties.tile_height;

  auto [x_slices, x_overlap]
      = Ax::determine_overlap(tile_region_width, slice_width, properties.tile_overlap);
  auto [y_slices, y_overlap] = Ax::determine_overlap(
      tile_region_height, slice_height, properties.tile_overlap);

  std::vector<std::array<int, 4>> boxes{ { 0, 0, image_width - 1, image_height - 1 } };
  for (auto row = 0; row != y_slices; ++row) {
    for (auto col = 0; col != x_slices; ++col) {
      int x = col * (slice_width - x_overlap);
      int y = row * (slice_height - y_overlap);
      if (x + slice_width > tile_region_width) {
        x = std::max(tile_region_width - slice_width, 0);
      }
      if (y + slice_height > tile_region_height) {
        y = std::max(tile_region_height - slice_height, 0);
      }
      auto box = std::array<int, 4>{
        x + col_start,
        y + row_start,
        //  Remember this is a fully closed range
        std::min(static_cast<int>(x + slice_width) + col_start - 1,
            col_start + tile_region_width - 1),
        std::min(static_cast<int>(y + slice_height) + row_start - 1,
            row_start + tile_region_height - 1),
      };
      boxes.push_back(box);
    }
  }
  return boxes;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<addtiles_properties>();
  prop->meta_key = Ax::get_property(
      input, "meta_key", "addtiles_static_properties", prop->meta_key);
  prop->tile_size = Ax::get_property(
      input, "tile_size", "addtiles_static_properties", prop->tile_size);
  prop->tile_overlap = Ax::get_property(
      input, "tile_overlap", "addtiles_static_properties", prop->tile_overlap);
  prop->tile_position = Ax::get_property(
      input, "tile_position", "addtiles_static_properties", prop->tile_position);
  prop->model_width = Ax::get_property(
      input, "model_width", "addtiles_static_properties", prop->model_width);
  prop->model_height = Ax::get_property(
      input, "model_height", "addtiles_static_properties", prop->model_height);
  prop->json_file = Ax::get_property(
      input, "tile_json", "addtiles_static_properties", prop->json_file);
  if (!prop->json_file.empty()) {
    prop->tiles = load_tiles_from_json(prop->json_file, logger);
    //  This simply sets last_modified to the current modification time
    prop->last_modified = get_last_modified_time(prop->json_file);
    if (prop->tiles.empty()) {
      throw std::runtime_error("Failed to load tiles from JSON: " + prop->json_file);
    }
  } else {
    determine_slice_sizes(*prop, logger);
  }
  return prop;
}

extern "C" void
inplace(const AxDataInterface &interface, const addtiles_properties *details,
    unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  if (details->meta_key.empty()) {
    logger(AX_ERROR) << "inplace_addtiles: meta_key is empty" << std::endl;
    throw std::runtime_error("inplace_addtiles: meta_key is empty");
  }
  if (map.count(details->meta_key)) {
    logger(AX_ERROR) << "inplace_addtiles: meta_key (" << details->meta_key
                     << ") already exists" << std::endl;
    throw std::runtime_error("inplace_addtiles: meta_key already exists");
  }

  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error("addtiles works on video input only");
  }

  auto last_modified = get_last_modified_time(details->json_file);
  if (last_modified != details->last_modified) {
    details->last_modified = last_modified;
    auto new_tiles = load_tiles_from_json(details->json_file, logger);
    if (new_tiles.empty()) {
      logger(AX_ERROR) << "inplace_addtiles: Failed to reload tiles from JSON: "
                       << details->json_file << std::endl;
    } else {
      logger(AX_INFO)
          << "inplace_addtiles: Reloaded tiles from JSON: " << details->json_file
          << std::endl;
      details->tiles = new_tiles;
    }
  }
  auto &video = std::get<AxVideoInterface>(interface);
  auto tiles = determine_tiles(*details, video.info.width, video.info.height);
  if (!validate_tiles(tiles, video.info.width, video.info.height)) {
    logger(AX_ERROR) << "inplace_addtiles: Invalid tiles for image size "
                     << video.info.width << "x" << video.info.height << std::endl;
    throw std::runtime_error("inplace_addtiles: Invalid tiles for image size");
  }
  std::vector<box_xyxy> boxes(tiles.size());
  std::transform(tiles.begin(), tiles.end(), boxes.begin(), [](const auto &box) {
    return box_xyxy{ .x1 = box[0], .y1 = box[1], .x2 = box[2], .y2 = box[3] };
  });

  std::vector<float> scores(boxes.size(), 1.0);
  std::vector<int> class_ids(boxes.size(), -1);
  ax_utils::insert_meta<AxMetaObjDetectionTiles>(map, details->meta_key, "", 0,
      1, std::move(boxes), std::move(scores), std::move(class_ids));
}
