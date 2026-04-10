// Copyright Axelera AI, 2026
#include "AxRoiCropCommon.h"

#include <cmath>
#include "AxMetaMargin.hpp"
#include "AxUtils.hpp"

const std::unordered_set<std::string> &
roicrop_allowed_properties()
{
  static const std::unordered_set<std::string> props{
    "meta_key",
    "top",
    "left",
    "width",
    "height",
  };
  return props;
}

void
parse_roicrop_params(const std::unordered_map<std::string, std::string> &input,
    RoiCropParams &prop, const std::string &name, Ax::Logger &logger)
{
  prop.meta_key = Ax::get_property(input, "meta_key", name, prop.meta_key);
  prop.top = Ax::get_property(input, "top", name, prop.top);
  prop.left = Ax::get_property(input, "left", name, prop.left);
  prop.width = Ax::get_property(input, "width", name, prop.width);
  prop.height = Ax::get_property(input, "height", name, prop.height);

  if (prop.meta_key.empty()) {
    if (prop.top == -1 || prop.left == -1 || prop.width == -1 || prop.height == -1) {
      logger.throw_error(name + ": if meta_key is not provided, left, top, width and height must be provided");
    }
  } else {
    if (prop.top != -1 || prop.left != -1 || prop.width != -1 || prop.height != -1) {
      logger.throw_error(name + ": if meta_key is provided, left, top, width and height must not be provided");
    }
  }
}

float
roicrop_get_margin(std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map)
{
  auto it = meta_map.find("axelera-margin");
  if (it != meta_map.end()) {
    if (auto *p = dynamic_cast<AxMetaMargin *>(it->second.get())) {
      return p->margin;
    }
  }
  return 0.0F;
}

BboxXyxy
roicrop_get_roi_with_margin(const RoiCropParams &prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    const std::string &name, Ax::Logger &logger)
{
  if (!prop.meta_key.empty()) {
    if (number_of_subframes == 0) {
      return { 0, 0, 15, 15 };
    }
    if (meta_map.find(prop.meta_key) == meta_map.end()) {
      logger.throw_error(name + ": meta_key " + prop.meta_key + " not found in meta map");
    }
    AxMetaBbox *box_meta = dynamic_cast<AxMetaBbox *>(meta_map.at(prop.meta_key).get());
    if (!box_meta) {
      logger.throw_error(name + " has not been provided with AxMetaBbox");
    }
    if (number_of_subframes <= subframe_index) {
      logger.throw_error(name + ": subframe index must be less than number of subframes");
    }
    auto [x1, y1, x2, y2] = box_meta->get_box_xyxy(subframe_index);
    auto margin = roicrop_get_margin(meta_map);
    int x_margin = std::round((1 + x2 - x1) * margin);
    int y_margin = std::round((1 + y2 - y1) * margin);
    return { x1 - x_margin, y1 - y_margin, x2 + x_margin, y2 + y_margin };
  }
  //  Margin is not applied when the ROI is provided as static coordinates.
  return { prop.left, prop.top, prop.left + prop.width - 1, prop.top + prop.height - 1 };
}

AxDataInterface
roicrop_set_output_interface(const AxDataInterface &interface, const RoiCropParams &prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    const std::string &name, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error(name + " works on video input only");
  }
  AxDataInterface output = interface;
  auto &out_info = std::get<AxVideoInterface>(output).info;
  auto input_info = std::get<AxVideoInterface>(interface).info;

  auto [x1, y1, x2, y2] = roicrop_get_roi_with_margin(
      prop, subframe_index, number_of_subframes, meta_map, name, logger);
  if (roicrop_get_margin(meta_map) == 0.0F) {
    if (out_info.width <= x1) {
      logger.throw_error(name + ": x1 is out of bounds");
    }
    if (out_info.height <= y1) {
      logger.throw_error(name + ": y1 is out of bounds");
    }
    if (out_info.width <= x2) {
      logger(AX_WARN) << name << ": box exceeds image width, clipping to image width"
                      << std::endl;
      x2 = out_info.width - 1;
    }
    if (out_info.height <= y2) {
      logger(AX_WARN) << name << ": box exceeds image height, clipping to image height"
                      << std::endl;
      y2 = out_info.height - 1;
    }
  }

  out_info.width = 1 + x2 - x1;
  out_info.height = 1 + y2 - y1;
  if (x1 < 0 || input_info.width <= x2 || y1 < 0 || input_info.height <= y2) {
    //  If any part of the ROI is out of bounds we cannot pass crop info downstream.
    out_info.x_offset = 0;
    out_info.y_offset = 0;
    out_info.cropped = false;
    return output;
  }
  out_info.x_offset = x1;
  out_info.y_offset = y1;
  out_info.cropped = true;
  return output;
}
