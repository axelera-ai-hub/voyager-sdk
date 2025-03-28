#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxOpUtils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <vector>


namespace rtmdet
{

using lookups = std::array<float, 256>;
struct properties {
  std::string meta_name{};
  float confidence{ 0.18F };
  std::vector<float> zero_points;
  std::vector<float> scales;
  int model_width{ 320 };
  int model_height{ 320 };
  std::vector<lookups> sigmoid_tables{};
};

float
sigmoid(int value, const float *lookups)
{
  int index = value + 128;
  return lookups[index];
}

AxTensorsInterface
icdf_tensors(const AxTensorsInterface &tensors)
{
  auto width = 10;
  auto height = 10;
  auto channels = 4;
  const auto total_size = 268800;
  if (tensors[0].total() != total_size) {
    char buf[100];
    std::sprintf(buf, "Invalid tensor size: %lu", tensors[0].total());
    throw std::runtime_error(buf);
  }

  auto *data = static_cast<int8_t *>(tensors[0].data);

  auto p0 = AxTensorInterface{ { 1, height * 4, width * 4, channels },
    tensors[0].bytes, data };
  auto offset = height * 4 * width * 4 * channels * 16;
  auto p1 = AxTensorInterface{ { 1, height * 2, width * 2, channels },
    tensors[0].bytes, data + offset };
  offset += (height * 2 * width * 2 * channels * 16);
  auto p2 = AxTensorInterface{ { 1, height, width, channels }, tensors[0].bytes,
    data + offset };
  offset += (height * width * channels * 16);

  auto c0 = AxTensorInterface{ { 1, height * 4, width * 4, 1 },
    tensors[0].bytes, data + offset };
  offset += (height * 4 * width * 4 * channels * 16);
  auto c1 = AxTensorInterface{ { 1, height * 2, width * 2, 1 },
    tensors[0].bytes, data + offset };
  offset += (height * 2 * width * 2 * channels * 16);
  auto c2 = AxTensorInterface{ { 1, height, width, 1 }, tensors[0].bytes, data + offset };

  return { p0, p1, p2, c0, c1, c2 };
}
} // namespace rtmdet

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<rtmdet::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "rtmdet_decode_static_properties", props->meta_name);
  props->zero_points = Ax::get_property(input, "zero_points",
      "rtmdet_decode_static_properties", props->zero_points);
  props->scales = Ax::get_property(
      input, "scales", "rtmdet_decode_static_properties", props->scales);
  props->model_width = Ax::get_property(input, "model_width",
      "rtmdet_decode_static_properties", props->model_width);
  props->model_height = Ax::get_property(input, "model_height",
      "rtmdet_decode_static_properties", props->model_height);

  if (props->zero_points.size() != props->scales.size()) {
    logger(AX_ERROR) << "rtmdeti_decode_static_properties : zero_points and scales must be the same "
                        "size"
                     << std::endl;
    throw std::runtime_error(
        "rtmdet_decode_static_properties : zero_points and scales must be the same size");
  }
  props->sigmoid_tables
      = ax_utils::build_sigmoid_tables(props->zero_points, props->scales);
  return props;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "zero_points",
    "scales",
    "confidence_threshold",
    "model_width",
    "model_height",
  };
  return allowed_properties;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    rtmdet::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const rtmdet::properties *prop,
    unsigned int subframe_index, unsigned int subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{

  constexpr int confidence_tensor_index_offset = 3;
  constexpr int aipu_pixel_padding = 64;
  AxTensorsInterface tensors;
  std::vector<BboxXyxy> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  if (in_tensors.size() == 1) {
    tensors = rtmdet::icdf_tensors(in_tensors);
  } else {
    tensors = in_tensors;
  }
  auto video_info = std::get<AxVideoInterface>(video_interface).info;

  for (auto i = 0; i < tensors.size() / 2; ++i) {
    auto scale = tensors[i].sizes.at(1);
    auto scale_factor = prop->model_width / scale;
    for (auto y = 0; y < tensors[i].sizes.at(1); ++y) {
      for (auto x = 0; x < tensors[i].sizes.at(2); ++x) {
        auto offset = aipu_pixel_padding * ((y * scale) + x);
        auto *deconf_ptr
            = static_cast<int8_t *>(tensors[i + confidence_tensor_index_offset].data) + offset;
        float conf = rtmdet::sigmoid(*deconf_ptr,
            prop->sigmoid_tables[i + confidence_tensor_index_offset].data());
        if (conf > prop->confidence) {
          auto *gridcell = static_cast<int8_t *>(tensors[i].data) + offset;
          BboxXyxy box;
          box.x1 = (scale_factor
                       * (x
                           - ax_utils::dequantize(gridcell[0],
                               prop->scales.at(i), prop->zero_points.at(i))))
                   * (static_cast<float>(video_info.width) / prop->model_width);
          box.y1 = (scale_factor
                       * (y
                           - ax_utils::dequantize(gridcell[1],
                               prop->scales.at(i), prop->zero_points.at(i))))
                   * (static_cast<float>(video_info.height) / prop->model_height);
          box.x2 = (scale_factor
                       * (x
                           + ax_utils::dequantize(gridcell[2],
                               prop->scales.at(i), prop->zero_points.at(i))))
                   * (static_cast<float>(video_info.width) / prop->model_width);
          box.y2 = (scale_factor
                       * (y
                           + ax_utils::dequantize(gridcell[3],
                               prop->scales.at(i), prop->zero_points.at(i))))
                   * (static_cast<float>(video_info.height) / prop->model_height);
          boxes.push_back(std::move(box));
          class_ids.push_back(0);
          scores.push_back(conf);
        }
      }
    }
  }

  map[prop->meta_name] = std::make_unique<AxMetaObjDetection>(
      std::move(boxes), std::move(scores), std::move(class_ids));
}
