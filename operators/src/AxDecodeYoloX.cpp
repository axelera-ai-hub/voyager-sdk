// Copyright Axelera AI, 2024
// Optimized anchor-free YOLOX decoder

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <vector>

#include <chrono>
#include <thread>

namespace yolox_decode
{
using lookups = std::array<float, 256>;
using inferences = ax_utils::inferences;

struct properties {
  std::vector<lookups> dequantize_tables{};
  std::vector<lookups> exponential_tables{};
  std::vector<std::vector<int>> padding{};
  std::vector<float> zero_points{};
  std::vector<float> scales{};
  std::vector<std::string> class_labels{};
  std::vector<int> filter{};

  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 2000 };
  bool multiclass{ true };
  std::string meta_name{};
  std::string decoder_name{};
  bool scale_up{ true };
  bool letterbox{ true };
  int model_width{};
  int model_height{};
};


/// @brief Sort the tensors into the order that they are expected to be in and
///        pairs the box prediction with the correspong confidence predictions.
///        We determine which is which from the channel size.
/// @param tensors - The tensors to sort
/// @param num_classes - The number of classes
/// @param kpts_per_box - The number of kpts per box
/// @param logger - The logger to use for logging
/// @return The sorted tensors

struct tensor_pair {
  int conf_idx;
  int box_idx;
  int objectness_idx;
};

float
lut(int8_t value, const float *the_table, int offset = 128)
{
  int index = value + offset;
  return the_table[index];
}

std::vector<tensor_pair>
sort_tensors(const AxTensorsInterface &tensors, int num_classes, Ax::Logger &logger)
{
  if (tensors.size() % 3 != 0) {
    logger(AX_ERROR) << "The number of tensors must be multiple of 3" << std::endl;
    throw std::runtime_error("The number of tensors must be multiple of 3");
  }
  std::vector<int> indices(tensors.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  std::sort(std::begin(indices), std::end(indices), [&tensors](auto a, auto b) {
    const int width_or_height_idx = 2;
    const int channels_idx = 3;
    return std::tie(tensors[a].sizes[width_or_height_idx], tensors[a].sizes[channels_idx])
           > std::tie(tensors[b].sizes[width_or_height_idx], tensors[b].sizes[channels_idx]);
  });

  auto swap = num_classes < 4;
  std::vector<tensor_pair> tensor_pairs;
  for (auto i = size_t{}; i != indices.size(); i += 3) {
    (swap) ?
        tensor_pairs.push_back(tensor_pair{ indices[i + 1], indices[i], indices[i + 2] }) :
        tensor_pairs.push_back(tensor_pair{ indices[i], indices[i + 1], indices[i + 2] });
  }


  return tensor_pairs;
}
// TODO: doxygen
/// @brief Decode a single cell of the tensor
/// @param box_data - pointer to the raw tensor box data
/// @param score_data - pointer to the raw tensor score
/// @param props - properties of the model
/// @param score_level - index of score tensor
/// @param box_level - index of box tensor
/// @param recip_width - scale facror
/// @param xpos - x position of the cell
/// @param ypos - y position of the cell
/// @param outputs - output inferences
/// @return - The number of predictions added
int
decode_cell(const int8_t *box_data, const int8_t *score_data, const int8_t *objectness_data,
    const properties &props, int score_level, int box_level, int objectness_level,
    float recip_width, int xpos, int ypos, inferences &outputs)
{
  const auto &objectness_lookups = props.dequantize_tables[objectness_level].data();
  const auto objectness = yolox_decode::lut(objectness_data[0], objectness_lookups);

  if (objectness < props.confidence) {
    return 0;
  }
  const auto &score_lookups = props.dequantize_tables[score_level].data();
  const auto num_predictions = ax_utils::decode_scores(score_data, score_lookups,
      1, props.filter, props.confidence, props.multiclass, outputs, objectness);

  if (num_predictions != 0) {
    const auto &box_lookups = props.dequantize_tables[box_level].data();
    const auto &exponential_lookups = props.exponential_tables[box_level].data();
    auto *box_ptr = box_data;

    const auto cx = recip_width * (yolox_decode::lut(box_ptr[0], box_lookups) + xpos);
    const auto cy = recip_width * (yolox_decode::lut(box_ptr[1], box_lookups) + ypos);

    auto half_width
        = recip_width * yolox_decode::lut(box_ptr[2], exponential_lookups) / 2;
    auto half_height
        = recip_width * yolox_decode::lut(box_ptr[3], exponential_lookups) / 2;

    outputs.boxes.insert(outputs.boxes.end(), num_predictions,
        {
            std::clamp(cx - half_width, 0.0F, 1.0F),
            std::clamp(cy - half_height, 0.0F, 1.0F),
            std::clamp(cx + half_width, 0.0F, 1.0F),
            std::clamp(cy + half_height, 0.0F, 1.0F),
        });
  }

  return num_predictions;
}

///
/// @brief Decode a single feature map tensor
/// @param tensors - The tensor data
/// @param score_idx - The index of the score tensor
/// @param box_idx - The index of the box tensor
/// @param props - The properties of the model
/// @param level - which of features maps this tensor is
/// @param outputs - output inferences
/// @param logger - The logger to use for logging
/// @return - The number of predictions added
int
decode_tensor(const AxTensorsInterface &tensors, int score_idx, int box_idx, int objectness_idx,
    const properties &props, int level, inferences &outputs, Ax::Logger &logger)
{
  auto [box_width, box_height, box_depth] = ax_utils::get_dims(tensors, box_idx, true);
  auto [score_width, score_height, score_depth]
      = ax_utils::get_dims(tensors, score_idx, true);
  auto [objectness_width, objectness_height, objectness_depth]
      = ax_utils::get_dims(tensors, objectness_idx, true);

  if (box_width != score_width || box_height != score_height) {
    logger(AX_ERROR) << "decode_tensor : box and score tensors must be the same size"
                     << std::endl;
    return 0;
  }
  const auto box_x_stride = box_depth;
  const auto box_y_stride = box_x_stride * box_width;

  const auto score_x_stride = score_depth;
  const auto score_y_stride = score_x_stride * score_width;

  const auto objectness_x_stride = objectness_depth;
  const auto objectness_y_stride = objectness_x_stride * objectness_width;

  const auto box_tensor = tensors[box_idx];
  const auto score_tensor = tensors[score_idx];
  const auto objectness_tensor = tensors[objectness_idx];
  const auto *box_data = static_cast<const int8_t *>(box_tensor.data);
  const auto *score_data = static_cast<const int8_t *>(score_tensor.data);
  const auto *objectness_data = static_cast<const int8_t *>(objectness_tensor.data);

  auto total = 0;
  const auto recip_width = 1.0F / std::max(box_width, box_height);

  for (auto y = 0; y != box_height; ++y) {
    auto *box_ptr = std::next(box_data, box_y_stride * y);
    auto *score_ptr = std::next(score_data, score_y_stride * y);
    auto *objectness_ptr = std::next(objectness_data, objectness_y_stride * y);

    for (auto x = 0; x != box_width; ++x) {
      total += decode_cell(box_ptr, score_ptr, objectness_ptr, props, score_idx,
          box_idx, objectness_idx, recip_width, x, y, outputs);
      box_ptr = std::next(box_ptr, box_x_stride);
      score_ptr = std::next(score_ptr, score_x_stride);
      objectness_ptr = std::next(objectness_ptr, objectness_x_stride);
    }
  }
  return total;
}

AxTensorsInterface
depad_tensors(const AxTensorsInterface &tensors, const std::vector<std::vector<int>> &padding)
{
  auto depadded = tensors;
  for (auto i = 0; i != padding.size(); ++i) {
    depadded[i].sizes[3] = tensors[i].sizes[3] - padding[i][7] - padding[i][6];
  }
  return depadded;
}

/// @brief Decode the tensors into a set of inferences
/// @param tensors - The input tensors
/// @param prop - The properties of the model
/// @param padding - The padding for each tensor
/// @param logger - The logger to use for logging
/// @return The resulting inferences

inferences
decode_tensors(const AxTensorsInterface &tensors, const properties &prop,
    const std::vector<std::vector<int>> &padding, Ax::Logger &logger)
{
  auto depadded = depad_tensors(tensors, padding);
  auto tensor_order = sort_tensors(depadded, prop.num_classes, logger);

  inferences predictions(1000);
  predictions.kpts_shape = { 0, 0 };
  for (int level = 0; level != tensor_order.size(); ++level) {
    const auto [conf_tensor, loc_tensor, objectness_tensor] = tensor_order[level];
    auto num = decode_tensor(tensors, conf_tensor, loc_tensor,
        objectness_tensor, prop, level, predictions, logger);
  }
  return predictions;
}

} // namespace yolox_decode

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors,
    const yolox_decode::properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  if (!prop) {
    logger(AX_ERROR) << "decode_to_meta : properties not set" << std::endl;
    throw std::runtime_error("decode_to_meta : properties not set");
  }
  auto tensors = in_tensors;
  auto padding = prop->padding;

  auto predictions = yolox_decode::decode_tensors(tensors, *prop, padding, logger);
  predictions = ax_utils::topk(predictions, prop->topk);

  auto boxes = ax_utils::scale_boxes(predictions.boxes,
      std::get<AxVideoInterface>(video_interface), prop->model_width,
      prop->model_height, prop->scale_up, prop->letterbox);

  map[prop->meta_name] = std::make_unique<AxMetaObjDetection>(std::move(boxes),
      std::move(predictions.scores), std::move(predictions.class_ids));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_INFO) << "decode_to_meta : Decoding took " << duration.count()
                  << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "classlabels_file",
    "confidence_threshold",
    "label_filter",
    "topk",
    "zero_points",
    "scales",
    "multiclass",
    "classes",
    "padding",
    "decoder_name",
    "scale_up",
    "letterbox",
    "model_width",
    "model_height",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<yolox_decode::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "detection_static_properties", props->meta_name);
  props->zero_points = Ax::get_property(
      input, "zero_points", "detection_static_properties", props->zero_points);
  props->scales = Ax::get_property<float>(
      input, "scales", "detection_static_properties", props->scales);
  props->num_classes = Ax::get_property(
      input, "classes", "detection_static_properties", props->num_classes);
  props->decoder_name = Ax::get_property(
      input, "decoder_name", "detection_static_properties", props->decoder_name);
  auto topk
      = Ax::get_property(input, "topk", "detection_static_properties", props->topk);
  if (topk > 0) {
    props->topk = topk;
  }
  props->scale_up = Ax::get_property(
      input, "scale_up", "detection_static_properties", props->scale_up);

  props->model_width = Ax::get_property(
      input, "model_width", "detection_static_properties", props->model_width);
  props->model_height = Ax::get_property(
      input, "model_height", "detection_static_properties", props->model_height);
  if (props->model_height == 0 || props->model_width == 0) {
    logger(AX_ERROR) << "detection_static_properties : model_width and model_height must be "
                        "provided"
                     << std::endl;
    throw std::runtime_error(
        "detection_static_properties : model_width and model_height must be provided");
  }

  auto filename = Ax::get_property(
      input, "classlabels_file", "yolo_decode_static_properties", std::string{});
  if (!filename.empty()) {
    props->class_labels = ax_utils::read_class_labels(
        filename, "yolox_decode_static_properties", logger);
  }
  if (props->num_classes == 0) {
    props->num_classes = props->class_labels.size();
  }

  props->multiclass = Ax::get_property(
      input, "multiclass", "detection_static_properties", props->multiclass);

  if (props->zero_points.size() != props->scales.size()) {
    logger(AX_ERROR) << "detection_static_properties : zero_points and scales must have the same "
                        "number of elements."
                     << std::endl;
    throw std::runtime_error(
        "detection_static_properties : zero_points and scales must be the same size");
  }

  if (props->num_classes == 0) {
    if (!props->class_labels.empty()) {
      props->num_classes = props->class_labels.size();
    }
  }
  if (props->num_classes != 0) {
    ax_utils::validate_classes(props->class_labels, props->num_classes,
        "yolox_decode_static_properties", logger);
  }

  props->exponential_tables = ax_utils::build_exponential_tables_with_zero_point(
      props->zero_points, props->scales);
  props->dequantize_tables
      = ax_utils::build_dequantization_tables(props->zero_points, props->scales);

  props->filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", props->filter);
  if (props->filter.empty()) {
    auto size = props->num_classes;
    props->filter.resize(size);
    std::iota(props->filter.begin(), props->filter.end(), 0);
  }
  props->padding = Ax::get_property(
      input, "padding", "detection_static_properties", props->padding);
  std::sort(props->filter.begin(), props->filter.end());
  props->filter.erase(std::unique(props->filter.begin(), props->filter.end()),
      props->filter.end());
  props->letterbox = Ax::get_property(
      input, "letterbox", "detection_static_properties", props->letterbox);
  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolox_decode::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}
