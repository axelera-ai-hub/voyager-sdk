// Copyright Axelera AI, 2025
// Optimized anchor-free YOLO(v10) decoder

#include "AxDataInterface.h"
#include "AxLog.hpp"
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

namespace yolov10_decode_simple
{
using lookups = std::array<float, 256>;
using inferences = ax_utils::inferences;

constexpr auto weights_size = 16;

struct properties {
  std::vector<std::string> class_labels{};
  std::vector<int> filter{};

  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 2000 };
  std::string meta_name{};
  std::string master_meta{};
  std::string association_meta{};
  bool scale_up{ true };
  bool letterbox{ true };
  int model_width{};
  int model_height{};
};
static ax_utils::fbox
get_box(const float *data, int width, int height)
{
  // First 4 values are box coordinates (already relative to model dimensions)
  auto x1 = data[0] / width;
  auto y1 = data[1] / height;
  auto x2 = data[2] / width;
  auto y2 = data[3] / height;

  return ax_utils::fbox{ x1, y1, x2, y2 };
}
} // namespace yolov10_decode_simple

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors,
    const yolov10_decode_simple::properties *prop, unsigned int subframe_index,
    unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  ax_utils::inferences predictions(1000, 0);


  auto start_time = std::chrono::high_resolution_clock::now();
  if (!prop) {
    logger(AX_ERROR) << "decode_to_meta : properties not set" << std::endl;
    throw std::runtime_error("decode_to_meta : properties not set");
  }

  if (in_tensors.size() != 1) {
    throw std::runtime_error("decode_to_meta : Number of input tensors should be 1");
  }
  auto tensor = in_tensors[0];
  auto data = static_cast<float *>(tensor.data);
  // Process each grid cell
  const int grid_cells = tensor.sizes[2];
  const int features = tensor.sizes[3];

  for (int cell = 0; cell < grid_cells; ++cell) {
    // Point to start of current grid cell's data
    const float *cell_data = data + (features * cell);

    // Get box coordinates and process confidence scores
    auto box = yolov10_decode_simple::get_box(cell_data, prop->model_width, prop->model_height);
    cell_data += 4;
    float conf = *cell_data;
    if (conf <= prop->confidence) {
      continue;
    }
    int class_id = static_cast<int>(cell_data[1]);
    if (std::find(prop->filter.begin(), prop->filter.end(), class_id)
        == prop->filter.end()) {
      continue;
    }
    predictions.boxes.push_back(box);
    predictions.scores.push_back(*cell_data);
    predictions.class_ids.push_back(class_id);
  }

  // respect top_k
  if (prop->topk > 0) {
    predictions = ax_utils::topk(std::move(predictions), prop->topk);
  }


  auto base_box = ax_utils::get_master_box(prop->master_meta, prop->association_meta,
      video_interface, subframe_index, map, "yolov10_decode");
  auto pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, base_box,
      prop->model_width, prop->model_height, true, prop->letterbox);

  ax_utils::insert_and_associate_meta<AxMetaObjDetection>(map, prop->meta_name,
      prop->master_meta, subframe_index, number_of_subframes,
      prop->association_meta, std::move(pixel_boxes),
      std::move(predictions.scores), std::move(predictions.class_ids));

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_DEBUG) << "decode_to_meta : Decoding took " << duration.count()
                   << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "master_meta",
    "association_meta",
    "classlabels_file",
    "confidence_threshold",
    "label_filter",
    "topk",
    "classes",
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
  auto props = std::make_shared<yolov10_decode_simple::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "detection_static_properties", props->meta_name);
  props->master_meta = Ax::get_property(
      input, "master_meta", "detection_static_properties", props->master_meta);
  props->association_meta = Ax::get_property(input, "association_meta",
      "detection_static_properties", props->association_meta);
  props->num_classes = Ax::get_property(
      input, "classes", "detection_static_properties", props->num_classes);
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
        filename, "yolo_decode_static_properties", logger);
  }
  if (props->num_classes == 0) {
    props->num_classes = props->class_labels.size();
  }

  if (props->num_classes == 0) {
    if (!props->class_labels.empty()) {
      props->num_classes = props->class_labels.size();
    }
  }
  if (props->num_classes != 0) {
    ax_utils::validate_classes(props->class_labels, props->num_classes,
        "yolo_decode_static_properties", logger);
  }

  props->filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", props->filter);
  if (props->filter.empty()) {
    auto size = props->num_classes;
    props->filter.resize(size);
    std::iota(props->filter.begin(), props->filter.end(), 0);
  }
  props->letterbox = Ax::get_property(
      input, "letterbox", "detection_static_properties", props->letterbox);
  std::sort(props->filter.begin(), props->filter.end());
  props->filter.erase(std::unique(props->filter.begin(), props->filter.end()),
      props->filter.end());

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolov10_decode_simple::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}
