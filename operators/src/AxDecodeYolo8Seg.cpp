// Copyright Axelera AI, 2024
// Optimized anchor-free YOLO(v8) decoder

#include <opencv2/opencv.hpp>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaSegmentsDetection.hpp"
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

namespace yolov8seg_decode
{
using lookups = std::array<float, 256>;
using inferences = ax_utils::inferences;
constexpr auto weights_size = 16;
struct properties {
  std::vector<lookups> sigmoid_tables{};
  std::vector<lookups> softmax_tables{};
  std::vector<lookups> dequantize_tables{};
  std::vector<std::vector<int>> padding{};
  std::vector<float> zero_points{};
  std::vector<float> scales{};
  std::vector<std::string> class_labels{};
  std::vector<int> filter{};
  std::array<float, weights_size> weights;

  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 2000 };
  bool multiclass{ true };
  std::string meta_name{};
  std::string decoder_name{};
  int model_width{};
  int model_height{};
  std::string master_meta{};
  bool letterbox{ true };
};


struct tensor_pair {
  int conf_idx;
  int box_idx;
  int mask_idx;
  int prototype_idx;
};

void
build_prototype_vector(const AxTensorsInterface &depadded, int prototype_stride,
    const std::vector<tensor_pair> &tensor_order, const properties &props, inferences &outputs)
{

  const auto [unused1, unused2, unused3, idx] = tensor_order[0];
  auto [prototype_width, prototype_height, prototype_depth]
      = ax_utils::get_dims(depadded, idx, true);
  const auto prototype_tensor = depadded[idx];
  const auto *prototype_data = static_cast<const int8_t *>(prototype_tensor.data);

  const auto &prototype_lookups = props.dequantize_tables[idx].data();

  // Dequantize and remove padding of prototype maps
  const auto prototype_size = prototype_width * prototype_height * prototype_depth;
  const auto prototype_tensor_size = prototype_width * prototype_height * prototype_stride;
  outputs.prototype_coefs.resize(prototype_size);
  auto it = outputs.prototype_coefs.begin();

  for (auto i = 0; i < prototype_tensor_size; i += prototype_stride) {
    std::transform(prototype_data + i, prototype_data + i + prototype_depth, it,
        [&prototype_lookups](
            int8_t index) { return prototype_lookups[128 + index]; });
    it = std::next(it, prototype_depth);
  }
  outputs.set_prototype_dims(prototype_width, prototype_height, prototype_depth);
}

/// @brief Sort the tensors into the order that they are expected to be in and
///        pairs the box prediction with the correspong confidence predictions.
///        We determine which is which from the channel size.
/// @param tensors - The tensors to sort
/// @param num_classes - The number of classes
/// @param logger - The logger to use for logging
/// @return The sorted tensors
std::vector<tensor_pair>
sort_tensors(const AxTensorsInterface &tensors, int num_classes, Ax::Logger &logger)
{
  if ((tensors.size() - 1) % 3 != 0) {
    logger(AX_ERROR) << "The number of tensors - 1 must be multiple of 3" << std::endl;
    throw std::runtime_error("The number of tensors - 1 must be multiple of 3");
  }
  std::vector<int> indices(tensors.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  std::sort(std::begin(indices), std::end(indices), [&tensors](auto a, auto b) {
    const int width_or_height_idx = 2;
    const int channels_idx = 3;
    return std::tie(tensors[a].sizes[width_or_height_idx], tensors[a].sizes[channels_idx])
           > std::tie(tensors[b].sizes[width_or_height_idx], tensors[b].sizes[channels_idx]);
  });

  std::vector<tensor_pair> tensor_pairs;

  for (auto i = size_t{ 1 }; i < indices.size(); i += 3) {
    if (num_classes < 32) {
      tensor_pairs.push_back(
          tensor_pair{ indices[i + 2], indices[i], indices[i + 1], indices[0] });
    } else if (num_classes >= 32 && num_classes <= 64) {
      tensor_pairs.push_back(
          tensor_pair{ indices[i + 1], indices[i], indices[i + 2], indices[0] });
    } else {
      tensor_pairs.push_back(
          tensor_pair{ indices[i], indices[i + 1], indices[i + 2], indices[0] });
    }
  }
  return tensor_pairs;
}

/// @brief Decode a single cell of the tensor
/// @param box_data - pointer to the raw tensor box data
/// @param score_data - pointer to the raw tensor score
/// @param mask_data - pointer to the raw tensor mask coefs
/// @param props - properties of the model
/// @param score_level - index of score tensor
/// @param box_level - index of box tensor
/// @param mask_level - index of mask tensor
/// @param recip_width - scale facror
/// @param xpos - x position of the cell
/// @param ypos - y position of the cell
/// @param outputs - output inferences
/// @return - The number of predictions added
int
decode_cell(const int8_t *box_data, const int8_t *score_data,
    const int8_t *mask_data, const properties &props, int score_level, int box_level,
    int mask_level, float recip_width, int xpos, int ypos, inferences &outputs)
{

  const auto dummy = float{};
  const auto &lookups = props.sigmoid_tables[score_level].data();
  const auto confidence = props.confidence;
  const auto num_predictions = ax_utils::decode_scores(score_data, lookups, 1,
      props.filter, props.confidence, props.multiclass, outputs);
  if (num_predictions != 0) {
    const auto &softmax_lookups = props.softmax_tables[box_level].data();
    //  Here we need to decode the box
    constexpr auto box_size = 4;
    std::array<float, box_size> box;
    std::array<float, weights_size> softmaxed;
    auto *box_ptr = box_data;
    auto next_box = softmaxed.size();
    for (auto &b : box) {
      ax_utils::softmax(box_ptr, softmaxed.size(), 1, softmax_lookups, softmaxed.data());
      b = std::transform_reduce(
          props.weights.begin(), props.weights.end(), softmaxed.begin(), 0.0F);
      box_ptr = std::next(box_ptr, next_box);
    }

    const auto x1 = (xpos + 0.5F - box[0]) * recip_width;
    const auto y1 = (ypos + 0.5F - box[1]) * recip_width;
    const auto x2 = (xpos + 0.5F + box[2]) * recip_width;
    const auto y2 = (ypos + 0.5F + box[3]) * recip_width;

    outputs.boxes.insert(outputs.boxes.end(), num_predictions,
        {
            std::clamp(x1, 0.0F, 1.0F),
            std::clamp(y1, 0.0F, 1.0F),
            std::clamp(x2, 0.0F, 1.0F),
            std::clamp(y2, 0.0F, 1.0F),
        });

    const auto &mask_lookups = props.dequantize_tables[mask_level].data();
    std::vector<float> mask_coefs(outputs.prototype_depth);
    std::transform(mask_data, mask_data + outputs.prototype_depth, mask_coefs.begin(),
        [&mask_lookups](int8_t index) { return mask_lookups[128 + index]; });

    segment_func task([x1, y1, x2, y2, &props, prototype_width = outputs.prototype_width,
                          prototype_height = outputs.prototype_height,
                          prototype_depth = outputs.prototype_depth,
                          mask_coefs](const std::vector<float> &prototype_coefs,
                          size_t out_width, size_t out_height) {
      if (prototype_coefs.empty()) {
        throw std::runtime_error("invalid prototype tensor");
      }

      // TODO: Use OpenCL kernel for MVM and sigmoid
      auto ratio = static_cast<float>(out_width) / out_height;
      auto xmin{ 0 }, xmax{ prototype_width }, ymin{ 0 }, ymax{ prototype_height };

      // Bounding boxex coordinates cannot be in the area of letterboxing/pillarboxing, if we get bbox that in this are
      // we clamping the coordinates to visible area of the image
      if (ratio > 1.0F) {
        ymin = std::round(prototype_height * (ratio - 1.0) / (2.0 * ratio));
        ymax = prototype_height - ymin;
      } else if (ratio < 1.0F) {
        xmin = std::round(prototype_width * (1.0 - ratio) / 2.0);
        xmax = prototype_width - xmin;
      }

      auto bbox = std::array{
        std::clamp(static_cast<int>(std::round(x1 * prototype_width)), xmin, xmax),
        std::clamp(static_cast<int>(std::round(y1 * prototype_height)), ymin, ymax),
        std::clamp(static_cast<int>(std::round(x2 * prototype_width)), xmin, xmax),
        std::clamp(static_cast<int>(std::round(y2 * prototype_height)), ymin, ymax)
      };

      const auto row_offset = prototype_width * prototype_depth;
      const auto segment_map_size = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]);
      std::vector<float> segment_map(segment_map_size);
      auto idx = 0;
      for (int sy = bbox[1]; sy < bbox[3]; ++sy) {
        for (int sx = bbox[0]; sx < bbox[2]; ++sx) {
          auto prototype = prototype_coefs.begin() + sy * row_offset + sx * prototype_depth;
          auto dot = std::transform_reduce(mask_coefs.begin(), mask_coefs.end(),
              prototype, 0.0F, std::plus<>(), std::multiplies<>());
          segment_map[idx++] = ax_utils::sigmoid(dot, 0);
        }
      }
      return ax_utils::segment{ bbox[0], bbox[1], bbox[2], bbox[3], std::move(segment_map) };
    });

    outputs.seg_funcs.insert(outputs.seg_funcs.end(), num_predictions, task);
  }

  return num_predictions;
}

///
/// @brief Decode a single feature map tensor
/// @param tensors - The tensor data
/// @param score_idx - The index of the score tensor
/// @param box_idx - The index of the box tensor
/// @param mask_idx - The index of the mask tensor
/// @param props - The properties of the model
/// @param level - which of features maps this tensor is
/// @param outputs - output inferences
/// @param logger - The logger to use for logging
/// @return - The number of predictions added
int
decode_tensor(const AxTensorsInterface &tensors, int score_idx, int box_idx, int mask_idx,
    const properties &props, int level, inferences &outputs, Ax::Logger &logger)
{
  auto [box_width, box_height, box_depth] = ax_utils::get_dims(tensors, box_idx, true);
  auto [score_width, score_height, score_depth]
      = ax_utils::get_dims(tensors, score_idx, true);
  auto [mask_width, mask_height, mask_depth] = ax_utils::get_dims(tensors, mask_idx, true);

  if (box_width != score_width || box_height != score_height) {
    logger(AX_ERROR) << "decode_tensor : box and score tensors must be the same size"
                     << std::endl;
    return 0;
  }
  const auto box_x_stride = box_depth;
  const auto box_y_stride = box_x_stride * box_width;

  const auto score_x_stride = score_depth;
  const auto score_y_stride = score_x_stride * score_width;

  const auto mask_x_stride = mask_depth;
  const auto mask_y_stride = mask_x_stride * mask_width;

  const auto box_tensor = tensors[box_idx];
  const auto score_tensor = tensors[score_idx];
  const auto mask_tensor = tensors[mask_idx];
  const auto *box_data = static_cast<const int8_t *>(box_tensor.data);
  const auto *score_data = static_cast<const int8_t *>(score_tensor.data);
  const auto *mask_data = static_cast<const int8_t *>(mask_tensor.data);

  auto total = 0;
  const auto recip_width = 1.0F / std::max(box_width, box_height);


  for (auto y = 0; y != box_height; ++y) {
    auto *box_ptr = std::next(box_data, box_y_stride * y);
    auto *score_ptr = std::next(score_data, score_y_stride * y);
    auto *mask_ptr = std::next(mask_data, mask_y_stride * y);

    for (auto x = 0; x != box_width; ++x) {
      total += decode_cell(box_ptr, score_ptr, mask_ptr, props, score_idx,
          box_idx, mask_idx, recip_width, x, y, outputs);
      box_ptr = std::next(box_ptr, box_x_stride);
      score_ptr = std::next(score_ptr, score_x_stride);
      mask_ptr = std::next(mask_ptr, mask_x_stride);
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
/// @param pheightg - The prototype tensor height
/// @param logger - The logger to use for logging
/// @return The resulting inferences

inferences
decode_tensors(const AxTensorsInterface &tensors, const properties &prop,
    const std::vector<std::vector<int>> &padding, Ax::Logger &logger)

{
  constexpr int aipu_allignment = 64;
  auto depadded = depad_tensors(tensors, padding);
  auto tensor_order = sort_tensors(depadded, prop.num_classes, logger);


  inferences predictions(1000);
  auto idx = tensor_order[0].prototype_idx;
  auto prototype_stride = ax_utils::get_dims(tensors, idx, true).depth;
  yolov8seg_decode::build_prototype_vector(
      depadded, prototype_stride, tensor_order, prop, predictions);
  for (int level = 0; level != tensor_order.size(); ++level) {
    const auto [conf_tensor, loc_tensor, mask_tensor, _] = tensor_order[level];
    auto num = decode_tensor(tensors, conf_tensor, loc_tensor, mask_tensor,
        prop, level, predictions, logger);
  }
  return predictions;
}

} // namespace yolov8seg_decode

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors,
    const yolov8seg_decode::properties *prop, unsigned int subframe_index,
    unsigned int number_of_subframes,
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
  if (tensors.size() != prop->sigmoid_tables.size() && tensors[0].bytes == 1) {
    throw std::runtime_error(
        "ssd_decode_to_meta : Number of input tensors or dequantize parameters is incorrect");
  }
  auto predictions = yolov8seg_decode::decode_tensors(tensors, *prop, padding, logger);
  predictions = ax_utils::topk(predictions, prop->topk);

  std::vector<BboxXyxy> pixel_boxes;
  size_t width{}, height{};
  // std::vector<int> base_box;
  BboxXyxy base_box;
  if (prop->master_meta.empty()) {
    auto vinfo = std::get<AxVideoInterface>(video_interface);
    pixel_boxes = ax_utils::scale_boxes(predictions.boxes, vinfo,
        prop->model_width, prop->model_height, true, prop->letterbox);
    base_box = { 0, 0, vinfo.info.width, vinfo.info.height };
  } else {
    auto master_meta
        = ax_utils::get_meta<AxMetaBbox>(prop->master_meta, map, "yolov8seg_decode");
    auto master_box = master_meta->get_box_xyxy(subframe_index);
    pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, master_box,
        prop->model_width, prop->model_height, true, prop->letterbox);
    base_box = { master_box.x1, master_box.y1, master_box.x2, master_box.y2 };
  }

  auto sizes = SegmentShape{ static_cast<size_t>(predictions.prototype_width),
    static_cast<size_t>(predictions.prototype_height) };

  ax_utils::insert_meta<AxMetaSegmentsDetection>(map, prop->meta_name,
      prop->master_meta, subframe_index, number_of_subframes,
      std::move(pixel_boxes), std::move(predictions.seg_funcs),
      std::move(predictions.scores), std::move(predictions.class_ids), sizes,
      std::move(predictions.prototype_coefs), std::move(base_box), prop->decoder_name);

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
    "max_boxes",
    "label_filter",
    "topk",
    "zero_points",
    "scales",
    "multiclass",
    "classes",
    "padding",
    "decoder_name",
    "model_width",
    "model_height",
    "master_meta",
    "letterbox",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<yolov8seg_decode::properties>();
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
  props->master_meta = Ax::get_property(
      input, "master_meta", "detection_static_properties", props->master_meta);
  auto topk
      = Ax::get_property(input, "topk", "detection_static_properties", props->topk);
  if (topk > 0) {
    props->topk = topk;
  }
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

  props->multiclass = Ax::get_property(
      input, "multiclass", "detection_static_properties", props->multiclass);

  //  Build the lookup tables
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
        "yolo_decode_static_properties", logger);
  }

  props->sigmoid_tables
      = ax_utils::build_sigmoid_tables(props->zero_points, props->scales);
  props->softmax_tables
      = ax_utils::build_exponential_tables(props->zero_points, props->scales);
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
  std::iota(props->weights.begin(), props->weights.end(), 0);
  props->letterbox = Ax::get_property(
      input, "letterbox", "detection_static_properties", props->letterbox);
  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolov8seg_decode::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}
