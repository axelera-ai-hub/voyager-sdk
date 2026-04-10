// Copyright Axelera AI, 2025
// Optimized anchor-free YOLO(v8) decoder

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <execution>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <vector>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaPoseSegmentsDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

namespace yolov8multihead_decode
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
  std::vector<uint8_t> filter{};
  std::array<float, weights_size> weights;

  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 2000 };
  bool multiclass{ true };
  std::vector<int> kpts_shape{ 0, 0 };
  std::string meta_name{};
  std::string association_meta{};
  std::string decoder_name{};
  bool scale_up{ true };
  int model_width{};
  int model_height{};
};
struct tensor_pair {
  int box_idx;
  int kpt_idx;
  int mask_idx;
  int conf_idx;
  int prototype_idx;
};

float
dequantize(int8_t value, const float *the_table)
{
  int index = value + 128;
  return the_table[index];
}

void
build_prototype_vector(const AxTensorsInterface &depadded, int prototype_stride,
    const std::vector<tensor_pair> &tensor_order, const properties &props, inferences &outputs)
{
  const auto [unused1, unused2, unused3, unused4, idx] = tensor_order[0];
  auto [prototype_width, prototype_height, prototype_depth]
      = ax_utils::get_dims(depadded, idx, true);
  const auto prototype_tensor = depadded[idx];
  const auto *prototype_data = static_cast<const int8_t *>(prototype_tensor.data);

  auto scale = props.scales[idx];
  auto zero = props.zero_points[idx];

  // Dequantize and remove padding of prototype maps
  const size_t prototype_size = prototype_width * prototype_height * prototype_depth;
  const auto prototype_tensor_size = prototype_width * prototype_height * prototype_stride;
  ax_utils::prototype_details proto_details = {
    prototype_width,
    prototype_height,
    prototype_depth,
    scale,
    zero,
    std::make_unique_for_overwrite<uint8_t[]>(prototype_size),
    prototype_size,
  };
  auto *p = proto_details.coefs.get();
  for (int i = 0; i < prototype_tensor_size; i += prototype_stride) {
    std::memcpy(p, &prototype_data[i], prototype_depth);
    p += prototype_depth;
  }
  outputs.set_prototype(std::move(proto_details));
}

/// @brief Sort the tensors into the order that they are expected to be in and
///        pairs the box prediction with the correspong confidence predictions.
///        We determine which is which from the channel size.
/// @param tensors - The tensors to sort
/// @param logger - The logger to use for logging
/// @return The sorted tensors
std::vector<tensor_pair>
sort_tensors(const AxTensorsInterface &tensors, Ax::Logger &logger)
{
  if ((tensors.size() - 1) % 4 != 0) {
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
  for (auto i = size_t{ 1 }; i < indices.size(); i += 4) {
    tensor_pairs.push_back(tensor_pair{ indices[i], indices[i + 1],
        indices[i + 2], indices[i + 3], indices[0] });
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
    const int8_t *kpts_data, const int8_t *mask_data, const properties &props,
    int score_level, int box_level, int kpt_level, int mask_level,
    float recip_width, int xpos, int ypos, inferences &outputs)
{
  const auto &lookups = props.sigmoid_tables[score_level].data();

  auto num_predictions = ax_utils::decode_scores(score_data, lookups,
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
      ax_utils::softmax(box_ptr, softmaxed.size(), softmax_lookups, softmaxed.data());
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

    const auto &dequantize_lookups = props.dequantize_tables[kpt_level].data();
    const auto &sigmoid_lookups = props.sigmoid_tables[kpt_level].data();
    auto *kpts_ptr = kpts_data;

    std::vector<ax_utils::fkpt> kpts;
    kpts.reserve(props.kpts_shape[0]);
    for (auto i = 0; i < props.kpts_shape[0]; ++i) {
      const auto x = (xpos + 2.0F * yolov8multihead_decode::dequantize(kpts_ptr[0], dequantize_lookups))
                     * recip_width;
      const auto y = (ypos + 2.0F * yolov8multihead_decode::dequantize(kpts_ptr[1], dequantize_lookups))
                     * recip_width;
      const auto v = props.kpts_shape[1] == 3 ?
                         ax_utils::sigmoid(kpts_ptr[2], sigmoid_lookups) :
                         1.0F;

      kpts.push_back({ std::clamp(x, 0.0F, 1.0F), std::clamp(y, 0.0F, 1.0F), v });
      kpts_ptr = std::next(kpts_ptr, props.kpts_shape[1]);
    }

    for (auto i = 0; i < num_predictions; ++i) {
      outputs.kpts.insert(outputs.kpts.end(), kpts.begin(),
          std::next(kpts.begin(), props.kpts_shape[0]));
    }

    const auto scale = props.scales[mask_level];
    const auto zero = props.zero_points[mask_level];

    auto seg_details = segment_details{
      x1,
      y1,
      x2,
      y2,
      scale,
      zero,
      std::vector<int8_t>(mask_data, mask_data + outputs.prototype.depth),
    };
    outputs.seg_info.insert(outputs.seg_info.end(), num_predictions, std::move(seg_details));
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
decode_tensor(const AxTensorsInterface &tensors, int score_idx, int box_idx,
    int kpt_idx, int mask_idx, const properties &props, int level,
    inferences &outputs, Ax::Logger &logger)
{
  auto [box_width, box_height, box_depth] = ax_utils::get_dims(tensors, box_idx, true);
  auto [score_width, score_height, score_depth]
      = ax_utils::get_dims(tensors, score_idx, true);

  auto [kpts_width, kpts_height, kpts_depth] = ax_utils::get_dims(tensors, kpt_idx, true);
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

  const auto kpts_x_stride = kpts_depth;
  const auto kpts_y_stride = kpts_x_stride * kpts_width;

  const auto mask_x_stride = mask_depth;
  const auto mask_y_stride = mask_x_stride * mask_width;

  const auto box_tensor = tensors[box_idx];
  const auto score_tensor = tensors[score_idx];
  const auto mask_tensor = tensors[mask_idx];
  const auto kpts_tensor = tensors[kpt_idx];
  const auto *box_data = static_cast<const int8_t *>(box_tensor.data);
  const auto *score_data = static_cast<const int8_t *>(score_tensor.data);
  const auto *kpts_data = static_cast<const int8_t *>(kpts_tensor.data);
  const auto *mask_data = static_cast<const int8_t *>(mask_tensor.data);

  auto total = 0;
  const auto recip_width = 1.0F / std::max(box_width, box_height);


  for (auto y = 0; y != box_height; ++y) {
    auto *box_ptr = std::next(box_data, box_y_stride * y);
    auto *score_ptr = std::next(score_data, score_y_stride * y);
    auto *kpts_ptr = std::next(kpts_data, kpts_y_stride * y);
    auto *mask_ptr = std::next(mask_data, mask_y_stride * y);

    for (auto x = 0; x != box_width; ++x) {
      total += decode_cell(box_ptr, score_ptr, kpts_ptr, mask_ptr, props,
          score_idx, box_idx, kpt_idx, mask_idx, recip_width, x, y, outputs);
      box_ptr = std::next(box_ptr, box_x_stride);
      score_ptr = std::next(score_ptr, score_x_stride);
      kpts_ptr = std::next(kpts_ptr, kpts_x_stride);
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
  auto depadded = depad_tensors(tensors, padding);
  auto tensor_order = sort_tensors(depadded, logger);

  constexpr int default_detections = 1000;
  inferences predictions(default_detections, default_detections * prop.kpts_shape[0]);
  predictions.kpts_shape = prop.kpts_shape;
  auto idx = tensor_order[0].prototype_idx;
  auto prototype_stride = ax_utils::get_dims(tensors, idx, true).depth;

  yolov8multihead_decode::build_prototype_vector(
      depadded, prototype_stride, tensor_order, prop, predictions);
  for (int level = 0; level != tensor_order.size(); ++level) {
    const auto [loc_tensor, kpt_tensor, mask_tensor, conf_tensor, _] = tensor_order[level];
    decode_tensor(tensors, conf_tensor, loc_tensor, kpt_tensor, mask_tensor,
        prop, level, predictions, logger);
  }
  return predictions;
}


} // namespace yolov8multihead_decode

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors,
    const yolov8multihead_decode::properties *prop, unsigned int subframe_index,
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

  if (tensors.size() == 1) {
    throw std::runtime_error(
        "decode_to_meta : Badly constructed pipeline, possible reason is setting handle_postamble or handle_all to true, please refer to docs/tutorials/application.md");
  }

  if (tensors.size() != padding.size()) {
    throw std::runtime_error(
        "decode_to_meta : number of tensors: " + std::to_string(tensors.size())
        + " and padding size " + std::to_string(padding.size()) + " do not match");
  }

  if (tensors.size() != prop->sigmoid_tables.size() && tensors[0].bytes == 1) {
    throw std::runtime_error(
        "decode_to_meta : Number of input tensors or dequantize parameters is incorrect");
  }

  auto predictions
      = yolov8multihead_decode::decode_tensors(tensors, *prop, padding, logger);

  predictions = ax_utils::topk(std::move(predictions), prop->topk);


  auto base_box = ax_utils::get_master_box(prop->meta_name, prop->association_meta,
      video_interface, subframe_index, map, "yolov8multihead_decode");
  auto pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, base_box,
      prop->model_width, prop->model_height, prop->scale_up, false);
  auto pixel_kpts = ax_utils::scale_shift_kpts(predictions.kpts, base_box,
      prop->model_width, prop->model_height, prop->scale_up, false);
  auto sizes = SegmentShape{ static_cast<size_t>(predictions.prototype.width),
    static_cast<size_t>(predictions.prototype.height) };

  std::vector<int> ids;
  ax_utils::insert_meta<AxMetaPoseSegmentsDetection>(map, prop->meta_name, "",
      subframe_index, number_of_subframes, std::move(pixel_boxes), std::move(pixel_kpts),
      std::move(predictions.seg_info), std::move(predictions.scores),
      std::move(predictions.class_ids), ids, sizes, std::move(predictions.prototype),
      prop->kpts_shape, std::move(base_box), prop->decoder_name);

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
    "kpts_shape",
    "zero_points",
    "scales",
    "multiclass",
    "classes",
    "padding",
    "decoder_name",
    "scale_up",
    "model_width",
    "model_height",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<yolov8multihead_decode::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "detection_static_properties", props->meta_name);
  props->zero_points = Ax::get_property(
      input, "zero_points", "detection_static_properties", props->zero_points);
  props->scales = Ax::get_property<float>(
      input, "scales", "detection_static_properties", props->scales);
  props->num_classes = Ax::get_property(
      input, "classes", "detection_static_properties", props->num_classes);
  props->kpts_shape = Ax::get_property(
      input, "kpts_shape", "detection_static_properties", props->kpts_shape);
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
        filename, "yolo_decode_static_properties", logger);
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
        "yolo_decode_static_properties", logger);
  }

  props->sigmoid_tables
      = ax_utils::build_sigmoid_tables(props->zero_points, props->scales);
  props->softmax_tables
      = ax_utils::build_exponential_tables(props->zero_points, props->scales);
  props->dequantize_tables
      = ax_utils::build_dequantization_tables(props->zero_points, props->scales);
  auto filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", std::vector<int>{});
  props->filter = ax_utils::build_filter(filter, props->num_classes);

  props->padding = Ax::get_property(
      input, "padding", "detection_static_properties", props->padding);
  std::iota(props->weights.begin(), props->weights.end(), 0);

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolov8multihead_decode::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}
