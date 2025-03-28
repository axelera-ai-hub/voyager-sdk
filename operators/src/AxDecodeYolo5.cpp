// Copyright Axelera AI, 2024
// Highly optimized anchor-based YOLO decoder

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

namespace yolov5
{
using lookups = ax_utils::lookups;
using inferences = ax_utils::inferences;

struct properties {
  std::string meta_name{};
  std::vector<lookups> sigmoid_tables{};
  std::vector<float> anchors{};
  std::vector<std::string> class_labels{};
  std::vector<int> filter{};
  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 3 * 21 * 20 * 20 }; // Maximum number of boxes for 640*640 yolo
  bool multiclass{ false };
  bool transpose{};
  bool sigmoid_in_postprocess{ true };
  std::string master_meta{};
  int model_width{};
  int model_height{};
  bool scale_up{ true };
  bool letterbox{ true };
};

AxTensorsInterface
icdf_tensors(const AxTensorsInterface &tensors)
{
  auto width = 20;
  auto height = 20;
  auto channels = 256;
  const auto square_yolo_size = 2150400;
  const auto four_three_yolo_size = 1612800;
  if (tensors[0].total() == four_three_yolo_size) {
    width = 20;
    height = 15;
  } else if (tensors[0].total() != square_yolo_size) {
    //  Not a recognised icdf model may be a lite yolo
    return tensors;
  }

  auto *data = static_cast<int8_t *>(tensors[0].data);
  auto t0 = AxTensorInterface{ { 1, height * 4, width * 4, channels },
    tensors[0].bytes, data };
  auto t1 = AxTensorInterface{ { 1, height * 2, width * 2, channels },
    tensors[0].bytes, data + (height * width * channels * 16) };

  auto t2 = AxTensorInterface{ { 1, height, width, channels }, tensors[0].bytes,
    data + (height * width * channels * 20) };
  return { t0, t1, t2 };
}


/// The feature maps do not come out the inference engine in the same order as
/// the anchors.  Ideally they would be sorted by size, but they are not.  This
/// function builds a map of which feature map is at which level, and we process
/// them in that order.
std::vector<int>
build_feature_map_levels(const AxTensorsInterface &tensors)
{
  std::vector<int> map_levels(tensors.size());
  std::iota(std::begin(map_levels), std::end(map_levels), 0);
  //  sizes[2] is the width or height of the tensor dependent on transpose
  //  We want to sort the tensors by descending size so that the order
  //  corresponds to the strides, which is the same ordering as the anchors
  std::sort(std::begin(map_levels), std::end(map_levels), [&tensors](int i, int j) {
    return tensors[i].sizes[2] > tensors[j].sizes[2];
  });

  return map_levels;
}

///
/// Dequantize, decode and filter classes according to score
/// confidence.
/// @param data - pointer to the raw tensor data
/// @param sigmoids - lookup table dequantizing values and applying ax_utils::sigmoid
/// @param confidence - minimum confidence score to keep a box
/// @param z_stride - working along z axis, stride to next element
/// @param props - properties of the model
/// @param outputs - output inferences
/// @return number of boxes added to outputs
///
template <bool multiclass, typename input_type>
int
decode_scores(const input_type *data, const float *sigmoids, float confidence,
    int z_stride, const properties &props, inferences &outputs)
{
  const auto objectness = 4 * z_stride;
  const auto first_class = 5 * z_stride;

  // Get out if the objectness is too low, we don't care about this element
  auto object_score = ax_utils::sigmoid(data[objectness], sigmoids);
  if (object_score < confidence) {
    return 0;
  }

  return ax_utils::decode_scores<multiclass>(data + first_class, sigmoids,
      z_stride, props.filter, props.confidence, object_score, outputs);
}

template <typename input_type>
int
decode_scores(const input_type *data, const float *sigmoids, float confidence,
    int z_stride, const properties &props, inferences &outputs)
{
  return props.multiclass ?
             decode_scores<true>(data, sigmoids, confidence, z_stride, props, outputs) :
             decode_scores<false>(data, sigmoids, confidence, z_stride, props, outputs);
}

/// @brief Decode a single cell of the tensor
/// @param data - pointer to the raw tensor data
/// @param props - properties of the model
/// @param level - which of features maps is being decoded
/// @param anchor_level - which anchor level this tensor is
/// @param num_anchors - number of anchors for this level
/// @param which_anchor - which anchor we are decoding
/// @param recip_width - 1 / width of the feature map (for normalising coords)
/// @param z_stride - working along z axis, stride to next element
/// @param xpos - x position of the cell
/// @param ypos - y position of the cell
/// @param outputs - output inferences
/// @return - The number of predictions added
template <typename input_type>
int
decode_cell(const input_type *data, const properties &props, int level,
    int anchor_level, int num_anchors, int which_anchor, float recip_width,
    int z_stride, int xpos, int ypos, inferences &outputs)
{
  float dummy{};
  const auto &sigmoids
      = props.sigmoid_tables.empty() ? &dummy : props.sigmoid_tables[level].data();
  auto num_classes = props.num_classes;
  auto confidence = props.confidence;
  auto num_predictions
      = decode_scores(data, sigmoids, confidence, z_stride, props, outputs);
  if (num_predictions != 0) {
    // Create the bounding box
    auto *anchor = std::next(
        props.anchors.data(), 2 * (anchor_level * num_anchors + which_anchor));
    float x = (ax_utils::sigmoid(data[0], sigmoids) * 2.0F - 0.5F + xpos) * recip_width;
    float y = (ax_utils::sigmoid(data[z_stride], sigmoids) * 2.0F - 0.5F + ypos) * recip_width;
    float w = std::pow(ax_utils::sigmoid(data[2 * z_stride], sigmoids) * 2.0F, 2)
              * anchor[0] * recip_width;
    float h = std::pow(ax_utils::sigmoid(data[3 * z_stride], sigmoids) * 2.0F, 2)
              * anchor[1] * recip_width;

    for (int i = 0; i != num_predictions; ++i) {
      outputs.boxes.push_back({
          x - w / 2,
          y - h / 2,
          x + w / 2,
          y + h / 2,
      });
    }
  }
  return num_predictions;
}

///
/// @brief Decode a single feature map tensor
/// @param data - pointer to the raw tensor data
/// @param props - properties of the model
/// @param width - width of the feature map
/// @param height - height of the feature map
/// @param depth - depth of the feature map
/// @param level - which of features maps this tensor is
/// @param anchor_level - which anchor level this tensor is
/// @param num_anchors - number of anchors for this level
/// @param outputs - output inferences
/// @return - The number of predictions added
template <typename input_type>
int
decode_tensor(const input_type *tensor, const properties &props, int width, int height,
    int depth, int level, int anchor_level, int num_anchors, inferences &outputs)
{
  auto x_stride = props.transpose ? depth : 1;
  auto y_stride = x_stride * width;
  auto z_stride = props.transpose ? 1 : height * y_stride;
  const auto tensor_size = props.num_classes + 5;
  auto total = 0;
  auto recip_width = 1.0F / std::max(width, height);
  for (auto y = 0; y != height; ++y) {
    auto *ptr = std::next(tensor, y_stride * y);
    for (auto x = 0; x != width; ++x) {
      for (auto which = size_t{}; which != num_anchors; ++which) {
        auto *p = std::next(ptr, tensor_size * z_stride * which);
        total += decode_cell(p, props, level, anchor_level, num_anchors, which,
            recip_width, z_stride, x, y, outputs);
      }
      ptr = std::next(ptr, x_stride);
    }
  }
  return total;
}

///
/// yolo outputs three tensors of sizes:
/// (batch, num_anchors * (5 + num_classes), h, w)
/// Where h is [height / stride for stride in strides] and
/// w is [width / stride for stride in strides]
/// and strides is typically [8, 16, 32]
/// but maybe [16, 32] on tiny models and [8, 16, 32, 64] on large models
///
inferences
decode_tensors(const AxTensorsInterface &tensors, const properties &props)
{
  const int outputs_guess = 1000; //  Guess at the number of outputs to avoid allocs
  inferences output{ outputs_guess };
  auto quot_rem = std::div(int(props.anchors.size()), 2 * tensors.size());
  if (quot_rem.rem != 0) {
    throw std::runtime_error("decode_tensors : anchors must be a multiple of number of tensors");
  }
  const auto num_anchors = quot_rem.quot;
  auto mpa_levels = build_feature_map_levels(tensors);
  for (int lev = 0; lev != tensors.size(); ++lev) {
    //  Assumes NHWC format
    //  Extract the correct tensor for this level
    auto level = mpa_levels[lev];
    auto [width, height, depth] = ax_utils::get_dims(tensors, level, props.transpose);
    if (num_anchors * (props.num_classes + 5) > depth) {
      throw std::runtime_error("decode_tensors : too many anchors for the depth of the tensor");
    }
    if (tensors[level].bytes == 1) {
      if (props.sigmoid_tables.empty()) {
        throw std::runtime_error(
            "decode_tensors : zero_points and scales must be provided for dequantization");
      }
      decode_tensor(static_cast<const int8_t *>(tensors[level].data), props,
          width, height, depth, level, lev, num_anchors, output);
    } else if (tensors[level].bytes == 4) {
      decode_tensor(static_cast<const float *>(tensors[level].data), props,
          width, height, depth, level, lev, num_anchors, output);
    } else {
      throw std::runtime_error("decode_tensors : tensors must be int8_t or float32");
    }
  }

  return output;
}


} // namespace yolov5


extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<yolov5::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "yolo_decode_static_properties", props->meta_name);
  auto zero_points = Ax::get_property(input, "zero_points",
      "yolo_decode_static_properties", std::vector<float>{});
  auto scales = Ax::get_property(
      input, "scales", "yolo_decode_static_properties", std::vector<float>{});
  props->anchors = Ax::get_property(
      input, "anchors", "yolo_decode_static_properties", props->anchors);
  props->num_classes = Ax::get_property(
      input, "classes", "yolo_decode_static_properties", props->num_classes);
  auto topk
      = Ax::get_property(input, "topk", "yolo_decode_static_properties", props->topk);
  if (topk > 0) {
    props->topk = topk;
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
      input, "multiclass", "yolo_decode_static_properties", props->multiclass);

  props->transpose = Ax::get_property(
      input, "transpose", "yolo_decode_static_properties", props->transpose);

  props->master_meta = Ax::get_property(
      input, "master_meta", "yolo_decode_static_properties", props->master_meta);

  //  Build the lookup tables
  if (zero_points.size() != scales.size()) {
    logger(AX_ERROR) << "yolo_decode_static_properties : zero_points and scales must be the same "
                        "size"
                     << std::endl;
    throw std::runtime_error(
        "yolo_decode_static_properties : zero_points and scales must be the same size");
  }

  if (props->anchors.empty()) {
    logger(AX_ERROR) << "yolo_decode_static_properties : anchors must be provided"
                     << std::endl;
    throw std::runtime_error("yolo_decode_static_properties : anchors must be provided");
  }

  props->scale_up = Ax::get_property(
      input, "scale_up", "yolo_decode_static_properties", props->scale_up);

  props->model_width = Ax::get_property(
      input, "model_width", "yolo_decode_static_properties", props->model_width);
  props->model_height = Ax::get_property(input, "model_height",
      "yolo_decode_static_properties", props->model_height);
  ax_utils::validate_classes(props->class_labels, props->num_classes,
      "yolo_decode_static_properties", logger);
  props->filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", props->filter);
  if (props->filter.empty()) {
    auto size = props->num_classes;
    props->filter.resize(size);
    std::iota(props->filter.begin(), props->filter.end(), 0);
  }
  std::sort(props->filter.begin(), props->filter.end());
  props->filter.erase(std::unique(props->filter.begin(), props->filter.end()),
      props->filter.end());

  if (props->model_height == 0 || props->model_width == 0) {
    logger(AX_ERROR) << "yolo_decode_static_properties : model_width and model_height must be "
                        "provided"
                     << std::endl;
    throw std::runtime_error(
        "yolo_decode_static_properties : model_width and model_height must be provided");
  }
  props->letterbox = Ax::get_property(
      input, "letterbox", "yolo_decode_static_properties", props->letterbox);
  props->sigmoid_in_postprocess = Ax::get_property(input, "sigmoid_in_postprocess",
      "yolo_decode_static_properties", props->sigmoid_in_postprocess);
  if (props->sigmoid_in_postprocess) {
    props->sigmoid_tables = ax_utils::build_sigmoid_tables(zero_points, scales);
  } else {
    props->sigmoid_tables = ax_utils::build_dequantization_tables(zero_points, scales);
  }
  return props;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "zero_points",
    "scales",
    "anchors",
    "classes",
    "topk",
    "multiclass",
    "classlabels_file",
    "confidence_threshold",
    "transpose",
    "label_filter",
    "sigmoid_in_postprocess",
    "master_meta",
    "scale_up",
    "letterbox",
    "model_width",
    "model_height",
  };
  return allowed_properties;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolov5::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const yolov5::properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{

  auto tensors = in_tensors;
  if (tensors.size() == 1) {
    tensors = yolov5::icdf_tensors(tensors);
  }
  if (tensors.size() != prop->sigmoid_tables.size() && tensors[0].bytes == 1) {
    std::stringstream ss;
    ss << "yolov5_decode_to_meta : Number of input tensors (" << tensors.size()
       << ") does not match the number of dequantize parameters ("
       << prop->sigmoid_tables.size() << ")";

    throw std::runtime_error(ss.str());
  }

  auto predictions = yolov5::decode_tensors(tensors, *prop);
  predictions = ax_utils::topk(predictions, prop->topk);

  // The boxes are currently normalized i.e. scaled to [0, 1.0)
  // We need to scale them to the original image size
  //  Determine which edge we originally scaled to
  //  Scale the other edge to match the aspect ratio of the output
  //  and then calculate the offsets of the letterboxed image
  std::vector<BboxXyxy> pixel_boxes;
  if (prop->master_meta.empty()) {
    pixel_boxes = ax_utils::scale_boxes(predictions.boxes,
        std::get<AxVideoInterface>(video_interface), prop->model_width,
        prop->model_height, prop->scale_up, prop->letterbox);
  } else {
    auto master_meta
        = ax_utils::get_meta<AxMetaBbox>(prop->master_meta, map, "yolov5_decode");
    auto master_box = master_meta->get_box_xyxy(subframe_index);
    pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, master_box,
        prop->model_width, prop->model_height, prop->scale_up, prop->letterbox);
  }
  auto [boxes, scores, class_ids] = ax_utils::remove_empty_boxes(
      pixel_boxes, predictions.scores, predictions.class_ids);

  ax_utils::insert_meta<AxMetaObjDetection>(map, prop->meta_name,
      prop->master_meta, subframe_index, number_of_subframes, std::move(boxes),
      std::move(scores), std::move(class_ids));
}
