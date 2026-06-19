// Copyright Axelera AI, 2023
//
// Highly optimized anchor-based YOLO decoder used by the gst pipeline.
//
// Supports the full anchor-based YOLO family. The xy/wh activation formula is
// driven by per-feature-map `scale_x_y` (float) and `new_coords` (bool) decoder
// properties.
//
//   Ultralytics v5/v7:        scale_x_y=2.0,  new_coords=1
//     bxy = 2 * sigmoid(t) - 0.5 + grid
//     bwh = (2 * sigmoid(t)) ** 2 * anchor
//
//   Classic Darknet (v3, v4): scale_x_y=1.0..1.2 (per cfg), new_coords=0
//     bxy = scale_x_y * sigmoid(t) - 0.5 * (scale_x_y - 1) + grid
//     bwh = exp(t_raw) * anchor
//
//   Darknet new_coords=1:     scale_x_y from cfg, new_coords=1
//     Same as Ultralytics-style wh, with cfg-driven xy scale.

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
  std::string master_meta{};
  std::string association_meta{};
  std::vector<lookups> sigmoid_tables{};
  // wh-decode LUTs per chip-tensor level, with the nc-specific transform
  // pre-baked so the inner loop is a single table read per box dimension:
  //   wh_tables_nc0[level][q+128] = exp(dequant(q))
  //   wh_tables_nc1[level][q+128] = (2 * sigmoid(t))^2, where dequant(q) is t
  //     when sigmoid_in_postprocess, else already sigmoid(t) (no extra sigmoid)
  // Each variant is populated only when at least one feature map uses it.
  std::vector<lookups> wh_tables_nc0{};
  std::vector<lookups> wh_tables_nc1{};
  std::vector<float> anchors{};
  std::vector<std::string> class_labels{};
  std::vector<uint8_t> filter{};
  // Per-feature-map activation params. Defaults reproduce Ultralytics v5
  // (`scale_x_y=2.0`, `new_coords=1`) when these are not populated.
  std::vector<float> scale_x_y{};
  std::vector<uint8_t> new_coords{};
  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 3 * 21 * 20 * 20 }; // Maximum number of boxes for 640*640 yolo
  bool multiclass{ false };
  bool sigmoid_in_postprocess{ true };
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
/// @param props - properties of the model
/// @param outputs - output inferences
/// @return number of boxes added to outputs
///
template <bool multiclass, typename input_type>
int
decode_scores(const input_type *data, const float *sigmoids, float confidence,
    const properties &props, inferences &outputs)
{
  const auto objectness = 4;
  const auto first_class = 5;

  // Get out if the objectness is too low, we don't care about this element
  auto object_score = ax_utils::sigmoid(data[objectness], sigmoids);
  if (object_score < confidence) {
    return 0;
  }

  return ax_utils::decode_scores<multiclass>(data + first_class, sigmoids,
      props.filter, props.confidence, object_score, outputs);
}

template <typename input_type>
int
decode_scores(const input_type *data, const float *sigmoids, float confidence,
    const properties &props, inferences &outputs)
{
  return props.multiclass ?
             decode_scores<true>(data, sigmoids, confidence, props, outputs) :
             decode_scores<false>(data, sigmoids, confidence, props, outputs);
}

/// @brief Decode a single cell of the tensor
/// @param data - pointer to the raw tensor data
/// @param props - properties of the model
/// @param level - which of features maps is being decoded
/// @param anchor_level - which anchor level this tensor is
/// @param num_anchors - number of anchors for this level
/// @param which_anchor - which anchor we are decoding
/// @param recip_width - 1 / width of the feature map (for normalising coords)
/// @param xpos - x position of the cell
/// @param ypos - y position of the cell
/// @param outputs - output inferences
/// @return - The number of predictions added
int
decode_cell(const int8_t *data, const properties &props, const float *sigmoids,
    const float *wh_lut, float sxy, float xy_offset, int anchor_level, int num_anchors,
    int which_anchor, float recip_width, int xpos, int ypos, inferences &outputs)
{
  auto confidence = props.confidence;
  auto num_predictions = decode_scores(data, sigmoids, confidence, props, outputs);
  if (num_predictions != 0) {
    auto *anchor = std::next(
        props.anchors.data(), 2 * (anchor_level * num_anchors + which_anchor));

    float x = (ax_utils::sigmoid(data[0], sigmoids) * sxy - xy_offset + xpos) * recip_width;
    float y = (ax_utils::sigmoid(data[1], sigmoids) * sxy - xy_offset + ypos) * recip_width;
    // wh_lut bakes the nc-specific transform at init time: exp(dequant(q)) for
    // new_coords=0, (2*sigmoid(dequant(q)))^2 for new_coords=1. Single table
    // read per dim, no branch and no transcendental in the hot loop.
    float w = ax_utils::sigmoid(data[2], wh_lut) * anchor[0] * recip_width;
    float h = ax_utils::sigmoid(data[3], wh_lut) * anchor[1] * recip_width;

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
int
decode_tensor(const int8_t *tensor, const properties &props, int width, int height,
    int depth, int level, int anchor_level, int num_anchors, inferences &outputs)
{
  auto x_stride = depth;
  auto y_stride = x_stride * width;
  const auto tensor_size = props.num_classes + 5;
  auto total = 0;
  auto recip_width = 1.0F / std::max(width, height);

  // Hoist per-level activation params and LUT pointers out of the cell loop.
  // Defaults match Ultralytics v5 when the property vectors are not populated.
  // `level` is the chip-tensor index (per-tensor dequant/quant scale).
  // `anchor_level` is the stride-ordered index (matches the cfg yolo-block
  // order in which scale_x_y / new_coords / anchors were emitted).
  float dummy{};
  const float *sigmoids
      = props.sigmoid_tables.empty() ? &dummy : props.sigmoid_tables[level].data();
  const float sxy = anchor_level < static_cast<int>(props.scale_x_y.size()) ?
                        props.scale_x_y[anchor_level] :
                        2.0F;
  const bool nc = anchor_level < static_cast<int>(props.new_coords.size()) ?
                      static_cast<bool>(props.new_coords[anchor_level]) :
                      true;
  const float xy_offset = 0.5F * (sxy - 1.0F);
  // Pick the per-level wh LUT for this anchor_level's nc variant. Falls back
  // to &dummy if the matching variant is unpopulated (input validation
  // upstream catches mismatched property combos).
  const auto &wh_variant_tables = nc ? props.wh_tables_nc1 : props.wh_tables_nc0;
  const float *wh_lut = level < static_cast<int>(wh_variant_tables.size()) ?
                            wh_variant_tables[level].data() :
                            &dummy;

  for (auto y = 0; y != height; ++y) {
    auto *ptr = std::next(tensor, y_stride * y);
    for (auto x = 0; x != width; ++x) {
      for (auto which = size_t{}; which != num_anchors; ++which) {
        auto *p = std::next(ptr, tensor_size * which);
        total += decode_cell(p, props, sigmoids, wh_lut, sxy, xy_offset,
            anchor_level, num_anchors, which, recip_width, x, y, outputs);
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
    auto [width, height, depth] = ax_utils::get_dims(tensors, level, 1);
    if (num_anchors * (props.num_classes + 5) > depth) {
      throw std::runtime_error("decode_tensors : too many anchors for the depth of the tensor");
    }
    if (tensors[level].bytes != 1) {
      throw std::runtime_error("decode_tensors : tensors must be int8_t");
    }
    if (props.sigmoid_tables.empty()) {
      throw std::runtime_error(
          "decode_tensors : zero_points and scales must be provided for dequantization");
    }
    decode_tensor(static_cast<const int8_t *>(tensors[level].data), props,
        width, height, depth, level, lev, num_anchors, output);
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
  props->master_meta = Ax::get_property(
      input, "master_meta", "yolo_decode_static_properties", props->master_meta);
  props->association_meta = Ax::get_property(input, "association_meta",
      "yolo_decode_static_properties", props->association_meta);
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
  auto filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", std::vector<int>{});
  props->filter = ax_utils::build_filter(filter, props->num_classes);


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

  props->scale_x_y = Ax::get_property(
      input, "scale_x_y", "yolo_decode_static_properties", std::vector<float>{});
  std::vector<int> nc_int = Ax::get_property(
      input, "new_coords", "yolo_decode_static_properties", std::vector<int>{});
  props->new_coords.assign(nc_int.begin(), nc_int.end());

  // Build per-level wh LUTs, one variant per nc value actually used by the
  // model. Default (empty `new_coords`) treats every level as nc=1.
  const bool any_nc0 = std::any_of(props->new_coords.begin(),
      props->new_coords.end(), [](uint8_t v) { return v == 0; });
  const bool any_nc1
      = props->new_coords.empty()
        || std::any_of(props->new_coords.begin(), props->new_coords.end(),
            [](uint8_t v) { return v != 0; });
  if (any_nc0) {
    // exp(dequant(q)) per level for classic-Darknet wh.
    props->wh_tables_nc0
        = ax_utils::build_exponential_tables_with_zero_point(zero_points, scales);
  }
  if (any_nc1) {
    // (2 * sigmoid(t))^2 per level for Ultralytics-style wh. When sigmoid is
    // not done in postprocess the network has already applied it, so the
    // dequantized value is sigmoid(t) and the table must not apply it again
    // (mirrors the sigmoid_tables split above).
    if (props->sigmoid_in_postprocess) {
      props->wh_tables_nc1 = ax_utils::build_general_dequantization_tables(
          zero_points, scales, [](float x) {
            float s = ax_utils::to_sigmoid(x) * 2.0F;
            return s * s;
          });
    } else {
      props->wh_tables_nc1 = ax_utils::build_general_dequantization_tables(
          zero_points, scales, [](float x) { return (2.0F * x) * (2.0F * x); });
    }
  }
  return props;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "master_meta",
    "association_meta",
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
    "scale_x_y",
    "new_coords",
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
  predictions = ax_utils::topk(std::move(predictions), prop->topk);

  // The boxes are currently normalized i.e. scaled to [0, 1.0)
  // We need to scale them to the original image size
  //  Determine which edge we originally scaled to
  //  Scale the other edge to match the aspect ratio of the output
  //  and then calculate the offsets of the letterboxed image
  auto base_box = ax_utils::get_master_box(prop->master_meta, prop->association_meta,
      video_interface, subframe_index, map, "yolov5_decode");
  auto pixel_boxes = ax_utils::scale_shift_boxes(predictions.boxes, base_box,
      prop->model_width, prop->model_height, true, prop->letterbox);

  ax_utils::insert_and_associate_meta<AxMetaObjDetection>(map, prop->meta_name,
      prop->master_meta, subframe_index, number_of_subframes,
      prop->association_meta, std::move(pixel_boxes),
      std::move(predictions.scores), std::move(predictions.class_ids));
}
