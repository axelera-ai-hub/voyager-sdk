#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <vector>


namespace ssd_decode
{

using lookups = std::array<float, 256>;
using inferences = ax_utils::inferences;

struct anchor {
  float width;
  float height;
};

struct ssd_properties {
  std::vector<lookups> sigmoid_tables{};
  mutable std::vector<std::vector<anchor>> anchors{};
  std::vector<std::vector<int>> padding{};
  std::vector<float> zero_points{};
  std::vector<float> scales{};
  std::vector<std::string> class_labels{};
  std::vector<int> filter{};
  float confidence{ 0.25F };
  int num_classes{ 0 };
  int topk{ 2000 };
  bool multiclass{ true };
  bool softmax{ false };
  std::string meta_name{};
  std::string saved_anchors{};
  bool transpose{ true };
  bool row_major{ true };
  int model_width{ 300 };
  int model_height{ 300 };
  bool scale_up{ false };
  bool letterbox{ false };
};

const std::array row_major_offsets = { 1, 0, 3, 2 };
const std::array col_major_offsets = { 0, 1, 2, 3 };

float
sigmoid(int value, const float *lookups)
{
  int index = value + 128;
  return lookups[index];
}

float
sigmoid(float value, const float * /*unused*/)
{
  return ax_utils::to_sigmoid(value);
}

float
exponential(int value, const float *lookups)
{
  int index = value + 255;
  return lookups[index];
}

/// @brief Create a sequence of floats of size steps, linearly spaced between min and max
/// @param min - The first element of the sequence
/// @param max  - The last element of the sequence
/// @param steps - The number of elements in the sequence
/// @return The sequence of floats
auto
linear_space(float min, float max, int steps) -> std::vector<float>
{
  auto space = std::vector<float>(steps);
  std::iota(std::begin(space), std::end(space), 0);
  std::transform(std::begin(space), std::end(space), std::begin(space),
      [min, max, steps](auto x) { return min + x * (max - min) / (steps - 1); });
  return space;
}

struct tensor_pair {
  int conf_idx;
  int box_idx;
};
struct anchor_gen_cfg {
  float scale_min;
  float scale_max;
  int scale_steps;
  std::vector<float> ratios;
  bool reduce_in_lowest_layer;
  float interpolated_scale_aspect_ratio;
  float base_anchor_height;
  float base_anchor_width;
};

int
calculate_anchor_size(const std::vector<tensor_pair> &order, const AxTensorsInterface &tensors)
{
  return std::accumulate(std::begin(order), std::end(order), 0, [&](auto acc, auto p) {
    auto size = tensors[p.box_idx].sizes[3];
    return acc + size * tensors[p.box_idx].sizes[2] * tensors[p.box_idx].sizes[1];
  });
}

const std::vector<std::vector<anchor>>
generate_level_anchors(std::vector<float> &anchors,
    std::vector<tensor_pair> &order, const AxTensorsInterface &tensors)
{
  std::vector<std::vector<anchor>> all_anchors;
  const auto *panchors = anchors.data();
  for (auto [conf_idx, loc_idx] : order) {
    auto size = tensors[loc_idx].sizes[3];
    auto num_anchors = size / 4;
    std::vector<anchor> level_anchors;
    for (auto i = 0; i != num_anchors; ++i) {
      level_anchors.push_back({ panchors[i * 4 + 2], panchors[i * 4 + 3] });
    }
    auto anchors_size
        = 4 * num_anchors * tensors[loc_idx].sizes[2] * tensors[loc_idx].sizes[1];
    panchors += anchors_size;
    all_anchors.push_back(level_anchors);
  }

  return all_anchors;
}

/// @brief Generate anchors for a given set of scales and apect ratios
/// @param cfg - The configuration for generating the anchors
/// @param tensors - The input tensors
/// @param padding - The extra padding to add/remvoe fro each tensor
/// @param order - The order of the tensors
/// @return - A collections of anchors for each scale
std::vector<float>
generate_anchors(const anchor_gen_cfg &cfg, const AxTensorsInterface &tensors,
    std::vector<tensor_pair> &order)
{
  auto scales = linear_space(cfg.scale_min, cfg.scale_max, cfg.scale_steps);
  auto &ratios = cfg.ratios;
  bool reduce_in_lowest_layer = cfg.reduce_in_lowest_layer;
  const float interpolated_scale_aspect_ratio = cfg.interpolated_scale_aspect_ratio;
  const auto base_anchor_height = cfg.base_anchor_height;
  const auto base_anchor_width = cfg.base_anchor_width;

  auto num_levels = order.size();
  if (interpolated_scale_aspect_ratio > 0) {
    scales.push_back(1.0F);
    ++num_levels;
  }

  auto first = std::begin(scales);
  auto last = std::next(first, num_levels);
  if (interpolated_scale_aspect_ratio > 0) {
    last = std::prev(last);
  }

  auto anchors = std::vector<float>();

  int idx = 0;
  if (reduce_in_lowest_layer) {
    int width = tensors[order[idx].box_idx].sizes[2];
    int height = tensors[order[idx].box_idx].sizes[1];

    const std::array<float, 3> scales{ 0.1F, *first, *first };
    const std::array<float, 3> ratios{ 1.0F, 2.0F, 0.5F };

    for (int y = 0; y != height; ++y) {
      for (int x = 0; x != width; ++x) {
        for (auto i = std::size_t{}; i != scales.size(); ++i) {
          const auto scale = scales[i];
          const auto ratio = ratios[i];
          const auto anchor_height = base_anchor_height * scale / std::sqrt(ratio);
          const auto anchor_width = base_anchor_width * scale * std::sqrt(ratio);
          anchors.push_back(x);
          anchors.push_back(y);
          anchors.push_back(anchor_width);
          anchors.push_back(anchor_height);
        }
      }
    }
    ++idx;
    ++first;
  }

  for (; first != last; ++first) {
    int width = tensors[order[idx].box_idx].sizes[2];
    int height = tensors[order[idx].box_idx].sizes[1];
    for (int y = 0; y != height; ++y) {
      for (int x = 0; x != width; ++x) {

        const auto scale = *first;
        for (auto ratio : ratios) {
          const auto anchor_height = base_anchor_height * scale / std::sqrt(ratio);
          const auto anchor_width = base_anchor_width * scale * std::sqrt(ratio);
          anchors.push_back(x);
          anchors.push_back(y);
          anchors.push_back(anchor_width);
          anchors.push_back(anchor_height);
        }
        if (interpolated_scale_aspect_ratio > 0.0) {
          const auto interpolated_scale = std::sqrt(scale * *std::next(first));
          const auto interpolated_anchor_height = base_anchor_height * interpolated_scale;
          const auto interpolated_anchor_width = base_anchor_width * interpolated_scale;
          anchors.push_back(x);
          anchors.push_back(y);
          anchors.push_back(interpolated_anchor_width);
          anchors.push_back(interpolated_anchor_height);
        }
      }
    }
    ++idx;
  }
  return anchors;
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
  if (tensors.size() % 2 != 0) {
    logger(AX_ERROR) << "The number of tensors must be even" << std::endl;
    throw std::runtime_error("The number of tensors must be even");
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
  for (auto i = size_t{}; i != indices.size(); i += 2) {
    tensor_pairs.push_back(swap ? tensor_pair{ indices[i + 1], indices[i] } :
                                  tensor_pair{ indices[i], indices[i + 1] });
  }
  return tensor_pairs;
}

struct decode_details {
  float recip_width;

  float center_variance_x;
  float center_variance_y;

  float size_variance_x;
  float size_variance_y;

  float box_dequant_zero;
  float box_dequant_scale;

  float score_dequant_zero;
  float score_dequant_scale;
};

template <typename input_type>
float
find_max_score(input_type *data, int z_stride, int num_classes, const decode_details &details)
{
  typename std::remove_const<input_type>::type current = 0;
  for (auto i = 0; i != num_classes; ++i) {
    current = std::max(current, data[i * z_stride]);
  }
  return (current - details.score_dequant_zero) * details.score_dequant_scale;
}

int8_t
find_max_score(const int8_t *data, int z_stride, int num_classes)
{
  auto current = data[0];
  for (auto i = 1; i != num_classes; ++i) {
    current = std::max(current, data[i * z_stride]);
  }
  return current;
}

///
/// Dequantize, decode and filter classes according to score
/// confidence.
/// @param data - pointer to the raw tensor data
/// @param lookups - lookup table dequantizing values and applying sigmoid
/// @param props - properties of the model
/// @param z_stride - working along z axis, stride to next element
/// @param outputs - output inferences
/// @return number of scores added to outputs
///
template <bool multiclass, typename input_type>
int
decode_scores(const input_type *data, const float *lookups, const ssd_properties &props,
    int z_stride, const decode_details &details, inferences &outputs)
{
  //  We ignore the background class
  if (multiclass) {
    if (props.softmax) {
      std::vector<float> softmaxed_scores(props.num_classes + 1);
      ax_utils::softmax(data, props.num_classes + 1, z_stride, lookups,
          softmaxed_scores.data());
      for (auto i : props.filter) {
        auto score = softmaxed_scores[i + 1];
        if (props.confidence <= score) {
          outputs.scores.push_back(score);
          outputs.class_ids.push_back(i);
        }
      }
    } else {
      auto *first = data + z_stride;
      for (auto i : props.filter) {
        auto score = sigmoid(first[i * z_stride], lookups);
        if (props.confidence <= score) {
          outputs.scores.push_back(score);
          outputs.class_ids.push_back(i);
        }
      }
    }
  } else if (props.softmax) {
    std::vector<float> softmaxed_scores(props.num_classes + 1);
    ax_utils::softmax(
        data, props.num_classes + 1, z_stride, lookups, softmaxed_scores.data());
    auto high_score = std::max_element(
        std::next(std::begin(softmaxed_scores)), std::end(softmaxed_scores));
    auto score = *high_score;
    if (props.confidence <= score) {
      outputs.scores.push_back(score);
      auto i = std::distance(std::next(std::begin(softmaxed_scores)), high_score);
      outputs.class_ids.push_back(i);
    }
  } else {
    auto *first = data + z_stride;
    auto i = std::max_element(std::begin(props.filter), std::end(props.filter),
        [=](auto a, auto b) { return first[a * z_stride] < first[b * z_stride]; });
    auto idx = *i;
    auto score = sigmoid(first[idx * z_stride], lookups);
    if (props.confidence <= score) {
      outputs.scores.push_back(score);
      outputs.class_ids.push_back(idx);
    }
  }
  return outputs.scores.size() - outputs.boxes.size();
}

template <typename input_type>
int
decode_scores(const input_type *data, const float *lookups, int z_stride,
    const ssd_properties &props, const decode_details &details, inferences &outputs)
{
  return props.multiclass ?
             decode_scores<true>(data, lookups, props, z_stride, details, outputs) :
             decode_scores<false>(data, lookups, props, z_stride, details, outputs);
}

/// @brief Decode a single cell of the tensor
/// @param box_data - pointer to the raw tensor box data
/// @param score_data - pointer to the raw tensor score
/// @param props - properties of the model
/// @param level - which of features maps is being decoded
/// @param anchor - which anchor to match
/// @param details - parameters for decoding
/// @param box_z_stride - working along z axis, stride to next element
/// @param xpos - x position of the cell
/// @param ypos - y position of the cell
/// @param outputs - output inferences
/// @return - The number of predictions added
int
decode_cell(const int8_t *box_data, const int8_t *score_data, const ssd_properties &props,
    int level, const anchor &anchor, const decode_details &details,
    int box_z_stride, int score_z_stride, int xpos, int ypos, inferences &outputs)
{
  const auto dummy = float{};
  const auto &lookups
      = props.sigmoid_tables.empty() ? &dummy : props.sigmoid_tables[level].data();
  const auto confidence = props.confidence;
  const auto num_predictions
      = decode_scores(score_data, lookups, score_z_stride, props, details, outputs);
  if (num_predictions != 0) {

    const auto &offsets = props.row_major ? row_major_offsets : col_major_offsets;
    const auto [x_idx, y_idx, w_idx, h_idx] = offsets;

    const auto scaled_x = (box_data[x_idx * box_z_stride] - details.box_dequant_zero)
                          * details.box_dequant_scale;
    const auto scaled_y = (box_data[y_idx * box_z_stride] - details.box_dequant_zero)
                          * details.box_dequant_scale;
    const auto scaled_w = (box_data[w_idx * box_z_stride] - details.box_dequant_zero)
                          * details.box_dequant_scale;
    const auto scaled_h = (box_data[h_idx * box_z_stride] - details.box_dequant_zero)
                          * details.box_dequant_scale;

    const auto x = scaled_x * details.center_variance_x * anchor.width
                   + ((xpos + 0.5F) * details.recip_width);
    const auto y = scaled_y * details.center_variance_y * anchor.height
                   + ((ypos + 0.5F) * details.recip_width);
    const auto half_w = std::exp(scaled_w * details.size_variance_x) * anchor.width * 0.5F;
    const auto half_h = std::exp(scaled_h * details.size_variance_y) * anchor.height * 0.5F;

    outputs.boxes.insert(outputs.boxes.end(), num_predictions,
        {
            std::clamp(x - half_w, 0.0F, 1.0F),
            std::clamp(y - half_h, 0.0F, 1.0F),
            std::clamp(x + half_w, 0.0F, 1.0F),
            std::clamp(y + half_h, 0.0F, 1.0F),
        });
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
decode_tensor(const AxTensorsInterface &tensors, int score_idx, int box_idx,
    const ssd_properties &props, int level, inferences &outputs, Ax::Logger &logger)
{
  auto [box_width, box_height, box_depth]
      = ax_utils::get_dims(tensors, box_idx, props.transpose);
  auto [score_width, score_height, score_depth]
      = ax_utils::get_dims(tensors, score_idx, props.transpose);

  if (box_width != score_width || box_height != score_height) {
    logger(AX_ERROR) << "decode_tensor : box and score tensors must be the same size"
                     << std::endl;
    return 0;
  }
  auto box_x_stride = props.transpose ? box_depth : 1;
  auto box_y_stride = box_x_stride * box_width;
  auto box_z_stride = props.transpose ? 1 : box_height * box_y_stride;

  auto score_x_stride = props.transpose ? score_depth : 1;
  auto score_y_stride = score_x_stride * score_width;
  auto score_z_stride = props.transpose ? 1 : score_height * score_y_stride;

  const auto box_size = 4;
  const auto score_size = props.num_classes + 1; //  Add one for the background class

  auto box_tensor = tensors[box_idx];
  auto score_tensor = tensors[score_idx];
  auto *box_data = static_cast<const int8_t *>(box_tensor.data);
  auto *score_data = static_cast<const int8_t *>(score_tensor.data);

  auto total = 0;
  auto recip_width = 1.0F / std::max(box_width, box_height);

  const auto x_scale = 10.0F;
  const auto y_scale = 10.0F;
  const auto height_scale = 5.0F;
  const auto width_scale = 5.0F;

  decode_details details{
    1.0F / std::max(box_width, box_height),
    1.0F / x_scale,
    1.0F / y_scale,
    1.0F / width_scale,
    1.0F / height_scale,
    props.zero_points[box_idx],
    props.scales[box_idx],
    props.zero_points[score_idx],
    props.scales[score_idx],
  };

  for (auto y = 0; y != box_height; ++y) {
    auto *box_ptr = std::next(box_data, box_y_stride * y);
    auto *score_ptr = std::next(score_data, score_y_stride * y);

    for (auto x = 0; x != box_width; ++x) {
      const auto num_anchors = props.anchors[level].size();
      for (auto which = size_t{}; which != num_anchors; ++which) {
        auto *box_p = std::next(box_ptr, box_size * box_z_stride * which);
        auto *score_p = std::next(score_ptr, score_size * score_z_stride * which);

        total += decode_cell(box_p, score_p, props, score_idx, props.anchors[level][which],
            details, box_z_stride, score_z_stride, x, y, outputs);
      }
      box_ptr = std::next(box_ptr, box_x_stride);
      score_ptr = std::next(score_ptr, score_x_stride);
    }
  }
  return total;
}

std::streamsize
file_size(const std::string &filename)
{
  std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}

std::vector<float>
load_anchors(const std::string &filename)
{
  auto size = file_size(filename);
  std::vector<float> anchors(size / sizeof(float));
  auto f = std::ifstream(filename, std::ios::binary);
  f.read(reinterpret_cast<char *>(anchors.data()), size);
  return anchors;
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
decode_tensors(const AxTensorsInterface &tensors, const ssd_properties &prop, Ax::Logger &logger)
{
  const std::vector<std::vector<int>> &padding = prop.padding;
  auto depadded = depad_tensors(tensors, padding);
  auto tensor_order = sort_tensors(depadded, prop.num_classes, logger);
  if (prop.anchors.empty()) {
    auto anchor_cfg = anchor_gen_cfg{
      0.2F, // scale_min
      0.95F, // scale_max
      6, // scale_steps
      { 1.0F, 2.0F, 0.5F, 3.0F, 0.3333F }, //  ratios
      true, // reduce_in_lowest_layer
      1.0F, //  interpolated_scale_aspect_ratio
      1.0F, //  base_anchor_height
      1.0F //  base_anchor_width
    };

    //  If we have no saved anchors, generate them
    auto anchors = prop.saved_anchors.empty() ?
                       generate_anchors(anchor_cfg, tensors, tensor_order) :
                       load_anchors(prop.saved_anchors);

    auto anchors_size = calculate_anchor_size(tensor_order, depadded);
    if (anchors_size != anchors.size()) {
      logger(AX_ERROR)
          << "generate_anchors : anchors size mismatch expected size = " << anchors_size
          << " anchors.size() = " << anchors.size() << std::endl;
      throw std::runtime_error("generate_anchors : anchors size mismatch");
    }
    prop.anchors = generate_level_anchors(anchors, tensor_order, depadded);
  }

  inferences predictions(1000);
  for (int level = 0; level != tensor_order.size(); ++level) {
    const auto [conf_tensor, loc_tensor] = tensor_order[level];
    auto num = decode_tensor(
        tensors, conf_tensor, loc_tensor, prop, level, predictions, logger);
  }
  return predictions;
}

} // namespace ssd_decode

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors,
    const ssd_decode::ssd_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  if (!prop) {
    logger(AX_ERROR) << "decode_to_meta : properties not set" << std::endl;
    throw std::runtime_error("decode_to_meta : properties not set");
  }
  if (in_tensors.size() != prop->sigmoid_tables.size() && in_tensors[0].bytes == 1) {
    throw std::runtime_error(
        "ssd_decode_to_meta : Number of input tensors or dequantize parameters is incorrect");
  }
  auto predictions = ssd_decode::decode_tensors(in_tensors, *prop, logger);
  predictions = ax_utils::topk(predictions, prop->topk);

  auto video_info = std::get<AxVideoInterface>(video_interface).info;

  auto scaled_boxes = ax_utils::scale_boxes(predictions.boxes, video_info.width,
      video_info.height, prop->model_width, prop->model_height, prop->scale_up,
      prop->letterbox);

  map[prop->meta_name] = std::make_unique<AxMetaObjDetection>(std::move(scaled_boxes),
      std::move(predictions.scores), std::move(predictions.class_ids));
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
    "transpose",
    "softmax",
    "zero_points",
    "scales",
    "class_agnostic",
    "classes",
    "saved_anchors",
    "row_major",
    "padding",
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
  std::shared_ptr<ssd_decode::ssd_properties> props
      = std::make_shared<ssd_decode::ssd_properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "ssd_decode_static_properties", props->meta_name);
  props->zero_points = Ax::get_property(
      input, "zero_points", "ssd_decode_static_properties", props->zero_points);
  props->scales = Ax::get_property<float>(
      input, "scales", "ssd_decode_static_properties", props->scales);
  props->num_classes = Ax::get_property(
      input, "classes", "ssd_decode_static_properties", props->num_classes);
  auto topk
      = Ax::get_property(input, "topk", "ssd_decode_static_properties", props->topk);
  if (topk > 0) {
    props->topk = topk;
  }

  auto filename = Ax::get_property(
      input, "classlabels_file", "detection_static_properties", std::string{});
  if (!filename.empty()) {
    props->class_labels
        = ax_utils::read_class_labels(filename, "detection_static_properties", logger);
  }

  auto class_agnostic = Ax::get_property(input, "class_agnostic",
      "ssd_decode_static_properties", !props->multiclass);
  props->multiclass = !class_agnostic;

  props->softmax = Ax::get_property(
      input, "softmax", "ssd_decode_static_properties", props->softmax);

  props->transpose = Ax::get_property(
      input, "transpose", "ssd_decode_static_properties", props->transpose);

  //  Build the lookup tables
  if (props->zero_points.size() != props->scales.size()) {
    logger(AX_ERROR) << "ssd_decode_static_properties : zero_points and scales must have the same "
                        "number of elements."
                     << std::endl;
    throw std::runtime_error(
        "ssd_decode_static_properties : zero_points and scales must be the same size");
  }

  if (props->num_classes == 0) {
    if (props->class_labels.empty()) {
      logger(AX_ERROR) << "ssd_decode_static_properties : you must either provide classes or a"
                          "classlabels_file (or both)"
                       << std::endl;
      throw std::runtime_error("ssd_decode_static_properties : classes must be provided");
    } else {
      props->num_classes = props->class_labels.size();
    }
  }
  ax_utils::validate_classes(props->class_labels, props->num_classes,
      "ssd_decode_static_properties", logger);

  props->sigmoid_tables = (props->softmax ? ax_utils::build_exponential_tables :
                                            ax_utils::build_sigmoid_tables)(
      props->zero_points, props->scales);
  //  Should we get different variations of the anchor config in the
  //  future, we can add these to the options
  props->filter = Ax::get_property(
      input, "label_filter", "detection_static_properties", props->filter);
  if (props->filter.empty()) {
    auto size = props->num_classes;
    props->filter.resize(size);
    std::iota(props->filter.begin(), props->filter.end(), 0);
  } else if (props->softmax) {
    logger(AX_ERROR) << "ssd_decode_static_properties : label_filter cannot be used with softmax"
                     << std::endl;
  }
  props->row_major = Ax::get_property(
      input, "row_major", "ssd_decode_static_properties", props->row_major);

  props->padding = Ax::get_property(
      input, "padding", "ssd_decode_static_properties", props->padding);
  props->saved_anchors = Ax::get_property(input, "saved_anchors",
      "ssd_decode_static_properties", props->saved_anchors);

  std::sort(props->filter.begin(), props->filter.end());
  props->filter.erase(std::unique(props->filter.begin(), props->filter.end()),
      props->filter.end());

  props->scale_up = Ax::get_property(
      input, "scale_up", "ssd_decode_static_properties", props->scale_up);
  props->letterbox = Ax::get_property(
      input, "letterbox", "ssd_decode_static_properties", props->letterbox);
  props->model_width = Ax::get_property(
      input, "model_width", "ssd_decode_static_properties", props->model_width);
  props->model_height = Ax::get_property(input, "model_height",
      "ssd_decode_static_properties", props->model_height);


  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    ssd_decode::ssd_properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "detection_dynamic_properties", prop->confidence);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}
