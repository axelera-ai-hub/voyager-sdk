// Copyright Axelera AI, 2026
// YOLACT instance segmentation decoder: SSD-style prior decode, per-class
// fast NMS and prototype mask assembly

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaSegmentsDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

namespace yolact_decode
{

using lookups = std::array<float, 256>;

float
dequantize(int8_t value, const float *table)
{
  return table[value + 128];
}

struct properties {
  std::vector<lookups> exponential_tables{};
  std::vector<lookups> dequantize_tables{};
  std::vector<lookups> tanh_tables{};
  std::vector<std::vector<int>> padding{};
  std::vector<float> zero_points{};
  std::vector<float> scales{};
  std::vector<std::string> class_labels{};
  std::vector<float> anchor_scales{ 24.0F, 48.0F, 96.0F, 192.0F, 384.0F };
  std::vector<float> aspect_ratios{ 1.0F, 0.5F, 2.0F };
  float center_variance{ 0.1F };
  float size_variance{ 0.2F };
  float confidence{ 0.05F };
  float nms_iou_threshold{ 0.5F };
  int max_nms_boxes{ 200 };
  int topk{ 100 };
  int num_classes{ 0 };
  int num_seg_masks{ 32 };
  int crop_padding{ 1 };
  int model_width{};
  int model_height{};
  std::string meta_name{};
  std::string master_meta{};
  std::string association_meta{};
  std::string decoder_name{};
};

/// One pyramid level: the three head tensors that share its grid.
struct level_tensors {
  int loc;
  int conf;
  int mask;
};

/// An anchor that cleared the confidence threshold under at least one class.
/// The mask coefficients stay as a pointer into the level tensor, paired with
/// the lookup table of the tensor they point into, so they are read at most
/// once per surviving detection rather than once per candidate.
struct candidates {
  std::vector<ax_utils::fbox> boxes{};
  std::vector<const int8_t *> coefficients{};
  std::vector<const float *> coefficient_tables{};
  std::vector<float> scores{}; // num_classes foreground scores per candidate
};

/// A detection that survived fast NMS, still carrying its coefficient source.
struct detection {
  ax_utils::fbox box;
  const int8_t *coefficients;
  const float *coefficient_table;
  float score;
  int class_id;
};

int
depadded_depth(const AxTensorsInterface &tensors,
    const std::vector<std::vector<int>> &padding, int index)
{
  return tensors[index].sizes[3] - padding[index][6] - padding[index][7];
}

/// @brief Group the head tensors into the prototype plus one triple per level.
///
/// The compiled model emits, per level, a loc tensor of 4 channels per anchor,
/// a conf tensor of num_classes + 1 and a coefficient tensor of num_seg_masks,
/// so the depadded depth identifies each one. Levels are ordered coarsest grid
/// last, matching the order the prior scales are listed in.
std::pair<int, std::vector<level_tensors>>
sort_tensors(const AxTensorsInterface &tensors, const std::vector<std::vector<int>> &padding,
    const properties &props, Ax::Logger &logger)
{
  const auto num_anchors = static_cast<int>(props.aspect_ratios.size());
  const auto expected_levels = static_cast<int>(props.anchor_scales.size());
  if (tensors.size() != 1 + 3 * expected_levels) {
    logger.throw_error("libdecode_yolact: expected " + std::to_string(1 + 3 * expected_levels)
                       + " tensors (one prototype plus loc, conf and coefficients for each of "
                       + std::to_string(expected_levels) + " levels), but got "
                       + std::to_string(tensors.size()));
  }

  std::vector<int> indices(tensors.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  std::sort(std::begin(indices), std::end(indices), [&tensors](auto a, auto b) {
    return std::tie(tensors[a].sizes[2], tensors[a].sizes[3])
           > std::tie(tensors[b].sizes[2], tensors[b].sizes[3]);
  });

  const auto prototype_idx = indices[0];
  if (depadded_depth(tensors, padding, prototype_idx) != props.num_seg_masks) {
    logger.throw_error(
        "libdecode_yolact: the prototype tensor has "
        + std::to_string(depadded_depth(tensors, padding, prototype_idx))
        + " channels but num_seg_masks is " + std::to_string(props.num_seg_masks));
  }

  std::vector<level_tensors> levels;
  levels.reserve(expected_levels);
  for (auto i = size_t{ 1 }; i != indices.size(); i += 3) {
    auto level = level_tensors{ -1, -1, -1 };
    for (auto j = i; j != i + 3; ++j) {
      const auto depth = depadded_depth(tensors, padding, indices[j]);
      if (depth == 4 * num_anchors) {
        level.loc = indices[j];
      } else if (depth == (props.num_classes + 1) * num_anchors) {
        level.conf = indices[j];
      } else if (depth == props.num_seg_masks * num_anchors) {
        level.mask = indices[j];
      } else {
        logger.throw_error("libdecode_yolact: cannot identify a head tensor with "
                           + std::to_string(depth) + " channels; expected loc ("
                           + std::to_string(4 * num_anchors) + "), conf ("
                           + std::to_string((props.num_classes + 1) * num_anchors) + ") or coefficients ("
                           + std::to_string(props.num_seg_masks * num_anchors) + ")");
      }
    }
    if (level.loc == -1 || level.conf == -1 || level.mask == -1) {
      logger.throw_error("libdecode_yolact: a pyramid level is missing one of its head tensors");
    }
    levels.push_back(level);
  }
  return { prototype_idx, levels };
}

/// @brief Collect the anchors of one level whose best foreground score clears
///        the confidence threshold, decoding their boxes against the priors.
///
/// The prior for a cell is centred at ((x + 0.5) / grid_width, (y + 0.5) /
/// grid_height) and is square with side scale * sqrt(ratio) / max(model_width,
/// model_height), so it is a pure function of the level grid. Boxes come out in
/// normalized xyxy and are deliberately not clamped: the crop and the pixel
/// conversion clamp later, exactly as the reference decode does.
void
decode_level(const AxTensorsInterface &tensors, const level_tensors &level,
    const properties &props, float anchor_scale, candidates &out)
{
  const auto [width, height, loc_stride] = ax_utils::get_dims(tensors, level.loc, true);
  const auto conf_stride = tensors[level.conf].sizes[3];
  const auto mask_stride = tensors[level.mask].sizes[3];
  const auto *loc_data = static_cast<const int8_t *>(tensors[level.loc].data);
  const auto *conf_data = static_cast<const int8_t *>(tensors[level.conf].data);
  const auto *mask_data = static_cast<const int8_t *>(tensors[level.mask].data);

  const auto *loc_lookups = props.dequantize_tables[level.loc].data();
  const auto *conf_lookups = props.exponential_tables[level.conf].data();
  // Upstream's mask head ends in tanh, and the compiler leaves that activation
  // out of the AIPU graph, so the coefficients arrive raw and it is applied here.
  const auto *mask_lookups = props.tanh_tables[level.mask].data();

  // The softmax denominator is at least exp(largest logit), so no foreground
  // score can exceed exp((best foreground - largest) * scale). Comparing that
  // bound against the threshold on the raw values rejects the great majority of
  // anchors, for which the background class wins by a wide margin, without
  // paying for a softmax over every class.
  const auto conf_scale = props.scales[level.conf];
  const auto log_confidence = std::log(props.confidence);

  const auto scores_per_anchor = props.num_classes + 1;
  const auto longest_side
      = static_cast<float>(std::max(props.model_width, props.model_height));
  const auto recip_width = 1.0F / width;
  const auto recip_height = 1.0F / height;

  std::vector<float> softmaxed(scores_per_anchor);
  for (auto y = 0; y != height; ++y) {
    for (auto x = 0; x != width; ++x) {
      const auto *loc_cell = std::next(loc_data, (y * width + x) * loc_stride);
      const auto *conf_cell = std::next(conf_data, (y * width + x) * conf_stride);
      const auto *mask_cell = std::next(mask_data, (y * width + x) * mask_stride);

      for (auto which = size_t{}; which != props.aspect_ratios.size(); ++which) {
        const auto *conf_anchor = std::next(conf_cell, which * scores_per_anchor);
        const auto best_foreground = *std::max_element(
            std::next(conf_anchor), conf_anchor + scores_per_anchor);
        const auto largest = std::max(conf_anchor[0], best_foreground);
        if ((best_foreground - largest) * conf_scale <= log_confidence) {
          continue;
        }
        ax_utils::softmax(
            conf_anchor, scores_per_anchor, conf_lookups, softmaxed.data());
        const auto best
            = std::max_element(std::next(std::begin(softmaxed)), std::end(softmaxed));
        if (*best <= props.confidence) {
          continue;
        }

        const auto *loc_anchor = std::next(loc_cell, which * 4);
        const auto side = anchor_scale * std::sqrt(props.aspect_ratios[which]) / longest_side;
        const auto centre_x
            = (x + 0.5F) * recip_width
              + dequantize(loc_anchor[0], loc_lookups) * props.center_variance * side;
        const auto centre_y
            = (y + 0.5F) * recip_height
              + dequantize(loc_anchor[1], loc_lookups) * props.center_variance * side;
        const auto half_width
            = side * std::exp(dequantize(loc_anchor[2], loc_lookups) * props.size_variance)
              * 0.5F;
        const auto half_height
            = side * std::exp(dequantize(loc_anchor[3], loc_lookups) * props.size_variance)
              * 0.5F;

        out.boxes.push_back({ centre_x - half_width, centre_y - half_height,
            centre_x + half_width, centre_y + half_height });
        out.coefficients.push_back(std::next(mask_cell, which * props.num_seg_masks));
        out.coefficient_tables.push_back(mask_lookups);
        out.scores.insert(out.scores.end(), std::next(std::begin(softmaxed)),
            std::end(softmaxed));
      }
    }
  }
}

float
intersection_over_union(const ax_utils::fbox &a, const ax_utils::fbox &b)
{
  const auto overlap_width = std::max(std::min(a.x2, b.x2) - std::max(a.x1, b.x1), 0.0F);
  const auto overlap_height = std::max(std::min(a.y2, b.y2) - std::max(a.y1, b.y1), 0.0F);
  const auto intersection = overlap_width * overlap_height;
  const auto area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const auto area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / (area_a + area_b - intersection);
}

/// @brief Per-class fast NMS followed by a global cap on the detection count.
///
/// Fast NMS suppresses a box when a higher scoring box of the same class
/// overlaps it, and unlike greedy NMS it never consults the boxes that were
/// themselves suppressed, so it is a single pass over the score-ordered list
/// rather than a transitive sweep. A candidate is judged once per class, so it
/// can be emitted more than once with different class ids.
std::vector<detection>
fast_nms(const candidates &all, const properties &props)
{
  const auto num_candidates = static_cast<int>(all.boxes.size());
  //  A starting capacity, not a bound: the topk cap is applied after
  //  suppression, and a candidate can be kept once per class it wins.
  std::vector<detection> kept;
  kept.reserve(props.topk);
  std::vector<int> order;
  order.reserve(num_candidates);
  for (auto class_id = 0; class_id != props.num_classes; ++class_id) {
    order.clear();
    for (auto i = 0; i != num_candidates; ++i) {
      if (all.scores[i * props.num_classes + class_id] > props.confidence) {
        order.push_back(i);
      }
    }
    const auto *scores = all.scores.data() + class_id;
    std::stable_sort(std::begin(order), std::end(order), [scores, &props](auto a, auto b) {
      return scores[a * props.num_classes] > scores[b * props.num_classes];
    });
    if (static_cast<int>(order.size()) > props.max_nms_boxes) {
      order.resize(props.max_nms_boxes);
    }

    for (auto j = 0; j != static_cast<int>(order.size()); ++j) {
      const auto &box = all.boxes[order[j]];
      const auto suppressed = std::any_of(
          std::begin(order), std::next(std::begin(order), j), [&](auto earlier) {
            return intersection_over_union(all.boxes[earlier], box) > props.nms_iou_threshold;
          });
      if (!suppressed) {
        kept.push_back({ box, all.coefficients[order[j]], all.coefficient_tables[order[j]],
            scores[order[j] * props.num_classes], class_id });
      }
    }
  }

  std::stable_sort(std::begin(kept), std::end(kept),
      [](const auto &a, const auto &b) { return a.score > b.score; });
  if (static_cast<int>(kept.size()) > props.topk) {
    kept.resize(props.topk);
  }
  return kept;
}

/// @brief Order a normalized coordinate pair, scale it to a pixel grid and pad.
std::pair<float, float>
to_pixels(float low, float high, int size, int padding)
{
  low *= size;
  high *= size;
  return { std::max(std::min(low, high) - padding, 0.0F),
    std::min(std::max(low, high) + padding, static_cast<float>(size)) };
}

/// @brief Assemble a detection's mask from the prototypes and crop it to the box.
///
/// The mask is the linear combination of the prototype maps weighted by the
/// detection's coefficients, binarized where it is positive. Upstream applies a
/// sigmoid and thresholds at 0.5, which is the same predicate.
///
/// The coefficients are dequantized through their tanh table, because tanh
/// reweights the terms of the channel sum and the sign of that sum is what the
/// threshold reads. The prototypes stay quantized: their scale is one positive
/// factor common to every term, so it cannot change that sign.
ax_utils::segment
build_segment(const detection &detected, const AxTensorInterface &prototype, int prototype_depth,
    int prototype_zero, const properties &props, const BboxXyxy &base_box)
{
  const auto width = prototype.sizes[2];
  const auto height = prototype.sizes[1];
  const auto row_stride = width * prototype.sizes[3];
  const auto *prototype_data = static_cast<const int8_t *>(prototype.data);

  const auto [left, right]
      = to_pixels(detected.box.x1, detected.box.x2, width, props.crop_padding);
  const auto [top, bottom]
      = to_pixels(detected.box.y1, detected.box.y2, height, props.crop_padding);
  const auto x1 = static_cast<int>(std::ceil(left));
  const auto x2 = static_cast<int>(std::ceil(right));
  const auto y1 = static_cast<int>(std::ceil(top));
  const auto y2 = static_cast<int>(std::ceil(bottom));
  if (x2 <= x1 || y2 <= y1) {
    // The box misses the prototype grid, so the crop leaves nothing.
    return { base_box.x1, base_box.y1, base_box.x2, base_box.y2, 0, 0, 1, 1,
      std::vector<uint8_t>(1, 0) };
  }

  std::vector<uint8_t> map;
  map.reserve((x2 - x1) * (y2 - y1));
  for (auto y = y1; y != y2; ++y) {
    for (auto x = x1; x != x2; ++x) {
      const auto *prototype_cell
          = std::next(prototype_data, y * row_stride + x * prototype.sizes[3]);
      auto dot = 0.0F;
      for (auto k = 0; k != prototype_depth; ++k) {
        dot += static_cast<float>(prototype_cell[k] - prototype_zero)
               * dequantize(detected.coefficients[k], detected.coefficient_table);
      }
      map.push_back(dot > 0.0F ? 255 : 0);
    }
  }
  return { base_box.x1, base_box.y1, base_box.x2, base_box.y2, x1, y1, x2, y2,
    std::move(map) };
}

} // namespace yolact_decode

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const yolact_decode::properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  if (!prop) {
    logger(AX_ERROR) << "decode_to_meta : properties not set" << std::endl;
    throw std::runtime_error("decode_to_meta : properties not set");
  }
  if (in_tensors.size() != prop->padding.size()) {
    throw std::runtime_error(
        "libdecode_yolact : " + std::to_string(in_tensors.size()) + " tensors but "
        + std::to_string(prop->padding.size()) + " padding entries");
  }
  if (in_tensors.size() != prop->scales.size()) {
    throw std::runtime_error(
        "libdecode_yolact : " + std::to_string(in_tensors.size()) + " tensors but "
        + std::to_string(prop->scales.size()) + " dequantize parameters");
  }
  if (std::any_of(std::begin(prop->padding), std::end(prop->padding),
          [](const auto &entry) { return entry.size() != 8; })) {
    throw std::runtime_error("libdecode_yolact : every padding entry needs two values per axis");
  }

  const auto [prototype_idx, levels]
      = yolact_decode::sort_tensors(in_tensors, prop->padding, *prop, logger);

  //  An estimate, not a bound: every anchor of every level can in principle
  //  clear the threshold, which is two orders more than ever does.
  yolact_decode::candidates all;
  all.boxes.reserve(prop->max_nms_boxes);
  all.coefficients.reserve(prop->max_nms_boxes);
  all.coefficient_tables.reserve(prop->max_nms_boxes);
  all.scores.reserve(static_cast<size_t>(prop->max_nms_boxes) * prop->num_classes);
  for (auto level = size_t{}; level != levels.size(); ++level) {
    yolact_decode::decode_level(
        in_tensors, levels[level], *prop, prop->anchor_scales[level], all);
  }
  auto detections = yolact_decode::fast_nms(all, *prop);

  const auto base_box = ax_utils::get_master_box(prop->master_meta,
      prop->association_meta, video_interface, subframe_index, map, prop->decoder_name);
  const auto &prototype = in_tensors[prototype_idx];
  const auto prototype_depth
      = yolact_decode::depadded_depth(in_tensors, prop->padding, prototype_idx);
  const auto prototype_zero = static_cast<int>(prop->zero_points[prototype_idx]);

  std::vector<ax_utils::fbox> boxes;
  std::vector<float> scores;
  std::vector<int> classes;
  SegmentList segments;
  boxes.reserve(detections.size());
  scores.reserve(detections.size());
  classes.reserve(detections.size());
  segments.reserve(detections.size());
  for (const auto &detected : detections) {
    boxes.push_back(detected.box);
    scores.push_back(detected.score);
    classes.push_back(detected.class_id);
    segments.push_back(yolact_decode::build_segment(
        detected, prototype, prototype_depth, prototype_zero, *prop, base_box));
  }

  // YOLACT stretches its input, so the boxes scale straight onto the frame.
  auto pixel_boxes = ax_utils::scale_shift_boxes(
      boxes, base_box, prop->model_width, prop->model_height, true, false);
  auto sizes = SegmentShape{ static_cast<size_t>(prototype.sizes[2]),
    static_cast<size_t>(prototype.sizes[1]) };
  std::vector<int> ids;
  ax_utils::insert_and_associate_meta<AxMetaSegmentsDetection>(map,
      prop->meta_name, prop->master_meta, subframe_index, number_of_subframes,
      prop->association_meta, std::move(pixel_boxes), std::move(segments),
      std::move(scores), std::move(classes), ids, sizes, prop->decoder_name);
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "master_meta",
    "association_meta",
    "decoder_name",
    "classlabels_file",
    "classes",
    "confidence_threshold",
    "nms_iou_threshold",
    "max_nms_boxes",
    "topk",
    "zero_points",
    "scales",
    "padding",
    "num_seg_masks",
    "anchor_scales",
    "aspect_ratios",
    "center_variance",
    "size_variance",
    "crop_padding",
    "model_width",
    "model_height",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<yolact_decode::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "yolact_static_properties", props->meta_name);
  props->master_meta = Ax::get_property(
      input, "master_meta", "yolact_static_properties", props->master_meta);
  props->association_meta = Ax::get_property(input, "association_meta",
      "yolact_static_properties", props->association_meta);
  props->decoder_name = Ax::get_property(
      input, "decoder_name", "yolact_static_properties", props->decoder_name);
  props->zero_points = Ax::get_property<float>(
      input, "zero_points", "yolact_static_properties", props->zero_points);
  props->scales = Ax::get_property<float>(
      input, "scales", "yolact_static_properties", props->scales);
  props->padding
      = Ax::get_property(input, "padding", "yolact_static_properties", props->padding);
  props->num_classes = Ax::get_property(
      input, "classes", "yolact_static_properties", props->num_classes);
  props->num_seg_masks = Ax::get_property(
      input, "num_seg_masks", "yolact_static_properties", props->num_seg_masks);
  props->anchor_scales = Ax::get_property<float>(
      input, "anchor_scales", "yolact_static_properties", props->anchor_scales);
  props->aspect_ratios = Ax::get_property<float>(
      input, "aspect_ratios", "yolact_static_properties", props->aspect_ratios);
  props->center_variance = Ax::get_property(input, "center_variance",
      "yolact_static_properties", props->center_variance);
  props->size_variance = Ax::get_property(
      input, "size_variance", "yolact_static_properties", props->size_variance);
  props->crop_padding = Ax::get_property(
      input, "crop_padding", "yolact_static_properties", props->crop_padding);
  props->max_nms_boxes = Ax::get_property(
      input, "max_nms_boxes", "yolact_static_properties", props->max_nms_boxes);
  auto topk = Ax::get_property(input, "topk", "yolact_static_properties", props->topk);
  if (topk > 0) {
    props->topk = topk;
  }
  props->model_width = Ax::get_property(
      input, "model_width", "yolact_static_properties", props->model_width);
  props->model_height = Ax::get_property(
      input, "model_height", "yolact_static_properties", props->model_height);
  if (props->model_width == 0 || props->model_height == 0) {
    throw std::runtime_error(
        "yolact_static_properties : model_width and model_height must be provided");
  }

  auto filename = Ax::get_property(
      input, "classlabels_file", "yolact_static_properties", std::string{});
  if (!filename.empty()) {
    props->class_labels
        = ax_utils::read_class_labels(filename, "yolact_static_properties", logger);
  }
  if (props->num_classes == 0) {
    props->num_classes = props->class_labels.size();
  }
  if (props->num_classes == 0) {
    throw std::runtime_error("yolact_static_properties : you must provide either classes or a "
                             "classlabels_file");
  }
  ax_utils::validate_classes(props->class_labels, props->num_classes,
      "yolact_static_properties", logger);

  if (props->zero_points.size() != props->scales.size()) {
    throw std::runtime_error(
        "yolact_static_properties : zero_points and scales must be the same size");
  }
  if (props->anchor_scales.empty() || props->aspect_ratios.empty()) {
    throw std::runtime_error(
        "yolact_static_properties : anchor_scales and aspect_ratios must not be empty");
  }
  // The head tensors of a level are told apart by their depth, so the three
  // depths have to differ.
  const auto anchors = static_cast<int>(props->aspect_ratios.size());
  const auto depths = std::array{ 4 * anchors,
    (props->num_classes + 1) * anchors, props->num_seg_masks * anchors };
  if (depths[0] == depths[1] || depths[0] == depths[2] || depths[1] == depths[2]) {
    throw std::runtime_error(
        "yolact_static_properties : the loc, conf and coefficient tensors cannot be told "
        "apart because two of their depths are equal ("
        + std::to_string(depths[0]) + ", " + std::to_string(depths[1]) + ", "
        + std::to_string(depths[2]) + ")");
  }
  props->exponential_tables
      = ax_utils::build_exponential_tables(props->zero_points, props->scales);
  props->dequantize_tables
      = ax_utils::build_dequantization_tables(props->zero_points, props->scales);
  props->tanh_tables = ax_utils::build_general_dequantization_tables(
      props->zero_points, props->scales, [](float x) { return std::tanh(x); });

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    yolact_decode::properties *prop, Ax::Logger &logger)
{
  prop->confidence = Ax::get_property(input, "confidence_threshold",
      "yolact_dynamic_properties", prop->confidence);
  prop->nms_iou_threshold = Ax::get_property(input, "nms_iou_threshold",
      "yolact_dynamic_properties", prop->nms_iou_threshold);
  logger(AX_DEBUG) << "prop->confidence_threshold is " << prop->confidence << std::endl;
}
