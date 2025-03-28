// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <unordered_set>

struct filterdetections_properties {
  std::string meta_key;
  std::string which;
  int top_k = 0;
  std::vector<int> classes_to_keep;
  int min_width = 0;
  int min_height = 0;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "meta_key",
    "which", "top_k", "classes_to_keep", "min_width", "min_height" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<filterdetections_properties>();
  prop->meta_key = Ax::get_property(
      input, "meta_key", "filterdetections_properties", prop->meta_key);
  prop->which = Ax::get_property(input, "which", "filterdetections_properties", prop->which);
  prop->top_k = Ax::get_property(input, "top_k", "filterdetections_properties", prop->top_k);
  prop->classes_to_keep = Ax::get_property(input, "classes_to_keep",
      "filterdetections_properties", prop->classes_to_keep);
  prop->min_width = Ax::get_property(
      input, "min_width", "filterdetections_properties", prop->min_width);
  prop->min_height = Ax::get_property(
      input, "min_height", "filterdetections_properties", prop->min_height);
  return prop;
}

extern "C" void
inplace(const AxDataInterface &interface,
    const filterdetections_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  if (prop->meta_key.empty()) {
    throw std::runtime_error("inplace_filterdetections.cc: No meta key given");
  }
  auto meta_itr = meta_map.find(prop->meta_key);
  if (meta_itr == meta_map.end()) {
    throw std::runtime_error(
        "inplace_filterdetections.cc: " + prop->meta_key + " not found in meta map");
  }
  auto *obj_det_meta = dynamic_cast<AxMetaObjDetection *>(meta_itr->second.get());
  auto *kpt_det_meta = dynamic_cast<AxMetaKptsDetection *>(meta_itr->second.get());
  auto *seg_det_meta = dynamic_cast<AxMetaSegmentsDetection *>(meta_itr->second.get());
  if (!kpt_det_meta && !obj_det_meta && !seg_det_meta) {
    throw std::runtime_error(
        "inplace_filterdetections.cc: " + prop->meta_key
        + " is neither AxMetaObjDetection nor AxMetaKptsDetection nor AxMetaSegmentsDetection");
  }

  std::vector<box_xyxy> boxes{};
  std::vector<float> scores{};
  std::vector<int> classes{};
  KptXyvVector kpts{};
  std::vector<ax_utils::segment> segments{};

  auto *box_meta = dynamic_cast<AxMetaBbox *>(meta_itr->second.get());
  for (size_t i = 0; i < box_meta->num_elements(); ++i) {
    auto box = box_meta->get_box_xyxy(i);
    if ((prop->min_width && box.x2 - box.x1 < prop->min_width)
        || (prop->min_height && box.y2 - box.y1 < prop->min_height)) {
      continue;
    }
    if (!prop->classes_to_keep.empty()) {
      int class_id;
      if (obj_det_meta) {
        class_id = obj_det_meta->class_id(i);

      } else if (seg_det_meta && seg_det_meta->is_multi_class()) {
        class_id = seg_det_meta->class_id(i);
      } else {
        throw std::runtime_error(
            "filterdetections : classes_to_keep is set but no class id found");
      }

      if (std::find(prop->classes_to_keep.begin(), prop->classes_to_keep.end(), class_id)
          == prop->classes_to_keep.end()) {
        continue;
      }
    }

    boxes.push_back(box);
    if (obj_det_meta) {
      scores.push_back(obj_det_meta->score(i));
      classes.push_back(obj_det_meta->class_id(i));
    } else if (kpt_det_meta) {
      scores.push_back(kpt_det_meta->score(i));
      int nk = kpt_det_meta->get_kpts_shape()[0];
      for (int j = 0; j < nk; ++j) {
        kpts.push_back(kpt_det_meta->get_kpt_xy(nk * i + j));
      }
    } else if (seg_det_meta) {
      scores.push_back(seg_det_meta->score(i));
      auto good_segment = std::move(
          const_cast<AxMetaSegmentsDetection *>(seg_det_meta)->get_segment(i));
      segments.push_back(std::move(good_segment));
      if (seg_det_meta->is_multi_class()) {
        classes.push_back(seg_det_meta->class_id(i));
      }
    }
  }

  if (prop->top_k && prop->top_k < static_cast<int>(boxes.size())) {
    std::vector<int> indices;
    if (prop->which == "NONE") {
      throw std::runtime_error("filterdetections : top_k is set but which is NONE");
    } else if (prop->which == "SCORE") {
      indices = ax_utils::indices_for_topk(scores, prop->top_k);
    } else if (prop->which == "CENTER") {
      if (!std::holds_alternative<AxVideoInterface>(interface)) {
        throw std::runtime_error("filterdetections : CENTER requires video interface");
      }
      const auto &info = std::get<AxVideoInterface>(interface).info;
      indices = ax_utils::indices_for_topk_center(
          boxes, prop->top_k, info.width, info.height);
    } else if (prop->which == "AREA") {
      indices = ax_utils::indices_for_topk_area(boxes, prop->top_k);
    } else {
      throw std::runtime_error("filterdetections : which must be one of SCORE, AREA, CENTER");
    }

    std::vector<box_xyxy> new_boxes{};
    std::vector<float> new_scores{};
    std::vector<int> new_classes{};
    KptXyvVector new_kpts{};
    std::vector<ax_utils::segment> new_segments{};
    for (auto idx : indices) {
      new_boxes.push_back(boxes[idx]);
      new_scores.push_back(scores[idx]);
      if (!classes.empty()) {
        new_classes.push_back(classes[idx]);
      }
      if (!kpts.empty()) {
        int nk = kpt_det_meta->get_kpts_shape()[0];
        for (int j = 0; j < nk; ++j) {
          new_kpts.push_back(kpts[nk * idx + j]);
        }
      }
      if (!segments.empty()) {
        new_segments.push_back(segments[idx]);
      }
    }
    boxes.swap(new_boxes);
    scores.swap(new_scores);
    classes.swap(new_classes);
    kpts.swap(new_kpts);
    segments.swap(new_segments);
  }

  if (obj_det_meta) {
    meta_map[prop->meta_key] = std::make_unique<AxMetaObjDetection>(
        std::move(boxes), std::move(scores), std::move(classes));
  } else if (kpt_det_meta) {
    meta_map[prop->meta_key] = std::make_unique<AxMetaKptsDetection>(
        std::move(boxes), std::move(kpts), std::move(scores),
        kpt_det_meta->get_kpts_shape(), kpt_det_meta->get_decoder_name());
  } else if (seg_det_meta) {
    auto shape = seg_det_meta->get_segments_shape();
    auto sizes = SegmentShape{ shape[2], shape[1] };
    meta_map[prop->meta_key] = std::make_unique<AxMetaSegmentsDetection>(std::move(boxes),
        std::move(segments), std::move(scores), std::move(classes), sizes,
        seg_det_meta->get_base_box(), std::move(seg_det_meta->get_decoder_name()));
  } else {
    throw std::logic_error(
        "inplace_filterdetections.cc: Neither AxMetaObjDetection nor AxMetaKptsDetection nor AxMetaSegmentsDetection");
  }
}
