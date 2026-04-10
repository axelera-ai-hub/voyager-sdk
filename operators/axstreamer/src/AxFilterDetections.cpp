// Copyright Axelera AI, 2025

#include "AxFilterDetections.hpp"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"

void
Ax::filter_detections(const AxDataInterface &interface,
    const FilterDetectionsProperties &prop, Ax::MetaMap &meta_map, Ax::Logger &logger)
{
  auto meta_itr = meta_map.find(prop.input_meta_key);
  if (meta_itr == meta_map.end()) {
    std::string valid_keys;
    for (const auto &pair : meta_map) {
      valid_keys += pair.first + ", ";
    }
    throw std::runtime_error("filterdetections: " + prop.input_meta_key
                             + " not found in meta map " + valid_keys);
  }
  auto *base_meta = meta_itr->second.get();
  auto *box_meta = dynamic_cast<AxMetaBbox *>(base_meta);
  if (!box_meta) {
    throw std::runtime_error("filterdetections: " + prop.input_meta_key
                             + " is not derived from AxMetaBbox");
  }
  auto *obj_det_meta = dynamic_cast<AxMetaObjDetection *>(base_meta);
  auto *kpt_det_meta = dynamic_cast<AxMetaKptsDetection *>(base_meta);
  auto *seg_det_meta = dynamic_cast<AxMetaSegmentsDetection *>(base_meta);

  std::vector<box_xyxy> boxes{};
  std::vector<int> ids{};
  std::vector<float> scores{};
  std::vector<int> classes{};
  KptXyvVector kpts{};
  std::vector<ax_utils::segment> segments{};

  size_t num_elements = box_meta->num_elements();

  for (size_t i = 0; i < num_elements; ++i) {
    auto box = box_meta->get_box_xyxy(i);
    int id = box_meta->get_id(i);
    id = id == -1 ? i : id;
    if ((prop.min_width && box.x2 - box.x1 + 1 < prop.min_width)
        || (prop.min_height && box.y2 - box.y1 + 1 < prop.min_height)) {
      continue;
    }
    if (!prop.classes_to_keep.empty()) {
      if (!box_meta->has_class_id()) {
        throw std::runtime_error(
            "filterdetections : classes_to_keep is set but no class id found");
      }
      auto class_id = box_meta->class_id(i);
      if (std::find(prop.classes_to_keep.begin(), prop.classes_to_keep.end(), class_id)
          == prop.classes_to_keep.end()) {
        continue;
      }
    }

    const auto score = box_meta->score(i);
    if (prop.score != 0.0 && score < prop.score) {
      continue;
    }

    boxes.push_back(box);
    scores.push_back(score);
    ids.push_back(id);
    if (box_meta->has_class_id()) {
      if (obj_det_meta) {
        classes.push_back(obj_det_meta->class_id(i));
      } else if (seg_det_meta) {
        classes.push_back(seg_det_meta->class_id(i));
      } else {
        logger(AX_WARN) << "filterdetections: Box has class ID but no specialized meta with class_id() method, using default class 0"
                        << std::endl;
        classes.push_back(0); // Default class ID as fallback
      }
    }
    if (kpt_det_meta) {
      int nk = kpt_det_meta->get_kpts_shape()[0];
      for (int j = 0; j < nk; ++j) {
        kpts.push_back(kpt_det_meta->get_kpt_xy(nk * i + j));
      }
    } else if (seg_det_meta) {
      try {
        auto good_segment = seg_det_meta->get_segment(i);
        segments.push_back(std::move(good_segment));
      } catch (const std::exception &e) {
        logger(AX_ERROR)
            << "filterdetections: Exception while getting segment: " << e.what()
            << std::endl;
      }
    }
  }

  if (prop.which != Which::None && prop.top_k < static_cast<int>(boxes.size())) {
    logger(AX_DEBUG) << "filterdetections: Applying top-" << prop.top_k << " filtering"
                     << ", found " << boxes.size() << " boxes" << std::endl;

    std::vector<int> indices;
    if (prop.which == Which::Score) {
      indices = ax_utils::indices_for_topk(scores, prop.top_k);
    } else if (prop.which == Which::Center) {
      if (!std::holds_alternative<AxVideoInterface>(interface)) {
        throw std::runtime_error("filterdetections : CENTER requires video interface");
      }
      try {
        const auto &info = std::get<AxVideoInterface>(interface).info;
        if (info.width <= 0 || info.height <= 0) {
          throw std::runtime_error("filterdetections : Invalid video dimensions in interface");
        }
        indices = ax_utils::indices_for_topk_center(
            boxes, prop.top_k, info.width, info.height);
      } catch (const std::exception &e) {
        logger(AX_WARN)
            << "filterdetections: Error accessing video interface: " << e.what()
            << std::endl;
        indices = ax_utils::indices_for_topk(scores, prop.top_k);
      }
    } else if (prop.which == Which::Area) {
      indices = ax_utils::indices_for_topk_area(boxes, prop.top_k);
    }

    std::vector<box_xyxy> new_boxes{};
    std::vector<int> new_ids{};
    std::vector<float> new_scores{};
    std::vector<int> new_classes{};
    KptXyvVector new_kpts{};
    std::vector<ax_utils::segment> new_segments{};
    for (auto idx : indices) {
      new_boxes.push_back(boxes[idx]);
      new_ids.push_back(ids[idx]);
      if (!scores.empty()) {
        new_scores.push_back(scores[idx]);
      }
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
    ids.swap(new_ids);
    scores.swap(new_scores);
    classes.swap(new_classes);
    kpts.swap(new_kpts);
    segments.swap(new_segments);
  }

  bool enable_extern = base_meta->enable_extern;

  std::string metaType = obj_det_meta ? "AxMetaObjDetection" :
                         kpt_det_meta ? "AxMetaKptsDetection" :
                         seg_det_meta ? "AxMetaSegmentsDetection" :
                                        "AxMetaBbox";
  logger(AX_INFO) << "filterdetections: Creating " << metaType << " with "
                  << boxes.size() << " boxes" << std::endl;

  std::unique_ptr<AxMetaBbox> new_meta;
  if (obj_det_meta) {
    new_meta = std::make_unique<AxMetaObjDetection>(
        std::move(boxes), std::move(scores), std::move(classes), std::move(ids));
  } else if (kpt_det_meta) {
    try {
      auto shape = kpt_det_meta->get_kpts_shape();
      auto decoder_name = kpt_det_meta->get_decoder_name();
      new_meta = std::make_unique<AxMetaKptsDetection>(std::move(boxes),
          std::move(kpts), std::move(scores), std::move(ids), shape, decoder_name);
    } catch (const std::exception &e) {
      logger(AX_ERROR) << "filterdetections: Exception creating AxMetaKptsDetection: "
                       << e.what() << std::endl;
      throw;
    }
  } else if (seg_det_meta) {
    try {
      auto shape = seg_det_meta->get_segments_shape();
      auto sizes = SegmentShape{ shape[2], shape[1] };
      auto decoder_name = seg_det_meta->get_decoder_name();
      new_meta = std::make_unique<AxMetaSegmentsDetection>(std::move(boxes),
          std::move(segments), std::move(scores), std::move(classes),
          std::move(ids), sizes, std::move(decoder_name));
    } catch (const std::exception &e) {
      logger(AX_ERROR) << "filterdetections: Exception creating AxMetaSegmentsDetection: "
                       << e.what() << std::endl;
      throw;
    }
  } else if (box_meta) {
    new_meta = std::make_unique<AxMetaBbox>(
        std::move(boxes), std::move(scores), std::move(classes), std::move(ids));
  } else {
    logger(AX_ERROR) << "filterdetections: No appropriate meta type found" << std::endl;
    throw std::logic_error("filterdetections: No appropriate meta type");
  }

  new_meta->enable_extern = enable_extern;
  if (prop.hide_output_meta) {
    new_meta->enable_extern = false;
  }
  meta_map.insert_or_assign(prop.output_meta_key, std::move(new_meta));
}
