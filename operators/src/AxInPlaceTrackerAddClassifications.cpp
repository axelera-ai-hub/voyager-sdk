// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaClassification.hpp"
#include "AxMetaTracker.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <unordered_set>

struct trackeraddclassifications_properties {
  std::string tracking_meta_key;
  std::string classification_meta_key;
};

std::shared_ptr<AxMetaBase>
set_classification_key_to_class_id_score_pair(int first_id, int frame_id,
    uint8_t key, const std::unordered_map<int, TrackingElement> &frame_id_to_element)
{
  int best_class_id = -1;
  float best_score = 0.0;

  for (int id = first_id; id <= frame_id; ++id) {
    const auto itr_element = frame_id_to_element.find(id);
    if (itr_element == frame_id_to_element.end()) {
      continue;
    }
    auto itr_class_id_score = itr_element->second.frame_data_map.find(key);
    if (itr_class_id_score == itr_element->second.frame_data_map.end()) {
      continue;
    }
    auto classification_meta_ptr
        = dynamic_cast<AxMetaClassification *>(itr_class_id_score->second.get());
    if (classification_meta_ptr == nullptr) {
      throw std::runtime_error("trackeraddclassifications: meta not of type AxMetaClassification");
    }
    auto current_class_id = classification_meta_ptr->get_classes()[0][0];
    auto current_score = classification_meta_ptr->get_scores()[0][0];
    if (current_score > best_score) {
      best_class_id = current_class_id;
      best_score = current_score;
    }
  }
  return std::make_shared<AxMetaClassification>(
      std::vector<std::vector<float>>{ { best_score } },
      std::vector<std::vector<int32_t>>{ { best_class_id } },
      std::vector<std::vector<std::string>>{ { "" } });
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "tracking_meta_key",
    "classification_meta_key" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<trackeraddclassifications_properties>();
  prop->tracking_meta_key = Ax::get_property(input, "tracking_meta_key",
      "trackeraddclassifications_properties", prop->tracking_meta_key);
  prop->classification_meta_key = Ax::get_property(input, "classification_meta_key",
      "trackeraddclassifications_properties", prop->classification_meta_key);

  if (prop->classification_meta_key.empty()) {
    throw std::runtime_error("trackeraddclassifications: classification meta key not provided");
  }

  return prop;
}

extern "C" void
inplace(const AxDataInterface &data,
    const trackeraddclassifications_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  auto *tracker_meta = ax_utils::get_meta<AxMetaTracker>(
      prop->tracking_meta_key, map, "trackeraddclassifications");

  auto classifications = tracker_meta->submeta_map->get<AxMetaClassification>(
      prop->classification_meta_key);

  int classification_meta_index = 0;
  for (auto &[track_id, tracking_descriptor] : tracker_meta->track_id_to_tracking_descriptor) {
    if (!tracker_meta->num_subtask_runs
        || tracking_descriptor.frame_id < tracker_meta->num_subtask_runs) {
      const auto &frame
          = tracking_descriptor.collection->get_frame(tracking_descriptor.frame_id);
      auto &video_info = std::get<AxVideoInterface>(data).info;
      if (frame.bbox.x1 < 0 || frame.bbox.y1 < 0 || frame.bbox.x2 >= video_info.width
          || frame.bbox.y2 >= video_info.height) {
        continue;
      }

      if (classification_meta_index >= classifications.size()) {
        throw std::runtime_error("trackeraddclassifications: number of assigned classifications "
                                 + std::to_string(classification_meta_index + 1) + " exceeds number of available classifications "
                                 + std::to_string(classifications.size()));
      }
      auto classification_meta = classifications[classification_meta_index++];
      auto latest_class_id = classification_meta->get_classes()[0][0];
      auto latest_score = classification_meta->get_scores()[0][0];
      tracking_descriptor.collection->set_frame_data_map(tracking_descriptor.frame_id,
          prop->classification_meta_key,
          std::make_shared<AxMetaClassification>(
              std::vector<std::vector<float>>{ { latest_score } },
              std::vector<std::vector<int32_t>>{ { latest_class_id } },
              std::vector<std::vector<std::string>>{ { "" } }),
          set_classification_key_to_class_id_score_pair);
    }
  }

  if (classifications.size() != classification_meta_index) {
    throw std::runtime_error("trackeraddclassifications: number of assigned classifications "
                             + std::to_string(classification_meta_index) + " does not match number of available classifications "
                             + std::to_string(classifications.size()));
  }

  tracker_meta->submeta_map->erase(prop->classification_meta_key);
}
