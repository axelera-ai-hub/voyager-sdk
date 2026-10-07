// Copyright Axelera AI, 2025
#include "AxMetaClassification.hpp"
#include "AxMetaTracker.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

extern "C" std::unique_ptr<AxMetaBase>
determine_object_attribute(const void *props, int first_id, int frame_id, uint8_t key,
    const std::unordered_map<int, TrackingElement> &frame_id_to_element, Ax::Logger &)
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
      throw std::runtime_error("set_best_score: meta not of type AxClassificationMeta");
    }
    auto current_class_id = classification_meta_ptr->get_classes()[0][0];
    auto current_score = classification_meta_ptr->get_scores()[0][0];
    if (current_score > best_score) {
      best_class_id = current_class_id;
      best_score = current_score;
    }
  }
  return std::make_unique<AxMetaClassification>(
      std::vector<std::vector<float>>{ { best_score } },
      std::vector<std::vector<int32_t>>{ { best_class_id } },
      std::vector<std::vector<std::string>>{ { "" } });
}
