// Copyright Axelera AI, 2025
#include "AxMetaClassification.hpp"
#include "AxMetaTracker.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

extern "C" std::unique_ptr<AxMetaBase>
determine_object_attribute(const void *props, int first_id, int frame_id, uint8_t key,
    const std::unordered_map<int, TrackingElement> &frame_id_to_element, Ax::Logger &)
{
  std::unordered_map<int, std::pair<float, int>> class_id_to_score_and_frequency;
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
      throw std::runtime_error("set_most_frequent: meta not of type AxClassificationMeta");
    }
    auto current_class_id = classification_meta_ptr->get_classes()[0][0];
    auto current_score = classification_meta_ptr->get_scores()[0][0];
    auto [itr, newly] = class_id_to_score_and_frequency.try_emplace(
        current_class_id, current_score, 1);
    if (!newly) {
      float sum = itr->second.first * itr->second.second;
      itr->second.second += 1;
      itr->second.first = (sum + current_score) / itr->second.second;
    }
  }
  float avg_score = 0.0f;
  int most_frequent_class_id = -1;
  int max_frequency = 0;
  for (const auto &pair : class_id_to_score_and_frequency) {
    if (pair.second.second > max_frequency) {
      max_frequency = pair.second.second;
      most_frequent_class_id = pair.first;
      avg_score = pair.second.first;
    } else if (pair.second.second == max_frequency) {
      if (pair.second.first > avg_score) {
        most_frequent_class_id = pair.first;
        avg_score = pair.second.first;
      }
    }
  }

  return std::make_unique<AxMetaClassification>(
      std::vector<std::vector<float>>{ { avg_score } },
      std::vector<std::vector<int32_t>>{ { most_frequent_class_id } },
      std::vector<std::vector<std::string>>{ { "" } });
}
