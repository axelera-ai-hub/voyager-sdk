// Copyright Axelera AI, 2025
#pragma once

#include <vector>

#include "AxMetaBBox.hpp"

class AxMetaObjDetection : public AxMetaBbox
{
  public:
  AxMetaObjDetection(std::vector<box_xyxy> boxes, std::vector<float> scores,
      std::vector<int> classes)
      : AxMetaBbox(std::move(boxes)), scores_(std::move(scores)),
        classes_(std::move(classes))
  {
    if (num_elements() != scores_.size()
        || (num_elements() != classes_.size() && !classes_.empty())) {
      throw std::logic_error(
          "AxMetaObjDetection: scores and classes must have the same size as boxes");
    }
  }
  ///
  /// @brief Update class_id and score of bbox
  /// @param idx - Index of the requested element
  //  @param score - Detection score
  //  @class_id - Class id
  ///
  void update_detection(int idx, float score, int class_id)
  {
    scores_.at(idx) = score;
    classes_.at(idx) = class_id;
  }

  ///
  /// @brief Get the elements score at the given index. If index is out of
  /// bounds std::out_of_range is thrown.
  /// @param idx - Index of the requested element
  /// @return The score of the given element
  ///
  float score_at(size_t idx) const
  {
    return scores_.at(idx);
  }

  ///
  /// @brief Get the elements score at the given index. No bounds checking is
  /// performed. The behaviour is undefined if idx is out of range
  /// @param idx - Index of the requested element
  /// @return The score of the given element
  ///
  float score(size_t idx) const
  {
    return scores_[idx];
  }

  ///
  /// @brief Get the class id of the given element. If idx is out of bounds
  /// std::out_of_range is thrown.
  /// @param idx - Index of the requested element
  /// @return The class id of the given element
  ///
  int class_id_at(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_.at(idx);
  }

  ///
  /// @brief Get the class id of the given element. No boubds checking is
  /// performed. The behaviour is undefined if idx is out of range
  /// @param idx - Index of the requested element
  /// @return The class id of the given element
  ///
  int class_id(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_[idx];
  }

  ///
  /// @brief Check if we have more than a single class
  /// @return True if we have more than a single class
  ///
  bool is_multi_class() const
  {
    return !classes_.empty();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *object_meta = "ObjectDetectionMeta";
    auto meta1 = extern_meta{ object_meta, "scores", int(scores_.size() * sizeof(float)),
      reinterpret_cast<const char *>(scores_.data()) };
    auto meta2 = extern_meta{ object_meta, "classes", int(classes_.size() * sizeof(int)),
      reinterpret_cast<const char *>(classes_.data()) };

    auto meta = AxMetaBbox::get_extern_meta();
    meta[0].type = object_meta;
    meta.push_back(meta1);
    meta.push_back(meta2);
    return meta;
  }

  const float *get_score_data() const
  {
    return scores_.data();
  }

  const int *get_classes_data() const
  {
    return classes_.data();
  }

  private:
  std::vector<float> scores_;
  std::vector<int> classes_;
};
