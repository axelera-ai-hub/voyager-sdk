// Copyright Axelera AI, 2023
#include <fstream>
#include <numeric>
#include <unordered_set>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaClassification.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "master_meta",
    "classlabels_file",
    "box_meta",
    "top_k",
    "sorted",
    "largest",
    "softmax",
  };
  return allowed_properties;
}

struct classification_properties {
  std::string meta_key
      = "meta_" + std::to_string(reinterpret_cast<long long unsigned int>(this));
  std::vector<std::string> classlabels;
  std::string box_meta{};
  std::string master_meta{};
  int top_k;
  int sorted;
  int largest;
  int softmax;
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<classification_properties> prop
      = std::make_shared<classification_properties>();

  auto classlabels = Ax::get_property(input, "classlabels_file",
      "classification_static_properties", std::string{});
  if (!classlabels.empty()) {
    prop->classlabels = ax_utils::read_class_labels(
        classlabels, "classification_static_properties", logger);
  }

  prop->meta_key = Ax::get_property(
      input, "meta_key", "classification_static_properties", prop->meta_key);

  prop->box_meta = Ax::get_property(
      input, "box_meta", "classification_static_properties", prop->box_meta);
  prop->master_meta = Ax::get_property(input, "master_meta",
      "classification_static_properties", prop->master_meta);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    classification_properties *prop, Ax::Logger &logger)
{
  prop->top_k = Ax::get_property(input, "top_k", "classification_dynamic_properties", 1);
  //  Currently this property is ignored as the output is always sorted.
  //  Added purely for completeness
  prop->sorted
      = Ax::get_property(input, "sorted", "classification_dynamic_properties", false);
  prop->largest
      = Ax::get_property(input, "largest", "classification_dynamic_properties", true);
  prop->softmax
      = Ax::get_property(input, "softmax", "classification_dynamic_properties", true);
}

extern "C" void
decode_to_meta(const AxTensorsInterface &tensors, const classification_properties *prop,
    unsigned int current_frame, unsigned int total_frames,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &, Ax::Logger &logger)
{
  if (total_frames <= current_frame) {
    throw std::runtime_error("classification_decode_to_meta: Current frame is out of bounds");
  }
  if (1 != tensors.size()) {
    throw std::runtime_error("classification_decode_to_meta: Number of tensors must be 1");
  }
  auto &tensor = tensors[0];
  size_t total = tensor.total();
  if (!prop->classlabels.empty() && prop->classlabels.size() != total) {
    logger(AX_WARN) << "classification_decode_to_meta: Number of classes from NN ("
                    << total << ") must match that of classes file ("
                    << prop->classlabels.size() << ")" << std::endl;
  }
  if (4 != tensor.bytes) {
    throw std::runtime_error("classification_decode_to_meta: NN must return float");
  }

  std::vector<int> indices(total);

  auto *mat_data = static_cast<const float *>(tensor.data);
  std::vector<float> mat_copy{};

  if (prop->softmax) {
    mat_copy.resize(total);
    float max_for_shift = *std::max_element(mat_data, mat_data + total);
    std::transform(mat_data, mat_data + total, mat_copy.begin(),
        [max_for_shift](float a) { return std::exp(a - max_for_shift); });
    float denominator = std::accumulate(mat_copy.begin(), mat_copy.end(), 0.0);
    std::transform(mat_copy.begin(), mat_copy.end(), mat_copy.begin(),
        [denominator](float a) { return a / denominator; });
    mat_data = mat_copy.data();
  }

  std::iota(indices.begin(), indices.end(), 0);

  auto top_k = std::min(std::size_t(prop->top_k), total);
  if (prop->largest) {
    std::partial_sort(indices.begin(), std::next(indices.begin(), top_k),
        indices.end(),
        [mat_data](int i, int j) { return mat_data[i] > mat_data[j]; });
  } else {
    std::partial_sort(indices.begin(), std::next(indices.begin(), top_k),
        indices.end(),
        [mat_data](int i, int j) { return mat_data[i] < mat_data[j]; });
  }

  std::vector<float> scores;
  std::vector<std::string> labels;
  scores.reserve(top_k);
  labels.reserve(top_k);
  indices.resize(top_k);

  for (auto idx : indices) {
    scores.push_back(mat_data[idx]);
    if (prop->classlabels.empty()) {
      labels.push_back("Class: " + std::to_string(idx));
    } else {
      labels.push_back(prop->classlabels[idx]);
    }
  }

  if (!prop->box_meta.empty()) {
    auto box_position = map.find(prop->box_meta);
    if (box_position != map.end()) {
      auto *meta_ptr = dynamic_cast<AxMetaObjDetection *>(box_position->second.get());
      if (current_frame < meta_ptr->num_elements()) {
        meta_ptr->update_detection(current_frame, scores.at(0), indices.at(0));
      } else {
        throw std::runtime_error("classification_decode_to_meta: current_frame out of range");
      }
    }
  } else if (!prop->master_meta.empty()) {
    ax_utils::insert_meta<AxMetaClassification>(map, prop->meta_key, prop->master_meta,
        current_frame, total_frames, AxMetaClassification::scores_vec{ scores },
        AxMetaClassification::classes_vec{ indices },
        AxMetaClassification::labels_vec{ labels });
  } else {
    auto position = map.find(prop->meta_key);
    if (position == map.end()) {
      AxMetaClassification::scores_vec scores_vec(total_frames);
      AxMetaClassification::classes_vec indices_vec(total_frames);
      AxMetaClassification::labels_vec labels_vec(total_frames);
      scores_vec[current_frame] = std::move(scores);
      indices_vec[current_frame] = std::move(indices);
      labels_vec[current_frame] = std::move(labels);
      auto ptr = std::make_unique<AxMetaClassification>(
          std::move(scores_vec), std::move(indices_vec), std::move(labels_vec));
      map[prop->meta_key] = std::move(ptr);
    } else {
      auto *meta_ptr = dynamic_cast<AxMetaClassification *>(position->second.get());
      if (!meta_ptr) {
        throw std::runtime_error(
            "classification_decode_to_meta: Meta key already exists but with a different type");
      }
      if (meta_ptr->get_number_of_subframes() != total_frames) {
        throw std::runtime_error(
            "classification_decode_to_meta: Meta key already exists but with a different number of frames");
      }
      meta_ptr->replace(current_frame, std::move(scores), std::move(indices),
          std::move(labels));
    }
  }
}
