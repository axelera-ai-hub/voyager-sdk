// Copyright Axelera AI, 2025
#include <algorithm>
#include <iomanip> // For std::fixed and std::setprecision
#include <iostream>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxMetaClassification.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxOpUtils.hpp"

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "master_task_name",
    "subtask_name",
  };
  return allowed_properties;
}

struct inplace_tu9_properties {
  std::string master_task_name{};
  std::string subtask_name{};
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &)
{
  std::shared_ptr<inplace_tu9_properties> prop
      = std::make_shared<inplace_tu9_properties>();
  prop->master_task_name = Ax::get_property(input, "master_task_name",
      "inplace_tu9_properties", prop->master_task_name);
  prop->subtask_name = Ax::get_property(
      input, "subtask_name", "inplace_tu9_properties", prop->subtask_name);
  return prop;
}

extern "C" void
inplace(const AxDataInterface &, const inplace_tu9_properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  // Get the master detection metadata
  AxMetaObjDetection *master_meta = ax_utils::get_meta<AxMetaObjDetection>(
      prop->master_task_name, map, "inplace_tu9");
  int num_master_boxes = master_meta->num_elements();

  // Check if the subtask exists in the submeta names
  auto submeta_names = master_meta->submeta_names();
  if (std::none_of(submeta_names.begin(), submeta_names.end(),
          [&prop](const char *name) { return prop->subtask_name == name; })) {
    std::cout << "Subtask name " << prop->subtask_name
              << " not found in submeta names. This is normal for frames with all objects filtered out."
              << std::endl;
    return;
  }

  // Get all submetas (some may be null if the object wasn't classified)
  std::vector<AxMetaClassification *> submetas
      = master_meta->get_submetas<AxMetaClassification>(prop->subtask_name);

  // Safety check for vector size - should match the number of detections
  // Except if all detections were filtered out, then submetas can be empty
  if (submetas.size() && submetas.size() != num_master_boxes) {
    throw std::runtime_error("Submetas exist but number of submetas ("
                             + std::to_string(submetas.size()) + ") doesn't match number of master boxes ("
                             + std::to_string(num_master_boxes) + ").");
  }

  std::cout << "\n--- OPTION 1: Processing all detections with their classifications ---"
            << std::endl;

  // Set floating point precision for more readable output
  std::cout << std::fixed << std::setprecision(2);

  // Process each detection box
  for (int i = 0; i < num_master_boxes; i++) {
    auto bbox = master_meta->get_box_xyxy(i);
    int master_class_id = master_meta->class_id(i);
    float master_score = master_meta->score(i);
    auto midpoint_x = (bbox.x1 + bbox.x2) / 2;
    auto midpoint_y = (bbox.y1 + bbox.y2) / 2;

    // Check if this box has classification results (only filtered fruit/bowl objects should)
    if (i < submetas.size() && submetas[i] != nullptr) {
      auto class_id = submetas[i]->get_classes()[0][0]; // Always a 2D array
      auto score = submetas[i]->get_scores()[0][0]; // Always a 2D array

      // Note: If softmax is not enabled in the decoder, raw scores will be higher
      // Users can either enable softmax in the decoder to normalize scores to [0,1]
      // or experiment to find an appropriate threshold, which can save computation time
      float threshold = 2.0f; // Threshold for raw scores (not softmaxed)
      // Alternatively, apply softmax here: score = std::exp(score) / (std::exp(score) + 1)

      if (score > threshold) {
        std::cout << "The box with midpoint (" << midpoint_x << ", "
                  << midpoint_y << ") has classification class ID " << class_id
                  << " with confidence " << score << std::endl;
      } else {
        std::cout << "The box with midpoint (" << midpoint_x << ", "
                  << midpoint_y << ") has low confidence classification ("
                  << score << "), class ID " << class_id << std::endl;
      }
    } else {
      // Show the actual object class from master detection
      std::cout << "The box with midpoint (" << midpoint_x << ", " << midpoint_y
                << ") has no classification - detected as master class ID "
                << master_class_id << " with score " << master_score << std::endl;
    }
  }

  std::cout << "\n--- OPTION 2: Direct access to a specific detection's classification ---"
            << std::endl;

  // Example of directly accessing a specific submeta
  if (num_master_boxes > 1) {
    // Check if the second detection has classification metadata
    int target_idx = 1; // Second detection (index 1)

    // First check if it exists and is within range
    if (target_idx < submetas.size() && submetas[target_idx] != nullptr) {
      // Method 1: Access via the vector we already have
      auto class_id_method1 = submetas[target_idx]->get_classes()[0][0];
      auto score_method1 = submetas[target_idx]->get_scores()[0][0];

      // Method 2: Direct access using get_submeta (alternative method)
      AxMetaClassification *second_submeta = master_meta->get_submeta<AxMetaClassification>(
          prop->subtask_name, target_idx, num_master_boxes);

      if (second_submeta != nullptr) {
        auto class_id_method2 = second_submeta->get_classes()[0][0];
        auto score_method2 = second_submeta->get_scores()[0][0];

        // Verify both methods return the same result
        if (class_id_method1 == class_id_method2 && score_method1 == score_method2) {
          std::cout << "The second box (index " << target_idx << ") has classification class ID "
                    << class_id_method2 << " with score " << score_method2
                    << " (verified with two different access methods)" << std::endl;
        } else {
          throw std::runtime_error("Inconsistency between metadata access methods!");
        }
      }
    } else {
      std::cout << "The second box (index " << target_idx
                << ") has no classification metadata" << std::endl;
    }
  }
}
