// Copyright Axelera AI, 2025
#include <string>
#include "AxDataInterface.h"
#include "AxInferenceNet.hpp"

namespace Ax
{

enum class Which { None, Score, Center, Area };

struct FilterDetectionsProperties {
  /// @brief Key in the meta map to find the input detections.
  std::string input_meta_key{};
  /// @brief Key in the meta map to store the output detections.
  std::string output_meta_key{};
  /// @brief If true, the output meta will be hidden and not rendered
  bool hide_output_meta = true;

  /// filter_detections can have multiple filters, that are processed in the
  /// followoing order

  /// @brief If not zero, any detecitons smaller than min_width or min_height
  /// will be removed.
  int min_width = 0;
  int min_height = 0;

  /// @brief If not empty, the detections will be filtered based on the classes to keep.
  /// The classes are specified by their class id.
  /// If empty, all classes are kept.
  /// This property requires the input meta to have a class id, so one of
  /// AxMetaObjDetection, or a multi-class AxMetaSegmentsDetection.
  std::vector<int> classes_to_keep{};

  /// @brief If not zero, any detections below this score will be removed.
  float score = 0.0f;

  /// @brief If not None, the remaining detections affter the above filters will ordered
  /// by either score, proximity to the center of the frame, or by the area of the detection.
  /// And the top `topk` items will be selected.
  /// If None, all detections are kept by this filter, and topk should be set to 0.
  Which which = Which::None;
  int top_k = 0;
};

/// @brief Get a default logger instance for filter_detections
inline Ax::Logger &
getDefaultLogger()
{
  static Ax::Logger defaultLogger(Ax::Severity::warning);
  return defaultLogger;
}

/// @brief Filter detections in the AxDataInterface based on the properties.
/// The input meta must contain one of AxMetaObjDetection, AxMetaKptsDetection,
/// or AxMetaSegmentsDetection filters are based
void filter_detections(const AxDataInterface &interface,
    const FilterDetectionsProperties &prop, Ax::MetaMap &meta_map,
    Ax::Logger &logger = getDefaultLogger());


template <typename Destination>
InferenceDoneCallback
filter_detections_to(Destination &dest, const FilterDetectionsProperties &prop)
{
  return [&](auto &done) {
    if (!done.end_of_input) {
      filter_detections(done.video, prop, *done.meta_map);
    }
    forward(dest, done);
  };
}


} // namespace Ax
