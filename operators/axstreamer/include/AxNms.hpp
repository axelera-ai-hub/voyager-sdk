// Copyright Axelera AI, 2025
#pragma once

#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"
#include "AxOpenCl.hpp"

class CLNms
{
  public:
  using buffer = ax_utils::CLProgram::ax_buffer;
  using kernel = ax_utils::CLProgram::ax_kernel;

  explicit CLNms(int max_size, Ax::Logger &logger);

  AxMetaObjDetection run(const AxMetaObjDetection &meta, float threshold,
      int class_agnostic, int max_size, int &error);

  static AxMetaObjDetection remove_suppressed(
      const AxMetaObjDetection &meta, const std::vector<char> &keep);

  private:
  ax_utils::CLProgram program;
  int error{};
  buffer boxes;
  buffer scores;
  buffer classes;
  buffer ious;
  buffer suppressed;
  kernel map_kernel;
  kernel reduce_kernel;
  int max_size_;
};


///
/// @brief Remove boxes that overlap too much
/// @param meta   The meta boxes to remove from
/// @param threshold   The threshold to use
/// @param class_agnostic  If true, all boxes are considered, otherwise only
/// boxes of the same class are considered
/// @return The boxes that were not removed
///
/// Boxes assumed to be in the format [x1, y1, x2, y2]
///
AxMetaObjDetection non_max_suppression(const AxMetaObjDetection &meta,
    float threshold, bool class_agnostic, int max_boxes);


///
/// @brief Remove keypoints from boxes  that overlap too much
/// @param meta   The meta with boxes to remove from
/// @param threshold   The threshold to use
/// @param class_agnostic  If true, all boxes are considered, otherwise only
/// boxes of the same class are considered
/// @return The keypoints that were not removed
///
/// Keypoints  assumed to be in the format [x, y, visibility]
///
AxMetaKptsDetection non_max_suppression(const AxMetaKptsDetection &meta,
    float threshold, bool class_agnostic, int max_boxes);

/// TODO: fix doxygen
/// @brief Remove keypoints from boxes  that overlap too much
/// @param meta   The meta with boxes to remove from
/// @param threshold   The threshold to use
/// @param class_agnostic  If true, all boxes are considered, otherwise only
/// boxes of the same class are considered
/// @return The keypoints that were not removed
///
/// Keypoints  assumed to be in the format [x, y, visibility]
///
AxMetaSegmentsDetection non_max_suppression(const AxMetaSegmentsDetection &meta,
    float threshold, bool class_agnostic, int max_boxes);
