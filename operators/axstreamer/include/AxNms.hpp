// Copyright Axelera AI, 2025
#pragma once

#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaPoseSegmentsDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"

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
AxMetaObjDetection non_max_suppression(AxMetaObjDetection &meta,
    float threshold, bool class_agnostic, int max_boxes, bool merge);

///
/// @brief Remove boxes that overlap too much
/// @param meta   The meta boxes to remove from
/// @param threshold   The threshold to use
/// @param class_agnostic  If true, all boxes are considered, otherwise only
/// boxes of the same class are considered
/// @return The boxes that were not removed
///
/// Boxes assumed to be in the format [x11, y11, x12, y12, x21, y21, x22, y22]
///
AxMetaObjDetectionOBB non_max_suppression(AxMetaObjDetectionOBB &meta,
    float threshold, bool class_agnostic, int max_boxes, bool merge);

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
AxMetaKptsDetection non_max_suppression(AxMetaKptsDetection &meta,
    float threshold, bool class_agnostic, int max_boxes, bool merge);

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
AxMetaSegmentsDetection non_max_suppression(AxMetaSegmentsDetection &meta,
    float threshold, bool class_agnostic, int max_boxes, bool merge);

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
AxMetaPoseSegmentsDetection non_max_suppression(const AxMetaPoseSegmentsDetection &meta,
    float threshold, bool class_agnostic, int max_boxes, bool merge);


std::unique_ptr<AxMetaSegmentsDetection> non_max_suppression(
    std::span<AxMetaBase *> metas, float threshold, bool class_agnostic, int max_boxes);
