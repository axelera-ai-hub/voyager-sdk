// Copyright Axelera AI, 2025
#include "AxNms.hpp"

#include <chrono>
#include <iostream>
#include <numeric>
#include <span>
#include <stdio.h>
#include <string>
#include <type_traits>

#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"


namespace
{
struct nms_properties {
  float nms_threshold;
  bool class_agnostic;
};

///
/// @brief Caluclate the area
/// @param r The rectangle
/// @return area of rectangle
///
float
area(const box_xyxy &rect)
{
  return (1 + rect.x2 - rect.x1) * (1 + rect.y2 - rect.y1);
}
struct OBBPoint {
  float x;
  float y;
};

///
/// @brief Calculate the intersection over union
/// @param lhs
/// @param rhs
/// @return The intersection over union of the two box_xyxys
///
float
IntersectionOverUnion(const box_xyxy &lhs, const box_xyxy &rhs)
{
  const float ix = 1 + std::min(lhs.x2, rhs.x2) - std::max(lhs.x1, rhs.x1);
  const float iy = 1 + std::min(lhs.y2, rhs.y2) - std::max(lhs.y1, rhs.y1);

  const float intersection = std::max(0.0f, ix) * std::max(0.0f, iy);
  return intersection / (area(lhs) + area(rhs) - intersection);
}

float
IntersectionOverUnionOBB(const box_xywhr &lhs, const box_xywhr &rhs)
{
  // Get covariance matrix components from oriented bounding box in xywhr format
  auto get_covariance_matrix
      = [](const box_xywhr &xywhr) -> std::tuple<float, float, float> {
    float a = (xywhr.w * xywhr.w) / 12.0;
    float b = (xywhr.h * xywhr.h) / 12.0;
    float cos_r = std::cos(xywhr.r);
    float sin_r = std::sin(xywhr.r);
    float cos2 = cos_r * cos_r;
    float sin2 = sin_r * sin_r;

    float A = a * cos2 + b * sin2;
    float B = a * sin2 + b * cos2;
    float C = (a - b) * cos_r * sin_r;

    return std::make_tuple(A, B, C);
  };

  float x1 = lhs.x;
  float y1 = lhs.y;
  float x2 = rhs.x;
  float y2 = rhs.y;

  // Get covariance matrix components
  auto [a1, b1, c1] = get_covariance_matrix(lhs);
  auto [a2, b2, c2] = get_covariance_matrix(rhs);

  // Calculate probabilistic IoU according to the Python implementation
  const float eps = 1e-7;
  float denom = (a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2) + eps;
  float t1 = (((a1 + a2) * (y1 - y2) * (y1 - y2) + (b1 + b2) * (x1 - x2) * (x1 - x2)) / denom)
             * 0.25f;
  float t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5f;
  float num3 = (a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2);
  float term1 = std::max(0.0f, a1 * b1 - c1 * c1);
  float term2 = std::max(0.0f, a2 * b2 - c2 * c2);
  float den3 = 4.0f * std::sqrt(term1 * term2) + eps;
  float t3 = 0.5f * std::log(num3 / den3 + eps);
  float bd = std::clamp(t1 + t2 + t3, eps, 100.0f);
  float hd = std::sqrt(1.0f - std::exp(-bd) + eps);
  return static_cast<float>(1.0f - hd);
}


///
/// @brief Copies the remaining values from the input object detection metadata
/// @param meta - The input metadata
/// @param first - The first index to keep
/// @param last - The last index to keep
/// @return -  The metadata in the range [first, last)
///
AxMetaObjDetection
nms_results(const AxMetaObjDetection &meta, std::vector<int>::iterator first,
    std::vector<int>::iterator last)
{

  std::vector<box_xyxy> boxes_xyxy{};
  std::vector<float> scores{};
  std::vector<int> class_ids{};

  const int num_boxes = std::distance(first, last);
  boxes_xyxy.reserve(num_boxes);
  scores.reserve(num_boxes);

  class_ids.reserve(meta.has_class_id() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xyxy.push_back(meta.get_box_xyxy(idx));
    scores.push_back(meta.score(idx));
    if (meta.has_class_id()) {
      class_ids.push_back(meta.class_id(idx));
    }
  }
  std::vector<int> ids;
  return AxMetaObjDetection{ std::move(boxes_xyxy), std::move(scores),
    std::move(class_ids), std::move(ids) };
}


///
/// @brief Copies the remaining values from the input object detection metadata
/// @param meta - The input metadata
/// @param first - The first index to keep
/// @param last - The last index to keep
/// @return -  The metadata in the range [first, last)
///
AxMetaObjDetectionOBB
nms_results(const AxMetaObjDetectionOBB &meta, std::vector<int>::iterator first,
    std::vector<int>::iterator last)
{

  std::vector<box_xywhr> boxes_xywhr{};
  std::vector<float> scores{};
  std::vector<int> class_ids{};

  const int num_boxes = std::distance(first, last);
  boxes_xywhr.reserve(num_boxes);
  scores.reserve(num_boxes);

  class_ids.reserve(meta.has_class_id() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xywhr.push_back(meta.get_box_xywhr(idx));
    scores.push_back(meta.score(idx));
    if (meta.has_class_id()) {
      class_ids.push_back(meta.class_id(idx));
    }
  }
  std::vector<int> ids;
  return AxMetaObjDetectionOBB{ std::move(boxes_xywhr), std::move(scores),
    std::move(class_ids), std::move(ids) };
}

///
/// @brief Copies the remaining values from the input keypoint detection metadata
/// @param meta - The input metadata
/// @param first - The first index to keep
/// @param last - The last index to keep
/// @return -  The metadata in the range [first, last)
///

AxMetaKptsDetection
nms_results(const AxMetaKptsDetection &meta, std::vector<int>::iterator first,
    std::vector<int>::iterator last)
{
  std::vector<box_xyxy> boxes_xyxy{};
  std::vector<float> scores{};
  std::vector<int> class_ids{};
  std::vector<kpt_xyv> kpts_xyv{};

  const int num_boxes = std::distance(first, last);
  boxes_xyxy.reserve(num_boxes);
  scores.reserve(num_boxes);

  kpts_xyv.reserve(num_boxes * meta.get_kpts_shape()[0]);

  class_ids.reserve(meta.has_class_id() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xyxy.push_back(meta.get_box_xyxy(idx));
    scores.push_back(meta.score(idx));
    if (meta.has_class_id()) {
      class_ids.push_back(meta.class_id(idx));
    }
    auto kpts_shape = meta.get_kpts_shape();
    auto good_kpts = meta.get_kpts_xyv(idx, kpts_shape[0]);
    kpts_xyv.insert(kpts_xyv.end(), good_kpts.begin(), good_kpts.end());
  }
  std::vector<int> ids;
  return AxMetaKptsDetection{ std::move(boxes_xyxy), std::move(kpts_xyv),
    std::move(scores), ids, meta.get_kpts_shape(), std::move(meta.get_decoder_name()) };
}

///
/// @brief Copies the remaining values from the input segment detection metadata
/// @param meta - The input metadata
/// @param first - The first index to keep
/// @param last - The last index to keep
/// @return -  The metadata in the range [first, last)
///
AxMetaSegmentsDetection
nms_results(const AxMetaSegmentsDetection &meta,
    std::vector<int>::iterator first, std::vector<int>::iterator last)
{
  std::vector<box_xyxy> boxes_xyxy{};
  std::vector<float> scores{};
  std::vector<int> class_ids{};
  std::vector<ax_utils::segment> segment_maps{};

  const int num_boxes = std::distance(first, last);
  boxes_xyxy.reserve(num_boxes);
  scores.reserve(num_boxes);

  segment_maps.reserve(num_boxes);

  class_ids.reserve(meta.has_class_id() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xyxy.push_back(meta.get_box_xyxy(idx));
    scores.push_back(meta.score(idx));
    if (meta.has_class_id()) {
      class_ids.push_back(meta.class_id(idx));
    }
    segment_maps.push_back(meta.get_segment(idx));
  }

  auto shape = meta.get_segments_shape();
  auto sizes = SegmentShape{ shape[2], shape[1] };
  std::vector<int> ids;
  return AxMetaSegmentsDetection{ std::move(boxes_xyxy),
    std::move(segment_maps), std::move(scores), std::move(class_ids), ids,
    sizes, std::move(meta.get_decoder_name()) };
}

template <typename meta_type>
box_xyxy
merge_boxes(const meta_type &meta, const box_xyxy &lhs, int other_box_idx)
{
  auto rhs = meta.get_box_xyxy(other_box_idx);
  return { std::min(lhs.x1, rhs.x1), std::min(lhs.y1, rhs.y1),
    std::max(lhs.x2, rhs.x2), std::max(lhs.y2, rhs.y2) };
}

/// @brief Merge the keypoints of two boxes, preferring most visible keypoints
/// @param meta - The metadata containing the keypoints
/// @param other_box_idx - The index of the other box
/// @param merged - The current set of merged keypoints to be updated
void
merge_kpts(const AxMetaKptsDetection &meta, int other_box_idx, std::vector<kpt_xyv> &merged)
{
  const auto kpts_shape = meta.get_kpts_shape();
  const auto num_kpts = kpts_shape[0];
  auto kpts1 = merged;
  auto kpts2 = meta.get_kpts_xyv(other_box_idx, num_kpts);
  merged.resize(num_kpts);
  std::transform(kpts1.begin(), kpts1.end(), kpts2.begin(), merged.begin(),
      [](const kpt_xyv &a, const kpt_xyv &b) {
        return a.visibility < b.visibility ? b : a;
      });
}

float
IntersectionOverSmallest(const box_xyxy &lhs, const box_xyxy &rhs)
{
  const float ix = 1 + std::min(lhs.x2, rhs.x2) - std::max(lhs.x1, rhs.x1);
  const float iy = 1 + std::min(lhs.y2, rhs.y2) - std::max(lhs.y1, rhs.y1);

  const float intersection = std::max(0.0f, ix) * std::max(0.0f, iy);
  return intersection / std::min(area(lhs), area(rhs));
}

void
get_keypoints(const AxMetaKptsDetection &meta, size_t idx, std::vector<kpt_xyv> &keypoints)
{
  auto num_keypoints = meta.get_kpts_shape()[0];
  auto kpts = meta.get_kpts_xyv(idx, num_keypoints);
  keypoints.resize(num_keypoints);
  std::copy(kpts.begin(), kpts.end(), keypoints.begin());
}

template <typename meta_type>
void
get_keypoints(const meta_type &meta, size_t idx, std::vector<kpt_xyv> &keypoints)
{
}

template <typename meta_type, typename F1>
std::vector<int>::iterator
merge_boxes_impl(meta_type &meta, std::vector<int>::iterator first,
    std::vector<int>::iterator last, F1 &&is_adjacent, float threshold)
{
  const float ios_threshold = 0.9F;
  std::vector<kpt_xyv> this_kpts{};
  while (first != last) {
    const int i = *first++;
    auto this_box = meta.get_box_xyxy(i);
    auto this_class = meta.class_id(i);
    auto this_score = meta.score(i);
    get_keypoints(meta, i, this_kpts);
    auto dest = first;
    auto pos = first;
    for (; pos != last; ++pos) {
      const auto idx = *pos;
      if (this_class == meta.class_id(idx)
          && (is_adjacent(this_box, meta.get_box_xyxy(idx))
              || ::IntersectionOverSmallest(this_box, meta.get_box_xyxy(idx)) >= ios_threshold
              || ::IntersectionOverUnion(this_box, meta.get_box_xyxy(idx)) >= threshold)) {
        this_box = merge_boxes(meta, this_box, idx);
        this_score = std::max(this_score, meta.score(idx));
        if constexpr (std::is_same_v<meta_type, AxMetaKptsDetection>) {
          merge_kpts(meta, idx, this_kpts);
        }
      } else {
        *dest++ = idx;
      }
    }
    if (dest != last) {
      //  We merged some boxes, so we need to update the merged box
      meta.set_box_xyxy(i, this_box);
      meta.set_score(i, this_score);
      if constexpr (std::is_same_v<meta_type, AxMetaKptsDetection>) {
        meta.set_kpts_xyv(i, this_kpts);
      }
    }
    last = dest;
  }
  return last;
}

template <typename T>
T
merge_boxes(T &meta, std::vector<int>::iterator first,
    std::vector<int>::iterator last, float threshold)
{
  //  This probably needs adjusting dependent on the video size
  const int MERGE_THRESHOLD = 1;

  auto is_adjacent_horizontal = [](const box_xyxy &lhs, const box_xyxy &rhs) {
    auto diff1 = std::abs(lhs.x1 - rhs.x2);
    auto diff2 = std::abs(lhs.x2 - rhs.x1);
    if (diff1 > MERGE_THRESHOLD && diff2 > MERGE_THRESHOLD) {
      return false;
    }
    auto ydiff1 = std::abs(lhs.y1 - rhs.y1);
    auto ydiff2 = std::abs(lhs.y2 - rhs.y2);
    return ydiff1 <= MERGE_THRESHOLD && ydiff2 <= MERGE_THRESHOLD;
  };

  std::sort(first, last, [&meta](int a, int b) {
    return meta.get_box_xyxy(a).x2 < meta.get_box_xyxy(b).x2;
  });

  auto last_merged = merge_boxes_impl(meta, first, last, is_adjacent_horizontal, threshold);

  auto is_adjacent_vertical = [](const box_xyxy &lhs, const box_xyxy &rhs) {
    auto diff1 = std::abs(lhs.y2 - rhs.y1);
    auto diff2 = std::abs(lhs.y1 - rhs.y2);
    if (diff1 > MERGE_THRESHOLD && diff2 > MERGE_THRESHOLD) {
      return false;
    }
    auto ydiff1 = std::abs(lhs.x1 - rhs.x1);
    auto ydiff2 = std::abs(lhs.x2 - rhs.x2);
    return ydiff1 <= MERGE_THRESHOLD && ydiff2 <= MERGE_THRESHOLD;
  };

  std::sort(first, last_merged, [&meta](int a, int b) {
    return meta.get_box_xyxy(a).y2 < meta.get_box_xyxy(b).y2;
  });
  last_merged = merge_boxes_impl(meta, first, last_merged, is_adjacent_vertical, threshold);

  last_merged = merge_boxes_impl(
      meta, first, last_merged,
      [](const box_xyxy &lhs, const box_xyxy &rhs) { return false; }, threshold);
  return nms_results(meta, first, last_merged);
}

AxMetaPoseSegmentsDetection
nms_results(const AxMetaPoseSegmentsDetection &meta,
    std::vector<int>::iterator first, std::vector<int>::iterator last)
{
  std::vector<box_xyxy> boxes_xyxy{};
  std::vector<kpt_xyv> kpts_xyv{};
  std::vector<float> scores{};
  std::vector<int> class_ids{};
  std::vector<ax_utils::segment> segment_maps{};


  const int num_boxes = std::distance(first, last);
  kpts_xyv.reserve(num_boxes * meta.get_kpts_shape()[0]);
  boxes_xyxy.reserve(num_boxes);
  scores.reserve(num_boxes);

  segment_maps.reserve(num_boxes);

  class_ids.reserve(meta.has_class_id() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xyxy.push_back(meta.get_box_xyxy(idx));
    scores.push_back(meta.score(idx));
    if (meta.has_class_id()) {
      class_ids.push_back(meta.class_id(idx));
    }
    auto kpts_shape = meta.get_kpts_shape();
    auto good_kpts = meta.get_kpts_xyv(idx, kpts_shape[0]);
    kpts_xyv.insert(kpts_xyv.end(), good_kpts.begin(), good_kpts.end());

    segment_maps.push_back(meta.get_segment(idx));
  }

  auto shape = meta.get_segments_shape();
  auto sizes = SegmentShape{ shape[2], shape[1] };
  std::vector<int> ids;
  return AxMetaPoseSegmentsDetection{ std::move(boxes_xyxy),
    std::move(kpts_xyv), std::move(segment_maps), std::move(scores),
    std::move(class_ids), ids, sizes, meta.get_kpts_shape(),
    std::move(meta.get_base_box()), std::move(meta.get_decoder_name()) };
}
} // namespace


///
/// @brief Remove boxes that overlap too much
/// @param meta    The meta with boxes to remove from
/// @param threshold   The threshold to use
/// @param class_agnostic  If true, all boxes are considered, otherwise only
/// boxes of the same class are considered
/// @return The met with boxes that were not removed
///
/// Boxes assumed to be in the format [x1, y1, x2, y2]
///
template <typename T>
T
non_max_suppression_impl(
    T &meta, float threshold, bool class_agnostic, int max_boxes, bool merge)
{

  //  Preconditions
  std::vector<int> indices(meta.num_elements());
  auto first = std::begin(indices);
  auto last = std::end(indices);
  std::iota(first, last, 0);

  if (merge) {
    if constexpr (std::is_same_v<T, AxMetaObjDetection>
                  || std::is_same_v<T, AxMetaKptsDetection>) {
      return merge_boxes(meta, first, last, threshold);
    } else {
      auto id = typeid(T).name();
      throw std::runtime_error("non_max_suppression_impl : Merging not supported for this metadata type: "
                               + std::string(id));
    }
  }

  std::sort(first, last,
      [&meta](int a, int b) { return meta.score(a) > meta.score(b); });
  int count = 0;
  while (first != last && count != max_boxes) {
    ++count;
    const int i = *first++;
    auto this_box = meta.get_box_xyxy(i);
    auto this_class = meta.class_id(i);
    last = std::remove_if(first, last, [&](auto idx) {
      auto other_box = meta.get_box_xyxy(idx);
      return (class_agnostic || this_class == meta.class_id(idx))
             && ::IntersectionOverUnion(this_box, other_box) >= threshold;
    });
  }
  //  When here, the range std::begin(indices) to last contains the indices of
  //  the boxes that should be kept.
  return nms_results(meta, std::begin(indices), first);
}

AxMetaObjDetectionOBB
non_max_suppression_impl_obb(AxMetaObjDetectionOBB &meta, float threshold,
    bool class_agnostic, int max_boxes)
{
  std::vector<int> indices(meta.num_elements());
  auto first = std::begin(indices);
  auto last = std::end(indices);
  std::iota(first, last, 0);

  std::sort(first, last,
      [&meta](int a, int b) { return meta.score(a) > meta.score(b); });
  int count = 0;
  while (first != last && count != max_boxes) {
    ++count;
    const int i = *first++;
    auto this_box = meta.get_box_xywhr(i);
    auto this_class = meta.class_id(i);
    last = std::remove_if(first, last, [&](auto idx) {
      auto other_box = meta.get_box_xywhr(idx);

      return (class_agnostic || this_class == meta.class_id(idx))
             && ::IntersectionOverUnionOBB(this_box, other_box) >= threshold;
    });
  }
  //  When here, the range std::begin(indices) to last contains the indices of
  //  the boxes that should be kept.
  return nms_results(meta, std::begin(indices), first);
}


AxMetaObjDetection
non_max_suppression(AxMetaObjDetection &meta, float threshold,
    bool class_agnostic, int max_boxes, bool merge)
{
  return non_max_suppression_impl(meta, threshold, class_agnostic, max_boxes, merge);
}

AxMetaKptsDetection
non_max_suppression(AxMetaKptsDetection &meta, float threshold,
    bool class_agnostic, int max_boxes, bool merge)
{
  return non_max_suppression_impl(meta, threshold, class_agnostic, max_boxes, merge);
}

AxMetaSegmentsDetection
non_max_suppression(AxMetaSegmentsDetection &meta, float threshold,
    bool class_agnostic, int max_boxes, bool merge)
{
  return non_max_suppression_impl(meta, threshold, class_agnostic, max_boxes, merge);
}

AxMetaPoseSegmentsDetection
non_max_suppression(const AxMetaPoseSegmentsDetection &meta, float threshold,
    bool class_agnostic, int max_boxes, bool merge)
{
  return non_max_suppression_impl(meta, threshold, class_agnostic, max_boxes, merge);
}

AxMetaObjDetectionOBB
non_max_suppression(AxMetaObjDetectionOBB &meta, float threshold,
    bool class_agnostic, int max_boxes, bool merge)
{
  std::ignore = merge;
  return non_max_suppression_impl_obb(meta, threshold, class_agnostic, max_boxes);
}

template <typename Iterator>
std::unique_ptr<AxMetaSegmentsDetection>
unflattened_nms_results(std::span<AxMetaBase *> metas, Iterator first, Iterator last)
{
  auto detections = std::make_unique<AxMetaSegmentsDetection>();
  int current_index = 0;
  for (auto *p : metas) {
    auto *m = dynamic_cast<AxMetaSegmentsDetection *>(p);
    auto last_index = current_index + m->num_elements();
    auto new_end = std::partition(
        first, last, [last_index](auto idx) { return idx < last_index; });
    //  Now [first, new_end) contains the indices for this metadata
    std::transform(first, new_end, first,
        [current_index](auto idx) { return idx - current_index; });
    detections->extend(nms_results(*m, first, new_end));
    m->enable_extern = false;
    current_index = last_index;
    first = new_end;
  }
  return detections;
}

struct boxinfo {
  box_xyxy box;
  float score;
  int class_id;
};

std::unique_ptr<AxMetaSegmentsDetection>
non_max_suppression(std::span<AxMetaBase *> metas, float threshold,
    bool class_agnostic, int max_boxes)
{
  //  First let's create a table of bboxes, scores and class ids that we can
  //  perform NMS on. This is a temporarily flattened structure

  auto temp_boxes = std::vector<boxinfo>{};
  for (auto *p : metas) {
    auto *m = dynamic_cast<AxMetaSegmentsDetection *>(p);
    if (!m) {
      throw std::runtime_error("non_max_suppression : Expected AxMetaSegmentsDetection metadata");
    }
    for (size_t i = 0; i < m->num_elements(); ++i) {
      temp_boxes.push_back({ m->get_box_xyxy(i), m->score(i), m->class_id(i) });
    }
  }

  std::vector<int> indices(temp_boxes.size());
  auto first = std::begin(indices);
  auto last = std::end(indices);
  std::iota(first, last, 0);

  std::sort(first, last, [&temp_boxes](int a, int b) {
    return temp_boxes[a].score > temp_boxes[b].score;
  });
  int count = 0;
  while (first != last && count != max_boxes) {
    ++count;
    const int i = *first++;
    auto this_box = temp_boxes[i].box;
    auto this_class = temp_boxes[i].class_id;
    last = std::remove_if(first, last, [&](auto idx) {
      auto other_box = temp_boxes[idx].box;
      return (class_agnostic || this_class == temp_boxes[idx].class_id)
             && ::IntersectionOverUnion(this_box, other_box) >= threshold;
    });
  }
  //  When here, the range std::begin(indices) to last contains the indices of
  //  the boxes that should be kept.
  //  Now we need to reconstruct the metadata from the kept indices
  return unflattened_nms_results(metas, std::begin(indices), first);
}
