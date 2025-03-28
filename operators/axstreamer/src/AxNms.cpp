// Copyright Axelera AI, 2025
#include "AxNms.hpp"

#include <chrono>
#include <iostream>
#include <numeric>
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

  class_ids.reserve(meta.is_multi_class() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xyxy.push_back(meta.get_box_xyxy(idx));
    scores.push_back(meta.score(idx));
    if (meta.is_multi_class()) {
      class_ids.push_back(meta.class_id(idx));
    }
  }
  return AxMetaObjDetection{ std::move(boxes_xyxy), std::move(scores), std::move(class_ids) };
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

  kpts_xyv.reserve(num_boxes * meta.kpts_shape[0]);

  class_ids.reserve(meta.is_multi_class() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xyxy.push_back(meta.get_box_xyxy(idx));
    scores.push_back(meta.score(idx));
    if (meta.is_multi_class()) {
      class_ids.push_back(meta.class_id(idx));
    }
    auto kpts_shape = meta.get_kpts_shape();
    auto good_kpts = meta.get_kpts_xyv(kpts_shape[0] * idx, kpts_shape[0]);
    kpts_xyv.insert(kpts_xyv.end(), good_kpts.begin(), good_kpts.end());
  }
  return AxMetaKptsDetection{ std::move(boxes_xyxy), std::move(kpts_xyv),
    std::move(scores), meta.get_kpts_shape(), std::move(meta.get_decoder_name()) };
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

  class_ids.reserve(meta.is_multi_class() ? num_boxes : 0);
  for (auto it = first; it != last; ++it) {
    const auto idx = *it;
    boxes_xyxy.push_back(meta.get_box_xyxy(idx));
    scores.push_back(meta.score(idx));
    if (meta.is_multi_class()) {
      class_ids.push_back(meta.class_id(idx));
    }
    segment_maps.push_back(const_cast<AxMetaSegmentsDetection &>(meta).get_segment(idx));
  }

  auto shape = meta.get_segments_shape();
  auto sizes = SegmentShape{ shape[2], shape[1] };
  return AxMetaSegmentsDetection{ std::move(boxes_xyxy),
    std::move(segment_maps), std::move(scores), std::move(class_ids), sizes,
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
non_max_suppression_impl(const T &meta, float threshold, bool class_agnostic, int max_boxes)
{

  //  Preconditions
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


AxMetaObjDetection
non_max_suppression(const AxMetaObjDetection &meta, float threshold,
    bool class_agnostic, int max_boxes)
{
  return non_max_suppression_impl(meta, threshold, class_agnostic, max_boxes);
}

AxMetaKptsDetection
non_max_suppression(const AxMetaKptsDetection &meta, float threshold,
    bool class_agnostic, int max_boxes)
{
  return non_max_suppression_impl(meta, threshold, class_agnostic, max_boxes);
}

AxMetaSegmentsDetection
non_max_suppression(const AxMetaSegmentsDetection &meta, float threshold,
    bool class_agnostic, int max_boxes)
{
  return non_max_suppression_impl(meta, threshold, class_agnostic, max_boxes);
}

//  The CL NMS is based on the algorithm described in https://ieeexplore.ieee.org/document/9646917]
#ifdef OPENCL

/// GPU NMS - Uses the algorithm described
const char *KernelSource = R"###(
typedef struct  {
  int x1, y1, x2, y2;
} box_xyxy;

float
area (__global const box_xyxy *rect)
{
  return (1 + rect->x2 - rect->x1) * (1 + rect->y2 - rect->y1);
}

float
IntersectionOverUnion (__global const box_xyxy *lhs, __global const box_xyxy *rhs)
{
  const float ix = min (lhs->x2, rhs->x2) - max (lhs->x1, rhs->x1);
  const float iy = min (lhs->y2, rhs->y2) - max (lhs->y1, rhs->y1);

  const float intersection = max (0.0f, ix) * max (0.0f, iy);
  return intersection / (area (lhs) + area (rhs) - intersection);
}

__kernel void calculate_ious(__global const box_xyxy *boxes, __global const float *scores, __global const int *classes,
                             __global char *suppressed, int count, float threshold, int agnostic, int max_items) {

  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < count && y < count) {
    bool iou = (agnostic || classes[x] == classes[y]) &&
                IntersectionOverUnion (&boxes[x], &boxes[y]) > threshold;
    bool suppress = scores[x] < scores[y];
    suppressed[y * max_items + x] =  iou && suppress;
  }
}

__kernel void determine_suppressed(__global const char *ious, __global char *suppressed, int count, int max_items) {
  int x = get_global_id(0);

  if (x < count) {
    bool suppress = false;
    for (int i = 0; i != count && !suppress; ++i) {
      suppress = suppress || ious[i * max_items + x];
    }
    suppressed[x] = !suppress;
  }
}
)###";

CLNms::CLNms(int max_size, Ax::Logger &logger)
    : program(KernelSource, logger),
      error(CL_SUCCESS), boxes{ program.create_buffer(sizeof(box_xyxy),
                             max_size, CL_MEM_READ_ONLY, (void *) nullptr) },
      scores{ program.create_buffer(
          sizeof(float), max_size, CL_MEM_READ_ONLY, (void *) nullptr) },
      classes{ program.create_buffer(
          sizeof(int), max_size, CL_MEM_READ_ONLY, (void *) nullptr) },
      ious{ program.create_buffer(sizeof(char), max_size * max_size,
          CL_MEM_READ_WRITE, (void *) nullptr) },
      suppressed{ program.create_buffer(
          sizeof(char), max_size, CL_MEM_WRITE_ONLY, (void *) nullptr) },
      map_kernel{ program.get_kernel("calculate_ious") }, reduce_kernel{ program.get_kernel(
                                                              "determine_suppressed") },
      max_size_(max_size)
{
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Unable to create OpenCL kernels.");
  }
}

AxMetaObjDetection
topk(const AxMetaObjDetection &predictions, int topk)
{
  if (predictions.num_elements() <= topk) {
    return predictions;
  }
  std::vector<int> indices(predictions.num_elements());
  std::iota(std::begin(indices), std::end(indices), 0);
  //  Note: We use nth_element here rather than partial_sort because we do not
  //  care about order of the elements that are in the topk and K will typically
  //  be quite large and generally, nth_element is the more efficient in this
  //  case. When K is small, partial_sort is usually more efficient.
  std::nth_element(std::begin(indices), std::next(std::begin(indices), topk),
      std::end(indices), [&predictions](int i, int j) {
        return predictions.score(i) > predictions.score(j);
      });
  indices.resize(topk);
  auto boxes = std::vector<box_xyxy>{};
  auto scores = std::vector<float>{};
  auto classes = std::vector<int>{};
  for (auto idx : indices) {
    boxes.push_back(predictions.get_box_xyxy(idx));
    scores.push_back(predictions.score(idx));
    classes.push_back(predictions.class_id(idx));
  }
  return AxMetaObjDetection(std::move(boxes), std::move(scores), std::move(classes));
}


AxMetaObjDetection
CLNms::run(const AxMetaObjDetection &meta, float threshold, int class_agnostic,
    int max_size, int &error)
{
  size_t count = meta.num_elements();
  if (count == 0) {
    return meta;
  }
  program.set_kernel_args(map_kernel, 0, *boxes, *scores, *classes, *ious,
      count, threshold, class_agnostic, max_size_);
  program.set_kernel_args(reduce_kernel, 0, *ious, *suppressed, count, max_size_);


  error = error == CL_SUCCESS ? program.write_buffer(
              boxes, sizeof(box_xyxy), count, meta.get_boxes_data()) :
                                error;
  error = error == CL_SUCCESS ?
              program.write_buffer(scores, sizeof(float), count, meta.get_score_data()) :
              error;
  if (error == CL_SUCCESS && !class_agnostic && meta.is_multi_class()) {
    error = program.write_buffer(classes, sizeof(int), count, meta.get_classes_data());
  }
  size_t dispatch_dims[] = { count, count, 1 };
  error = error == CL_SUCCESS ? program.execute_kernel(map_kernel, 2, dispatch_dims) : error;
  error = error == CL_SUCCESS ? program.execute_kernel(reduce_kernel, 1, dispatch_dims) : error;

  std::vector<char> results(count);
  error = error == CL_SUCCESS ?
              program.read_buffer(suppressed, sizeof(char), count, results.data()) :
              error;

  auto filtered = error == CL_SUCCESS ? remove_suppressed(meta, results) : meta;
  return topk(filtered, max_size);
}


AxMetaObjDetection
CLNms::remove_suppressed(const AxMetaObjDetection &meta, const std::vector<char> &keep)
{
  auto boxes = std::vector<box_xyxy>{};
  auto scores = std::vector<float>{};
  auto classes = std::vector<int>{};
  boxes.reserve(keep.size());
  scores.reserve(keep.size());
  classes.reserve(keep.size());

  for (int i = 0; i != keep.size(); ++i) {
    if (keep[i]) {
      boxes.push_back(meta.get_box_xyxy(i));
      scores.push_back(meta.score(i));
      if (meta.is_multi_class()) {
        classes.push_back(meta.class_id(i));
      }
    }
  }
  return AxMetaObjDetection(std::move(boxes), std::move(scores), std::move(classes));
}

#endif
