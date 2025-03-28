// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

using Segment = ax_utils::segment;
using SegmentList = std::vector<Segment>;

using segment_func = std::function<Segment(const std::vector<float> &, size_t, size_t)>;


class AxMetaSegments : public virtual AxMetaBase
{
  public:
  AxMetaSegments(size_t w, size_t h, SegmentList segments)
      : width(w), height(h), segmentlist(std::move(segments))
  {
    segment_shape = { segmentlist.size(), h, w };
  }

  AxMetaSegments(size_t w, size_t h, std::vector<segment_func> segment_funcs)
      : width(w), height(h), tasks(std::move(segment_funcs))
  {
    segment_shape = { segmentlist.size(), h, w };
  }

  std::vector<float> get_segment_map(size_t idx)
  {
    if (segmentlist.empty()) {
      return tasks[idx](prototype_tensor, width, height).map;
    }

    if (idx >= segmentlist.size()) {
      throw std::out_of_range("Segment Index out of range");
    }
    return segmentlist[idx].map;
  }

  Segment get_segment(size_t idx)
  {
    if (segmentlist.empty()) {
      auto seg = tasks[idx](prototype_tensor, width, height);
      return Segment{ seg.x1, seg.y1, seg.x2, seg.y2, std::move(seg.map) };
    }

    if (idx >= segmentlist.size()) {
      throw std::out_of_range("Segment Index out of range");
    }
    return segmentlist[idx];
  }
  void set_prototype_tensor(const std::vector<float> &tensor)
  {
    prototype_tensor = std::move(tensor);
  }

  size_t get_segments_count() const
  {
    return segmentlist.size();
  }
  size_t get_segment_size(int idx) const
  {
    if (segmentlist.empty()) {
      return 0;
    }

    if (idx >= segmentlist.size()) {
      throw std::out_of_range("get_segment_size: Segment Index out of range");
    }
    const auto &seg = segmentlist[idx];
    const auto bbox_size = (seg.x2 - seg.x1) * (seg.y2 - seg.y1);
    if (seg.map.size() != bbox_size) {
      throw std::runtime_error("get_segment_size: internal size error"); // TODO: use asert
    }
    return bbox_size;
  }

  const std::vector<size_t> &get_segments_shape() const
  {
    return segment_shape;
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    segment_vec.clear();
    for (const auto &seg : segmentlist) {
      std::transform(seg.map.begin(), seg.map.end(), std::back_inserter(segment_vec),
          [](float value) -> uint8_t { return value * 255; });
      bbox_vec.insert(bbox_vec.end(), { seg.x1, seg.y1, seg.x2, seg.y2 });
    }
    return { { "segments", "segment_maps",
                 static_cast<int>(segment_vec.size() * sizeof(uint8_t)),
                 reinterpret_cast<const char *>(segment_vec.data()) },
      { "segments", "segment_bboxs", static_cast<int>(bbox_vec.size() * sizeof(int)),
          reinterpret_cast<const char *>(bbox_vec.data()) } };
  }

  private:
  size_t width;
  size_t height;

  SegmentList segmentlist;
  std::vector<segment_func> tasks;
  std::vector<float> prototype_tensor;
  mutable std::vector<uint8_t> segment_vec; // Cache variable from flattening segment maps
  mutable std::vector<int> bbox_vec;
  std::vector<size_t> segment_shape;
};
