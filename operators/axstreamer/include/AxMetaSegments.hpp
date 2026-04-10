// Copyright Axelera AI, 2024
#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

using Segment = ax_utils::segment;
using segment_details = ax_utils::segment_details;
using SegmentList = std::vector<Segment>;
struct SegmentShape {
  size_t width;
  size_t height;
};

inline ax_utils::segment
decode_segment(const segment_details &details, const ax_utils::prototype_details &prototype,
    size_t out_width, size_t out_height, const box_xyxy &base_box)
{
  if (!prototype.coefs) {
    throw std::runtime_error("invalid prototype tensor");
  }

  // Calculate the scale to fit original image into prototype while preserving aspect ratio
  auto scale = std::min(static_cast<float>(prototype.width) / out_width,
      static_cast<float>(prototype.height) / out_height);

  auto scaled_width = out_width * scale;
  auto scaled_height = out_height * scale;

  // Calculate letterbox/pillarbox offsets
  int xmin = std::round((prototype.width - scaled_width) / 2.0);
  int xmax = prototype.width - xmin;
  int ymin = std::round((prototype.height - scaled_height) / 2.0);
  int ymax = prototype.height - ymin;

  // Use max dimension for scaling since coordinates were normalized using max(width, height)
  const auto proto_scale
      = static_cast<float>(std::max(prototype.width, prototype.height));
  auto bbox = std::array{
    std::clamp(static_cast<int>(std::round(details.x1 * proto_scale)), xmin, xmax),
    std::clamp(static_cast<int>(std::round(details.y1 * proto_scale)), ymin, ymax),
    std::clamp(static_cast<int>(std::round(details.x2 * proto_scale)), xmin, xmax),
    std::clamp(static_cast<int>(std::round(details.y2 * proto_scale)), ymin, ymax)
  };

  const auto row_offset = prototype.width * prototype.depth;
  const auto segment_map_size = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]);
  std::vector<uint8_t> segment_map(segment_map_size);
  auto idx = 0;
  for (int sy = bbox[1]; sy < bbox[3]; ++sy) {
    for (int sx = bbox[0]; sx < bbox[2]; ++sx) {
      auto *proto = prototype.coefs.get() + sy * row_offset + sx * prototype.depth;
      const auto dot = std::transform_reduce(details.mask_data.begin(),
          details.mask_data.end(), proto, 0.0F, std::plus<>(),
          [&details, &prototype](int8_t val, int8_t p) {
            auto v = prototype.scale * (p - prototype.zero);
            auto x = details.scale * (val - details.zero);
            return v * x;
          });
      // Note: Sigmoid is NOT applied to mask values - it's only applied to class scores
      // The mask is the result of (mask_coef @ prototype), directly clipped to [0, 255]
      segment_map[idx++] = std::clamp(dot * 255.0F, 0.0F, 255.0F);
    }
  }
  return ax_utils::segment{ base_box.x1, base_box.y1, base_box.x2, base_box.y2,
    bbox[0], bbox[1], bbox[2], bbox[3], std::move(segment_map) };
}


class AxMetaSegments : public virtual AxMetaBase
{
  public:
  AxMetaSegments(size_t w, size_t h, SegmentList segments)
      : width(w),
        height(h),
        segmentlist(std::move(segments))
  {
  }

  AxMetaSegments(size_t w, size_t h, BboxXyxy base_box_, std::vector<segment_details> seg_info)
      : width(w),
        height(h),
        base_box(std::move(base_box_)),
        segment_info(std::move(seg_info))
  {
  }

  std::vector<uint8_t> get_segment_map(size_t idx)
  {
    if (segmentlist.empty()) {
      return decode_segment(segment_info[idx], prototype_tensor, width, height, base_box)
          .map;
    }

    if (idx >= segmentlist.size()) {
      throw std::out_of_range("Segment Index out of range");
    }
    return segmentlist[idx].map;
  }

  Segment get_segment(size_t idx) const
  {
    if (segmentlist.empty()) {
      auto seg = decode_segment(segment_info[idx], prototype_tensor, width, height, base_box);
      return Segment{ base_box.x1, base_box.y1, base_box.x2, base_box.y2,
        seg.x1, seg.y1, seg.x2, seg.y2, std::move(seg.map) };
    }

    if (idx >= segmentlist.size()) {
      throw std::out_of_range("Segment Index out of range");
    }
    return segmentlist[idx];
  }
  void set_prototype(ax_utils::prototype_details tensor)
  {
    prototype_tensor = std::move(tensor);
  }

  size_t get_segments_count() const
  {
    return segmentlist.empty() ? segment_info.size() : segmentlist.size();
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
    segment_shape = { get_segments_count(), height, width };
    return segment_shape;
  }

  void extend(const AxMetaSegments &other)
  {
    width = other.width;
    height = other.height;
    segmentlist.insert(
        segmentlist.end(), other.segmentlist.begin(), other.segmentlist.end());
    segment_info.insert(segment_info.end(), other.segment_info.begin(),
        other.segment_info.end());
    segment_vec.clear();
  }

  void extend(AxMetaSegments &&other)
  {
    width = other.width;
    height = other.height;
    segmentlist.insert(segmentlist.end(),
        std::make_move_iterator(other.segmentlist.begin()),
        std::make_move_iterator(other.segmentlist.end()));
    segment_info.insert(segment_info.end(),
        std::make_move_iterator(other.segment_info.begin()),
        std::make_move_iterator(other.segment_info.end()));
    segment_vec.clear();
  }


  std::vector<extern_meta> get_extern_meta() const override
  {
    if (segment_vec.empty()) {
      // Check if materialization is needed but wasn't done
      if (segmentlist.empty() && !segment_info.empty()) {
        throw std::runtime_error(
            "Segment masks not materialized. Enable materialize_masks in decoder or run NMS operator");
      }

      auto seg_size = std::accumulate(segmentlist.begin(), segmentlist.end(), 0,
          [](size_t sum, const Segment &seg) { return sum + seg.map.size(); });

      segment_vec.resize(seg_size);
      bbox_vec.clear();
      base_boxes.clear();
      auto *out = segment_vec.data();
      for (const auto &seg : segmentlist) {
        std::memcpy(out, seg.map.data(), seg.map.size());
        out += seg.map.size();
        bbox_vec.insert(bbox_vec.end(), { seg.x1, seg.y1, seg.x2, seg.y2 });
        base_boxes.insert(base_boxes.end(),
            { seg.base_box_x1, seg.base_box_y1, seg.base_box_x2, seg.base_box_y2 });
      }
    }
    return { { "segments", "segment_maps",
                 static_cast<int>(segment_vec.size() * sizeof(uint8_t)),
                 reinterpret_cast<const char *>(segment_vec.data()) },
      { "segments", "segment_bboxs", static_cast<int>(bbox_vec.size() * sizeof(int)),
          reinterpret_cast<const char *>(bbox_vec.data()) },
      { "segments", "base_boxes", static_cast<int>(base_boxes.size() * sizeof(int)),
          reinterpret_cast<const char *>(base_boxes.data()) } };
  }

  private:
  size_t width;
  size_t height;
  BboxXyxy base_box;

  SegmentList segmentlist;
  std::vector<segment_details> segment_info;
  ax_utils::prototype_details prototype_tensor;
  mutable std::vector<uint8_t> segment_vec; // Cache variable from flattening segment maps
  mutable std::vector<int> bbox_vec;
  mutable std::vector<int> base_boxes;
  mutable std::vector<size_t> segment_shape; // Lazily computed cache
};
