// Copyright Axelera AI, 2025
#pragma once

#include <vector>

#include <AxMeta.hpp>
#include <AxUtils.hpp>
#include "AxMetaBBox.hpp"
#include "AxMetaKpts.hpp"
#include "AxMetaSegments.hpp"

class AxMetaPoseSegmentsDetection : public AxMetaBbox, public AxMetaKpts, public AxMetaSegments
{
  public:
  AxMetaPoseSegmentsDetection(std::vector<box_xyxy> boxes, KptXyvVector kpts,
      std::vector<ax_utils::segment> segments, std::vector<float> scores,
      std::vector<int> classes, std::vector<int> ids, const SegmentShape &segment_shape,
      std::vector<int> _kpts_shape, box_xyxy mbox, const std::string &decoder_name_ = "")
      : AxMetaBbox(std::move(boxes), std::move(scores), std::move(classes), std::move(ids)),
        AxMetaKpts(std::move(kpts)),
        AxMetaSegments(segment_shape.width, segment_shape.height, std::move(segments)),
        kpts_shape(_kpts_shape),
        base_box(std::move(mbox)),
        decoder_name(decoder_name_)
  {
  }

  AxMetaPoseSegmentsDetection(std::vector<box_xyxy> boxes, KptXyvVector kpts,
      std::vector<segment_details> segments_info, std::vector<float> scores,
      std::vector<int> classes, std::vector<int> ids, const SegmentShape &segment_shape,
      ax_utils::prototype_details prototype_tensor, std::vector<int> _kpts_shape,
      box_xyxy mbox, const std::string &decoder_name_ = "")
      : AxMetaBbox(std::move(boxes), std::move(scores), std::move(classes), std::move(ids)),
        AxMetaKpts(std::move(kpts)),
        AxMetaSegments(segment_shape.width, segment_shape.height, mbox,
            std::move(segments_info)),
        kpts_shape(_kpts_shape),
        base_box(std::move(mbox)),
        decoder_name(decoder_name_)
  {
    set_prototype(std::move(prototype_tensor));
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
  }

  size_t num_elements() const
  {
    return AxMetaBbox::num_elements();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *multihead_meta
        = decoder_name.size() == 0 ? "multihead_meta" : decoder_name.c_str();
    auto meta1 = AxMetaSegments::get_extern_meta();
    auto meta2 = extern_meta{ multihead_meta, "scores",
      int(scores_.size() * sizeof(float)),
      reinterpret_cast<const char *>(scores_.data()) };
    const auto &shape = AxMetaSegments::get_segments_shape();
    auto meta3 = extern_meta{ multihead_meta, "segment_shape",
      int(shape.size() * sizeof(size_t)), reinterpret_cast<const char *>(shape.data()) };
    auto meta = AxMetaBbox::get_extern_meta();
    auto meta4 = extern_meta{ multihead_meta, "classes",
      int(classes_.size() * sizeof(int)),
      reinterpret_cast<const char *>(classes_.data()) };

    auto meta5 = extern_meta{ multihead_meta, "kpts_shape",
      int(kpts_shape.size() * sizeof(int)),
      reinterpret_cast<const char *>(kpts_shape.data()) };
    auto meta6 = AxMetaKpts::get_extern_meta();
    auto meta7 = extern_meta{ multihead_meta, "base_box", int(sizeof(box_xyxy)),
      reinterpret_cast<const char *>(&base_box) };
    meta1[0].type = multihead_meta;
    meta6[0].type = multihead_meta;
    meta[0].type = multihead_meta;
    meta.push_back(meta1[0]);
    meta.push_back(meta2);
    meta.push_back(meta3);
    meta.push_back(meta4);
    meta.push_back(meta5);
    meta.push_back(meta6[0]);
    meta.push_back(meta7);
    meta1[1].type = multihead_meta;
    meta.push_back(meta1[1]);
    return meta;
  }

  const box_xyxy &get_base_box() const
  {
    return base_box;
  }

  std::vector<int> get_kpts_shape() const
  {
    return kpts_shape;
  }

  std::string get_decoder_name() const
  {
    return decoder_name;
  }

  private:
  std::vector<int> kpts_shape;
  box_xyxy base_box;
  std::string decoder_name;
};
