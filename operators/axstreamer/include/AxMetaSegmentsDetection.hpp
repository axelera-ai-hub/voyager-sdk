// Copyright Axelera AI, 2025
#pragma once

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaSegments.hpp"
#include "AxUtils.hpp"

class AxMetaSegmentsDetection : public AxMetaBbox, public AxMetaSegments
{
  public:
  AxMetaSegmentsDetection()
      : AxMetaBbox(),
        AxMetaSegments(0, 0, SegmentList{})
  {
  }

  AxMetaSegmentsDetection(std::vector<box_xyxy> boxes, std::vector<ax_utils::segment> segments,
      std::vector<float> scores, std::vector<int> classes, std::vector<int> ids,
      const SegmentShape &segment_shape, const std::string &decoder_name_ = "")
      : AxMetaBbox(std::move(boxes), std::move(scores), std::move(classes), std::move(ids)),
        AxMetaSegments(segment_shape.width, segment_shape.height, std::move(segments)),
        decoder_name(decoder_name_)
  {
  }

  AxMetaSegmentsDetection(std::vector<box_xyxy> boxes, std::vector<segment_details> segments_info,
      std::vector<float> scores, std::vector<int> classes, std::vector<int> ids,
      const SegmentShape &segment_shape, ax_utils::prototype_details prototype_tensor,
      box_xyxy mbox, const std::string &decoder_name_ = "")
      : AxMetaBbox(std::move(boxes), std::move(scores), std::move(classes), std::move(ids)),
        AxMetaSegments(segment_shape.width, segment_shape.height, mbox,
            std::move(segments_info)),
        decoder_name(decoder_name_)
  {
    set_prototype(std::move(prototype_tensor));
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
  }

  void extend(AxMetaSegmentsDetection &&other)
  {
    AxMetaBbox::extend(std::move(other));
    AxMetaSegments::extend(std::move(other));
    base_box = other.base_box;
    decoder_name = std::move(other.decoder_name);
  }

  void extend(const AxMetaSegmentsDetection &other)
  {
    AxMetaBbox::extend(other);
    AxMetaSegments::extend(other);
    base_box = other.base_box;
    decoder_name = other.decoder_name;
  }

  using AxMetaBbox::num_elements;

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *segment_meta
        = decoder_name.size() == 0 ? "segments" : decoder_name.c_str();
    auto meta1 = AxMetaSegments::get_extern_meta();
    auto meta2 = extern_meta{ segment_meta, "scores", int(scores_.size() * sizeof(float)),
      reinterpret_cast<const char *>(scores_.data()) };
    const auto &shape = AxMetaSegments::get_segments_shape();
    auto meta3 = extern_meta{ segment_meta, "segment_shape",
      int(shape.size() * sizeof(size_t)), reinterpret_cast<const char *>(shape.data()) };
    auto meta = AxMetaBbox::get_extern_meta();
    auto meta4 = extern_meta{ segment_meta, "classes", int(classes_.size() * sizeof(int)),
      reinterpret_cast<const char *>(classes_.data()) };
    meta1[0].type = segment_meta;
    meta[0].type = segment_meta;
    meta.push_back(meta1[0]);
    meta.push_back(meta2);
    meta.push_back(meta3);
    meta.push_back(meta4);

    if (meta1.size() == 3) {
      meta1[1].type = segment_meta;
      meta1[2].type = segment_meta;
      meta.push_back(meta1[1]);
      meta.push_back(meta1[2]);
    }
    return meta;
  }

  const box_xyxy &get_base_box() const
  {
    return base_box;
  }

  std::string get_decoder_name() const
  {
    return decoder_name;
  }

  private:
  box_xyxy base_box;
  std::string decoder_name;
};
