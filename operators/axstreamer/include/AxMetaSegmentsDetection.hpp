// Copyright Axelera AI, 2025
#pragma once

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaSegments.hpp"
#include "AxUtils.hpp"

struct SegmentShape {
  size_t width;
  size_t height;
};

class AxMetaSegmentsDetection : public AxMetaBbox, public AxMetaSegments
{
  public:
  AxMetaSegmentsDetection(std::vector<box_xyxy> boxes,
      std::vector<ax_utils::segment> segments, std::vector<float> scores,
      std::vector<int> ids, const SegmentShape &segment_shape, box_xyxy mbox,
      const std::string &decoder_name_ = "")
      : AxMetaBbox(std::move(boxes)),
        AxMetaSegments(segment_shape.width, segment_shape.height, std::move(segments)),
        scoresvec(std::move(scores)), class_ids(ids), base_box(std::move(mbox)),
        decoder_name(decoder_name_)
  {
  }

  AxMetaSegmentsDetection(std::vector<box_xyxy> boxes, std::vector<segment_func> segments_funcs,
      std::vector<float> scores, std::vector<int> ids,
      const SegmentShape &segment_shape, std::vector<float> prototype_tensor,
      box_xyxy mbox, const std::string &decoder_name_ = "")
      : AxMetaBbox(std::move(boxes)),
        AxMetaSegments(segment_shape.width, segment_shape.height, std::move(segments_funcs)),
        scoresvec(std::move(scores)), class_ids(ids), base_box(std::move(mbox)),
        decoder_name(decoder_name_)
  {
    set_prototype_tensor(std::move(prototype_tensor));
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
  }

  size_t num_elements() const
  {
    return AxMetaBbox::num_elements();
  }

  bool is_multi_class() const
  {
    return !class_ids.empty();
  }

  int class_id(size_t idx) const
  {
    // TODO: check out of bounds
    return class_ids[idx];
  }

  float score(size_t idx) const
  {
    return scoresvec[idx];
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *segment_meta
        = decoder_name.size() == 0 ? "segments" : decoder_name.c_str();
    auto meta1 = AxMetaSegments::get_extern_meta();
    auto meta2 = extern_meta{ segment_meta, "scores",
      int(scoresvec.size() * sizeof(float)),
      reinterpret_cast<const char *>(scoresvec.data()) };
    const auto &shape = AxMetaSegments::get_segments_shape();
    auto meta3 = extern_meta{ segment_meta, "segment_shape",
      int(shape.size() * sizeof(size_t)), reinterpret_cast<const char *>(shape.data()) };
    auto meta = AxMetaBbox::get_extern_meta();
    auto meta4 = extern_meta{ segment_meta, "classes",
      int(class_ids.size() * sizeof(int)),
      reinterpret_cast<const char *>(class_ids.data()) };
    meta1[0].type = segment_meta;
    meta[0].type = segment_meta;
    meta.push_back(meta1[0]);
    meta.push_back(meta2);
    meta.push_back(meta3);
    meta.push_back(meta4);

    auto meta5 = extern_meta{ segment_meta, "base_box", int(sizeof(box_xyxy)),
      reinterpret_cast<const char *>(&base_box) };
    meta.push_back(meta5);

    if (meta1.size() == 2) {
      meta1[1].type = segment_meta;
      meta.push_back(meta1[1]);
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
  std::vector<float> scoresvec;
  std::vector<int> class_ids;
  box_xyxy base_box;
  std::string decoder_name;
};
