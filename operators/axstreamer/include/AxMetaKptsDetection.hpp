// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaKpts.hpp"
#include "AxUtils.hpp"

class AxMetaKptsDetection : public AxMetaBbox, public AxMetaKpts
{
  public:
  AxMetaKptsDetection(std::vector<box_xyxy> boxes, KptXyvVector kpts,
      std::vector<float> scores, std::vector<int> kpts_shape_,
      const std::string &decoder_name_ = "")
      : AxMetaBbox(std::move(boxes)), AxMetaKpts(std::move(kpts)),
        scoresvec(std::move(scores)), kpts_shape(kpts_shape_), decoder_name(decoder_name_)
  {
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
  }

  size_t num_elements() const
  {
    return (AxMetaBbox::num_elements() != 0) ? AxMetaBbox::num_elements() :
                                               AxMetaKpts::num_elements();
  }

  bool is_multi_class() const
  {
    return false;
  }

  int class_id(size_t idx) const
  {
    return 0;
  }

  float score(size_t idx) const
  {
    return scoresvec[idx];
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *kpts_meta
        = decoder_name.size() == 0 ? "keypoints" : decoder_name.c_str();
    auto meta1 = AxMetaKpts::get_extern_meta();
    auto meta2 = extern_meta{ kpts_meta, "scores", int(scoresvec.size() * sizeof(float)),
      reinterpret_cast<const char *>(scoresvec.data()) };
    auto meta3 = extern_meta{ kpts_meta, "kpts_shape",
      int(kpts_shape.size() * sizeof(int)),
      reinterpret_cast<const char *>(kpts_shape.data()) };

    auto meta = AxMetaBbox::get_extern_meta();
    meta1[0].type = kpts_meta;
    meta[0].type = kpts_meta;
    meta.push_back(meta1[0]);
    meta.push_back(meta2);
    meta.push_back(meta3);
    return meta;
  }

  std::vector<int> get_kpts_shape() const
  {
    return kpts_shape;
  }

  std::string get_decoder_name() const
  {
    return decoder_name;
  }
  std::vector<float> scoresvec;
  std::vector<int> kpts_shape;
  std::string decoder_name;
};
