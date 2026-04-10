// Copyright Axelera AI, 2025
#pragma once

#include <vector>

#include "AxMetaBBox.hpp"

class AxMetaObjDetection : public AxMetaBbox
{
  public:
  AxMetaObjDetection() = default;
  AxMetaObjDetection(std::vector<box_xyxy> boxes, std::vector<float> scores,
      std::vector<int> class_ids, std::vector<int> ids = {})
      : AxMetaBbox(std::move(boxes), std::move(scores), std::move(class_ids), std::move(ids))
  {
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *object_meta = "ObjectDetectionMeta";
    auto meta1 = extern_meta{ object_meta, "scores", int(scores_size() * sizeof(float)),
      reinterpret_cast<const char *>(scores_data()) };
    auto meta2 = extern_meta{ object_meta, "classes", int(classes_size() * sizeof(int)),
      reinterpret_cast<const char *>(classes_data()) };

    auto meta = AxMetaBbox::get_extern_meta();
    meta[0].type = object_meta;
    meta.push_back(meta1);
    meta.push_back(meta2);
    return meta;
  }

  const float *get_score_data() const
  {
    return scores_data();
  }

  const int *get_classes_data() const
  {
    return classes_data();
  }
};

class AxMetaObjDetectionOBB : public AxMetaBboxXYWHR
{
  public:
  AxMetaObjDetectionOBB() = default;
  AxMetaObjDetectionOBB(std::vector<box_xywhr> boxes, std::vector<float> scores,
      std::vector<int> class_ids, std::vector<int> ids = {})
      : AxMetaBboxXYWHR(std::move(boxes), std::move(scores),
            std::move(class_ids), std::move(ids))
  {
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *object_meta = "ObjectDetectionMetaOBB";
    auto meta1 = extern_meta{ object_meta, "scores", int(scores_size() * sizeof(float)),
      reinterpret_cast<const char *>(scores_data()) };
    auto meta2 = extern_meta{ object_meta, "classes", int(classes_size() * sizeof(int)),
      reinterpret_cast<const char *>(classes_data()) };

    auto meta = AxMetaBboxXYWHR::get_extern_meta();
    meta[0].type = object_meta;
    meta[1].type = object_meta;
    meta.push_back(meta1);
    meta.push_back(meta2);
    return meta;
  }

  const float *get_score_data() const
  {
    return scores_data();
  }

  const int *get_classes_data() const
  {
    return classes_data();
  }
};

class AxMetaObjDetectionTiles : public AxMetaObjDetection
{
  public:
  AxMetaObjDetectionTiles() = default;
  AxMetaObjDetectionTiles(std::vector<box_xyxy> boxes, std::vector<float> scores,
      std::vector<int> class_ids, std::vector<int> ids = {})
      : AxMetaObjDetection(std::move(boxes), std::move(scores),
            std::move(class_ids), std::move(ids))
  {
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *object_meta = "ObjectDetectionMetaTiles";
    auto meta1 = extern_meta{ object_meta, "scores", int(scores_size() * sizeof(float)),
      reinterpret_cast<const char *>(scores_data()) };
    auto meta2 = extern_meta{ object_meta, "classes", int(classes_size() * sizeof(int)),
      reinterpret_cast<const char *>(classes_data()) };

    auto meta = AxMetaBbox::get_extern_meta();
    meta[0].type = object_meta;
    meta.push_back(meta1);
    meta.push_back(meta2);
    return meta;
  }
};
