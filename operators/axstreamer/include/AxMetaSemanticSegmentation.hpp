// Copyright Axelera AI, 2025
#pragma once

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaSegments.hpp"
#include "AxUtils.hpp"

class AxMetaSemanticSegmentation : public AxMetaBase
{
  public:
  AxMetaSemanticSegmentation(std::vector<int> _class_map,
      std::vector<int> data_shape, const std::string &decoder)
      : class_map(std::move(_class_map)), shape(data_shape), decoder_name(decoder)
  {
  }

  AxMetaSemanticSegmentation(std::vector<float> _probabilities,
      std::vector<int> data_shape, const std::string &decoder)
      : probabilities(std::move(_probabilities)), shape(data_shape),
        decoder_name(decoder)
  {
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
  }
  std::vector<extern_meta> get_extern_meta() const override
  {

    std::vector<extern_meta> metas;
    const char *segment_meta
        = decoder_name.size() == 0 ? "segments" : decoder_name.c_str();

    metas.emplace_back(extern_meta{ segment_meta, "data_shape",
        int(shape.size() * sizeof(int)), reinterpret_cast<const char *>(shape.data()) });


    if (!probabilities.empty()) {
      metas.emplace_back(extern_meta{ segment_meta, "segment_probabilities",
          int(probabilities.size() * sizeof(float)),
          reinterpret_cast<const char *>(probabilities.data()) });
    } else if (!class_map.empty()) {
      metas.emplace_back(extern_meta{ segment_meta, "segment_classes",
          int(class_map.size() * sizeof(int)),
          reinterpret_cast<const char *>(class_map.data()) });
    } else {
      // Error
    }
    return metas;
  }

  private:
  std::vector<float> probabilities;
  std::vector<int> class_map;
  const std::vector<int> shape;
  const std::string decoder_name;
};
