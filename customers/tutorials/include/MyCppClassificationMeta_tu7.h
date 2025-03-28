#pragma once

#include "AxMeta.hpp"


class MyCppClassificationMeta : public AxMetaBase
{
  public:
  MyCppClassificationMeta(std::vector<float> scores, std::vector<int32_t> classes, int num_classes)
      : scores(std::move(scores)), classes(std::move(classes)), num_classes{ num_classes }
  {
    if (scores.size() != classes.size()) {
      throw std::logic_error("scores and classes must have the same size");
    }
  }

  void draw(const AxVideoInterface &,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &) override
  {
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    auto results = std::vector<extern_meta>();
    results.push_back({ "MyPyClassificationMeta", "scores_subtype",
        int(scores.size() * sizeof(float)),
        reinterpret_cast<const char *>(scores.data()) });
    results.push_back({ "MyPyClassificationMeta", "classes_subtype",
        int(classes.size() * sizeof(int)),
        reinterpret_cast<const char *>(classes.data()) });
    results.push_back({ "MyPyClassificationMeta", "num_classes_subtype",
        int(sizeof(int)), reinterpret_cast<const char *>(&num_classes) });
    return results;
  }

  private:
  std::vector<float> scores;
  std::vector<int32_t> classes;
  int num_classes;
};
