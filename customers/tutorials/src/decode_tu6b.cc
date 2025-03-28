#include <unordered_set>
#include "AxDataInterface.h"
#include "AxMetaClassification.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "top_k",
  };
  return allowed_properties;
}

struct classification_properties {
  std::string key;
  int top_k;
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &)
{
  std::shared_ptr<classification_properties> prop
      = std::make_shared<classification_properties>();
  prop->key = Ax::get_property(input, "meta_key", "classification_properties", prop->key);
  prop->top_k = Ax::get_property(input, "top_k", "classification_properties", prop->top_k);
  return prop;
}


extern "C" void
decode_to_meta(const AxTensorsInterface &tensors,
    const classification_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &, Ax::Logger &)
{
  auto &tensor = tensors[0];
  size_t total = tensor.total();
  auto top_k = std::min(std::size_t(prop->top_k), total);

  auto *mat_data = static_cast<const float *>(tensor.data);
  std::vector<float> mat_copy(total);
  float max_for_shift = *std::max_element(mat_data, mat_data + total);
  std::transform(mat_data, mat_data + total, mat_copy.begin(),
      [max_for_shift](float a) { return std::exp(a - max_for_shift); });
  float denominator = std::accumulate(mat_copy.begin(), mat_copy.end(), 0.0);
  std::transform(mat_copy.begin(), mat_copy.end(), mat_copy.begin(),
      [denominator](float a) { return a / denominator; });
  mat_data = mat_copy.data();

  std::vector<int> indices(total);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), std::next(indices.begin(), top_k), indices.end(),
      [mat_data](int i, int j) { return mat_data[i] > mat_data[j]; });
  indices.resize(top_k);

  std::vector<float> scores;
  std::vector<std::string> labels;
  for (auto idx : indices) {
    scores.push_back(mat_data[idx]);
    labels.push_back(std::to_string(idx));
  }

  ax_utils::insert_meta<AxMetaClassification>(map, prop->key, {}, 0, 1,
      std::vector<std::vector<float>>{ scores },
      std::vector<std::vector<int32_t>>{ indices },
      std::vector<std::vector<std::string>>{ labels });
}
