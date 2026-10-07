// Copyright Axelera AI, 2025
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
    "master_meta_key",
    "association_meta_key",
  };
  return allowed_properties;
}

struct classification_properties {
  std::string key{};
  std::string master_meta_key{};
  std::string association_meta_key{};
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &)
{
  std::shared_ptr<classification_properties> prop
      = std::make_shared<classification_properties>();
  prop->key = Ax::get_property(input, "meta_key", "classification_properties", prop->key);
  prop->master_meta_key = Ax::get_property(input, "master_meta_key",
      "classification_properties", prop->master_meta_key);
  prop->association_meta_key = Ax::get_property(input, "association_meta_key",
      "classification_properties", prop->association_meta_key);
  return prop;
}


extern "C" void
decode_to_meta(const AxTensorsInterface &tensors, const classification_properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &, Ax::Logger &)
{
  auto &tensor = tensors[0];
  size_t total = tensor.total();
  auto *data = static_cast<const float *>(tensor.data);
  auto max_itr = std::max_element(data, data + total);
  auto max_score = *max_itr;
  int max_class_id = std::distance(data, max_itr);

  ax_utils::insert_and_associate_meta<AxMetaClassification>(map, prop->key,
      prop->master_meta_key, subframe_index, number_of_subframes,
      prop->association_meta_key, std::vector<std::vector<float>>{ { max_score } },
      std::vector<std::vector<int32_t>>{ { max_class_id } },
      std::vector<std::vector<std::string>>{ { std::to_string(max_class_id) } });
}
