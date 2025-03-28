#include <unordered_set>
#include "AxDataInterface.h"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"
#include "MyCppClassificationMeta_tu7.h"

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "num_classes",
  };
  return allowed_properties;
}

struct classification_properties {
  std::string key;
  int num_classes;
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &)
{
  std::shared_ptr<classification_properties> prop
      = std::make_shared<classification_properties>();
  prop->key = Ax::get_property(input, "meta_key", "classification_properties", prop->key);
  prop->num_classes = Ax::get_property(
      input, "num_classes", "classification_properties", prop->num_classes);
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
  auto *data = static_cast<const float *>(tensor.data);
  auto max_itr = std::max_element(data, data + total);
  auto max_score = *max_itr;
  int max_class_id = std::distance(data, max_itr);

  ax_utils::insert_meta<MyCppClassificationMeta>(map, prop->key, {}, 0, 1,
      std::vector<float>{ max_score }, std::vector<int32_t>{ max_class_id },
      prop->num_classes);
}
