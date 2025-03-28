#include <unordered_map>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaClassification.hpp"
#include "AxOpUtils.hpp"

extern "C" void
decode_to_meta(const AxTensorsInterface &tensors, const void *, unsigned int,
    unsigned int, std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &, Ax::Logger &)
{
  auto &tensor = tensors[0];
  size_t total = tensor.total();
  auto *data = static_cast<const float *>(tensor.data);
  auto max_itr = std::max_element(data, data + total);
  auto max_score = *max_itr;
  int max_class_id = std::distance(data, max_itr);

  ax_utils::insert_meta<AxMetaClassification>(map, "classifier", {}, 0, 1,
      std::vector<std::vector<float>>{ { max_score } },
      std::vector<std::vector<int32_t>>{ { max_class_id } },
      std::vector<std::vector<std::string>>{ { std::to_string(max_class_id) } });
}
