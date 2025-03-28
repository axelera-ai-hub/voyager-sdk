// Copyright Axelera AI, 2023
#include <unordered_set>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaLandmarks.hpp"
#include "AxUtils.hpp"

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "meta_key", "box_meta" };
  return allowed_properties;
}

struct landmarks_properties {
  std::string meta_key
      = "meta_" + std::to_string(reinterpret_cast<long long unsigned int>(this));
  std::string boxes_meta{};
};


extern "C" std::shared_ptr<void>
init_and_set_static_properties(const Ax::properties &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<landmarks_properties>();

  prop->meta_key = Ax::get_property(
      input, "meta_key", "landmarks_static_properties", prop->meta_key);

  prop->boxes_meta = Ax::get_property(
      input, "box_meta", "landmarks_static_properties", prop->boxes_meta);

  return prop;
}


extern "C" void
decode_to_meta(const AxTensorsInterface &tensors, const landmarks_properties *prop,
    unsigned int subframe_index, unsigned int subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &, Ax::Logger &logger)
{
  const std::string &boxes_meta_string = prop->boxes_meta;
  const std::string &landmarks_meta_string = prop->meta_key;

  //  In a multi-threaded environment, the map is not thread-safe.
  //  We might need some locking around this if we want to use it in a multi-threaded environment.
  if (map.count(landmarks_meta_string) == 0) {
    auto ptr = std::make_unique<AxMetaLandmarks>(subframe_number);
    map[landmarks_meta_string] = std::move(ptr);
  }

  auto &landmarks_meta = static_cast<AxMetaLandmarks &>(*map[landmarks_meta_string]);
  auto found = map.find(boxes_meta_string);
  if (found == std::end(map)) {
    throw std::runtime_error("decode_faciallandmarks has not been provided with AxMetaBbox");
  }

  const auto &boxes_meta = dynamic_cast<const AxMetaBbox &>(*found->second);
  if (boxes_meta.num_elements() < subframe_number) {
    throw std::runtime_error("More landmark detections provided than bounding boxes");
  }
  const float *input_data = static_cast<float *>(tensors[0].data);

  const auto &[x1, y1, x2, y2] = boxes_meta.get_box_xyxy(subframe_index);

  for (int idx = 0; idx < AxMetaLandmarks::total_landmarks; ++idx) {
    auto in_x = input_data[idx * 2];
    auto in_y = input_data[idx * 2 + 1];

    int lm_x = (in_x + 0.5) * (x2 - x1) + x1;
    int lm_y = (in_y + 0.5) * (y2 - y1) + y1;

    landmarks_meta.all_landmarks[subframe_index][idx].x = lm_x;
    landmarks_meta.all_landmarks[subframe_index][idx].y = lm_y;
  }
}
