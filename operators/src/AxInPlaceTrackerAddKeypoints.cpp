// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaTracker.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <unordered_set>

struct trackeraddkeypoints_properties {
  std::string tracking_meta_key;
  std::string keypoints_submeta_key;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "tracking_meta_key",
    "keypoints_submeta_key" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<trackeraddkeypoints_properties>();
  prop->tracking_meta_key = Ax::get_property(input, "tracking_meta_key",
      "trackeraddkeypoints_properties", prop->tracking_meta_key);
  prop->keypoints_submeta_key = Ax::get_property(input, "keypoints_submeta_key",
      "trackeraddkeypoints_properties", prop->keypoints_submeta_key);
  return prop;
}

extern "C" void
inplace(const AxDataInterface &, const trackeraddkeypoints_properties *prop,
    unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  if (prop->keypoints_submeta_key.empty()) {
    throw std::runtime_error("trackeraddkeypoints: keypoints submeta key not provided");
  }
  auto *tracker_meta = ax_utils::get_meta<AxMetaTracker>(
      prop->tracking_meta_key, map, "trackeraddkeypoints");
  auto submetas = tracker_meta->submeta_map->get(prop->keypoints_submeta_key);
  if (submetas.empty()) {
    return;
  }
  AxMetaKptsDetection *kpts_meta = dynamic_cast<AxMetaKptsDetection *>(submetas[0]);
  if (kpts_meta == nullptr) {
    throw std::runtime_error(
        "trackeraddkeypoints: keypoints submeta not of type AxMetaKptsDetection");
  }
  int kpts_per_box = kpts_meta->get_kpts_shape()[0];

  for (auto &[track_id, tracking_descriptor] : tracker_meta->track_id_to_tracking_descriptor) {
    if (tracking_descriptor.detection_meta_id < 0) {
      continue;
    }
    if (tracking_descriptor.detection_meta_id >= kpts_meta->num_elements()) {
      throw std::runtime_error("trackeraddkeypoints: detection_meta_id out of bounds");
    }
    KptXyvVector kpts = kpts_meta->get_kpts_xyv(
        tracking_descriptor.detection_meta_id * kpts_per_box, kpts_per_box);
    float score = kpts_meta->score(tracking_descriptor.detection_meta_id);

    tracking_descriptor.collection->set_frame_data_map(tracking_descriptor.frame_id,
        prop->keypoints_submeta_key,
        std::make_shared<AxMetaKptsDetection>(std::vector<box_xyxy>(1),
            std::move(kpts), std::vector<float>{ score },
            std::vector<int>{ 17, 3 }, "CocoBodyKeypointsMeta"));
  }
}
