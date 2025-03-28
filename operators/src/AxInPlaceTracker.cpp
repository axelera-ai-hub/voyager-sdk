// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaStreamId.hpp"
#include "AxMetaTracker.hpp"
#include "TrackerFactory.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_set>

struct PerStreamTracker {
  std::unique_ptr<ax::MultiObjTracker> tracker;
  std::unordered_map<int, TrackingDescriptor> static_tracker_id_to_tracking_descriptor;

  PerStreamTracker(std::string algorithm, TrackerParams algo_params)
      : tracker(CreateMultiObjTracker(algorithm, algo_params))
  {
  }
};

TrackerParams
DeserializeToTrackerParams(const nlohmann::json &j)
{
  TrackerParams params;

  for (const auto &el : j.items()) {
    if (el.value().is_boolean()) {
      params[el.key()] = el.value().get<bool>();
    } else if (el.value().is_number_integer()) {
      params[el.key()] = el.value().get<int>();
    } else if (el.value().is_number_float()) {
      params[el.key()] = el.value().get<float>();
    } else if (el.value().is_string()) {
      params[el.key()] = el.value().get<std::string>();
    } else {
      throw std::runtime_error("Unsupported type in JSON");
    }
  }
  return params;
}

struct tracker_properties {
  std::string meta_key{};
  std::string streamid_meta_key{ "stream_id" };
  size_t history_length{ 1 };
  std::string algorithm{ "oc-sort" };
  TrackerParams algo_params{};
  size_t num_subtask_runs{ 0 };
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "meta_key", "streamid_meta_key",
    "history_length", "algorithm", "algo_params_json", "num_subtask_runs" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<tracker_properties>();
  prop->meta_key = Ax::get_property(input, "meta_key", "tracker_properties", prop->meta_key);
  prop->streamid_meta_key = Ax::get_property(
      input, "streamid_meta_key", "tracker_properties", prop->streamid_meta_key);
  prop->history_length = Ax::get_property(
      input, "history_length", "tracker_properties", prop->history_length);
  prop->num_subtask_runs = Ax::get_property(
      input, "num_subtask_runs", "tracker_properties", prop->num_subtask_runs);
  prop->algorithm
      = Ax::get_property(input, "algorithm", "tracker_properties", prop->algorithm);

  auto filename = Ax::get_property(
      input, "algo_params_json", "tracker_properties", std::string{});
  if (!filename.empty()) {
    std::ifstream file(filename);
    if (!file) {
      logger(AX_ERROR) << "tracker_properties : algo_params_json not found" << std::endl;
      throw std::runtime_error("tracker_properties : algo_params_json not found");
    }
    nlohmann::json j;
    file >> j;
    prop->algo_params = DeserializeToTrackerParams(j);
  }
  return prop;
}

extern "C" void
inplace(const AxDataInterface &data, const tracker_properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  // Static map to hold Tracker Records for each stream_id
  static std::unordered_map<int, PerStreamTracker> stream_tracker_map;

  int stream_id = 0;
  if (!prop->streamid_meta_key.empty()) {
    auto stream_id_itr = map.find(prop->streamid_meta_key);
    if (stream_id_itr != map.end()) {
      auto stream_id_meta = dynamic_cast<AxMetaStreamId *>(stream_id_itr->second.get());
      if (stream_id_meta == nullptr) {
        throw std::runtime_error("inplace_tracker: streamid_meta_key not of type AxMetaStreamId");
      }
      stream_id = stream_id_meta->stream_id;
    }
  }

  if (!std::holds_alternative<AxVideoInterface>(data)) {
    throw std::runtime_error("Tracker requires video info");
  }
  auto &video_info = std::get<AxVideoInterface>(data).info;

  if (prop->meta_key.empty()) {
    logger(AX_ERROR) << "inplace_tracker: meta_key not provided" << std::endl;
    throw std::runtime_error("inplace_tracker: meta_key not provided");
  }
  auto itr = map.find(prop->meta_key);
  if (itr == map.end()) {
    logger(AX_ERROR) << "inplace_tracker: meta_key not found" << std::endl;
    throw std::runtime_error("inplace_tracker: meta_key not found");
  }

  auto tracker_storage
      = std::make_unique<AxMetaTracker>(prop->history_length, prop->num_subtask_runs);
  auto *tracker_meta = tracker_storage.get();
  tracker_meta->submeta_map->insert(prop->meta_key, subframe_index,
      number_of_subframes, std::move(itr->second));
  itr->second = std::move(tracker_storage);

  const auto *box_meta = dynamic_cast<const AxMetaBbox *>(
      tracker_meta->submeta_map->get(prop->meta_key, subframe_index, number_of_subframes));
  const auto *det_meta = dynamic_cast<const AxMetaObjDetection *>(box_meta);
  const auto *kpts_meta = dynamic_cast<const AxMetaKptsDetection *>(box_meta);
  if (det_meta == nullptr && kpts_meta == nullptr) {
    logger(AX_ERROR) << "inplace_tracker: meta_key not of type AxMetaObjDetection or AxMetaKptsDetection"
                     << std::endl;
    throw std::runtime_error(
        "inplace_tracker: meta_key not of type AxMetaObjDetection or AxMetaKptsDetection");
  }

  std::vector<ax::ObservedObject> convertedDetections(box_meta->num_elements());
  for (int i = 0; i < box_meta->num_elements(); ++i) {
    const auto &[x1, y1, x2, y2] = box_meta->get_box_xyxy(i);
    auto &det = convertedDetections[i];
    det.bbox.x1 = x1 / static_cast<float>(video_info.width);
    det.bbox.y1 = y1 / static_cast<float>(video_info.height);
    det.bbox.x2 = x2 / static_cast<float>(video_info.width);
    det.bbox.y2 = y2 / static_cast<float>(video_info.height);
    if (det_meta) {
      det.class_id = det_meta->class_id(i);
      det.score = det_meta->score(i);
    } else {
      det.class_id = kpts_meta->class_id(i);
      det.score = kpts_meta->score(i);
    }
  }

  auto &per_stream_tracker = stream_tracker_map
                                 .try_emplace(stream_id, prop->algorithm, prop->algo_params)
                                 .first->second;

  tracker_meta->track_object(per_stream_tracker.tracker->Update(convertedDetections),
      per_stream_tracker.static_tracker_id_to_tracking_descriptor,
      video_info.width, video_info.height);

  tracker_meta->create_box_meta(video_info.width, video_info.height);
}
