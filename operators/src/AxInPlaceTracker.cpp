// Copyright Axelera AI, 2025
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include "AxMetaClassification.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaStreamId.hpp"
#include "AxMetaTracker.hpp"
#include "MultiObjTracker.hpp"
#include "TrackerFactory.h"
#include "cmc.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_set>


struct PerStreamTracker {
  std::unique_ptr<ax::MultiObjTracker> tracker;
  std::unordered_map<int, TrackingDescriptor> static_tracker_id_to_tracking_descriptor;

  std::unique_ptr<CMCComputer> cmc_computer;

  PerStreamTracker(std::string algorithm, TrackerParams algo_params)
      : tracker(CreateMultiObjTracker(algorithm, algo_params))
  {
    if (algo_params.contains("cmc_enabled")
        && std::get<bool>(algo_params["cmc_enabled"]) == true) {
      cmc_computer = std::make_unique<CMCComputer>();
    }
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

template <typename T>
void
insert_into_callback_map(std::string_view options_string,
    std::unordered_map<std::string, T> &callback_map, Ax::Logger &logger)
{
  auto split_options = Ax::Internal::split(options_string, "?");
  for (const auto &split_option : split_options) {
    if (split_option.empty()) {
      continue;
    }
    auto options = Ax::extract_secondary_options(logger, std::string(split_option));
    auto found_key = options.find("key");
    if (found_key == options.end()) {
      throw std::runtime_error("Callback key not found");
    }
    auto found_lib = options.find("lib");
    if (found_lib == options.end()) {
      throw std::runtime_error("Callback lib not found");
    }
    auto key = std::move(found_key->second);
    options.erase(found_key);
    auto lib = std::move(found_lib->second);
    options.erase(found_lib);
    auto returned
        = callback_map.try_emplace(std::move(key), std::move(lib), options, logger);
    if (!returned.second) {
      throw std::runtime_error("Callback key already exists");
    }
  }
}

struct tracker_properties {
  std::string tracker_meta_key{};
  std::string input_meta_key{};
  std::string output_meta_key{};
  std::string streamid_meta_key{ "stream_id" };
  std::string embeddings_meta_key{};
  size_t history_length{ 1 };
  size_t max_lost{ 20 };
  std::string algorithm{ "oc-sort" };
  TrackerParams algo_params{};
  std::unordered_map<std::string, KeepBoxCallback> keep_box_callback_map;
  std::unordered_map<std::string, DetermineObjectAttributeCallback> determine_object_attribute_callback_map;

  // Store per-stream tracker state here instead of static variable
  // This ensures tracker state is destroyed when properties are destroyed (pipeline restart)
  mutable std::unordered_map<int, PerStreamTracker> stream_tracker_map;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "tracker_meta_key",
    "input_meta_key",
    "output_meta_key",
    "streamid_meta_key",
    "embeddings_meta_key",
    "history_length", // Buffer size
    "history_grace_period", // How long to keep after object loss
    "algorithm",
    "algo_params_json",
    "filter_callbacks",
    "determine_object_attribute_callbacks",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<tracker_properties>();
  prop->tracker_meta_key = Ax::get_property(
      input, "tracker_meta_key", "tracker_properties", prop->tracker_meta_key);
  prop->input_meta_key = Ax::get_property(
      input, "input_meta_key", "tracker_properties", prop->input_meta_key);
  prop->output_meta_key = Ax::get_property(
      input, "output_meta_key", "tracker_properties", prop->output_meta_key);
  prop->streamid_meta_key = Ax::get_property(
      input, "streamid_meta_key", "tracker_properties", prop->streamid_meta_key);
  prop->embeddings_meta_key = Ax::get_property(input, "embeddings_meta_key",
      "tracker_properties", prop->embeddings_meta_key);
  prop->history_length = Ax::get_property(
      input, "history_length", "tracker_properties", prop->history_length);
  prop->max_lost = Ax::get_property(
      input, "history_grace_period", "tracker_properties", prop->max_lost);
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

  if (auto found = input.find("filter_callbacks"); found != input.end()) {
    insert_into_callback_map(found->second, prop->keep_box_callback_map, logger);
  }

  if (auto found = input.find("determine_object_attribute_callbacks");
      found != input.end()) {
    insert_into_callback_map(
        found->second, prop->determine_object_attribute_callback_map, logger);
  }

  return prop;
}

void
create_box_meta(const std::string &output_meta_key,
    const std::unordered_map<int, TrackingDescriptor> &track_id_to_tracking_descriptor,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    int width, int height, const KeepBoxCallback *callback_ptr = nullptr)
{
  BboxXyxyVector bboxvec;
  std::vector<int> ids;
  std::vector<int> class_ids;
  std::vector<float> scores;
  for (const auto &[track_id, tracking_descriptor] : track_id_to_tracking_descriptor) {
    if (callback_ptr && !(*callback_ptr)(tracking_descriptor)) {
      continue;
    }
    const auto *element
        = tracking_descriptor.collection->get_frame(tracking_descriptor.frame_id);
    if (!element) {
      throw std::logic_error("create_box_meta: No frame found for track_id "
                             + std::to_string(track_id));
    }
    const auto &bbox = element->bbox;
    if (bbox.x2 < 0 || bbox.x1 >= width || bbox.y2 < 0 || bbox.y1 >= height) {
      continue;
    }
    bboxvec.push_back({ std::max(bbox.x1, 0), std::max(bbox.y1, 0),
        std::min(bbox.x2, width - 1), std::min(bbox.y2, height - 1) });
    ids.push_back(track_id);
    class_ids.push_back(tracking_descriptor.collection->detection_class_id);
    scores.push_back(tracking_descriptor.collection->detection_score);
  }
  ax_utils::insert_meta<AxMetaObjDetection>(map, output_meta_key, "", 0, 1,
      std::move(bboxvec), std::move(scores), std::move(class_ids), std::move(ids));
  map[output_meta_key]->enable_extern = false;
}


extern "C" void
inplace(const AxDataInterface &data, const tracker_properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(data)) {
    throw std::runtime_error("inplace_tracker: buffer not of type AxVideoInterface");
  }
  auto &video_info = std::get<AxVideoInterface>(data).info;

  int stream_id = 0;
  if (!prop->streamid_meta_key.empty()) {
    auto *stream_id_meta = ax_utils::get_meta<AxMetaStreamId>(
        prop->streamid_meta_key, map, "inplace_tracker");
    stream_id = stream_id_meta->stream_id;
  }

  auto *tracker_meta = ax_utils::insert_meta<AxMetaTracker>(map, prop->tracker_meta_key,
      "", subframe_index, number_of_subframes, prop->history_length);

  auto *box_meta = ax_utils::get_meta<AxMetaBbox>(prop->input_meta_key, map, "inplace_tracker");
  const auto *det_meta = dynamic_cast<const AxMetaObjDetection *>(box_meta);
  const auto *kpts_meta = dynamic_cast<const AxMetaKptsDetection *>(box_meta);
  if (det_meta == nullptr && kpts_meta == nullptr) {
    logger(AX_ERROR) << "inplace_tracker: input_meta_key not of type AxMetaObjDetectionMeta or AxMetaKptsDetectionMeta"
                     << std::endl;
    throw std::runtime_error(
        "inplace_tracker: input_meta_key not of type AxMetaObjDetection or AxMetaKptsDetection");
  }

  std::vector<ax::ObservedObject> convertedDetections(box_meta->num_elements());
  for (int i = 0; i < box_meta->num_elements(); ++i) {
    const auto &[x1, y1, x2, y2] = box_meta->get_box_xyxy(i);
    auto &det = convertedDetections[i];
    det.bbox.x1 = x1;
    det.bbox.y1 = y1;
    det.bbox.x2 = x2;
    det.bbox.y2 = y2;
    if (det_meta) {
      det.class_id = det_meta->class_id(i);
      det.score = det_meta->score(i);
    } else {
      det.class_id = kpts_meta->class_id(i);
      det.score = kpts_meta->score(i);
    }
  }

  std::vector<std::vector<float>> embeddings{};
  if (!prop->embeddings_meta_key.empty()) {
    auto embeddings_meta
        = box_meta->get_submetas<AxMetaEmbeddings>(prop->embeddings_meta_key);
    if (embeddings_meta.size() != convertedDetections.size()) {
      throw std::runtime_error("inplace_tracker: embeddings_meta size mismatch, number of embeddings "
                               + std::to_string(embeddings_meta.size()) + " != number of detections "
                               + std::to_string(convertedDetections.size()));
    }

    for (int i = 0; i < embeddings_meta.size(); ++i) {
      auto *embedding_meta = embeddings_meta[i];
      if (!embedding_meta) {
        throw std::runtime_error("inplace_tracker: embedding_meta is nullptr for index "
                                 + std::to_string(i));
      }
      auto current_embedding = embedding_meta->get_embeddings();
      if (current_embedding.size() != 1) {
        throw std::runtime_error(
            "inplace_tracker: embedding submeta contains more than one element");
      }
      embeddings.push_back(std::move(current_embedding[0]));
    }
  }

  TrackerParams algo_params = prop->algo_params;
  if (GetParamOrDefault<bool>(algo_params, "enable_id_recovery", false)) {
    algo_params["img_width"] = static_cast<int>(video_info.width);
    algo_params["img_height"] = static_cast<int>(video_info.height);
  }

  auto &per_stream_tracker = prop->stream_tracker_map
                                 .try_emplace(stream_id, prop->algorithm, algo_params)
                                 .first->second;

  // Compute CMC transform if enabled
  std::optional<Eigen::Matrix<float, 2, 3>> cmc_transform;
  if (per_stream_tracker.cmc_computer) {
    auto &input_video = std::get<AxVideoInterface>(data);
    if (input_video.data == nullptr) {
      logger(AX_WARN) << "inplace_tracker: Video data is NULL, CMC disabled for this frame"
                      << std::endl;
    } else {
      logger(AX_DEBUG) << "inplace_tracker: Video data available, processing CMC. Format: "
                       << static_cast<int>(input_video.info.format)
                       << ", Size: " << input_video.info.width << "x"
                       << input_video.info.height << std::endl;

      cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
          Ax::opencv_type_u8(input_video.info.format), input_video.data,
          input_video.info.stride);

      cv::Mat gray_frame;
      if (video_info.format == AxVideoFormat::RGB || video_info.format == AxVideoFormat::RGBA) {
        cv::cvtColor(input_mat, gray_frame, cv::COLOR_RGB2GRAY);
      } else if (video_info.format == AxVideoFormat::BGR
                 || video_info.format == AxVideoFormat::BGRA) {
        cv::cvtColor(input_mat, gray_frame, cv::COLOR_BGR2GRAY);
      } else {
        // Assume already grayscale or use as is
        gray_frame = input_mat;
      }

      // Create bounding box matrix from detections
      Eigen::MatrixXf bbox_matrix(convertedDetections.size(), 4);
      for (size_t i = 0; i < convertedDetections.size(); ++i) {
        const auto &bbox = convertedDetections[i].bbox;
        float *row = bbox_matrix.row(i).data();
        row[0] = bbox.x1;
        row[1] = bbox.y1;
        row[2] = bbox.x2;
        row[3] = bbox.y2;
      }

      try {
        cmc_transform = per_stream_tracker.cmc_computer->compute_affine(gray_frame, bbox_matrix);

        logger(AX_INFO) << "inplace_tracker: Computed CMC for stream "
                        << stream_id << " with " << convertedDetections.size()
                        << " detections" << std::endl;

      } catch (const std::exception &e) {
        logger(AX_WARN) << "inplace_tracker: CMC computation failed: " << e.what()
                        << std::endl;
      }
    }
  }

  std::vector<ax::TrackedObject> trackers = per_stream_tracker.tracker->Update(
      convertedDetections, embeddings, cmc_transform);

  for (const auto &tracker : trackers) {
    const auto &xyxy = tracker.GetXyxy();

    auto [itr, new_entry_created]
        = per_stream_tracker.static_tracker_id_to_tracking_descriptor.try_emplace(
            tracker.track_id, tracker.track_id, tracker.class_id, tracker.score,
            tracker_meta->history_length, prop->determine_object_attribute_callback_map);

    if (!new_entry_created) {
      itr->second.lost_since = 0;
      ++itr->second.frame_id;
    }
    itr->second.detection_meta_id = tracker.latest_detection_id;
    itr->second.collection->set_frame(itr->second.frame_id,
        TrackingElement{ BboxXyxy{ std::get<0>(xyxy), std::get<1>(xyxy),
            std::get<2>(xyxy), std::get<3>(xyxy) } });

    tracker_meta->track_id_to_tracking_descriptor.try_emplace(tracker.track_id, itr->second);
  }
  for (auto itr = per_stream_tracker.static_tracker_id_to_tracking_descriptor.begin();
       itr != per_stream_tracker.static_tracker_id_to_tracking_descriptor.end();) {
    if (!tracker_meta->track_id_to_tracking_descriptor.count(itr->first)) {
      ++itr->second.frame_id;
      ++itr->second.lost_since;
      if (itr->second.lost_since > prop->max_lost) {
        itr = per_stream_tracker.static_tracker_id_to_tracking_descriptor.erase(itr);
        continue;
      }
    }
    ++itr;
  }

  if (!prop->output_meta_key.empty()) {
    create_box_meta(prop->output_meta_key, tracker_meta->track_id_to_tracking_descriptor,
        map, video_info.width, video_info.height);
  }
  for (const auto &[callback_key, callback] : prop->keep_box_callback_map) {
    std::string output_meta_key
        = prop->tracker_meta_key + "_adapted_as_input_for_" + callback_key;
    create_box_meta(output_meta_key, tracker_meta->track_id_to_tracking_descriptor,
        map, video_info.width, video_info.height, &callback);
  }

  if (kpts_meta) {
    auto kpts_shape = kpts_meta->get_kpts_shape();
    int kpts_per_box = kpts_shape[0];

    for (auto &[track_id, tracking_descriptor] : tracker_meta->track_id_to_tracking_descriptor) {
      if (tracking_descriptor.detection_meta_id < 0) {
        continue;
      }
      if (tracking_descriptor.detection_meta_id >= kpts_meta->num_elements()) {
        throw std::runtime_error("inplace_tracker: detection_meta_id out of bounds");
      }
      auto kpts_span = kpts_meta->get_kpts_xyv(
          tracking_descriptor.detection_meta_id, kpts_per_box);
      KptXyvVector kpts(kpts_span.begin(), kpts_span.end());
      float score = kpts_meta->score(tracking_descriptor.detection_meta_id);

      std::vector<int> ids;
      tracking_descriptor.collection->set_frame_data_map(tracking_descriptor.frame_id,
          prop->input_meta_key,
          std::make_unique<AxMetaKptsDetection>(std::vector<box_xyxy>(1),
              std::move(kpts), std::vector<float>{ score }, ids, kpts_shape,
              kpts_meta->get_decoder_name()));
    }
  }
}
