// Copyright Axelera AI, 2025
#pragma once

#include <AxMetaBBox.hpp>
#include <AxStreamerUtils.hpp>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include "AxColor.hpp"

class TrackerMetaFrameDataKeys
{
  public:
  static uint8_t get(const std::string &meta_string)
  {
    {
      std::shared_lock lock(mutex);
      auto it = subtask_name_to_key.find(meta_string);
      if (it != subtask_name_to_key.end()) {
        return it->second;
      }
    }
    {
      std::unique_lock lock(mutex);
      if (subtask_keystrings.empty()) {
        subtask_keystrings.reserve(1024);
      }
      uint8_t key = subtask_name_to_key.size();
      size_t old_size = subtask_keystrings.size();
      size_t new_size
          = old_size + sizeof(uint8_t) + sizeof(size_t) + meta_string.size();
      subtask_keystrings.resize(new_size);
      char *data = subtask_keystrings.data() + old_size;
      *reinterpret_cast<uint8_t *>(data) = key;
      data += sizeof(uint8_t);
      *reinterpret_cast<size_t *>(data) = meta_string.size();
      data += sizeof(size_t);
      std::memcpy(data, meta_string.data(), meta_string.size());
      subtask_name_to_key[meta_string] = key;
      return key;
    }
  }

  static size_t size()
  {
    std::shared_lock lock(mutex);
    return subtask_name_to_key.size();
  }

  static const std::vector<char> &strings()
  {
    std::shared_lock lock(mutex);
    return subtask_keystrings;
  }

  private:
  inline static std::shared_mutex mutex;
  inline static std::unordered_map<std::string, uint8_t> subtask_name_to_key;
  inline static std::vector<char> subtask_keystrings;
};

template <typename Plugin, typename Ret, typename... Args> class TrackerCallback
{
  public:
  TrackerCallback(const TrackerCallback &) = delete;
  TrackerCallback &operator=(const TrackerCallback &) = delete;
  TrackerCallback(TrackerCallback &&) = delete;
  TrackerCallback &operator=(TrackerCallback &&) = delete;
  TrackerCallback(const std::string &lib_name, const Ax::StringMap &options, Ax::Logger &l)
      : logger(l),
        lib(logger, lib_name)
  {
    Ax::load_v1_plugin(lib, plugin);
    if (!plugin.init_and_set_static_properties) {
      if (!options.empty()) {
        throw std::runtime_error(
            "TrackerCallback: Function -init_and_set_static_properties- not found but options not empty");
      }
    } else {
      if (!options.empty()) {
        if (!plugin.allowed_properties) {
          throw std::runtime_error(
              "TrackerCallback: Function -allowed_properties- not found but options not empty");
        }
        const auto &allowed_properties_stringset = plugin.allowed_properties();
        for (const auto &opt : options) {
          if (allowed_properties_stringset.count(opt.first) == 0) {
            throw std::runtime_error(
                "TrackerCallback: Property not allowed - " + opt.first + ".");
          }
        }
      }
      subplugin_data = plugin.init_and_set_static_properties(options, logger);
    }
    if (plugin.set_dynamic_properties) {
      throw std::runtime_error(
          "TrackerCallback: Function -set_dynamic_properties- found but dynamic properties not available for tracker");
    }

    if constexpr (std::is_same_v<Plugin, Ax::V1Plugin::DetermineObjectAttribute>) {
      callback = plugin.determine_object_attribute;
    } else if constexpr (std::is_same_v<Plugin, Ax::V1Plugin::TrackerFilter>) {
      callback = plugin.filter;
    } else {
      throw std::logic_error("Invalid plugin type for TrackerCallback in AxMetaTracker");
    }
  }

  Ret operator()(Args... args) const
  {
    return callback(subplugin_data.get(), std::forward<Args>(args)..., logger);
  }

  private:
  Ax::Logger &logger;
  Ax::SharedLib lib;
  Plugin plugin;
  Ret (*callback)(const void *subplugin_properties, Args... args, Ax::Logger &logger)
      = nullptr;
  std::shared_ptr<void> subplugin_data;
};

struct TrackingElement {
  BboxXyxy bbox;
  std::unordered_map<uint8_t, std::unique_ptr<AxMetaBase>> frame_data_map;

  explicit TrackingElement(BboxXyxy bbox)
      : bbox{ bbox }
  {
  }
  TrackingElement(TrackingElement &&) = default;
  TrackingElement &operator=(TrackingElement &&) = default;
  TrackingElement(const TrackingElement &) = delete;
  TrackingElement &operator=(const TrackingElement &) = delete;
};

using DetermineObjectAttributeCallback
    = TrackerCallback<Ax::V1Plugin::DetermineObjectAttribute, std::unique_ptr<AxMetaBase>,
        int, int, uint8_t, const std::unordered_map<int, TrackingElement> &>;

class TrackingCollection
{
  public:
  TrackingCollection(int track_id, int detection_class_id,
      float detection_score, int history_length,
      const std::unordered_map<std::string, DetermineObjectAttributeCallback> &determine_object_attribute_map)
      : track_string{ "track_" + std::to_string(track_id) },
        detection_class_id{ detection_class_id },
        detection_score{ detection_score },
        history_length{ history_length },
        determine_object_attribute_map{ determine_object_attribute_map }
  {
    if (history_length < 1) {
      throw std::runtime_error("history_length must be at least 1");
    }
  }
  TrackingCollection(const TrackingCollection &) = delete;
  TrackingCollection &operator=(const TrackingCollection &) = delete;
  TrackingCollection(TrackingCollection &&) = delete;
  TrackingCollection &operator=(TrackingCollection &&) = delete;

  const TrackingElement *get_frame(int frame_id) const
  {
    std::shared_lock lock(mutex);
    auto itr = frame_id_to_element.find(frame_id);
    if (itr == frame_id_to_element.end()) {
      return nullptr;
    }
    return &itr->second;
  }

  std::vector<char> get_history(int frame_id) const
  {
    int first_id = std::max(0, frame_id - history_length + 1);
    int num_boxes = frame_id - first_id + 1;
    size_t size = sizeof(first_id) + sizeof(num_boxes) + num_boxes * sizeof(BboxXyxy);
    std::shared_lock lock(mutex);

    using subtask_results_t = std::map<uint8_t, std::vector<extern_meta>>;
    subtask_results_t subtask_results;
    for (const auto &[key, value] : frame_id_to_element.at(frame_id).frame_data_map) {
      auto it = subtask_results.try_emplace(key).first;
      it->second = value->get_extern_meta();
    }

    subtask_results_t subtask_frame_results;
    for (const auto &[key, value] : tracker_data_map) {
      auto it = subtask_frame_results.try_emplace(key).first;
      it->second = value->get_extern_meta();
    }

    auto add_sizes = [&size](const subtask_results_t &res) {
      size += sizeof(uint8_t) + sizeof(int);
      for (const auto &[key, vec] : res) {
        size += sizeof(uint8_t) + sizeof(int);
        for (const auto &meta : vec) {
          size += sizeof(int) + std::strlen(meta.type) + sizeof(int)
                  + std::strlen(meta.subtype) + sizeof(int) + meta.meta_size;
        }
      }
    };
    add_sizes(subtask_results);
    add_sizes(subtask_frame_results);

    std::vector<char> result(size);
    char *data = result.data();

    *reinterpret_cast<int *>(data) = detection_class_id;
    data += sizeof(int);
    *reinterpret_cast<int *>(data) = num_boxes;
    data += sizeof(int);
    for (int id = first_id; id <= frame_id; ++id) {
      auto itr = frame_id_to_element.find(id);
      if (itr != frame_id_to_element.end()) {
        std::memcpy(data, &itr->second.bbox, sizeof(BboxXyxy));
      }
      data += sizeof(BboxXyxy);
    }

    auto add_data = [&data](const subtask_results_t &res) {
      *reinterpret_cast<int *>(data) = res.size();
      data += sizeof(int);
      for (const auto &[key, vec] : res) {
        *reinterpret_cast<uint8_t *>(data) = key;
        data += sizeof(uint8_t);
        *reinterpret_cast<int *>(data) = vec.size();
        data += sizeof(int);
        for (const auto &meta : vec) {
          int meta_type_size = std::strlen(meta.type);
          *reinterpret_cast<int *>(data) = meta_type_size;
          data += sizeof(int);
          std::memcpy(data, meta.type, meta_type_size);
          data += meta_type_size;
          int meta_subtype_size = std::strlen(meta.subtype);
          *reinterpret_cast<int *>(data) = meta_subtype_size;
          data += sizeof(int);
          std::memcpy(data, meta.subtype, meta_subtype_size);
          data += meta_subtype_size;
          *reinterpret_cast<int *>(data) = meta.meta_size;
          data += sizeof(int);
          std::memcpy(data, meta.meta, meta.meta_size);
          data += meta.meta_size;
        }
      }
    };
    add_data(subtask_results);
    add_data(subtask_frame_results);

    return result;
  }

  void set_frame(int frame_id, TrackingElement &&element)
  {
    std::unique_lock lock(mutex);
    frame_id_to_element.insert_or_assign(frame_id, std::move(element));
  }

  void set_frame_data_map(int frame_id, const std::string &key_string,
      std::unique_ptr<AxMetaBase> value)
  {
    int first_id = std::max(0, frame_id - history_length + 1);
    std::unique_lock lock(mutex);
    uint8_t key = TrackerMetaFrameDataKeys::get(key_string);
    frame_id_to_element.at(frame_id).frame_data_map.insert_or_assign(key, std::move(value));
    auto itr = determine_object_attribute_map.find(key_string);
    if (itr != determine_object_attribute_map.end()) {
      tracker_data_map.insert_or_assign(
          key, itr->second(first_id, frame_id, key, frame_id_to_element));
    }
  }

  void delete_frame(int frame_id)
  {
    std::unique_lock lock(mutex);
    frame_id_to_element.erase(frame_id - history_length + 1);
  }

  const std::string track_string;
  const int detection_class_id;
  const float detection_score;
  const int history_length;

  private:
  mutable std::shared_mutex mutex;
  std::unordered_map<int, TrackingElement> frame_id_to_element;
  const std::unordered_map<std::string, DetermineObjectAttributeCallback> &determine_object_attribute_map;
  std::unordered_map<uint8_t, std::unique_ptr<AxMetaBase>> tracker_data_map;
};

struct TrackingDescriptor {
  int frame_id;
  int detection_meta_id = -1;
  int lost_since = 0;
  std::shared_ptr<TrackingCollection> collection;

  TrackingDescriptor(int track_id, int detection_class_id,
      float detection_score, int history_length,
      const std::unordered_map<std::string, DetermineObjectAttributeCallback> &determine_object_attribute_map)
      : frame_id{ 0 },
        collection{ std::make_shared<TrackingCollection>(track_id, detection_class_id,
            detection_score, history_length, determine_object_attribute_map) }
  {
  }
  TrackingDescriptor(const TrackingDescriptor &) = default;
  TrackingDescriptor &operator=(const TrackingDescriptor &) = default;
  TrackingDescriptor(TrackingDescriptor &&) = delete;
  TrackingDescriptor &operator=(TrackingDescriptor &&) = delete;
};

using KeepBoxCallback
    = TrackerCallback<Ax::V1Plugin::TrackerFilter, bool, const TrackingDescriptor &>;

class AxMetaTracker : public AxMetaBase
{
  public:
  std::unordered_map<int, TrackingDescriptor> track_id_to_tracking_descriptor;
  int history_length;
  mutable std::unordered_map<int, std::vector<char>> extern_meta_storage;

  explicit AxMetaTracker(int history_length = 1)
      : history_length{ history_length }
  {
    if (history_length < 1) {
      throw std::runtime_error("history_length must be at least 1");
    }
  }

  virtual ~AxMetaTracker()
  {
    for (auto &[track_id, tracking_descriptor] : track_id_to_tracking_descriptor) {
      tracking_descriptor.collection->delete_frame(tracking_descriptor.frame_id);
    }
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    if (video.info.format != AxVideoFormat::RGB && video.info.format != AxVideoFormat::RGBA) {
      throw std::runtime_error("Tracker results can only be drawn on RGB or RGBA");
    }
    cv::Mat frame(cv::Size(video.info.width, video.info.height),
        Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);

    for (const auto &[track_id, tracking_descriptor] : track_id_to_tracking_descriptor) {
      cv::Scalar color = getColorForTracker(track_id);

      int first_id = std::max(0, tracking_descriptor.frame_id - history_length + 1);
      int start_id = -1;
      cv::Point prevCenter;
      for (int i = first_id; i < tracking_descriptor.frame_id; ++i) {
        if (const TrackingElement *element = tracking_descriptor.collection->get_frame(i)) {
          const auto &prevBbox = element->bbox;
          prevCenter = cv::Point(
              (prevBbox.x1 + prevBbox.x2) / 2, (prevBbox.y1 + prevBbox.y2) / 2);
          start_id = i;
          break;
        }
      }
      if (start_id < 0) {
        continue;
      }
      for (int i = start_id + 1; i <= tracking_descriptor.frame_id; ++i) {
        if (const TrackingElement *element = tracking_descriptor.collection->get_frame(i)) {
          const auto &currBbox = element->bbox;
          cv::Point currCenter(
              (currBbox.x1 + currBbox.x2) / 2, (currBbox.y1 + currBbox.y2) / 2);
          cv::line(frame, prevCenter, currCenter, color, 2);
          prevCenter = currCenter;
        }
      }
      const TrackingElement *element
          = tracking_descriptor.collection->get_frame(tracking_descriptor.frame_id);
      if (!element) {
        continue;
      }
      const auto &lastBbox = element->bbox;
      cv::Rect pixelBbox(cv::Point(lastBbox.x1, lastBbox.y1),
          cv::Point(lastBbox.x2, lastBbox.y2));
      cv::rectangle(frame, pixelBbox, color, 2);
      cv::putText(frame, std::to_string(track_id),
          cv::Point(pixelBbox.x, pixelBbox.y - 5), cv::FONT_HERSHEY_SIMPLEX,
          0.5, color, 2);
    }
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    std::vector<extern_meta> metas;
    for (const auto &[track_id, tracking_descriptor] : track_id_to_tracking_descriptor) {
      auto &data = extern_meta_storage
                       .try_emplace(track_id, tracking_descriptor.collection->get_history(
                                                  tracking_descriptor.frame_id))
                       .first->second;
      metas.push_back(
          { "tracking_meta", tracking_descriptor.collection->track_string.c_str(),
              static_cast<int>(data.size()), data.data() });
    }
    metas.push_back({ "tracking_meta", "objmeta_keys",
        static_cast<int>(TrackerMetaFrameDataKeys::strings().size()),
        TrackerMetaFrameDataKeys::strings().data() });
    return metas;
  }
};
