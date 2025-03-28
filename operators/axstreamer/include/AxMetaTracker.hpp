// Copyright Axelera AI, 2025
#pragma once

#include <AxMetaBBox.hpp>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include "AxMetaObjectDetection.hpp"

#include "MultiObjTracker.hpp"

static cv::Scalar
getColorForTracker(int trackId)
{
  static const uint32_t colormap[] = {
    0xff0000,
    0xffb27d,
    0xffffff,
    0x84b9ff,
    0x0009ff,
    0x000088,
    0x000000,
    0x730000,
    0xfc0000,
    0xffa46f,
    0xfffdf9,
    0x92c5ff,
    0x081cff,
    0x000096,
    0x00000a,
    0x650000,
    0xf10000,
    0xff9561,
    0xfffbee,
    0xa0d1ff,
    0x132eff,
    0x0000a4,
    0x000016,
    0x570000,
    0xe50000,
    0xff8653,
    0xfff7e2,
    0xaedbff,
    0x1f41ff,
    0x0000b2,
    0x000022,
    0x490000,
    0xd90000,
    0xff7546,
    0xfff2d6,
    0xbce4ff,
    0x2b53ff,
    0x0000bf,
    0x00002f,
    0x3c0000,
    0xcc0000,
    0xff6438,
    0xffecc9,
    0xc9ecff,
    0x3864ff,
    0x0000cc,
    0x00003c,
    0x2f0000,
    0xbf0000,
    0xff532b,
    0xffe4bc,
    0xd6f2ff,
    0x4675ff,
    0x0000d9,
    0x000049,
    0x220000,
    0xb20000,
    0xff411f,
    0xffdbae,
    0xe2f7ff,
    0x5386ff,
    0x0000e5,
    0x000057,
    0x160000,
    0xa40000,
    0xff2e13,
    0xffd1a0,
    0xeefbff,
    0x6195ff,
    0x0000f1,
    0x000065,
    0x0a0000,
    0x960000,
    0xff1c08,
    0xffc592,
    0xf9fdff,
    0x6fa4ff,
    0x0000fc,
    0x000073,
    0x000000,
    0x880000,
    0xff0900,
    0xffb984,
    0xfffeff,
    0x7db2ff,
    0x0000ff,
    0x000081,
    0x000000,
    0x7a0000,
    0xff0000,
    0xffab76,
    0xfffeff,
    0x8bbfff,
    0x0212ff,
    0x00008f,
    0x000005,
    0x6c0000,
    0xf60000,
    0xff9d68,
    0xfffcf4,
    0x99cbff,
    0x0d25ff,
    0x00009d,
    0x000010,
    0x5e0000,
    0xeb0000,
    0xff8e5a,
    0xfff9e8,
    0xa7d6ff,
    0x1938ff,
    0x0000ab,
    0x00001c,
    0x500000,
    0xdf0000,
    0xff7e4c,
    0xfff5dc,
    0xb5dfff,
    0x254aff,
    0x0000b8,
    0x000028,
    0x420000,
    0xd30000,
    0xff6d3f,
    0xffefcf,
    0xc2e8ff,
    0x325cff,
    0x0000c6,
    0x000035,
  };
  const auto color
      = colormap[static_cast<size_t>(std::abs(trackId)) % (sizeof(colormap) / sizeof(colormap[0]))];
  return cv::Scalar((color & 0xff0000) >> 16, (color & 0xff00) >> 8, (color & 0xff));
}

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

struct TrackingElement {
  BboxXyxy bbox;
  std::unordered_map<uint8_t, std::shared_ptr<AxMetaBase>> frame_data_map;
};

class TrackingCollection
{
  public:
  TrackingCollection(int track_id, int detection_class_id, int history_length)
      : track_string{ "track_" + std::to_string(track_id) },
        detection_class_id{ detection_class_id }, history_length{ history_length }
  {
    if (history_length < 1) {
      throw std::runtime_error("history_length must be at least 1");
    }
  }

  const std::string &get_track_string() const
  {
    return track_string;
  }

  const TrackingElement &get_frame(int frame_id) const
  {
    std::shared_lock lock(mutex);
    auto itr = frame_id_to_element.find(frame_id);
    if (itr == frame_id_to_element.end()) {
      throw std::runtime_error("Frame not found in get_frame with frame_id = "
                               + std::to_string(frame_id));
    }
    return itr->second;
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
      if (itr == frame_id_to_element.end()) {
        throw std::runtime_error(
            "Frame not found in add_history with frame_id = " + std::to_string(id));
      }
      std::memcpy(data, &itr->second.bbox, sizeof(BboxXyxy));
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
    frame_id_to_element[frame_id] = element;
  }

  void set_frame_data_map(int frame_id, const std::string &key_string,
      std::shared_ptr<AxMetaBase> value,
      std::shared_ptr<AxMetaBase> (*func)(
          int, int, uint8_t, const std::unordered_map<int, TrackingElement> &)
      = nullptr)
  {
    int first_id = std::max(0, frame_id - history_length + 1);
    std::unique_lock lock(mutex);
    uint8_t key = TrackerMetaFrameDataKeys::get(key_string);
    frame_id_to_element[frame_id].frame_data_map[key] = std::move(value);
    if (func) {
      tracker_data_map[key] = func(first_id, frame_id, key, frame_id_to_element);
    }
  }

  void delete_frame(int frame_id)
  {
    std::unique_lock lock(mutex);
    frame_id_to_element.erase(frame_id - history_length + 1);
  }

  private:
  mutable std::shared_mutex mutex;
  std::unordered_map<int, TrackingElement> frame_id_to_element;
  const std::string track_string;
  const int detection_class_id;
  const int history_length;
  std::unordered_map<uint8_t, std::shared_ptr<AxMetaBase>> tracker_data_map;
};

struct TrackingDescriptor {
  int frame_id;
  int detection_meta_id = -1;
  std::shared_ptr<TrackingCollection> collection;

  TrackingDescriptor(int track_id, int detection_class_id, int history_length)
      : frame_id{ 0 }, collection{ std::make_shared<TrackingCollection>(
                           track_id, detection_class_id, history_length) }
  {
  }
};

class AxMetaTracker : public AxMetaBbox
{
  public:
  std::unordered_map<int, TrackingDescriptor> track_id_to_tracking_descriptor;
  int history_length;
  int num_subtask_runs;
  mutable std::unordered_map<int, std::vector<char>> extern_meta_storage;

  AxMetaTracker() : AxMetaBbox({}), history_length{ 1 }, num_subtask_runs{ 0 }
  {
  }

  AxMetaTracker(int history_length, int num_subtask_runs)
      : AxMetaBbox({}), history_length{ history_length }, num_subtask_runs{ num_subtask_runs }
  {
    if (history_length < 1) {
      throw std::runtime_error("history_length must be at least 1");
    }
    if (num_subtask_runs < 0 || num_subtask_runs > history_length) {
      throw std::runtime_error(
          "num_subtask_runs must be at either zero or at least 1 and at most history_length");
    }
  }

  virtual ~AxMetaTracker()
  {
    for (auto &[track_id, tracking_descriptor] : track_id_to_tracking_descriptor) {
      tracking_descriptor.collection->delete_frame(tracking_descriptor.frame_id);
    }
  }

  void create_box_meta(int width, int height)
  {
    bboxvec.clear();
    for (const auto &[track_id, tracking_descriptor] : track_id_to_tracking_descriptor) {
      if (!num_subtask_runs || tracking_descriptor.frame_id < num_subtask_runs) {
        const auto &frame
            = tracking_descriptor.collection->get_frame(tracking_descriptor.frame_id);
        if (frame.bbox.x1 < 0 || frame.bbox.y1 < 0 || frame.bbox.x2 >= width
            || frame.bbox.y2 >= height) {
          continue;
        }
        bboxvec.push_back(tracking_descriptor.collection
                              ->get_frame(tracking_descriptor.frame_id)
                              .bbox);
      }
    }
  }

  void track_object(const std::vector<ax::TrackedObject> &trackers,
      std::unordered_map<int, TrackingDescriptor> &static_track_id_to_tracking_descriptor,
      int width, int height)
  {
    for (const auto &tracker : trackers) {
      const auto &xyxy = tracker.GetXyxy(width, height);

      auto [itr, success] = static_track_id_to_tracking_descriptor.try_emplace(
          tracker.track_id, tracker.track_id, tracker.class_id, history_length);

      if (!success) {
        ++itr->second.frame_id;
      }
      itr->second.detection_meta_id = tracker.latest_detection_id;
      itr->second.collection->set_frame(itr->second.frame_id,
          TrackingElement{ BboxXyxy{ std::get<0>(xyxy), std::get<1>(xyxy),
                               std::get<2>(xyxy), std::get<3>(xyxy) },
              {} });
      track_id_to_tracking_descriptor.insert(*itr);
    }
    for (auto itr = static_track_id_to_tracking_descriptor.begin();
         itr != static_track_id_to_tracking_descriptor.end();) {
      if (track_id_to_tracking_descriptor.count(itr->first)) {
        ++itr;
      } else {
        itr = static_track_id_to_tracking_descriptor.erase(itr);
      }
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
      for (int i = first_id + 1; i <= tracking_descriptor.frame_id; ++i) {
        const auto &prevBbox = tracking_descriptor.collection->get_frame(i - 1).bbox;
        const auto &currBbox = tracking_descriptor.collection->get_frame(i).bbox;
        cv::Point prevCenter((prevBbox.x1 + prevBbox.x2) * video.info.width / 2,
            (prevBbox.y1 + prevBbox.y2) * video.info.height / 2);
        cv::Point currCenter((currBbox.x1 + currBbox.x2) * video.info.width / 2,
            (currBbox.y1 + currBbox.y2) * video.info.height / 2);
        cv::line(frame, prevCenter, currCenter, color, 2);
      }

      const auto &lastBbox = tracking_descriptor.collection
                                 ->get_frame(tracking_descriptor.frame_id)
                                 .bbox;
      cv::Rect pixelBbox(
          cv::Point(lastBbox.x1 * video.info.width, lastBbox.y1 * video.info.height),
          cv::Point(lastBbox.x2 * video.info.width, lastBbox.y2 * video.info.height));
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
      metas.push_back({ "tracking_meta",
          tracking_descriptor.collection->get_track_string().c_str(),
          static_cast<int>(data.size()), data.data() });
    }
    metas.push_back({ "tracking_meta", "objmeta_keys",
        static_cast<int>(TrackerMetaFrameDataKeys::strings().size()),
        TrackerMetaFrameDataKeys::strings().data() });
    return metas;
  }
};
