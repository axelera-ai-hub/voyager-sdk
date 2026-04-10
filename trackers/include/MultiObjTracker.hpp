// Copyright Axelera AI, 2023
#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ax
{
enum TrackState { kUndefined = 0, kNew, kTracked, kLost, kRemoved };
std::string to_string(const TrackState &state);

struct Rect2f {
  float x1;
  float y1;
  float x2;
  float y2;
};

struct ObservedObject {
  Rect2f bbox;
  int class_id = -1;
  float score = 0.0f;

  ObservedObject() = default;

  // Factory method for xyxy format
  static ObservedObject FromXYXY(int x1, int y1, int x2, int y2, int classId, float score)
  {
    return ObservedObject(x1, y1, x2, y2, classId, score);
  }

  // Factory method for ltxywh format
  static ObservedObject FromLTWH(int x, int y, int w, int h, int classId, float score)
  {
    return ObservedObject(x, y, x + w, y + h, classId, score);
  }

  // Factory method for xywh format (x_center, y_center, w, h)
  static ObservedObject FromXYWH(int x, int y, int w, int h, int classId, float score)
  {
    return ObservedObject(x - w / 2, y - h / 2, x + w / 2, y + h / 2, classId, score);
  }

  private:
  // Private constructor to enforce the use of factory methods
  ObservedObject(float x1, float y1, float x2, float y2, int classId, float score)
      : bbox({ x1, y1, x2, y2 }),
        class_id(classId),
        score(score)
  {
  }
};

struct TrackedObject {
  Rect2f bbox;
  float score; // -1 means no tracking confidence
  int class_id; // -1 means no class id
  int track_id; // -1 means no assigned track id
  int latest_detection_id = -1; // -1 means no latest detection id
  TrackState state;

  // Sets the state of the track.
  void SetTrackState(TrackState new_state)
  {
    state = new_state;
  }

  // Gets the bounding box in absolute coordinates (xyxy format).
  std::tuple<int, int, int, int> GetXyxy() const
  {
    return std::make_tuple(bbox.x1, bbox.y1, bbox.x2, bbox.y2);
  }

  // Gets the bounding box in absolute coordinates (ltxywh format).
  std::tuple<int, int, int, int> Getltxywh() const
  {
    return std::make_tuple(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
  }

  TrackedObject(float x1, float y1, float x2, float y2, int trackId = -1,
      int classId = -1, float scoreVal = -1.0f, TrackState trackState = kUndefined)
      : bbox{ x1, y1, x2, y2 },
        score(scoreVal),
        class_id(classId),
        track_id(trackId),
        state(trackState)
  {
  }
};

inline std::ostream &
operator<<(std::ostream &os, const TrackedObject &obs)
{
  os << "TrackedObject: (" << obs.bbox.x1 << ", " << obs.bbox.y1 << "), ("
     << obs.bbox.x2 << ", " << obs.bbox.y2 << "), " << obs.track_id
     << ", state=" << to_string(obs.state);
  return os;
}

//************** Multiple Object Tracker (interface) ***********
class MultiObjTracker
{
  public:
  virtual ~MultiObjTracker() = default;
  virtual const std::vector<TrackedObject>
  Update(const std::vector<ObservedObject> &detections,
      const std::vector<std::vector<float>> &embeddings,
      const std::optional<Eigen::Matrix<float, 2, 3>> &transform = std::nullopt)
      = 0;
};

} // namespace ax
