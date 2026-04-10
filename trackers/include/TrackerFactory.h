// Copyright Axelera AI, 2024
#pragma once

#include <unordered_set>
#include "../axtracker/include/trackers.hpp"
#include "MultiObjTracker.hpp"

#ifdef HAVE_BYTETRACK
#include "../algorithms/bytetrack/include/BYTETracker.h"
#endif

#ifdef HAVE_OC_SORT
#include "../algorithms/oc_sort/include/OCSort.hpp"
#endif

#ifdef HAVE_TRACKTRACK
#include "../algorithms/tracktrack/include/TrackTrack.hpp"
#endif

//************** Multiple Object Tracker Factory ***********
// Supported algorithms:
//   ScalarMOT
//   SORT
//   bytetrack
//   OC_SORT (OC-SORT)
//
// Usage example:
//   TrackerParams params = { {"det_thresh", 0}, {"maxAge", 30}, {"minHits", 3},
//   {"iouThreshold", 0.3} }; auto tracker = CreateMultiObjTracker("sort",
//   params); // or CreateMultiObjTracker("sort", {}); for default params auto
//   results = tracker->Update(detections);

using TrackerParams
    = std::unordered_map<std::string, std::variant<bool, int, float, std::string>>;
// Helper function to create TrackerParams after initialization
inline TrackerParams
CreateTrackerParams(
    std::initializer_list<std::pair<std::string, std::variant<bool, int, float, std::string>>> list)
{
  return TrackerParams(list.begin(), list.end());
}
std::unique_ptr<ax::MultiObjTracker> CreateMultiObjTracker(
    const std::string &tracker_type_str, const TrackerParams &params);


//************** Adapters of each Multiple Object Tracker ***********
class ScalarMOTWrapper : public ax::MultiObjTracker
{
  public:
  ScalarMOTWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(const std::vector<ax::ObservedObject> &detections,
      const std::vector<std::vector<float>> &embeddings,
      const std::optional<Eigen::Matrix<float, 2, 3>> &transform = std::nullopt) override;

  private:
  axtracker::ScalarMOT tracker_;
};

class SORTWrapper : public ax::MultiObjTracker
{
  public:
  SORTWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(const std::vector<ax::ObservedObject> &detections,
      const std::vector<std::vector<float>> &embeddings,
      const std::optional<Eigen::Matrix<float, 2, 3>> &transform = std::nullopt) override;

  private:
  axtracker::SORT tracker_;
  int min_hits_;
};

#ifdef HAVE_BYTETRACK
class BytetrackWrapper : public ax::MultiObjTracker
{
  public:
  BytetrackWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(const std::vector<ax::ObservedObject> &detections,
      const std::vector<std::vector<float>> &embeddings,
      const std::optional<Eigen::Matrix<float, 2, 3>> &transform = std::nullopt) override;

  private:
  BYTETracker tracker_;
  bool return_all_states_;
  std::unordered_set<int> previous_removed_ids_;
};
#endif

#ifdef HAVE_OC_SORT
class OCSortWrapper : public ax::MultiObjTracker
{
  public:
  OCSortWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(const std::vector<ax::ObservedObject> &detections,
      const std::vector<std::vector<float>> &embeddings,
      const std::optional<Eigen::Matrix<float, 2, 3>> &transform = std::nullopt) override;

  private:
  ocsort::OCSort tracker_;
  bool return_all_states_;
};
#endif

#ifdef HAVE_TRACKTRACK
class TrackTrackWrapper : public ax::MultiObjTracker
{
  public:
  TrackTrackWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(const std::vector<ax::ObservedObject> &detections,
      const std::vector<std::vector<float>> &embeddings,
      const std::optional<Eigen::Matrix<float, 2, 3>> &transform = std::nullopt) override;

  private:
  std::unique_ptr<tracktrack::TrackTrack> tracker_;
  std::vector<tracktrack::Detection> prev_detections_; // For finding deleted detections
  bool return_all_states_ = false;
};
#endif

// Helper function to get a parameter value or a default value if not found
template <typename T>
T
GetParamOrDefault(const TrackerParams &params, const std::string &key, T defaultValue)
{
  auto it = params.find(key);
  if (it != params.end() && std::holds_alternative<T>(it->second)) {
    return std::get<T>(it->second);
  }
  return defaultValue;
}
