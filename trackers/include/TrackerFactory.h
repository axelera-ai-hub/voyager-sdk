// Copyright Axelera AI, 2023
#pragma once

#include "../axtracker/include/trackers.hpp"
#include "MultiObjTracker.hpp"

#ifdef HAVE_BYTETRACK
#include "../3rd_parties/bytetrack/include/BYTETracker.h"
#endif

#ifdef HAVE_OC_SORT
#include "../3rd_parties/oc_sort/include/OCSort.hpp"
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
TrackerParams
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
  const std::vector<ax::TrackedObject> Update(
      const std::vector<ax::ObservedObject> &detections) override;

  private:
  axtracker::ScalarMOT tracker_;
};

class SORTWrapper : public ax::MultiObjTracker
{
  public:
  SORTWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(
      const std::vector<ax::ObservedObject> &detections) override;

  private:
  axtracker::SORT tracker_;
};

#ifdef HAVE_BYTETRACK
class BytetrackWrapper : public ax::MultiObjTracker
{
  public:
  BytetrackWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(
      const std::vector<ax::ObservedObject> &detections) override;

  private:
  BYTETracker tracker_;
};
#endif

#ifdef HAVE_OC_SORT
class OCSortWrapper : public ax::MultiObjTracker
{
  public:
  OCSortWrapper(const TrackerParams &params);
  const std::vector<ax::TrackedObject> Update(
      const std::vector<ax::ObservedObject> &detections) override;

  private:
  ocsort::OCSort tracker_;
};
#endif
