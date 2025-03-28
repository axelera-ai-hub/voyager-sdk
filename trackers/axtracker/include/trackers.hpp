// Copyright Axelera AI, 2023
#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace axtracker
{
/************** Single Object Tracker (Kalman filter) ***********/
struct BboxXyxyRelative {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int class_id;
};

inline std::ostream &
operator<<(std::ostream &os, const BboxXyxyRelative &bbox)
{
  os << "[" << bbox.x1 << ", " << bbox.y1 << ", " << bbox.x2 << ", " << bbox.y2 << "]"
     << " score: " << bbox.score << " class_id: " << bbox.class_id;
  return os;
}

inline cv::Mat
convert_bbox_to_z(const BboxXyxyRelative &bbox)
{
  float w = bbox.x2 - bbox.x1;
  float h = bbox.y2 - bbox.y1;
  float x = bbox.x1 + w / 2.0f;
  float y = bbox.y1 + h / 2.0f;
  float s = w * h; // scale is just area
  float r = w / h; // aspect ratio
  return (cv::Mat_<float>(4, 1) << x, y, s, r);
}

inline BboxXyxyRelative
convert_x_to_bbox(const cv::Mat &x_state)
{
  float w = sqrt(x_state.at<float>(2) * x_state.at<float>(3));
  float h = x_state.at<float>(2) / w;
  float x = x_state.at<float>(0) - w / 2.0f;
  float y = x_state.at<float>(1) - h / 2.0f;
  return { x, y, x + w, y + h };
}

class KalmanBoxTracker
{
  public:
  static int count;

  int time_since_update = 0;
  int hits = 0;
  int hit_streak = 0;
  int age = 0;
  int max_history_size;

  KalmanBoxTracker(BboxXyxyRelative bbox, int max_history = 100);

  int getTrackId() const
  {
    return track_id;
  }

  int getClassId() const
  {
    return history.back().class_id;
  }

  BboxXyxyRelative get_state() const
  {
    return convert_x_to_bbox(kf.statePost);
  }

  const std::vector<BboxXyxyRelative> &get_history() const
  {
    return history;
  }

  void update(BboxXyxyRelative bbox);

  BboxXyxyRelative predict();

  BboxXyxyRelative get_absolute_xyxy(int width, int height) const;

  private:
  cv::KalmanFilter kf;
  std::vector<BboxXyxyRelative> history;
  int track_id;
};

//************** Multiple Object Tracker (SORT) ***********

std::vector<std::vector<double>> iou_batch(const std::vector<BboxXyxyRelative> &bb_test,
    const std::vector<BboxXyxyRelative> &bb_gt);

std::vector<std::pair<int, int>> linear_assignment(
    const std::vector<std::vector<double>> &costMatrix);

// simple online and realtime tracking
class SORT
{
  public:
  SORT(int maxAge = 30, int minHits = 3, float iouThreshold = 0.3)
      : maxAge(maxAge), minHits(minHits), iouThreshold(iouThreshold), frame_count(0)
  {
  }

  const std::vector<KalmanBoxTracker> &getTrackers() const
  {
    return active_trackers;
  }

  void update(const std::vector<BboxXyxyRelative> &dets);

  private:
  int maxAge;
  int minHits;
  float iouThreshold;
  int frame_count;
  std::vector<KalmanBoxTracker> trackers;
  // this is used for getting history of trackers; it can be improved by
  // associating a tracker state with each tracker, so that we can just
  // identify if a tracker is active or not without copying the trackers
  std::vector<KalmanBoxTracker> active_trackers;

  std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
  associateDetectionsToTrackers(const std::vector<BboxXyxyRelative> &detections,
      const std::vector<BboxXyxyRelative> &trackers);
};

//*********** Multiple Object Tracker (nearest with simple logic association)

class ScalarMOT
{
  public:
  ScalarMOT(int maxLostFrames = 30) : maxLostFrames_(maxLostFrames)
  {
  }

  void update(const std::vector<BboxXyxyRelative> &detections);

  const std::vector<KalmanBoxTracker> &getTrackers() const
  {
    return trackers_;
  }

  private:
  std::vector<KalmanBoxTracker> trackers_;
  int maxLostFrames_;

  void associateDetectionsToTrackers(const std::vector<BboxXyxyRelative> &detections);

  float calculateDistance(const BboxXyxyRelative &det, const BboxXyxyRelative &trackerState);

  void removeLostTrackers();
};

} // namespace axtracker
