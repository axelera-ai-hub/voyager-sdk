// Copyright Axelera AI, 2023
#include "trackers.hpp"

#include <algorithm>
#include <numeric>

#include "lapjv.h"

namespace axtracker
{

KalmanBoxTracker::KalmanBoxTracker(BboxXyxyRelative bbox, int max_history)
    : max_history_size(max_history), track_id(KalmanBoxTracker::count++)
{
  int stateSize = 7;
  int measSize = 4;
  kf.init(stateSize, measSize, CV_32F);

  // Transition State Matrix F
  // [ 1 0 0 0 1 0 0 ]
  // [ 0 1 0 0 0 1 0 ]
  // [ 0 0 1 0 0 0 1 ]
  // [ 0 0 0 1 0 0 0 ]
  // [ 0 0 0 0 1 0 0 ]
  // [ 0 0 0 0 0 1 0 ]
  // [ 0 0 0 0 0 0 1 ]

  // clang-format off
    kf.transitionMatrix = (cv::Mat_<float>(7, 7) <<
          1, 0, 0, 0, 1, 0, 0,
          0, 1, 0, 0, 0, 1, 0,
          0, 0, 1, 0, 0, 0, 1,
          0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 1);
  // clang-format on

  // Measure Matrix H
  // [ 1 0 0 0 0 0 0 ]
  // [ 0 1 0 0 0 0 0 ]
  // [ 0 0 1 0 0 0 0 ]
  // [ 0 0 0 1 0 0 0 ]
  kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
  for (int i = 0; i < measSize; ++i) {
    kf.measurementMatrix.at<float>(i, i) = 1.0f;
  }

  // Process Noise Covariance Matrix Q
  setIdentity(kf.processNoiseCov, cv::Scalar(1));
  kf.processNoiseCov.at<float>(4, 4) = 1e-2;
  kf.processNoiseCov.at<float>(5, 5) = 1e-2;
  kf.processNoiseCov.at<float>(6, 6) = 1e-4;

  // Measures Noise Covariance Matrix R
  setIdentity(kf.measurementNoiseCov, cv::Scalar(1));
  kf.measurementNoiseCov.at<float>(2, 2) = 10.0f;
  kf.measurementNoiseCov.at<float>(3, 3) = 10.0f;

  // Error Covariance Matrix P
  setIdentity(kf.errorCovPost, cv::Scalar(10.0));
  for (int i = 4; i < 7; i++) {
    kf.errorCovPost.at<float>(i, i) *= 1000.0f; // High uncertainty for initial velocities
  }

  // print P, Q, R
  // std::cout << "R: " << kf.measurementNoiseCov << std::endl;
  // std::cout << "Q: " << kf.processNoiseCov << std::endl;
  // std::cout << "P: " << kf.errorCovPost << std::endl;

  // Set initial state
  cv::Mat initialState = convert_bbox_to_z(bbox);
  for (int i = 0; i < initialState.rows; i++) {
    kf.statePost.at<float>(i) = initialState.at<float>(i, 0);
  }
  history.push_back(bbox);
}

void
KalmanBoxTracker::update(BboxXyxyRelative bbox)
{
  time_since_update = 0;
  hits++;
  hit_streak++;
  kf.correct(convert_bbox_to_z(bbox));
}

BboxXyxyRelative
KalmanBoxTracker::predict()
{
  if ((kf.statePost.at<float>(6) + kf.statePost.at<float>(2)) <= 0) {
    kf.statePost.at<float>(6) *= 0.0;
  }
  kf.predict();

  age++;
  if (time_since_update > 0) {
    hit_streak = 0;
  }
  time_since_update++;

  auto predicted_bbox = convert_x_to_bbox(kf.statePost);
  // copy score and class_id from history
  BboxXyxyRelative bbox = history.empty() ? predicted_bbox : history.back();
  predicted_bbox.score = bbox.score;
  predicted_bbox.class_id = bbox.class_id;

  history.push_back(predicted_bbox);
  if (history.size() > max_history_size) {
    history.erase(history.begin());
  }

  return !history.empty() ? history.back() : BboxXyxyRelative();
}

BboxXyxyRelative
KalmanBoxTracker::get_absolute_xyxy(int width, int height) const
{
  auto bbox = convert_x_to_bbox(kf.statePost);

  // Clamp bounding box coordinates
  bbox.x1 = std::max(0.0f, bbox.x1);
  bbox.y1 = std::max(0.0f, bbox.y1);
  bbox.x2 = std::min(1.0f, bbox.x2);
  bbox.y2 = std::min(1.0f, bbox.y2);

  bbox.x1 *= width;
  bbox.y1 *= height;
  bbox.x2 *= width;
  bbox.y2 *= height;

  return bbox;
}

// start from 1 which is required by the MOT benchmark
int KalmanBoxTracker::count = 1;

std::vector<std::vector<double>>
iou_batch(const std::vector<BboxXyxyRelative> &bb_test,
    const std::vector<BboxXyxyRelative> &bb_gt)
{
  std::vector<std::vector<double>> iouMatrix(
      bb_test.size(), std::vector<double>(bb_gt.size()));

  for (size_t i = 0; i < bb_test.size(); ++i) {
    for (size_t j = 0; j < bb_gt.size(); ++j) {
      float xx1 = std::max(bb_test[i].x1, bb_gt[j].x1);
      float yy1 = std::max(bb_test[i].y1, bb_gt[j].y1);
      float xx2 = std::min(bb_test[i].x2, bb_gt[j].x2);
      float yy2 = std::min(bb_test[i].y2, bb_gt[j].y2);

      float w = std::max(0.0f, xx2 - xx1);
      float h = std::max(0.0f, yy2 - yy1);
      float overlap = w * h;

      float area1
          = (bb_test[i].x2 - bb_test[i].x1) * (bb_test[i].y2 - bb_test[i].y1);
      float area2 = (bb_gt[j].x2 - bb_gt[j].x1) * (bb_gt[j].y2 - bb_gt[j].y1);
      float total_area = area1 + area2 - overlap;

      iouMatrix[i][j] = (total_area > 0.0f) ? overlap / total_area : 0.0;
    }
  }

  return iouMatrix;
}

std::vector<std::pair<int, int>>
linear_assignment(const std::vector<std::vector<double>> &costMatrix)
{
  int n = costMatrix.size();
  cost_t **cost = new cost_t *[n];
  int_t *x = new int_t[n];
  int_t *y = new int_t[n];

  // Convert cost matrix to C array
  for (int i = 0; i < n; ++i) {
    cost[i] = new cost_t[n];
    for (int j = 0; j < n; ++j) {
      cost[i][j] = static_cast<cost_t>(costMatrix[i][j]);
    }
  }

  lapjv_internal(n, cost, x, y);

  // Convert result back to C++ data structure
  std::vector<std::pair<int, int>> result;
  for (int i = 0; i < n; ++i) {
    if (x[i] >= 0) {
      result.push_back(std::make_pair(i, x[i]));
    }
  }

  for (int i = 0; i < n; ++i) {
    delete[] cost[i];
  }
  delete[] cost;
  delete[] x;
  delete[] y;

  return result;
}

void
SORT::update(const std::vector<BboxXyxyRelative> &dets)
{
  frame_count++;
  std::vector<BboxXyxyRelative> trks;
  std::vector<int> to_del;

  // Prepare predicted locations from existing trackers
  for (auto &tracker : trackers) {
    auto prediction = tracker.predict();
    if (!std::isnan(prediction.x1) && !std::isnan(prediction.y1)
        && !std::isnan(prediction.x2) && !std::isnan(prediction.y2)) {
      trks.push_back(prediction);
      // std::cout << "Predicted State: " << tracker.get_state() << std::endl;
    } else {
      std::cerr << "Warning: NaN values in tracker prediction, delete tracker "
                << tracker.getTrackId() << std::endl;
      to_del.push_back(tracker.getTrackId());
    }
  }

  // Associate detections to trackers
  auto [matched, unmatched_dets, unmatched_trks]
      = associateDetectionsToTrackers(dets, trks);
  // std::cout << frame_count << " - detections: " << dets.size()
  //       << " - matched: " << matched.size()
  //       << ", unmatched_dets: " << unmatched_dets.size()
  //       << ", unmatched_trks: " << unmatched_trks.size() << std::endl;

  // Update matched trackers
  for (auto &m : matched) {
    // std::cout << "fill trackers " << m.second << " with detection "
    //           << dets[m.first] << std::endl;
    trackers[m.second].update(dets[m.first]);
  }

  // Create new trackers for unmatched detections
  for (auto &u_det : unmatched_dets) {
    KalmanBoxTracker tracker(dets[u_det]);
    trackers.push_back(tracker);
  }

  // Remove dead tracklets
  trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                     [this](const KalmanBoxTracker &t) {
                       return t.time_since_update > maxAge;
                     }),
      trackers.end());

  // Get active trackers
  active_trackers.clear();
  for (auto &tracker : trackers) {
    if (tracker.time_since_update < 1
        && (tracker.hit_streak >= minHits || frame_count <= minHits)) {
      active_trackers.push_back(tracker);
    }
  }
  // std::cout << "active trackers: " << active_trackers.size()
  //   << ", trackers: " << trackers.size() << std::endl;
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
SORT::associateDetectionsToTrackers(const std::vector<BboxXyxyRelative> &detections,
    const std::vector<BboxXyxyRelative> &trackers)

{
  if (trackers.empty()) {
    std::vector<int> unmatchedDetections(detections.size());
    std::iota(unmatchedDetections.begin(), unmatchedDetections.end(), 0);
    return { std::vector<std::pair<int, int>>(), unmatchedDetections, std::vector<int>() };
  }

  auto iouMatrix = iou_batch(detections, trackers);
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatchedDetections;
  std::vector<int> unmatchedTrackers;

  // Adapt linear_assignment only if any row or column in the IOU matrix has
  // more than one element exceeding the threshold
  bool useLinearAssignment = false;
  for (size_t i = 0; i < iouMatrix.size() && !useLinearAssignment; ++i) {
    int aboveThresholdCount = std::count_if(iouMatrix[i].begin(), iouMatrix[i].end(),
        [this](double score) { return score > this->iouThreshold; });

    if (aboveThresholdCount > 1) {
      useLinearAssignment = true;
      break;
    }
  }
  // Check columns
  if (!useLinearAssignment && iouMatrix.size()) {
    for (size_t j = 0; j < iouMatrix[0].size(); ++j) {
      int count = 0;
      for (size_t i = 0; i < iouMatrix.size(); ++i) {
        if (iouMatrix[i][j] > this->iouThreshold) {
          count++;
        }
      }
      if (count > 1) {
        useLinearAssignment = true;
        break;
      }
    }
  }

  // std::cout << "useLinearAssignment: " << useLinearAssignment << std::endl;
  if (useLinearAssignment) {
    // Invert iouMatrix for linear_assignment
    for (auto &row : iouMatrix) {
      std::transform(
          row.begin(), row.end(), row.begin(), [](double iou) { return -iou; });
    }

    auto matchedIndices = linear_assignment(iouMatrix);

    // Process matched indices
    for (const auto &match : matchedIndices) {
      if (-iouMatrix[match.first][match.second] >= iouThreshold) {
        // std::cout << "multi - matched score: " <<
        // -iouMatrix[match.first][match.second] << std::endl;
        matches.push_back(match);
      } else {
        unmatchedDetections.push_back(match.first);
        unmatchedTrackers.push_back(match.second);
      }
    }
  } else {
    // Directly use IOU threshold to determine matches
    for (size_t i = 0; i < iouMatrix.size(); ++i) {
      for (size_t j = 0; j < iouMatrix[i].size(); ++j) {
        if (iouMatrix[i][j] >= iouThreshold) {
          // std::cout << "single - matched score: " << iouMatrix[i][j] <<
          // std::endl;
          matches.emplace_back(i, j);
        }
      }
    }
  }

  // Determine unmatched detections and trackers
  for (size_t i = 0; i < detections.size(); ++i) {
    if (std::none_of(matches.begin(), matches.end(),
            [i](const std::pair<int, int> &m) { return m.first == i; })) {
      unmatchedDetections.push_back(i);
    }
  }
  for (size_t j = 0; j < trackers.size(); ++j) {
    if (std::none_of(matches.begin(), matches.end(),
            [j](const std::pair<int, int> &m) { return m.second == j; })) {
      unmatchedTrackers.push_back(j);
    }
  }

  return { matches, unmatchedDetections, unmatchedTrackers };
}

void
ScalarMOT::update(const std::vector<BboxXyxyRelative> &detections)
{
  for (auto &tracker : trackers_)
    tracker.predict();
  associateDetectionsToTrackers(detections);
  removeLostTrackers();
}

void
ScalarMOT::associateDetectionsToTrackers(const std::vector<BboxXyxyRelative> &detections)
{
  // Simple nearest detection logic for associating detections with trackers
  for (const auto &det : detections) {
    float minDistance = std::numeric_limits<float>::max();
    KalmanBoxTracker *nearestTracker = nullptr;

    for (auto &tracker : trackers_) {
      auto current_state = tracker.get_history().back();
      if (current_state.class_id != det.class_id) {
        continue;
      }
      float distance = calculateDistance(det, current_state);
      if (distance < minDistance) {
        minDistance = distance;
        nearestTracker = &tracker;
      }
    }

    float distThreshold = std::min((det.x2 - det.x1), (det.y2 - det.y1)) / 2.0f;
    if (nearestTracker && minDistance < distThreshold) {
      nearestTracker->update(det);
    } else {
      // create new trackers for unmatched detections
      trackers_.push_back(KalmanBoxTracker(det));
    }
  }
}

float
ScalarMOT::calculateDistance(const BboxXyxyRelative &det, const BboxXyxyRelative &trackerState)
{
  float detCenterX = (det.x1 + det.x2) / 2.0f;
  float detCenterY = (det.y1 + det.y2) / 2.0f;
  float trackerCenterX = (trackerState.x1 + trackerState.x2) / 2.0f;
  float trackerCenterY = (trackerState.y1 + trackerState.y2) / 2.0f;

  float dx = detCenterX - trackerCenterX;
  float dy = detCenterY - trackerCenterY;
  return sqrt(dx * dx + dy * dy);
}

void
ScalarMOT::removeLostTrackers()
{
  // Remove trackers that have been lost for more than maxLostFrames
  trackers_.erase(std::remove_if(trackers_.begin(), trackers_.end(),
                      [this](const KalmanBoxTracker &tracker) {
                        return tracker.time_since_update > maxLostFrames_;
                      }),
      trackers_.end());
}

} // namespace axtracker
