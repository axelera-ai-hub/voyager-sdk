// Copyright Axelera AI, 2025
#include "Track.hpp"
#include <algorithm>
#include <cmath>
#include "KalmanFilter.hpp"

namespace tracktrack
{

Track::Track(const Detection &det, int track_id, int frame_id, float alpha)
    : track_id_(track_id),
      class_id_(det.class_id),
      state_(TrackState::New),
      score_(det.score),
      start_frame_(frame_id),
      end_frame_(frame_id),
      tracklet_len_(1),
      time_since_update_(0),
      alpha_(alpha)
{
  latest_detection_id_ = det.original_index;

  // Initialize Kalman filter
  kf_ = std::make_unique<KalmanFilter>();
  kf_->initiate(det.to_xyah());

  // Initialize features if available
  if (det.features.size() > 0) {
    features_ = det.features;
  }

  // Add to history
  FrameData data;
  data.bbox = det.bbox;
  data.score = det.score;
  data.mean = kf_->get_state();
  data.covariance = kf_->get_covariance();
  data.features = det.features;
  history_[frame_id] = data;
}

Track::~Track() = default; // Defined here where KalmanFilter is complete

void
Track::mark_lost()
{
  state_ = TrackState::Lost;
}

void
Track::mark_removed()
{
  state_ = TrackState::Removed;
}

void
Track::mark_tracked()
{
  state_ = TrackState::Tracked;
}

void
Track::update(const Detection &det, int frame_id)
{
  // Update Kalman filter with confidence-based noise scaling
  kf_->update(det.to_xyah(), det.score);

  // Update track attributes
  score_ = det.score;
  class_id_ = det.class_id;
  end_frame_ = frame_id;
  tracklet_len_++;
  time_since_update_ = 0;
  latest_detection_id_ = det.original_index;

  // Update features using confidence-dependent EMA (matches reference TrackTrack)
  // High-confidence detections update appearance more conservatively (trust existing model),
  // low-confidence detections are discounted (keep existing features).
  if (det.features.size() > 0) {
    if (features_.size() == 0) {
      features_ = det.features;
    } else {
      float beta = alpha_ + (1.0f - alpha_) * (1.0f - det.score);
      features_ = beta * features_ + (1.0f - beta) * det.features;
      float norm = features_.norm();
      if (norm > 0) {
        features_ /= norm;
      }
    }
  }

  // Update state based on history length (matching Python)
  if (state_ == TrackState::New && history_.size() >= 3) {
    state_ = TrackState::Tracked;
  } else if (state_ == TrackState::Lost) {
    state_ = TrackState::Tracked;
  }

  // Add to history
  FrameData data;
  data.bbox = det.bbox;
  data.score = det.score;
  data.mean = kf_->get_state();
  data.covariance = kf_->get_covariance();
  data.features = det.features;
  history_[frame_id] = data;
}

void
Track::predict()
{
  kf_->predict();
  time_since_update_++;
}

void
Track::apply_cmc(const Eigen::Matrix<float, 2, 3> &warp_matrix)
{
  // Extract rotation and translation
  Eigen::Matrix2f rot = warp_matrix.block<2, 2>(0, 0);
  Eigen::Vector2f trans = warp_matrix.block<2, 1>(0, 2);

  // Create 8x8 rotation matrix using Kronecker product
  Eigen::MatrixXf rot_8x8 = Eigen::MatrixXf::Zero(8, 8);
  for (int i = 0; i < 4; ++i) {
    rot_8x8.block<2, 2>(i * 2, i * 2) = rot;
  }

  // Apply to mean
  Eigen::VectorXf mean = kf_->get_state();
  mean = rot_8x8 * mean;
  mean.head<2>() += trans;
  kf_->set_state(mean);

  // Apply to covariance
  Eigen::MatrixXf cov = kf_->get_covariance();
  cov = rot_8x8 * cov * rot_8x8.transpose();
  kf_->set_covariance(cov);
}

Eigen::Vector4f
Track::get_bbox() const
{
  Eigen::Vector4f xyah = kf_->project();
  return state_to_bbox(xyah);
}

Eigen::Vector4f
Track::get_x1y1wh() const
{
  Eigen::Vector4f bbox = get_bbox();
  return Eigen::Vector4f(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
}

Eigen::Matrix<float, 4, 2>
Track::get_velocities(int delta_t) const
{
  Eigen::Matrix<float, 4, 2> velocities = Eigen::Matrix<float, 4, 2>::Zero();

  // Find frame delta_t frames ago
  int target_frame = end_frame_ - delta_t;
  if (history_.find(target_frame) == history_.end()) {
    return velocities;
  }

  // Get current and previous boxes
  Eigen::Vector4f curr_box = get_bbox();
  Eigen::Vector4f prev_box = history_.at(target_frame).bbox;

  // Compute deltas
  Eigen::Vector4f deltas = curr_box - prev_box;

  // Compute normalized velocities for each corner
  // Top-left
  float norm_tl = std::sqrt(deltas[0] * deltas[0] + deltas[1] * deltas[1]) + 1e-5f;
  velocities(0, 0) = deltas[0] / norm_tl;
  velocities(0, 1) = deltas[1] / norm_tl;

  // Top-right
  float norm_tr = std::sqrt(deltas[2] * deltas[2] + deltas[1] * deltas[1]) + 1e-5f;
  velocities(1, 0) = deltas[2] / norm_tr;
  velocities(1, 1) = deltas[1] / norm_tr;

  // Bottom-left
  float norm_bl = std::sqrt(deltas[0] * deltas[0] + deltas[3] * deltas[3]) + 1e-5f;
  velocities(2, 0) = deltas[0] / norm_bl;
  velocities(2, 1) = deltas[3] / norm_bl;

  // Bottom-right
  float norm_br = std::sqrt(deltas[2] * deltas[2] + deltas[3] * deltas[3]) + 1e-5f;
  velocities(3, 0) = deltas[2] / norm_br;
  velocities(3, 1) = deltas[3] / norm_br;

  return velocities;
}

Eigen::Vector4f
Track::get_prev_bbox(int delta_t) const
{
  // Get bbox from delta_t frames ago
  int target_frame = end_frame_ - delta_t;

  // Look for the target frame in history
  auto it = history_.find(target_frame);
  if (it != history_.end()) {
    return it->second.bbox;
  }

  // If not found, try to find the closest available frame
  for (int i = 1; i < delta_t; ++i) {
    it = history_.find(end_frame_ - delta_t + i);
    if (it != history_.end()) {
      return it->second.bbox;
    }
  }

  // If still not found, return the oldest available bbox
  if (!history_.empty()) {
    return history_.begin()->second.bbox;
  }

  // Fallback to current bbox
  return get_bbox();
}

float
Track::get_prev_score() const
{
  // Get score from previous frame for confidence projection
  // Python uses: frame_ids[min(1, len(frame_ids) - 1)]

  if (history_.size() <= 1) {
    // If no previous frame, return current score
    return score_;
  }

  // Find the second most recent frame
  int prev_frame = end_frame_ - 1;
  auto it = history_.find(prev_frame);
  if (it != history_.end()) {
    return it->second.score;
  }

  // If not found, try to find any previous frame
  for (int i = 2; i <= 5; ++i) {
    it = history_.find(end_frame_ - i);
    if (it != history_.end()) {
      return it->second.score;
    }
  }

  return score_; // Fallback
}


Eigen::VectorXf
Track::get_mean() const
{
  return kf_->get_state();
}

Eigen::MatrixXf
Track::get_covariance() const
{
  return kf_->get_covariance();
}

void
Track::set_mean(const Eigen::VectorXf &mean)
{
  kf_->set_state(mean);
}

void
Track::set_covariance(const Eigen::MatrixXf &cov)
{
  kf_->set_covariance(cov);
}

void
Track::add_history(int frame_id, const FrameData &data)
{
  history_[frame_id] = data;
}

bool
Track::has_history(int frame_id) const
{
  return history_.find(frame_id) != history_.end();
}

const Track::FrameData &
Track::get_history(int frame_id) const
{
  return history_.at(frame_id);
}

Eigen::Vector4f
Track::state_to_bbox(const Eigen::VectorXf &state) const
{
  // Convert from (cx, cy, a, h) to (x1, y1, x2, y2)
  float cx = state[0];
  float cy = state[1];
  float a = state[2]; // aspect ratio
  float h = state[3];

  float w = a * h;
  float x1 = cx - w / 2.0f;
  float y1 = cy - h / 2.0f;
  float x2 = cx + w / 2.0f;
  float y2 = cy + h / 2.0f;

  return Eigen::Vector4f(x1, y1, x2, y2);
}

} // namespace tracktrack
