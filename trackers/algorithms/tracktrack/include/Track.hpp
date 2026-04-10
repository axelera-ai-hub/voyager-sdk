// Copyright Axelera AI, 2025
#ifndef TRACKTRACK_TRACK_HPP
#define TRACKTRACK_TRACK_HPP

#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tracktrack
{

// Forward declaration
class KalmanFilter;

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

struct Detection {
  Eigen::Vector4f bbox; // x1, y1, x2, y2
  float score;
  int class_id;
  int original_index = -1; // Index in the caller's detection list
  Eigen::VectorXf features; // Optional appearance features

  Detection()
      : score(0.0f),
        class_id(-1)
  {
  }

  Detection(float x1, float y1, float x2, float y2, float s, int cls = -1)
      : bbox(x1, y1, x2, y2),
        score(s),
        class_id(cls)
  {
  }

  // Convert to center-width-height format
  Eigen::Vector4f to_xyah() const
  {
    float cx = (bbox[0] + bbox[2]) / 2.0f;
    float cy = (bbox[1] + bbox[3]) / 2.0f;
    float w = bbox[2] - bbox[0];
    float h = bbox[3] - bbox[1];
    float a = w / h; // aspect ratio
    return Eigen::Vector4f(cx, cy, a, h);
  }

  // Convert to x1y1wh format
  Eigen::Vector4f to_x1y1wh() const
  {
    return Eigen::Vector4f(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
  }
};

class Track
{
  public:
  // Constructor
  Track(const Detection &det, int track_id, int frame_id, float alpha = 0.95f);

  // Destructor (needed because of unique_ptr with forward declaration)
  ~Track();

  // State management
  void mark_lost();
  void mark_removed();
  void mark_tracked();
  TrackState get_state() const
  {
    return state_;
  }

  // Update track with new detection
  void update(const Detection &det, int frame_id);

  // Predict next position using Kalman filter
  void predict();

  // Apply camera motion compensation
  void apply_cmc(const Eigen::Matrix<float, 2, 3> &warp_matrix);

  // Getters
  int get_track_id() const
  {
    return track_id_;
  }
  int get_class_id() const
  {
    return class_id_;
  }
  float get_score() const
  {
    return score_;
  }
  int get_latest_detection_id() const
  {
    return latest_detection_id_;
  }
  Eigen::Vector4f get_bbox() const; // Returns x1y1x2y2
  Eigen::Vector4f get_x1y1wh() const;

  // For velocity computation
  Eigen::Matrix<float, 4, 2> get_velocities(int delta_t = 3) const;
  Eigen::Vector4f get_prev_bbox(int delta_t = 3) const;
  float get_prev_score() const; // Get previous frame's score for confidence projection

  // Feature management
  void update_features(const Eigen::VectorXf &new_features);
  Eigen::VectorXf get_features() const
  {
    return features_;
  }

  // Frame tracking
  int get_start_frame() const
  {
    return start_frame_;
  }
  int get_end_frame() const
  {
    return end_frame_;
  }
  int get_tracklet_len() const
  {
    return tracklet_len_;
  }

  // For Kalman filter access
  Eigen::VectorXf get_mean() const;
  Eigen::MatrixXf get_covariance() const;
  void set_mean(const Eigen::VectorXf &mean);
  void set_covariance(const Eigen::MatrixXf &cov);

  // History management
  struct FrameData {
    Eigen::Vector4f bbox;
    float score;
    Eigen::VectorXf mean;
    Eigen::MatrixXf covariance;
    Eigen::VectorXf features;
  };

  void add_history(int frame_id, const FrameData &data);
  bool has_history(int frame_id) const;
  const FrameData &get_history(int frame_id) const;
  size_t get_history_size() const
  {
    return history_.size();
  }

  private:
  // Track identification
  int track_id_;
  int class_id_;
  TrackState state_;

  // Track metrics
  float score_;
  int start_frame_;
  int end_frame_;
  int tracklet_len_;
  int time_since_update_;
  int latest_detection_id_ = -1;

  // Kalman filter for motion prediction
  std::unique_ptr<KalmanFilter> kf_;

  // Appearance features (exponentially weighted average)
  Eigen::VectorXf features_;
  float alpha_; // EMA weight for feature update

  // History for velocity computation
  std::unordered_map<int, FrameData> history_;

  // Helper to convert Kalman state to bbox
  Eigen::Vector4f state_to_bbox(const Eigen::VectorXf &state) const;
};

// Track counter for unique IDs
class TrackCounter
{
  public:
  static TrackCounter &instance()
  {
    static TrackCounter instance;
    return instance;
  }

  int get_next_id()
  {
    return ++track_count_;
  }

  void reset()
  {
    track_count_ = 0;
  }

  private:
  TrackCounter()
      : track_count_(0)
  {
  }
  int track_count_;
};

} // namespace tracktrack

#endif // TRACKTRACK_TRACK_HPP
