// Copyright Axelera AI, 2025
#ifndef TRACKTRACK_TRACKTRACK_HPP
#define TRACKTRACK_TRACKTRACK_HPP

#include <memory>
#include <string>
#include <vector>
#include "Assignment.hpp"
#include "Track.hpp"

// Forward declaration for CMC
class CMCComputer;

namespace tracktrack
{

class TrackTrack
{
  public:
  struct Params {
    // Detection thresholds
    float det_thr = 0.6f; // Detection threshold for high confidence
    float init_thr = 0.6f; // Initialization threshold for new tracks

    // Matching parameters
    float match_thr = 0.8f; // Initial matching threshold
    float tai_thr = 0.55f; // Track-aware NMS threshold
    float penalty_p = 0.20f; // Penalty for low confidence detections
    float penalty_q = 0.40f; // Penalty for deleted detections
    float reduce_step = 0.05f; // Step reduction for iterative matching

    // Track management
    int max_time_lost = 30; // Frames before removing lost track
    int min_len = 3; // Minimum track length for Tracked state
    int min_box_area = 100; // Minimum box area

    // Feature parameters
    float alpha = 0.95f; // EMA weight for appearance features

    // Optional components
    bool use_cmc = true; // Enable camera motion compensation
    bool use_aflink = false; // Enable AFLink post-processing
    std::string aflink_model; // Path to AFLink ONNX model

    // Dataset-specific
    std::string dataset_type = "MOT"; // "MOT" or "DanceTrack"
  };

  // Constructor
  TrackTrack(const Params &params, const std::string &video_name = "");
  ~TrackTrack();

  // Main update function
  std::vector<Track *> update(const std::vector<Detection> &detections,
      const std::vector<Detection> &detections_95, // High confidence detections
      const Eigen::Matrix<float, 2, 3> &cmc_transform
      = Eigen::Matrix<float, 2, 3>::Identity());

  // Update without detections (for frames with no detections)
  std::vector<Track *> update_without_detections();

  // Get all tracks
  std::vector<Track *> get_tracks() const
  {
    return tracks_;
  }

  // Get active tracks (Tracked state only)
  std::vector<Track *> get_active_tracks() const;

  // Reset tracker
  void reset();

  private:
  // Parameters
  Params params_;

  // Track management
  std::vector<Track *> tracks_;
  int frame_id_;

  // CMC computer
  std::unique_ptr<CMCComputer> cmc_computer_;
  bool use_builtin_cmc_;

  // Initialize new tracks
  void init_tracks(const std::vector<Detection> &detections);

  // Clean up removed tracks
  void cleanup_tracks();

  // Split tracks by state
  std::vector<Track *> get_tracked_lost_tracks() const;
  std::vector<Track *> get_new_tracks() const;

  // Apply CMC to tracks
  void apply_cmc_to_tracks(
      std::vector<Track *> &tracks, const Eigen::Matrix<float, 2, 3> &transform);

  // Update frame rate-dependent parameters
  void update_max_time_lost(int frame_rate);
};

} // namespace tracktrack

#endif // TRACKTRACK_TRACKTRACK_HPP
