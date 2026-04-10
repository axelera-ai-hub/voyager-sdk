// Copyright Axelera AI, 2025
#include "TrackTrack.hpp"
#include <algorithm>
#include <iostream>
#include "cmc.hpp"

namespace tracktrack
{

TrackTrack::TrackTrack(const Params &params, const std::string &video_name)
    : params_(params),
      frame_id_(0),
      use_builtin_cmc_(false)
{

  // Initialize CMC if enabled
  if (params_.use_cmc) {
    // Check if pre-computed CMC file exists
    // For now, use built-in CMC computer
    cmc_computer_ = std::make_unique<CMCComputer>();
    use_builtin_cmc_ = true;
  }

  // Reset track counter
  TrackCounter::instance().reset();
}

TrackTrack::~TrackTrack()
{
  // Clean up all tracks
  for (Track *track : tracks_) {
    delete track;
  }
}

std::vector<Track *>
TrackTrack::update(const std::vector<Detection> &detections,
    const std::vector<Detection> &detections_95, const Eigen::Matrix<float, 2, 3> &cmc_transform)
{

  // Update frame ID
  frame_id_++;

  // Debug output disabled - uncomment for troubleshooting
  // if (frame_id_ <= 15) {
  //     std::cout << "\nFrame " << frame_id_ << " - Update called with "
  //               << detections.size() << " detections" << std::endl;
  // }

  // Find deleted detections (from NMS)
  std::vector<Detection> dets_del = find_deleted_detections(detections, detections_95);

  // Split detections by confidence
  std::vector<Detection> dets_high, dets_low;
  for (const auto &det : detections) {
    // Debug output disabled
    // if (frame_id_ <= 3 && detections.size() <= 5) {
    //     std::cout << "    Detection score: " << det.score
    //               << " vs threshold " << params_.det_thr
    //               << " -> " << (det.score > params_.det_thr ? "HIGH" : "LOW") << std::endl;
    // }
    if (det.score > params_.det_thr) {
      dets_high.push_back(det);
    } else {
      dets_low.push_back(det);
    }
  }

  // Split deleted detections by confidence
  std::vector<Detection> dets_del_high;
  for (const auto &det : dets_del) {
    if (det.score > params_.det_thr) {
      dets_del_high.push_back(det);
    }
  }

  // Debug output disabled
  // if (frame_id_ <= 15) {
  //     std::cout << "  det_thr=" << params_.det_thr
  //               << ", dets_high=" << dets_high.size()
  //               << ", dets_low=" << dets_low.size()
  //               << ", dets_del=" << dets_del.size() << std::endl;
  // }

  // Get tracks by state
  std::vector<Track *> tracked_lost = get_tracked_lost_tracks();
  std::vector<Track *> new_tracks = get_new_tracks();

  // Apply CMC if enabled
  if (params_.use_cmc) {
    apply_cmc_to_tracks(tracked_lost, cmc_transform);
    apply_cmc_to_tracks(new_tracks, cmc_transform);
  }

  // Predict all tracks
  for (Track *track : tracked_lost) {
    track->predict();
  }
  for (Track *track : new_tracks) {
    track->predict();
  }

  // First association: tracked/lost tracks with all detections
  auto [matches1, u_tracks1, u_dets1] = iterative_assignment(tracked_lost,
      dets_high, dets_low, dets_del_high, params_.match_thr, params_.penalty_p,
      params_.penalty_q, params_.reduce_step, frame_id_);

  // Combine all detections for indexing
  std::vector<Detection> all_dets;
  all_dets.insert(all_dets.end(), dets_high.begin(), dets_high.end());
  all_dets.insert(all_dets.end(), dets_low.begin(), dets_low.end());
  all_dets.insert(all_dets.end(), dets_del_high.begin(), dets_del_high.end());

  // Update matched tracks
  for (const auto &[track_idx, det_idx] : matches1) {
    tracked_lost[track_idx]->update(all_dets[det_idx], frame_id_);
  }

  // Mark unmatched tracks as lost
  for (int idx : u_tracks1) {
    tracked_lost[idx]->mark_lost();
  }

  // Get remaining high confidence detections
  std::vector<Detection> dets_high_left;
  for (int idx : u_dets1) {
    if (idx < dets_high.size()) {
      dets_high_left.push_back(dets_high[idx]);
    }
  }

  // Second association: new tracks with remaining high confidence detections
  auto [matches2, u_tracks2, u_dets2] = iterative_assignment(new_tracks,
      dets_high_left, {}, {}, params_.match_thr, params_.penalty_p,
      params_.penalty_q, params_.reduce_step, frame_id_);

  // Update matched new tracks
  for (const auto &[track_idx, det_idx] : matches2) {
    new_tracks[track_idx]->update(dets_high_left[det_idx], frame_id_);
  }

  // Mark unmatched new tracks as removed
  for (int idx : u_tracks2) {
    new_tracks[idx]->mark_removed();
  }

  // Mark old lost tracks as removed
  for (Track *track : tracks_) {
    if (frame_id_ - track->get_end_frame() > params_.max_time_lost) {
      track->mark_removed();
    }
  }

  // Clean up removed tracks
  cleanup_tracks();

  // Initialize new tracks from unmatched high confidence detections
  std::vector<Detection> new_dets;
  for (int idx : u_dets2) {
    new_dets.push_back(dets_high_left[idx]);
  }

  // Debug output disabled
  // if (frame_id_ <= 15 && !new_dets.empty()) {
  //     std::cout << "  Initializing " << new_dets.size() << " new tracks" << std::endl;
  // }

  init_tracks(new_dets);

  // Return active tracks
  auto active_tracks = get_active_tracks();

  // Debug output disabled
  // if (frame_id_ <= 15) {
  //     std::cout << "  Frame " << frame_id_ << ": returning " << active_tracks.size()
  //               << " active tracks (from " << tracks_.size() << " total)" << std::endl;
  //     for (const auto& track : active_tracks) {
  //         std::cout << "    Track " << track->get_track_id()
  //                   << ": state=" << static_cast<int>(track->get_state())
  //                   << ", history_size=" << track->get_history_size() << std::endl;
  //     }
  // }

  return active_tracks;
}

std::vector<Track *>
TrackTrack::update_without_detections()
{
  // Update frame ID
  frame_id_++;

  // Remove all new tracks
  for (Track *track : tracks_) {
    if (track->get_state() == TrackState::New) {
      track->mark_removed();
    }
  }

  // Get remaining tracks
  std::vector<Track *> active_tracks;
  for (Track *track : tracks_) {
    if (track->get_state() != TrackState::Removed && track->get_state() != TrackState::New) {
      active_tracks.push_back(track);
    }
  }

  // Apply CMC if enabled (identity transform)
  if (params_.use_cmc) {
    apply_cmc_to_tracks(active_tracks, Eigen::Matrix<float, 2, 3>::Identity());
  }

  // Predict all tracks
  for (Track *track : active_tracks) {
    track->predict();
  }

  // Mark old lost tracks as removed
  for (Track *track : tracks_) {
    if (frame_id_ - track->get_end_frame() > params_.max_time_lost) {
      track->mark_removed();
    }
  }

  // Clean up removed tracks
  cleanup_tracks();

  return get_active_tracks();
}

std::vector<Track *>
TrackTrack::get_active_tracks() const
{
  std::vector<Track *> active;
  for (Track *track : tracks_) {
    if (track->get_state() == TrackState::Tracked) {
      // Apply additional filters
      auto bbox = track->get_x1y1wh();
      float area = bbox[2] * bbox[3];

      // Check minimum box area
      if (area > params_.min_box_area) {
        // Check aspect ratio for MOT datasets
        if (params_.dataset_type == "MOT") {
          float aspect_ratio = bbox[2] / bbox[3];
          if (aspect_ratio <= 1.6f) {
            active.push_back(track);
          }
        } else {
          active.push_back(track);
        }
      }
    }
  }
  return active;
}

void
TrackTrack::reset()
{
  // Delete all tracks
  for (Track *track : tracks_) {
    delete track;
  }
  tracks_.clear();

  // Reset frame counter
  frame_id_ = 0;

  // Reset track ID counter
  TrackCounter::instance().reset();
}

void
TrackTrack::init_tracks(const std::vector<Detection> &detections)
{
  if (detections.empty())
    return;

  // Get alive tracks for track-aware NMS
  std::vector<Track *> alive_tracks;
  for (Track *track : tracks_) {
    if (track->get_state() == TrackState::Tracked || track->get_state() == TrackState::New) {
      alive_tracks.push_back(track);
    }
  }

  // Run track-aware NMS
  std::vector<bool> keep
      = track_aware_nms(alive_tracks, detections, params_.tai_thr, params_.init_thr);

  // Debug output disabled
  // if (frame_id_ <= 15) {
  //     int keep_count = 0;
  //     for (bool k : keep) if (k) keep_count++;
  //     std::cout << "    Track-aware NMS: " << detections.size()
  //               << " detections -> " << keep_count << " kept"
  //               << " (tai_thr=" << params_.tai_thr
  //               << ", init_thr=" << params_.init_thr << ")" << std::endl;
  // }

  // Initialize new tracks
  for (size_t i = 0; i < detections.size(); ++i) {
    if (keep[i]) {
      int track_id = TrackCounter::instance().get_next_id();
      Track *new_track = new Track(detections[i], track_id, frame_id_, params_.alpha);
      tracks_.push_back(new_track);
    }
  }
}

void
TrackTrack::cleanup_tracks()
{
  std::vector<Track *> kept_tracks;
  for (Track *track : tracks_) {
    if (track->get_state() != TrackState::Removed) {
      kept_tracks.push_back(track);
    } else {
      delete track;
    }
  }
  tracks_ = kept_tracks;
}

std::vector<Track *>
TrackTrack::get_tracked_lost_tracks() const
{
  std::vector<Track *> result;
  for (Track *track : tracks_) {
    if (track->get_state() == TrackState::Tracked || track->get_state() == TrackState::Lost) {
      result.push_back(track);
    }
  }
  return result;
}

std::vector<Track *>
TrackTrack::get_new_tracks() const
{
  std::vector<Track *> result;
  for (Track *track : tracks_) {
    if (track->get_state() == TrackState::New) {
      result.push_back(track);
    }
  }
  return result;
}

void
TrackTrack::apply_cmc_to_tracks(
    std::vector<Track *> &tracks, const Eigen::Matrix<float, 2, 3> &transform)
{
  if (tracks.empty())
    return;

  // Apply transform to each track
  for (Track *track : tracks) {
    track->apply_cmc(transform);
  }
}

void
TrackTrack::update_max_time_lost(int frame_rate)
{
  params_.max_time_lost = frame_rate * 2;
}

} // namespace tracktrack
