// Copyright Axelera AI, 2024
#include "../include/OCSort.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include "iomanip"
#include "lapjv.hpp"

namespace ocsort
{
template <typename Matrix>
std::ostream &
operator<<(std::ostream &os, const std::vector<Matrix> &v)
{
  os << "{";
  for (auto it = v.begin(); it != v.end(); ++it) {
    os << "(" << *it << ")\n";
    if (it != v.end() - 1)
      os << ",";
  }
  os << "}\n";
  return os;
}

OCSort::OCSort(float det_thresh_, int max_age_, int min_hits_,
    float iou_threshold_, int delta_t_, float inertia_, float w_assoc_emb_,
    float alpha_fixed_emb_, int max_id_, bool aw_off_, float aw_param_,
    bool cmc_off_, bool enable_id_recovery_, int img_width_, int img_height_,
    int rec_image_rect_margin_, int rec_track_min_time_since_update_at_boundary_,
    int rec_track_min_time_since_update_inside_, int rec_track_min_age_,
    float rec_track_merge_lap_thresh_, int rec_track_memory_capacity_, int rec_track_memory_max_age_)
    : det_thresh(det_thresh_),
      max_age(max_age_),
      min_hits(min_hits_),
      iou_threshold(iou_threshold_),
      delta_t(delta_t_),
      inertia(inertia_),
      w_assoc_emb(w_assoc_emb_),
      alpha_fixed_emb(alpha_fixed_emb_),
      aw_off(aw_off_),
      aw_param(aw_param_),
      cmc_off(cmc_off_),
      enable_id_recovery(enable_id_recovery_),
      rec_image_rect_margin(rec_image_rect_margin_),
      rec_track_min_time_since_update_at_boundary(rec_track_min_time_since_update_at_boundary_),
      rec_track_min_time_since_update_inside(rec_track_min_time_since_update_inside_),
      rec_track_min_age(rec_track_min_age_),
      rec_track_merge_lap_thresh(rec_track_merge_lap_thresh_),
      rec_track_memory_capacity(rec_track_memory_capacity_),
      rec_track_memory_max_age(rec_track_memory_max_age_),
      frame_count(0)
{
  const auto make_error = [](const char *param, const std::string &detail) {
    return std::invalid_argument(
        "OCSort tracker: parameter '" + std::string(param) + "' " + detail);
  };

  const auto require = [&](bool condition, const char *param, const std::string &detail) {
    if (!condition)
      throw make_error(param, detail);
  };

  const auto require_finite = [&](float value, const char *param) {
    if (!std::isfinite(value))
      throw make_error(param, "must be finite.");
  };

  require_finite(det_thresh, "det_thresh");
  require(det_thresh >= 0.0f && det_thresh <= 1.0f, "det_thresh", "must be in the range [0, 1].");

  require(max_age >= 0, "max_age", "must be non-negative.");
  require(min_hits >= 0, "min_hits", "must be non-negative.");

  require_finite(iou_threshold, "iou_threshold");
  require(iou_threshold >= 0.0f && iou_threshold <= 1.0f, "iou_threshold",
      "must be in the range [0, 1].");

  require(delta_t > 0, "delta_t", "must be positive.");

  require_finite(inertia, "inertia");
  require(inertia >= 0.0f && inertia <= 1.0f, "inertia", "must be in the range [0, 1].");

  require_finite(w_assoc_emb, "w_assoc_emb");
  require(w_assoc_emb >= 0.0f, "w_assoc_emb", "must be non-negative.");

  require_finite(alpha_fixed_emb, "alpha_fixed_emb");
  require(alpha_fixed_emb >= 0.0f && alpha_fixed_emb <= 1.0f, "alpha_fixed_emb",
      "must be in the range [0, 1].");

  require(max_id_ >= 0, "max_id", "must be non-negative.");

  require_finite(aw_param, "aw_param");
  require(aw_param >= 0.0f, "aw_param", "must be non-negative.");

  require(img_width_ >= 0, "img_width", "must be non-negative.");
  require(img_height_ >= 0, "img_height", "must be non-negative.");

  require(rec_image_rect_margin >= 0, "rec_image_rect_margin", "must be non-negative.");
  require(rec_track_min_time_since_update_at_boundary >= 0,
      "rec_track_min_time_since_update_at_boundary", "must be non-negative.");
  require(rec_track_min_time_since_update_inside >= 0,
      "rec_track_min_time_since_update_inside", "must be non-negative.");
  require(rec_track_min_age >= 0, "rec_track_min_age", "must be non-negative.");

  require_finite(rec_track_merge_lap_thresh, "rec_track_merge_lap_thresh");
  require(rec_track_merge_lap_thresh >= 0.0f && rec_track_merge_lap_thresh <= 2.0f,
      "rec_track_merge_lap_thresh", "must be in the range [0, 2].");

  require(rec_track_memory_capacity >= 0, "rec_track_memory_capacity", "must be non-negative.");

  require(rec_track_memory_max_age >= 0, "rec_track_memory_max_age", "must be non-negative.");

  if (enable_id_recovery) {
    require(img_width_ > 0, "img_width", "must be positive when enable_id_recovery is true.");
    require(img_height_ > 0, "img_height", "must be positive when enable_id_recovery is true.");

    require(max_age >= rec_track_min_time_since_update_at_boundary, "max_age",
        "must be >= rec_track_min_time_since_update_at_boundary when enable_id_recovery is true; "
        "this is necessary for the ID recovery feature (memory bank) to work correctly; "
        "boundary reappearing trackers may never be added when max_age is smaller.");
    require(max_age >= rec_track_min_time_since_update_inside, "max_age",
        "must be >= rec_track_min_time_since_update_inside when enable_id_recovery is true; "
        "this is necessary for the ID recovery feature (memory bank) to work correctly; "
        "inside reappearing trackers may never be added when max_age is smaller.");
  }

  KalmanBoxTracker::max_id = max_id_;
  img_rect = Rect(0.0f, 0.0f, static_cast<float>(img_width_), static_cast<float>(img_height_));
}
std::ostream &
precision(std::ostream &os)
{
  os << std::fixed << std::setprecision(2);
  return os;
}

bool
isOutOfImageRect(Rect image_rect, Rect obj_rect, int dist_thresh)
{
  const float shrink = static_cast<float>(dist_thresh);
  const Rect inner_rect(image_rect.x + shrink, image_rect.y + shrink,
      std::max(0.0f, image_rect.width - 2.0f * shrink),
      std::max(0.0f, image_rect.height - 2.0f * shrink));

  const Rect inter = inner_rect.intersect(obj_rect);
  return inter.area() < obj_rect.area();
}


void
mergeTracksIfTheSame(std::vector<KalmanBoxTracker> &active_tracks,
    std::vector<KalmanBoxTracker> &reappearing_tracks, float merge_thresh = 0.09f)
{
  if (active_tracks.empty() || reappearing_tracks.empty())
    return;

  // Prepare embedding matrices
  int n_active = active_tracks.size();
  int n_reappear = reappearing_tracks.size();
  int emb_dim = active_tracks[0].get_emb().size();
  if (emb_dim == 0)
    return;

  Eigen::MatrixXf active_embs(n_active, emb_dim);
  Eigen::MatrixXf reappear_embs(n_reappear, emb_dim);

  for (int i = 0; i < n_active; ++i)
    active_embs.row(i) = active_tracks[i].get_emb().transpose();
  for (int i = 0; i < n_reappear; ++i)
    reappear_embs.row(i) = reappearing_tracks[i].get_emb().transpose();

  // Compute cosine distance matrix
  std::vector<std::vector<float>> cost_matrix(
      n_active, std::vector<float>(n_reappear, 1.0f));
  for (int i = 0; i < n_active; ++i) {
    for (int j = 0; j < n_reappear; ++j) {
      float dot = active_embs.row(i).dot(reappear_embs.row(j));
      float norm1 = active_embs.row(i).norm();
      float norm2 = reappear_embs.row(j).norm();
      float cosine_sim = (norm1 > 1e-6f && norm2 > 1e-6f) ? dot / (norm1 * norm2) : 0.0f;
      cost_matrix[i][j] = 1.0f - cosine_sim; // cosine distance
    }
  }

  // Solve LAP (Hungarian) assignment
  std::vector<int> rowsol, colsol;
  execLapjv(cost_matrix, rowsol, colsol, true, merge_thresh, false);

  // Threshold for merging (tune as needed, e.g., 0.2 means cosine similarity > 0.8)

  // Merge trackers
  std::vector<bool> merged_reappear(n_reappear, false);
  for (int i = 0; i < n_active; ++i) {
    int j = rowsol[i];
    if (j >= 0 && j < n_reappear && cost_matrix[i][j] < merge_thresh) {
      // Merge reappearing_tracks[j] into active_tracks[i]
      // You can choose to keep the one with longer history, or just remove the
      // duplicate Here, we remove the reappearing tracker
      merged_reappear[j] = true;
      active_tracks[i].id = reappearing_tracks[j].id; // Update ID if needed

      // Average embeddings
      active_tracks[i].update_emb(reappearing_tracks[j].get_emb(), 0.5);
    }
  }

  // Remove merged reappearing trackers
  std::vector<KalmanBoxTracker> new_reappearing;
  for (int j = 0; j < n_reappear; ++j) {
    if (!merged_reappear[j]) {
      new_reappearing.push_back(std::move(reappearing_tracks[j]));
    }
  }
  reappearing_tracks = std::move(new_reappearing);
}

std::vector<Eigen::RowVectorXf>
OCSort::update(Eigen::MatrixXf dets, Eigen::MatrixXf embs, Eigen::Matrix<float, 2, 3> cmc_transform)
{
  std::vector<KalmanBoxTracker> trackers;
  trackers.insert(trackers.end(), active_trackers.begin(), active_trackers.end()); // Add active trackers
  trackers.insert(trackers.end(), not_active_trackers.begin(),
      not_active_trackers.end()); // Add not active trackers
  trackers.insert(trackers.end(), lost_trackers.begin(), lost_trackers.end()); // Add lost trackers


  bool use_reid = false;
  if (embs.size() > 0) {
    use_reid = true;

    if (embs.rows() != dets.rows()) {
      std::string error_msg("DeepOCSort tracker: Error: The number of detections and embeddings do not match.");
      throw std::runtime_error(error_msg);
    }

    // Normalize each row of embs
    for (int i = 0; i < embs.rows(); ++i) {
      float magnitude = std::sqrt(embs.row(i).squaredNorm());
      if (magnitude > 0) {
        embs.row(i) /= magnitude;
      }
    }
  }

  frame_count += 1;

  Eigen::Matrix<float, Eigen::Dynamic, 4> xyxys = dets.leftCols(4);
  Eigen::Matrix<float, 1, Eigen::Dynamic> confs = dets.col(4);
  Eigen::Matrix<float, 1, Eigen::Dynamic> clss = dets.col(5);
  Eigen::MatrixXf output_results = dets;
  auto inds_low = confs.array() > 0.1;
  auto inds_high = confs.array() < det_thresh;
  auto inds_second = inds_low && inds_high;
  Eigen::Matrix<float, Eigen::Dynamic, 6> dets_second;
  Eigen::Matrix<bool, 1, Eigen::Dynamic> remain_inds = (confs.array() > det_thresh);
  Eigen::Matrix<float, Eigen::Dynamic, 6> dets_first;
  std::vector<int> map_dets_first_to_dets;
  Eigen::MatrixXf dets_embs = Eigen::MatrixXf::Zero(0, embs.cols());
  for (int i = 0; i < output_results.rows(); i++) {
    if (true == inds_second(i)) {
      dets_second.conservativeResize(dets_second.rows() + 1, Eigen::NoChange);
      dets_second.row(dets_second.rows() - 1) = output_results.row(i);
    }
    if (true == remain_inds(i)) {
      dets_first.conservativeResize(dets_first.rows() + 1, Eigen::NoChange);
      dets_first.row(dets_first.rows() - 1) = output_results.row(i);
      map_dets_first_to_dets.push_back(i);

      if (use_reid) {
        // Fill dets_embs with corresponding embeddings
        dets_embs.conservativeResize(dets_embs.rows() + 1, Eigen::NoChange);
        dets_embs.row(dets_embs.rows() - 1) = embs.row(i);
      }
    }
  }

  // Apply CMC affine correction if cmc is enabled
  if (!cmc_off) {
    for (auto &tracker : trackers) {
      tracker.apply_affine_correction(cmc_transform);
    }
  }

  // Dynamic Appearance, 3.3 from https://arxiv.org/pdf/2302.11813
  Eigen::VectorXf trust = (dets_first.col(4).array() - det_thresh) / (1 - det_thresh);
  float af = alpha_fixed_emb;
  Eigen::VectorXf dets_alpha = af + (1 - af) * (1 - trust.array());

  Eigen::MatrixXf trks = Eigen::MatrixXf::Zero(trackers.size(), 5);
  std::vector<int> to_del;

  int emb_len = 0;
  if (use_reid && trackers.size() > 0)
    emb_len = trackers[0].get_emb().size();

  Eigen::MatrixXf trk_embs = Eigen::MatrixXf::Zero(trackers.size(), emb_len);
  for (int i = 0; i < trks.rows(); i++) {
    Eigen::RowVectorXf pos = trackers[i].predict();
    trks.row(i) << pos(0), pos(1), pos(2), pos(3), 0;

    if (use_reid)
      trk_embs.row(i) = trackers[i].get_emb().transpose();
  }
  Eigen::MatrixXf velocities = Eigen::MatrixXf::Zero(trackers.size(), 2);
  Eigen::MatrixXf last_boxes = Eigen::MatrixXf::Zero(trackers.size(), 5);
  Eigen::MatrixXf k_observations = Eigen::MatrixXf::Zero(trackers.size(), 5);
  for (int i = 0; i < trackers.size(); i++) {
    velocities.row(i) = trackers[i].velocity;
    last_boxes.row(i) = trackers[i].last_observation;
    k_observations.row(i)
        = k_previous_obs(trackers[i].observations, trackers[i].age, delta_t);
  }
  /////////////////////////
  ///  Step1 First round of association
  ////////////////////////
  // Perform IOU association associate()
  std::vector<Eigen::Matrix<int, 1, 2>> matched;
  std::vector<int> unmatched_dets;
  std::vector<int> unmatched_trks;
  auto result = associate(dets_first, trks, dets_embs, trk_embs, iou_threshold,
      velocities, k_observations, inertia, w_assoc_emb, aw_off, aw_param);
  matched = std::get<0>(result);
  unmatched_dets = std::get<1>(result);
  unmatched_trks = std::get<2>(result);
  // Update matched tracks
  for (auto m : matched) {
    Eigen::Matrix<float, 5, 1> tmp_bbox;
    tmp_bbox = dets_first.block<1, 5>(m(0), 0);
    trackers[m(1)].update(&(tmp_bbox), dets_first(m(0), 5),
        map_dets_first_to_dets[m(0)], frame_count);
    if (use_reid)
      trackers[m(1)].update_emb(dets_embs.row(m(0)), dets_alpha(m(0)));
  }

  ///////////////////////
  /// Step2 Second round of association by OCR to find lost tracks back
  ///////////////////////
  if (unmatched_dets.size() > 0 && unmatched_trks.size() > 0) {
    Eigen::MatrixXf left_dets(unmatched_dets.size(), 6);
    int inx_for_dets = 0;
    for (auto i : unmatched_dets) {
      left_dets.row(inx_for_dets++) = dets_first.row(i);
    }
    Eigen::MatrixXf left_trks(unmatched_trks.size(), last_boxes.cols());
    int indx_for_trk = 0;
    for (auto i : unmatched_trks) {
      left_trks.row(indx_for_trk++) = last_boxes.row(i);
    }
    Eigen::MatrixXf iou_left = giou_batch(left_dets, left_trks);
    if (iou_left.maxCoeff() > iou_threshold) {
      std::vector<std::vector<float>> iou_matrix(
          iou_left.rows(), std::vector<float>(iou_left.cols()));
      for (int i = 0; i < iou_left.rows(); i++) {
        for (int j = 0; j < iou_left.cols(); j++) {
          iou_matrix[i][j] = -iou_left(i, j);
        }
      }
      std::vector<int> rowsol, colsol;
      execLapjv(iou_matrix, rowsol, colsol, true, 0.01, false);
      std::vector<std::vector<int>> rematched_indices;
      for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol.at(i) >= 0) {
          rematched_indices.push_back({ colsol.at(rowsol.at(i)), rowsol.at(i) });
        }
      }
      // If still unmatched after reassignment, these need to be deleted
      std::vector<int> to_remove_det_indices;
      std::vector<int> to_remove_trk_indices;
      for (auto i : rematched_indices) {
        int det_ind = unmatched_dets[i.at(0)];
        int trk_ind = unmatched_trks[i.at(1)];
        if (iou_left(i.at(0), i.at(1)) < iou_threshold) {
          continue;
        }
        ////////////////////////////////
        ///  Step3  update status of second matched tracks
        ///////////////////////////////
        Eigen::Matrix<float, 5, 1> tmp_bbox;
        tmp_bbox = dets_first.block<1, 5>(det_ind, 0);
        trackers.at(trk_ind).update(&tmp_bbox, dets_first(det_ind, 5),
            map_dets_first_to_dets[det_ind], frame_count);
        if (use_reid)
          trackers.at(trk_ind).update_emb(dets_embs.row(det_ind), dets_alpha(det_ind));
        to_remove_det_indices.push_back(det_ind);
        to_remove_trk_indices.push_back(trk_ind);
      }
      std::vector<int> tmp_res(unmatched_dets.size());
      sort(unmatched_dets.begin(), unmatched_dets.end());
      sort(to_remove_det_indices.begin(), to_remove_det_indices.end());
      auto end = set_difference(unmatched_dets.begin(), unmatched_dets.end(),
          to_remove_det_indices.begin(), to_remove_det_indices.end(), tmp_res.begin());
      tmp_res.resize(end - tmp_res.begin());
      unmatched_dets = tmp_res;
      std::vector<int> tmp_res1(unmatched_trks.size());
      sort(unmatched_trks.begin(), unmatched_trks.end());
      sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());
      auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
          to_remove_trk_indices.begin(), to_remove_trk_indices.end(), tmp_res1.begin());
      tmp_res1.resize(end1 - tmp_res1.begin());
      unmatched_trks = tmp_res1;
    }
  }

  for (auto m : unmatched_trks) {
    trackers.at(m).update(nullptr, 0, -1, frame_count);
  }
  ///////////////////////////////
  /// Step4 Initialize new tracks and remove expired tracks
  ///////////////////////////////
  /* Create and initialize new trackers for unmatched detections */
  for (int i : unmatched_dets) {
    Eigen::RowVectorXf tmp_bbox = dets_first.block(i, 0, 1, 5);
    int cls_ = int(dets_first(i, 5));
    Eigen::VectorXf det_emb_;
    if (use_reid)
      det_emb_ = dets_embs.row(i);

    KalmanBoxTracker trk = KalmanBoxTracker(tmp_bbox, det_emb_, cls_,
        map_dets_first_to_dets[i], delta_t, frame_count);
    // Append newly created tracker to the end of trackers
    trackers.push_back(trk);
  }
  std::vector<Eigen::RowVectorXf> ret;
  active_trackers.clear();
  lost_trackers.clear();
  not_active_trackers.clear();
  for (int i = trackers.size() - 1; i >= 0; i--) {
    if (trackers.at(i).time_since_update > max_age) {
      trackers.at(i).release_id();
      continue; // Skip this tracker - don't add to any list
    }

    Eigen::Matrix<float, 1, 4> d;
    int last_observation_sum = trackers.at(i).last_observation.sum();
    if (last_observation_sum < 0) {
      d = trackers.at(i).get_state();
    } else {
      d = trackers.at(i).last_observation.block(0, 0, 1, 4);
    }
    const int age_thresh = 1;
    if (trackers.at(i).time_since_update < age_thresh
        && ((trackers.at(i).hit_streak >= min_hits) || (frame_count <= min_hits))) {
      Eigen::RowVectorXf tracking_res(8);
      tracking_res << d(0), d(1), d(2), d(3), trackers.at(i).id + 1,
          trackers.at(i).cls, trackers.at(i).conf, trackers.at(i).latest_detection_id;
      ret.push_back(tracking_res);

      active_trackers.push_back(trackers.at(i));
    } else if (trackers.at(i).time_since_update >= age_thresh) {
      lost_trackers.push_back(trackers.at(i));
    } else {
      not_active_trackers.push_back(trackers.at(i));
    }
  }

  if (enable_id_recovery) {
    std::vector<KalmanBoxTracker> lost_trackers_temp;
    lost_trackers_temp.reserve(lost_trackers.size());
    for (auto it = lost_trackers.begin(); it != lost_trackers.end(); ++it) {
      Rect obj_rect(it->last_observation(0), it->last_observation(1),
          it->last_observation(2) - it->last_observation(0),
          it->last_observation(3) - it->last_observation(1));

      if (it->time_since_update >= rec_track_min_time_since_update_at_boundary) {
        if (isOutOfImageRect(img_rect, obj_rect, rec_image_rect_margin)) {
          if (it->age > rec_track_min_age) {
            reappearing_trackers.push_back(*it);
          }
        } else if (it->time_since_update >= rec_track_min_time_since_update_inside) {
          if (it->age > rec_track_min_age) {
            reappearing_trackers.push_back(*it);
          }
        } else {
          lost_trackers_temp.push_back(*it);
        }
      } else {
        lost_trackers_temp.push_back(*it);
      }
    }

    lost_trackers = std::move(lost_trackers_temp);

    mergeTracksIfTheSame(active_trackers, reappearing_trackers, rec_track_merge_lap_thresh);

    const auto frames_since_update = [&](const KalmanBoxTracker &tracker) -> int {
      return std::max(0, frame_count - tracker.frame_of_last_update);
    };

    if (rec_track_memory_max_age >= 0) {
      for (auto it = reappearing_trackers.begin(); it != reappearing_trackers.end();) {
        if (frames_since_update(*it) > rec_track_memory_max_age) {
          it->release_id();
          it = reappearing_trackers.erase(it);
        } else {
          ++it;
        }
      }
    }

    if (rec_track_memory_capacity >= 0
        && reappearing_trackers.size() > static_cast<std::size_t>(rec_track_memory_capacity)) {
      auto nth = reappearing_trackers.begin() + rec_track_memory_capacity;
      std::nth_element(reappearing_trackers.begin(), nth, reappearing_trackers.end(),
          [&](const KalmanBoxTracker &lhs, const KalmanBoxTracker &rhs) {
            return frames_since_update(lhs) < frames_since_update(rhs);
          });

      for (auto it = nth; it != reappearing_trackers.end(); ++it) {
        it->release_id();
      }
      reappearing_trackers.erase(nth, reappearing_trackers.end());
    }
  }

  return ret;
}
} // namespace ocsort
