// Copyright Axelera AI, 2025
#include "Assignment.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_set>
#include "Hungarian.hpp"

namespace tracktrack
{

float
iou_distance(const Eigen::Vector4f &bbox1, const Eigen::Vector4f &bbox2)
{
  // Calculate intersection
  float x1 = std::max(bbox1[0], bbox2[0]);
  float y1 = std::max(bbox1[1], bbox2[1]);
  float x2 = std::min(bbox1[2], bbox2[2]);
  float y2 = std::min(bbox1[3], bbox2[3]);

  float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

  // Calculate union
  float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
  float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
  float union_area = area1 + area2 - inter_area;

  // IoU
  float iou = (union_area > 0) ? inter_area / union_area : 0.0f;
  return 1.0f - iou; // Return distance (1 - IoU)
}

float
h_iou_distance(const Eigen::Vector4f &bbox1, const Eigen::Vector4f &bbox2)
{
  // Height-aware IoU matching Python implementation
  // Calculate height IoU
  float h_inter = std::min(bbox1[3], bbox2[3]) - std::max(bbox1[1], bbox2[1]);
  float h_union = std::max(bbox1[3], bbox2[3]) - std::min(bbox1[1], bbox2[1]);
  float h_iou = (h_union > 0) ? h_inter / h_union : 0.0f;

  // Regular IoU
  float regular_iou = 1.0f - iou_distance(bbox1, bbox2);

  // Combine H-IoU with regular IoU (matching Python: h_iou * iou_sim)
  return 1.0f - (h_iou * regular_iou);
}

float
cosine_distance(const Eigen::VectorXf &feat1, const Eigen::VectorXf &feat2)
{
  if (feat1.size() == 0 || feat2.size() == 0) {
    return 0.0f; // No appearance features
  }

  float dot = feat1.dot(feat2);
  float norm1 = feat1.norm();
  float norm2 = feat2.norm();

  if (norm1 > 0 && norm2 > 0) {
    float cosine_sim = dot / (norm1 * norm2);
    return 1.0f - cosine_sim; // Return distance
  }

  return 1.0f;
}

float
confidence_distance(float current_score, float prev_score, float det_score)
{
  // Linear projection: projected_score = current_score + (current_score - prev_score)
  float projected_score = current_score + (current_score - prev_score);
  return std::abs(projected_score - det_score);
}

float
angle_distance(const Eigen::Matrix<float, 4, 2> &vel1,
    const Eigen::Matrix<float, 4, 2> &vel2)
{
  // Average angle distance across 4 corner velocities
  float total_dist = 0.0f;
  int valid_count = 0;

  for (int i = 0; i < 4; ++i) {
    Eigen::Vector2f v1 = vel1.row(i);
    Eigen::Vector2f v2 = vel2.row(i);

    float norm1 = v1.norm();
    float norm2 = v2.norm();

    if (norm1 > 1e-5f && norm2 > 1e-5f) {
      float dot = v1.dot(v2);
      float angle_sim = dot / (norm1 * norm2);
      angle_sim = std::max(-1.0f, std::min(1.0f, angle_sim)); // Clamp
      total_dist += (1.0f - angle_sim);
      valid_count++;
    }
  }

  return (valid_count > 0) ? total_dist / valid_count : 0.0f;
}

Eigen::Matrix<float, 4, 2>
compute_velocity_vectors(const Eigen::Vector4f &bbox1, const Eigen::Vector4f &bbox2)
{
  // Compute normalized velocity vectors from bbox1 to bbox2 for 4 corners
  // bbox format: [x1, y1, x2, y2]
  Eigen::Matrix<float, 4, 2> velocities;

  // Corner deltas: lt, lb, rt, rb
  float dx_l = bbox2[0] - bbox1[0]; // left x delta
  float dx_r = bbox2[2] - bbox1[2]; // right x delta
  float dy_t = bbox2[1] - bbox1[1]; // top y delta
  float dy_b = bbox2[3] - bbox1[3]; // bottom y delta

  // Compute normalized velocities for each corner
  // Left-top corner
  float norm_lt = std::sqrt(dx_l * dx_l + dy_t * dy_t) + 1e-5f;
  velocities(0, 0) = dx_l / norm_lt;
  velocities(0, 1) = dy_t / norm_lt;

  // Left-bottom corner
  float norm_lb = std::sqrt(dx_l * dx_l + dy_b * dy_b) + 1e-5f;
  velocities(1, 0) = dx_l / norm_lb;
  velocities(1, 1) = dy_b / norm_lb;

  // Right-top corner
  float norm_rt = std::sqrt(dx_r * dx_r + dy_t * dy_t) + 1e-5f;
  velocities(2, 0) = dx_r / norm_rt;
  velocities(2, 1) = dy_t / norm_rt;

  // Right-bottom corner
  float norm_rb = std::sqrt(dx_r * dx_r + dy_b * dy_b) + 1e-5f;
  velocities(3, 0) = dx_r / norm_rb;
  velocities(3, 1) = dy_b / norm_rb;

  return velocities;
}

Eigen::MatrixXf
compute_cost_matrix(const std::vector<Track *> &tracks,
    const std::vector<Detection> &detections, float penalty_p, float penalty_q,
    bool use_appearance, int delta_t)
{

  int num_tracks = tracks.size();
  int num_dets = detections.size();
  Eigen::MatrixXf cost_matrix(num_tracks, num_dets);

  for (int i = 0; i < num_tracks; ++i) {
    Track *track = tracks[i];
    Eigen::Vector4f track_bbox = track->get_bbox();
    Eigen::VectorXf track_feat = track->get_features();
    Eigen::Matrix<float, 4, 2> track_vel = track->get_velocities(delta_t);

    for (int j = 0; j < num_dets; ++j) {
      const Detection &det = detections[j];

      // Calculate distances matching Python
      float iou_dist = h_iou_distance(track_bbox, det.bbox);
      float app_dist = use_appearance ? cosine_distance(track_feat, det.features) : 0.0f;

      // Base cost: 0.50 * iou_dist + 0.50 * cos_dist (matching Python)
      float cost = 0.50f * iou_dist + 0.50f * app_dist;

      // Add confidence distance (0.10 weight) with linear projection
      float prev_score = track->get_prev_score();
      float conf_dist = confidence_distance(track->get_score(), prev_score, det.score);
      cost += 0.10f * conf_dist;

      // Add angle distance (0.05 weight)
      // Compute angle between track velocity and track-to-detection vector
      Eigen::Vector4f prev_bbox = track->get_prev_bbox(delta_t);
      Eigen::Matrix<float, 4, 2> vel_t_d = compute_velocity_vectors(prev_bbox, det.bbox);
      float angle_dist = angle_distance(track_vel, vel_t_d) * det.score;
      cost += 0.05f * angle_dist;

      // Note: Penalties are applied later in iterative_assignment
      // IoU constraint is also applied later

      cost_matrix(i, j) = cost;
    }
  }

  return cost_matrix;
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
linear_assignment(const Eigen::MatrixXf &cost_matrix, float threshold)
{
  // Use proper Hungarian algorithm
  return hungarian_algorithm(cost_matrix, threshold);
}

std::vector<std::pair<int, int>>
mutual_best_match(const Eigen::MatrixXf &cost_matrix, float threshold)
{

  std::vector<std::pair<int, int>> matches;
  int rows = cost_matrix.rows();
  int cols = cost_matrix.cols();

  // Find best match for each row
  std::vector<int> row_best(rows, -1);
  std::vector<float> row_best_cost(rows, std::numeric_limits<float>::max());

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (cost_matrix(i, j) < row_best_cost[i] && cost_matrix(i, j) < threshold) {
        row_best[i] = j;
        row_best_cost[i] = cost_matrix(i, j);
      }
    }
  }

  // Find best match for each column
  std::vector<int> col_best(cols, -1);
  std::vector<float> col_best_cost(cols, std::numeric_limits<float>::max());

  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (cost_matrix(i, j) < col_best_cost[j] && cost_matrix(i, j) < threshold) {
        col_best[j] = i;
        col_best_cost[j] = cost_matrix(i, j);
      }
    }
  }

  // Extract mutual best matches
  for (int i = 0; i < rows; ++i) {
    int j = row_best[i];
    if (j >= 0 && col_best[j] == i) {
      matches.push_back({ i, j });
    }
  }

  return matches;
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
iterative_assignment(const std::vector<Track *> &tracks,
    const std::vector<Detection> &dets_high, const std::vector<Detection> &dets_low,
    const std::vector<Detection> &dets_del, float match_threshold,
    float penalty_p, float penalty_q, float reduce_step, int frame_id)
{

  // Combine all detections
  std::vector<Detection> all_dets;
  all_dets.insert(all_dets.end(), dets_high.begin(), dets_high.end());
  all_dets.insert(all_dets.end(), dets_low.begin(), dets_low.end());
  all_dets.insert(all_dets.end(), dets_del.begin(), dets_del.end());

  // Compute initial cost matrix
  Eigen::MatrixXf cost_matrix
      = compute_cost_matrix(tracks, all_dets, penalty_p, penalty_q, true, 3);

  // Apply penalties
  int num_tracks = tracks.size();
  int num_high = dets_high.size();
  int num_low = dets_low.size();

  // Apply penalty_p to low confidence detections
  for (int i = 0; i < num_tracks; ++i) {
    for (int j = num_high; j < num_high + num_low; ++j) {
      cost_matrix(i, j) += penalty_p;
    }
  }

  // Apply penalty_q to deleted detections
  for (int i = 0; i < num_tracks; ++i) {
    for (int j = num_high + num_low; j < all_dets.size(); ++j) {
      cost_matrix(i, j) += penalty_q;
    }
  }

  // Apply IoU constraint matching Python
  for (int i = 0; i < num_tracks; ++i) {
    for (int j = 0; j < all_dets.size(); ++j) {
      float iou = 1.0f - h_iou_distance(tracks[i]->get_bbox(), all_dets[j].bbox);
      if (iou <= 0.10f) {
        cost_matrix(i, j) = 1.0f;
      }
    }
  }

  // Clip costs
  cost_matrix = cost_matrix.cwiseMin(1.0f).cwiseMax(0.0f);

  // Iterative matching
  std::vector<std::pair<int, int>> all_matches;
  Eigen::MatrixXf work_matrix = cost_matrix;
  float current_threshold = match_threshold;

  while (current_threshold > 0.0f) {
    // Find mutual best matches
    auto matches = mutual_best_match(work_matrix, current_threshold);

    if (matches.empty()) {
      current_threshold -= reduce_step;
      if (current_threshold <= 0.0f)
        break;
      continue;
    }

    // Add to all matches and mark as unavailable
    for (const auto &match : matches) {
      all_matches.push_back(match);
      // Set row and column to max cost
      work_matrix.row(match.first).setConstant(1.0f);
      work_matrix.col(match.second).setConstant(1.0f);
    }

    current_threshold -= reduce_step;
  }

  // Extract unmatched
  std::unordered_set<int> matched_tracks, matched_dets;
  for (const auto &match : all_matches) {
    matched_tracks.insert(match.first);
    matched_dets.insert(match.second);
  }

  std::vector<int> unmatched_tracks;
  for (int i = 0; i < tracks.size(); ++i) {
    if (matched_tracks.find(i) == matched_tracks.end()) {
      unmatched_tracks.push_back(i);
    }
  }

  std::vector<int> unmatched_dets;
  for (int j = 0; j < all_dets.size(); ++j) {
    if (matched_dets.find(j) == matched_dets.end()) {
      unmatched_dets.push_back(j);
    }
  }

  return { all_matches, unmatched_tracks, unmatched_dets };
}

std::vector<bool>
track_aware_nms(const std::vector<Track *> &tracks,
    const std::vector<Detection> &detections, float tai_threshold, float init_threshold)
{

  int num_dets = detections.size();
  std::vector<bool> keep(num_dets, true);

  // Get alive tracks (Tracked or New)
  std::vector<Track *> alive_tracks;
  for (Track *track : tracks) {
    if (track->get_state() == TrackState::Tracked || track->get_state() == TrackState::New) {
      alive_tracks.push_back(track);
    }
  }

  // Check each detection
  for (int i = 0; i < num_dets; ++i) {
    if (detections[i].score < init_threshold) {
      keep[i] = false;
      continue;
    }

    // Check overlap with existing tracks
    for (Track *track : alive_tracks) {
      float iou = 1.0f - iou_distance(track->get_bbox(), detections[i].bbox);
      if (iou > tai_threshold) {
        keep[i] = false;
        break;
      }
    }

    // Check overlap with higher scoring detections
    if (keep[i]) {
      for (int j = 0; j < i; ++j) {
        if (keep[j] && detections[j].score > detections[i].score) {
          float iou = 1.0f - iou_distance(detections[j].bbox, detections[i].bbox);
          if (iou > tai_threshold) {
            keep[i] = false;
            break;
          }
        }
      }
    }
  }

  return keep;
}

std::vector<Detection>
find_deleted_detections(const std::vector<Detection> &all_detections,
    const std::vector<Detection> &kept_detections)
{

  std::vector<Detection> deleted;

  // Create set of kept detection indices
  std::unordered_set<int> kept_indices;
  for (int i = 0; i < all_detections.size(); ++i) {
    for (const auto &kept : kept_detections) {
      if (all_detections[i].bbox == kept.bbox && all_detections[i].score == kept.score) {
        kept_indices.insert(i);
        break;
      }
    }
  }

  // Find deleted
  for (int i = 0; i < all_detections.size(); ++i) {
    if (kept_indices.find(i) == kept_indices.end()) {
      deleted.push_back(all_detections[i]);
    }
  }

  return deleted;
}

} // namespace tracktrack
