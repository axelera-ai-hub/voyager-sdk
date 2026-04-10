// Copyright Axelera AI, 2025
#ifndef TRACKTRACK_ASSIGNMENT_HPP
#define TRACKTRACK_ASSIGNMENT_HPP

#include <Eigen/Dense>
#include <tuple>
#include <vector>
#include "Track.hpp"

namespace tracktrack
{

// Distance metrics
float iou_distance(const Eigen::Vector4f &bbox1, const Eigen::Vector4f &bbox2);
float h_iou_distance(const Eigen::Vector4f &bbox1, const Eigen::Vector4f &bbox2);
float cosine_distance(const Eigen::VectorXf &feat1, const Eigen::VectorXf &feat2);
float confidence_distance(float current_score, float prev_score, float det_score);
float angle_distance(const Eigen::Matrix<float, 4, 2> &vel1,
    const Eigen::Matrix<float, 4, 2> &vel2);

// Helper function to compute velocity vectors between two bboxes
Eigen::Matrix<float, 4, 2> compute_velocity_vectors(
    const Eigen::Vector4f &bbox1, const Eigen::Vector4f &bbox2);

// Cost matrix computation
Eigen::MatrixXf compute_cost_matrix(const std::vector<Track *> &tracks,
    const std::vector<Detection> &detections, float penalty_p = 0.0f,
    float penalty_q = 0.0f, bool use_appearance = true, int delta_t = 3);

// Hungarian algorithm wrapper
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
linear_assignment(const Eigen::MatrixXf &cost_matrix, float threshold);

// Iterative assignment
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
iterative_assignment(const std::vector<Track *> &tracks,
    const std::vector<Detection> &dets_high, const std::vector<Detection> &dets_low,
    const std::vector<Detection> &dets_del, float match_threshold,
    float penalty_p, float penalty_q, float reduce_step, int frame_id);

// Track-aware NMS
std::vector<bool> track_aware_nms(const std::vector<Track *> &tracks,
    const std::vector<Detection> &detections, float tai_threshold, float init_threshold);

// Helper to find deleted detections (NMS survivors)
std::vector<Detection> find_deleted_detections(const std::vector<Detection> &all_detections,
    const std::vector<Detection> &kept_detections);

// Mutual best matching
std::vector<std::pair<int, int>> mutual_best_match(
    const Eigen::MatrixXf &cost_matrix, float threshold);

} // namespace tracktrack

#endif // TRACKTRACK_ASSIGNMENT_HPP
