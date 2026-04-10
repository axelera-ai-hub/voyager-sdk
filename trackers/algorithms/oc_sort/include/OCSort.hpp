// Copyright Axelera AI, 2024
#ifndef OC_SORT_CPP_OCSORT_HPP
#define OC_SORT_CPP_OCSORT_HPP

#include <algorithm>
#include <functional>
#include <unordered_map>
#include "Association.hpp"
#include "KalmanBoxTracker.hpp"
#include "lapjv.hpp"

namespace ocsort
{

struct Rect {
  float x{ 0.0f };
  float y{ 0.0f };
  float width{ 0.0f };
  float height{ 0.0f };

  Rect() = default;
  Rect(float x_, float y_, float width_, float height_)
      : x(x_),
        y(y_),
        width(width_),
        height(height_)
  {
  }

  float area() const
  {
    return (width > 0.0f && height > 0.0f) ? width * height : 0.0f;
  }

  Rect intersect(const Rect &other) const
  {
    const float nx = std::max(x, other.x);
    const float ny = std::max(y, other.y);
    const float rx = std::min(x + width, other.x + other.width);
    const float by = std::min(y + height, other.y + other.height);
    return Rect(nx, ny, std::max(0.0f, rx - nx), std::max(0.0f, by - ny));
  }
};

/**
 * @class OCSort
 * @brief Implementation of the OC-SORT and Deep-OCSORT multi-object tracking algorithms.
 *
 * This tracker supports ReID embeddings for association (Deep-OCSORT) and includes a memory bank
 * to recover the correct person ID when an individual leaves the scene and reappears later.
 *
 * The memory bank stores all lost tracks and, when a new track is created, checks whether the object
 * has been seen before (i.e., is in the memory bank) or is a new object. Tracks near the image boundary
 * are considered for recovery after a certain number of frames, and tracks inside the scene are considered
 * after a longer period. Tracks must meet certain criteria (e.g., minimum age) to be stored in the memory bank.
 * When a new track is created, it is compared with tracks in the memory bank using embedding similarity.
 * If the similarity is above a certain threshold, the track is merged with the corresponding memory bank entry.
 */
class OCSort
{
  public:
  /**
   * @brief Constructor for the OCSort tracker.
   *
   * @param det_thresh_ Detection threshold. Detections with confidence below this value are ignored.
   * @param max_age_ Maximum number of frames a track can remain unmatched before being removed.
   * @param min_hits_ Minimum number of consecutive matches required to consider a track as confirmed.
   * @param iou_threshold_ IOU threshold for association during the first round of matching.
   * @param delta_t_ Time step for velocity estimation in the Kalman filter.
   * @param inertia_ Inertia weight for velocity-based association.
   * @param w_assoc_emb Weight for embedding-based association. Higher values prioritize embeddings.
   * @param alpha_fixed_emb Fixed alpha value for dynamic appearance modeling.
   * @param max_id Maximum ID value for tracks. IDs will wrap around if this value is exceeded.
   * @param aw_off If true, disables adaptive weighting for association.
   * @param aw_param Adaptive weighting parameter for association.
   * @param cmc_off If true, disables CMC (camera motion compensation).
   * @param enable_id_recovery If true, enables the memory bank for ID recovery.
   * @param img_width Width of the image (required for ID recovery).
   * @param img_height Height of the image (required for ID recovery).
   * @param rec_image_rect_margin Margin for the image boundary when determining if a track is out of bounds(required for ID recovery).
   * @param rec_track_min_time_since_update_at_boundary Minimum time since update for tracks near the boundary to be considered for recovery (required for ID recovery).
   * @param rec_track_min_time_since_update_inside Minimum time since update for tracks inside the scene to be considered for recovery (required for ID recovery).
   * @param rec_track_min_age Minimum age of a track to be stored in the memory bank (required for ID recovery).
   * @param rec_track_merge_lap_thresh Threshold for merging tracks in the memory bank based on embedding similarity (required for ID recovery).
   * @param rec_track_memory_capacity Maximum number of tracks retained in the reappearing-track memory bank (required for ID recovery).
   * @param rec_track_memory_max_age Maximum number of frames a reappearing track stays in memory bank (required for ID recovery).
   */
  OCSort(float det_thresh_, int max_age_ = 3000, int min_hits_ = 3,
      float iou_threshold_ = 0.3, int delta_t_ = 3, float inertia_ = 0.2,
      float w_assoc_emb = 0.75, float alpha_fixed_emb = 0.95, int max_id = 0,
      bool aw_off = true, float aw_param = 0.5, bool cmc_off = true,
      bool enable_id_recovery = false, int img_width = 0, int img_height = 0,
      int rec_image_rect_margin = 20, int rec_track_min_time_since_update_at_boundary = 6,
      int rec_track_min_time_since_update_inside = 300, int rec_track_min_age = 30,
      float rec_track_merge_lap_thresh = 0.09f, int rec_track_memory_capacity = 1000,
      int rec_track_memory_max_age = 54000 /*30fps -- 30min*/);

  /**
   * @brief Update the tracker with new detections and embeddings.
   *
   * Takes a numpy array of detections and embeddings, and returns a numpy array of tracking results.
   *
   * @param dets A (N,6) numpy array of detections in the format [[x1, y1, x2, y2, score, class_id], ...],
   * where N is the number of detections.
   *
   * @param embs A (N, emb_len) numpy array of corresponding ReID embeddings for each detection in the format [[emb1], [emb2], ...],
   * where N is the number of detections and emb_len is the length of the embedding.
   * - The order of the embeddings must match the order of the detections.
   * - The embeddings are optional and can be empty if not used.
   * - If embs is not empty, the tracker will use the embeddings for association.
   *
   * @param cmc_transform A 2x3 matrix for camera motion compensation. Defaults to identity.
   *
   * @return A numpy array of tracking results in the format [[x1, y1, x2, y2, object_id, class_id, score, latest_detection_id], ...],
   * where:
   * - object_id is the ID assigned to the object by the tracker.
   * - latest_detection_id is the index (in dets) of the latest detection that was associated with this object.
   *
   * @note This method must be called once for each frame, even with empty detections (use an empty matrix for frames without detections).
   * @note The number of objects returned may differ from the number of detections provided.
   */
  std::vector<Eigen::RowVectorXf> update(Eigen::MatrixXf dets,
      Eigen::MatrixXf embs = Eigen::MatrixXf(),
      Eigen::Matrix<float, 2, 3> cmc_transform = Eigen::Matrix<float, 2, 3>::Identity());

  public:
  float det_thresh; ///< Detection threshold.
  int max_age; ///< Maximum number of frames a track can remain unmatched before it is removed; should be >= recovery thresholds when enable_id_recovery is true.
  int min_hits; ///< Minimum number of consecutive matches for confirmation.
  float iou_threshold; ///< IOU threshold for association.
  int delta_t; ///< Time step for velocity estimation.
  float inertia; ///< Inertia weight for velocity-based association.
  float w_assoc_emb; ///< Weight for embedding-based association.
  float alpha_fixed_emb; ///< Fixed alpha value for dynamic appearance modeling.
  bool aw_off; ///< Disable adaptive weighting for association.
  float aw_param; ///< Adaptive weighting parameter.
  bool cmc_off; ///< Disable camera motion compensation.
  bool enable_id_recovery; ///< Enable memory bank for ID recovery; requires max_age >= recovery thresholds.
  // id recovering track memory bank parameters
  int rec_image_rect_margin; ///< Margin for image boundary checks.
  int rec_track_min_time_since_update_at_boundary; ///< Minimum time for boundary recovery.
  int rec_track_min_time_since_update_inside; ///< Minimum time for inside recovery.
  int rec_track_min_age; ///< Minimum age for memory bank storage.
  float rec_track_merge_lap_thresh; ///< Threshold for merging tracks in memory bank.
  int rec_track_memory_capacity; ///< Maximum capacity of the reappearing tracker memory bank.
  int rec_track_memory_max_age; ///< Maximum allowed age for reappearing tracks in memory bank.

  int frame_count; ///< Frame counter.
  Rect img_rect; ///< Image rectangle for boundary checks.

  std::vector<KalmanBoxTracker> active_trackers; ///< Active trackers.
  std::vector<KalmanBoxTracker> not_active_trackers; ///< Inactive trackers.
  std::vector<KalmanBoxTracker> lost_trackers; ///< Lost trackers.
  std::vector<KalmanBoxTracker> reappearing_trackers; ///< Reappearing trackers.
};

} // namespace ocsort
#endif // OC_SORT_CPP_OCSORT_HPP
