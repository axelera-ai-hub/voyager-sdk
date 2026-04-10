// Copyright Axelera AI, 2026
#include "AxFaceAlignCommon.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace face_align
{

// ============================================================================
// Face alignment template keypoint arrays
// ============================================================================

// fmt: off
// Normalized coordinates for 5-point facial landmarks (eyes, nose, mouth corners)
const std::array<float, 5> template_5pt_x = {
  60.59f / 192, // Left eye
  131.06f / 192, // Right eye
  96.05f / 192, // Nose
  67.10f / 192, // Left mouth
  125.46f / 192 // Right mouth
};

const std::array<float, 5> template_5pt_y = {
  76.8f / 192, // Left eye (40% from top)
  76.8f / 192, // Right eye
  115.2f / 192, // Nose (60% from top)
  153.6f / 192, // Left mouth (80% from top)
  153.6f / 192 // Right mouth
};

// Normalized coordinates for 51-point facial landmarks
const std::array<float, 51> template_51pt_x = { 0.000213256, 0.0752622, 0.18113,
  0.29077, 0.393397, 0.586856, 0.689483, 0.799124, 0.904991, 0.98004, 0.490127,
  0.490127, 0.490127, 0.490127, 0.36688, 0.426036, 0.490127, 0.554217, 0.613373,
  0.121737, 0.187122, 0.265825, 0.334606, 0.260918, 0.182743, 0.645647, 0.714428,
  0.793132, 0.858516, 0.79751, 0.719335, 0.254149, 0.340985, 0.428858, 0.490127,
  0.551395, 0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532, 0.338094,
  0.290379, 0.428096, 0.490127, 0.552157, 0.689874, 0.553364, 0.490127, 0.42689 };

const std::array<float, 51> template_51pt_y = { 0.106454, 0.038915, 0.0187482,
  0.0344891, 0.0773906, 0.0773906, 0.0344891, 0.0187482, 0.038915, 0.106454, 0.203352,
  0.307009, 0.409805, 0.515625, 0.587326, 0.609345, 0.628106, 0.609345, 0.587326,
  0.216423, 0.178758, 0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852,
  0.178758, 0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578,
  0.727388, 0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192, 0.864805,
  0.784792, 0.778746, 0.785343, 0.778746, 0.784792, 0.824182, 0.831803, 0.824182 };
// fmt: on

// ============================================================================
// Face alignment matrix computation functions
// ============================================================================

bool
is_valid_matrix(const cv::Mat &M)
{
  if (M.empty() || M.rows < 2 || M.cols < 3) {
    return false;
  }

  // All matrices should be CV_32F (float type)
  if (M.type() != CV_32F) {
    return false;
  }

  // Check that all elements are finite
  for (int i = 0; i < M.rows; ++i) {
    for (int j = 0; j < M.cols; ++j) {
      if (!std::isfinite(M.at<float>(i, j))) {
        return false;
      }
    }
  }

  return true;
}

// Helper function to compute a simple scale + translation matrix for fallback
// Returns INVERSE matrix (destination → source) for use with warpAffine
// Always returns CV_32F matrix type
cv::Mat
compute_fallback_matrix(const box_xyxy &box, int output_width, int output_height)
{
  // Bounding boxes use closed ranges [x1, x2], so width = x2 - x1 + 1
  float box_width = box.x2 - box.x1 + 1;
  float box_height = box.y2 - box.y1 + 1;

  // Inverse scale: maps from destination back to source
  float inv_scale_x = box_width / output_width;
  float inv_scale_y = box_height / output_height;

  // Inverse matrix: x_src = x_dst * inv_scale_x + box.x1
  // cv::Mat_<float> ensures CV_32F type
  cv::Mat M_inv
      = (cv::Mat_<float>(2, 3) << inv_scale_x, 0, box.x1, 0, inv_scale_y, box.y1);

  return M_inv;
}

// Helper function to compute eye-based alignment matrix (for 5-point landmarks)
// Returns INVERSE matrix (destination → source) for use with warpAffine
// Returns fallback matrix if eye-based alignment cannot be computed
// Always returns CV_32F matrix type
cv::Mat
compute_self_normalizing_matrix(const std::vector<float> &X,
    const std::vector<float> &Y, const box_xyxy &box, int output_width, int output_height)
{
  if (X.size() < 2 || Y.size() < 2) {
    return compute_fallback_matrix(box, output_width, output_height);
  }

  cv::Point2f left_eye(X[0], Y[0]);
  cv::Point2f right_eye(X[1], Y[1]);

  float eye_distance = cv::norm(right_eye - left_eye);
  if (eye_distance < 10.0f) {
    return compute_fallback_matrix(box, output_width, output_height);
  }
  const float normalised_desired_eye_y = 0.4F;
  const float normalised_desired_eye_distance = 0.35F;
  float desired_eye_y = output_height * normalised_desired_eye_y;
  float desired_eye_center_x = output_width * 0.5F;
  float desired_eye_distance = output_width * normalised_desired_eye_distance;

  cv::Point2f eye_center = (left_eye + right_eye) * 0.5f;
  cv::Point2f eye_diff = right_eye - left_eye;
  float angle = std::atan2(eye_diff.y, eye_diff.x) * 180.0f / CV_PI;
  const float min_eye_scale = 0.1F;
  const float max_eye_scale = 5.0F;
  float scale = std::clamp(desired_eye_distance / eye_distance, min_eye_scale, max_eye_scale);

  // Compute forward matrix (source → destination)
  // Note: cv::getRotationMatrix2D returns CV_64F (double)
  cv::Mat rotation_matrix = cv::getRotationMatrix2D(eye_center, angle, scale);
  rotation_matrix.at<double>(0, 2) += desired_eye_center_x - eye_center.x;
  rotation_matrix.at<double>(1, 2) += desired_eye_y - eye_center.y;

  // Invert in double precision for better numerical accuracy
  cv::Mat rotation_matrix_inv_d;
  cv::invertAffineTransform(rotation_matrix, rotation_matrix_inv_d);

  // Convert to CV_32F for warpAffine
  cv::Mat rotation_matrix_inv;
  rotation_matrix_inv_d.convertTo(rotation_matrix_inv, CV_32F);
  return rotation_matrix_inv;
}

// Helper function to compute template-based Procrustes alignment matrix
// Returns INVERSE matrix (destination → source) for use with warpAffine
// Returns fallback matrix if template-based alignment cannot be computed
// Always returns CV_32F matrix type
cv::Mat
compute_template_based_matrix(std::vector<float> X_template,
    std::vector<float> Y_template, std::vector<float> XX, std::vector<float> YY,
    const box_xyxy &box, float padding, int output_width, int output_height)
{
  if (X_template.empty() || Y_template.empty() || XX.empty() || YY.empty()) {
    return compute_fallback_matrix(box, output_width, output_height);
  }

  if (X_template.size() != Y_template.size() || XX.size() != YY.size()
      || X_template.size() != XX.size()) {
    return compute_fallback_matrix(box, output_width, output_height);
  }

  // Convert normalized template coordinates to pixel coordinates with padding adjustment
  float inv_padding_factor = 1.0f / (2 * padding + 1);
  for (float &x : XX)
    x = (x + padding) * inv_padding_factor * output_width;
  for (float &y : YY)
    y = (y + padding) * inv_padding_factor * output_height;

  // Build point correspondences for OpenCV
  std::vector<cv::Point2f> src_points, dst_points;
  src_points.reserve(X_template.size());
  dst_points.reserve(X_template.size());

  for (size_t i = 0; i < X_template.size(); ++i) {
    src_points.push_back(cv::Point2f(X_template[i], Y_template[i]));
    dst_points.push_back(cv::Point2f(XX[i], YY[i]));
  }

  // Estimate similarity transform (rotation + uniform scale + translation) using OpenCV
  // Note: This uses uniform scaling (preserves aspect ratio), whereas the previous
  // implementation allowed different X/Y scale factors. Similarity transforms are more
  // appropriate for face alignment as they preserve shape without distortion.
  // LMEDS is used for robust estimation with outliers.
  cv::Mat M = cv::estimateAffinePartial2D(src_points, dst_points, cv::noArray(), cv::LMEDS);

  if (M.empty()) {
    return compute_fallback_matrix(box, output_width, output_height);
  }

  // Invert in double precision for better numerical accuracy (especially for near-singular matrices)
  cv::Mat M_inv_d;
  cv::invertAffineTransform(M, M_inv_d);

  // Convert to CV_32F for warpAffine
  cv::Mat M_inv;
  M_inv_d.convertTo(M_inv, CV_32F);

  // Return inverted matrix if valid, otherwise fallback (catches singular/degenerate matrices)
  return is_valid_matrix(M_inv) ?
             M_inv :
             compute_fallback_matrix(box, output_width, output_height);
}

} // namespace face_align
