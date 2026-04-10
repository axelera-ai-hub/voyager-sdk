// Copyright Axelera AI, 2026
#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <vector>
#include "AxMetaBBox.hpp"

namespace face_align
{

// ============================================================================
// Face alignment template keypoint arrays
// ============================================================================

// Normalized coordinates for 5-point facial landmarks (eyes, nose, mouth corners)
extern const std::array<float, 5> template_5pt_x;
extern const std::array<float, 5> template_5pt_y;

// Normalized coordinates for 51-point facial landmarks
extern const std::array<float, 51> template_51pt_x;
extern const std::array<float, 51> template_51pt_y;

// ============================================================================
// Face alignment matrix computation functions
// ============================================================================

// Helper function to validate that all matrix elements are finite
// Returns false if matrix is empty, wrong size, or contains non-finite values
// Only accepts CV_32F (float) matrices
bool is_valid_matrix(const cv::Mat &M);

// Compute a simple scale + translation matrix for fallback alignment
// Returns INVERSE matrix (destination → source) for use with warpAffine
// Always returns CV_32F matrix type
cv::Mat compute_fallback_matrix(const box_xyxy &box, int output_width, int output_height);

// Compute eye-based alignment matrix (for 5-point landmarks)
// Returns INVERSE matrix (destination → source) for use with warpAffine
// Returns fallback matrix if eye-based alignment cannot be computed
// Always returns CV_32F matrix type
cv::Mat compute_self_normalizing_matrix(const std::vector<float> &X,
    const std::vector<float> &Y, const box_xyxy &box, int output_width, int output_height);

// Compute template-based Procrustes alignment matrix
// Returns INVERSE matrix (destination → source) for use with warpAffine
// Returns fallback matrix if template-based alignment cannot be computed
// Always returns CV_32F matrix type
cv::Mat compute_template_based_matrix(std::vector<float> X_template,
    std::vector<float> Y_template, std::vector<float> XX, std::vector<float> YY,
    const box_xyxy &box, float padding, int output_width, int output_height);

} // namespace face_align
