// Copyright Axelera AI, 2025
#include "Hungarian.hpp"
#include <algorithm>
#include <limits>
#include <numeric>

namespace tracktrack
{

// Implementation based on the Hungarian (Kuhn-Munkres) algorithm
// for solving the assignment problem
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
hungarian_algorithm(const Eigen::MatrixXf &cost_matrix, float threshold)
{
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_a;
  std::vector<int> unmatched_b;

  int rows = cost_matrix.rows();
  int cols = cost_matrix.cols();

  if (rows == 0 || cols == 0) {
    for (int i = 0; i < rows; ++i)
      unmatched_a.push_back(i);
    for (int j = 0; j < cols; ++j)
      unmatched_b.push_back(j);
    return { matches, unmatched_a, unmatched_b };
  }

  // Create a square cost matrix by padding with high costs
  int n = std::max(rows, cols);
  Eigen::MatrixXf padded_cost = Eigen::MatrixXf::Constant(n, n, threshold + 1.0f);
  padded_cost.block(0, 0, rows, cols) = cost_matrix;

  // Convert to minimization problem
  Eigen::MatrixXf work_matrix = padded_cost;

  // Step 1: Subtract row minima
  for (int i = 0; i < n; ++i) {
    float row_min = work_matrix.row(i).minCoeff();
    if (row_min < std::numeric_limits<float>::max()) {
      work_matrix.row(i).array() -= row_min;
    }
  }

  // Step 2: Subtract column minima
  for (int j = 0; j < n; ++j) {
    float col_min = work_matrix.col(j).minCoeff();
    if (col_min < std::numeric_limits<float>::max()) {
      work_matrix.col(j).array() -= col_min;
    }
  }

  // Initialize assignments
  std::vector<int> row_assignment(n, -1);
  std::vector<int> col_assignment(n, -1);

  // Find initial zeros and make assignments
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (work_matrix(i, j) < 1e-6f && row_assignment[i] == -1 && col_assignment[j] == -1) {
        row_assignment[i] = j;
        col_assignment[j] = i;
      }
    }
  }

  // Main loop
  while (true) {
    // Check if we have a complete assignment
    int num_assigned = 0;
    for (int i = 0; i < n; ++i) {
      if (row_assignment[i] >= 0)
        num_assigned++;
    }

    if (num_assigned == n)
      break;

    // Find uncovered zeros
    std::vector<bool> covered_rows(n, false);
    std::vector<bool> covered_cols(n, false);

    for (int i = 0; i < n; ++i) {
      if (row_assignment[i] >= 0) {
        covered_rows[i] = true;
        covered_cols[row_assignment[i]] = true;
      }
    }

    // Find minimum uncovered value
    float min_val = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
      if (!covered_rows[i]) {
        for (int j = 0; j < n; ++j) {
          if (!covered_cols[j]) {
            min_val = std::min(min_val, work_matrix(i, j));
          }
        }
      }
    }

    if (min_val >= std::numeric_limits<float>::max() / 2)
      break;

    // Update matrix
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (covered_rows[i] && covered_cols[j]) {
          work_matrix(i, j) += min_val;
        } else if (!covered_rows[i] && !covered_cols[j]) {
          work_matrix(i, j) -= min_val;
        }
      }
    }

    // Try to find new assignments
    for (int i = 0; i < n; ++i) {
      if (row_assignment[i] == -1) {
        for (int j = 0; j < n; ++j) {
          if (col_assignment[j] == -1 && work_matrix(i, j) < 1e-6f) {
            row_assignment[i] = j;
            col_assignment[j] = i;
            break;
          }
        }
      }
    }
  }

  // Extract valid matches (within original matrix bounds and below threshold)
  for (int i = 0; i < rows; ++i) {
    if (row_assignment[i] >= 0 && row_assignment[i] < cols) {
      if (cost_matrix(i, row_assignment[i]) < threshold) {
        matches.push_back({ i, row_assignment[i] });
      }
    }
  }

  // Find unmatched
  std::vector<bool> matched_a(rows, false);
  std::vector<bool> matched_b(cols, false);

  for (const auto &[i, j] : matches) {
    matched_a[i] = true;
    matched_b[j] = true;
  }

  for (int i = 0; i < rows; ++i) {
    if (!matched_a[i])
      unmatched_a.push_back(i);
  }

  for (int j = 0; j < cols; ++j) {
    if (!matched_b[j])
      unmatched_b.push_back(j);
  }

  return { matches, unmatched_a, unmatched_b };
}

} // namespace tracktrack
