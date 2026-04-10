// Copyright Axelera AI, 2025
#ifndef TRACKTRACK_HUNGARIAN_HPP
#define TRACKTRACK_HUNGARIAN_HPP

#include <Eigen/Dense>
#include <tuple>
#include <vector>

namespace tracktrack
{

// Hungarian algorithm implementation
// Returns matches, unmatched_a, unmatched_b
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
hungarian_algorithm(const Eigen::MatrixXf &cost_matrix, float threshold);

} // namespace tracktrack

#endif // TRACKTRACK_HUNGARIAN_HPP
