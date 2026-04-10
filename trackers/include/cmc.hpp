// Copyright Axelera AI, 2025
#ifndef CMC_HPP
#define CMC_HPP

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class CMCComputer
{
  public:
  explicit CMCComputer(int minimum_features = 10);

  Eigen::Matrix<float, 2, 3> compute_affine(const cv::Mat &img, const Eigen::MatrixXf &bbox);

  private:
  Eigen::Matrix<float, 2, 3> affine_sparse_flow(const cv::Mat &frame, const cv::Mat &mask);

  int minimum_features;
  cv::Mat prev_img;
  std::vector<cv::Point2f> prev_desc;
};

#endif // CMC_HPP
