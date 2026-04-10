// Copyright Axelera AI, 2025
#include "../include/cmc.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>

CMCComputer::CMCComputer(int minimum_features)
    : minimum_features(minimum_features)
{
}

Eigen::Matrix<float, 2, 3>
CMCComputer::compute_affine(const cv::Mat &gray_img, const Eigen::MatrixXf &bbox)
{
  CV_Assert(!gray_img.empty() && gray_img.type() == CV_8UC1
            && "Input image must be a non-empty grayscale cv::Mat of type CV_8UC1");

  cv::Mat mask = cv::Mat::ones(gray_img.size(), CV_8U);
  for (int i = 0; i < bbox.rows(); ++i) {
    cv::Rect rect(bbox(i, 0), bbox(i, 1), bbox(i, 2) - bbox(i, 0),
        bbox(i, 3) - bbox(i, 1));
    cv::rectangle(mask, rect, cv::Scalar(0), cv::FILLED);
  }

  Eigen::Matrix<float, 2, 3> A = affine_sparse_flow(gray_img, mask);

  return A;
}


Eigen::Matrix<float, 2, 3>
CMCComputer::affine_sparse_flow(const cv::Mat &cur_img, const cv::Mat &mask)
{
  Eigen::Matrix<float, 2, 3> A = Eigen::Matrix<float, 2, 3>::Identity();

  std::vector<cv::Point2f> keypoints;
  cv::goodFeaturesToTrack(cur_img, keypoints, 3000, 0.01, 1, mask);

  if (cur_img.size() != prev_img.size()) {
    cur_img.copyTo(prev_img);
    prev_desc = keypoints;

    return A;
  }

  std::vector<cv::Point2f> next_pts;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_desc, next_pts, status, err);

  std::vector<cv::Point2f> src_pts, dst_pts;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i]) {
      src_pts.push_back(prev_desc[i]);
      dst_pts.push_back(next_pts[i]);
    }
  }

  if (src_pts.size() < minimum_features) {
    cur_img.copyTo(prev_img);
    prev_desc = keypoints;

    return A;
  }

  cv::Mat affine
      = cv::estimateAffinePartial2D(src_pts, dst_pts, cv::noArray(), cv::RANSAC);
  if (!affine.empty()) {
    cv::cv2eigen(affine, A);
  }

  cur_img.copyTo(prev_img);
  prev_desc = keypoints;
  return A;
}
