// Copyright Axelera AI, 2025
#include "KalmanFilter.hpp"
#include <cmath>

namespace tracktrack
{

KalmanFilter::KalmanFilter()
    : x_(8),
      P_(8, 8),
      F_(8, 8),
      H_(4, 8),
      Q_(8, 8),
      R_(4, 4),
      std_weight_position_(1.0f / 20.0f),
      std_weight_velocity_(1.0f / 160.0f)
{
  init_matrices();
}

void
KalmanFilter::init_matrices()
{
  // State transition matrix (constant velocity model)
  F_ = Eigen::MatrixXf::Identity(8, 8);
  F_(0, 4) = 1.0f; // x += vx
  F_(1, 5) = 1.0f; // y += vy
  F_(2, 6) = 1.0f; // a += va
  F_(3, 7) = 1.0f; // h += vh

  // Measurement matrix (observe position and size)
  H_ = Eigen::MatrixXf::Zero(4, 8);
  H_(0, 0) = 1.0f; // x
  H_(1, 1) = 1.0f; // y
  H_(2, 2) = 1.0f; // a
  H_(3, 3) = 1.0f; // h

  // Initialize noise matrices (will be updated dynamically)
  Q_ = Eigen::MatrixXf::Identity(8, 8);
  R_ = Eigen::MatrixXf::Identity(4, 4);
}

void
KalmanFilter::initiate(const Eigen::Vector4f &measurement)
{
  // Initialize state
  x_ = Eigen::VectorXf::Zero(8);
  x_.head<4>() = measurement;

  // Initialize covariance based on measurement
  P_ = Eigen::MatrixXf::Zero(8, 8);

  // Position uncertainty scales with object size
  float h = measurement[3];
  float w = measurement[2] * h; // a * h = w

  // Position covariance
  P_(0, 0) = 2.0f * std_weight_position_ * w; // x
  P_(1, 1) = 2.0f * std_weight_position_ * h; // y

  // Size covariance
  P_(2, 2) = 1e-2f; // aspect ratio
  P_(3, 3) = 2.0f * std_weight_position_ * h; // height

  // Velocity covariance (high initial uncertainty)
  P_(4, 4) = 10.0f * std_weight_velocity_ * w; // vx
  P_(5, 5) = 10.0f * std_weight_velocity_ * h; // vy
  P_(6, 6) = 1e-5f; // va
  P_(7, 7) = 10.0f * std_weight_velocity_ * h; // vh

  // Update measurement noise based on object size
  R_(0, 0) = std_weight_position_ * w;
  R_(1, 1) = std_weight_position_ * h;
  R_(2, 2) = 1e-3f;
  R_(3, 3) = std_weight_position_ * h;
}

void
KalmanFilter::predict()
{
  // Update process noise based on current state
  update_process_noise();

  // Predict state
  x_ = F_ * x_;

  // Predict covariance
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void
KalmanFilter::update(const Eigen::Vector4f &measurement, float confidence)
{
  // Innovation
  Eigen::Vector4f y = measurement - H_ * x_;

  // Adaptive measurement noise based on confidence (NSA)
  Eigen::Matrix4f R_adaptive = R_ * (2.0f - confidence);

  // Innovation covariance
  Eigen::Matrix4f S = H_ * P_ * H_.transpose() + R_adaptive;

  // Kalman gain
  Eigen::MatrixXf K = P_ * H_.transpose() * S.inverse();

  // Update state
  x_ = x_ + K * y;

  // Update covariance
  Eigen::MatrixXf I = Eigen::MatrixXf::Identity(8, 8);
  P_ = (I - K * H_) * P_;
}

Eigen::Vector4f
KalmanFilter::project() const
{
  return H_ * x_;
}

Eigen::Matrix4f
KalmanFilter::get_innovation_covariance() const
{
  return H_ * P_ * H_.transpose() + R_;
}

void
KalmanFilter::update_process_noise()
{
  // Get current height and width
  float h = x_[3];
  float w = x_[2] * h;

  // Process noise scales with object size
  Q_ = Eigen::MatrixXf::Zero(8, 8);

  // Position process noise
  Q_(0, 0) = std_weight_position_ * w;
  Q_(1, 1) = std_weight_position_ * h;

  // Size process noise
  Q_(2, 2) = 1e-2f;
  Q_(3, 3) = std_weight_position_ * h;

  // Velocity process noise
  Q_(4, 4) = std_weight_velocity_ * w;
  Q_(5, 5) = std_weight_velocity_ * h;
  Q_(6, 6) = 1e-5f;
  Q_(7, 7) = std_weight_velocity_ * h;
}

} // namespace tracktrack
