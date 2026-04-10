// Copyright Axelera AI, 2025
#ifndef TRACKTRACK_KALMANFILTER_HPP
#define TRACKTRACK_KALMANFILTER_HPP

#include <Eigen/Dense>

namespace tracktrack
{

class KalmanFilter
{
  public:
  // Constructor
  KalmanFilter();

  // Initialize filter with detection
  void initiate(const Eigen::Vector4f &measurement);

  // Predict next state
  void predict();

  // Update with new measurement
  void update(const Eigen::Vector4f &measurement, float confidence = 1.0f);

  // Getters
  Eigen::VectorXf get_state() const
  {
    return x_;
  }
  Eigen::MatrixXf get_covariance() const
  {
    return P_;
  }

  // Setters (for CMC)
  void set_state(const Eigen::VectorXf &state)
  {
    x_ = state;
  }
  void set_covariance(const Eigen::MatrixXf &cov)
  {
    P_ = cov;
  }

  // Convert state to measurement space (x, y, a, h)
  Eigen::Vector4f project() const;

  // Get innovation covariance for gating
  Eigen::Matrix4f get_innovation_covariance() const;

  private:
  // State vector: [x, y, a, h, vx, vy, va, vh]
  // x, y: center position
  // a: aspect ratio (w/h)
  // h: height
  // v*: velocities
  Eigen::VectorXf x_; // 8x1 state vector
  Eigen::MatrixXf P_; // 8x8 covariance matrix

  // System matrices
  Eigen::MatrixXf F_; // 8x8 state transition matrix
  Eigen::MatrixXf H_; // 4x8 measurement matrix
  Eigen::MatrixXf Q_; // 8x8 process noise covariance
  Eigen::MatrixXf R_; // 4x4 measurement noise covariance

  // Noise parameters
  float std_weight_position_;
  float std_weight_velocity_;

  // Initialize system matrices
  void init_matrices();

  // Update process noise based on current state
  void update_process_noise();
};

} // namespace tracktrack

#endif // TRACKTRACK_KALMANFILTER_HPP
