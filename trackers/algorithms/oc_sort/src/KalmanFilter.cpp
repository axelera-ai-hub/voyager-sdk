#include "../include/KalmanFilter.hpp"

#include <iostream>
namespace ocsort
{
KalmanFilterNew::KalmanFilterNew() {};
KalmanFilterNew::KalmanFilterNew(int dim_x_, int dim_z_)
{
  dim_x = dim_x_;
  dim_z = dim_z_;
  x = Eigen::VectorXf::Zero(dim_x_, 1);
  P = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
  Q = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
  B = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
  F = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
  H = Eigen::MatrixXf::Zero(dim_z_, dim_x_);
  R = Eigen::MatrixXf::Identity(dim_z_, dim_z_);
  M = Eigen::MatrixXf::Zero(dim_x_, dim_z_);
  z = Eigen::VectorXf::Zero(dim_z_, 1);
  K = Eigen::MatrixXf::Zero(dim_x_, dim_z_);
  y = Eigen::VectorXf::Zero(dim_x_, 1);
  S = Eigen::MatrixXf::Zero(dim_z_, dim_z_);
  SI = Eigen::MatrixXf::Zero(dim_z_, dim_z_);

  x_prior = x;
  P_prior = P;
  x_post = x;
  P_post = P;
};
void
KalmanFilterNew::predict()
{
  x = F * x;
  P.noalias() = _alpha_sq * ((F * P) * F.transpose()) + Q;
  x_prior = x;
  P_prior = P;
}

void
KalmanFilterNew::apply_affine_correction(
    const Eigen::Matrix<float, 2, 2> &m, const Eigen::Vector2f &t)
{
  // Apply affine correction to the state vector
  x.segment<2>(0) = m * x.segment<2>(0) + t; // Update position
  x.segment<2>(4) = m * x.segment<2>(4); // Update velocity

  // Apply affine correction to the covariance matrix
  P.block<2, 2>(0, 0) = m * P.block<2, 2>(0, 0) * m.transpose(); // Position covariance
  P.block<2, 2>(4, 4) = m * P.block<2, 2>(4, 4) * m.transpose(); // Velocity covariance

  // If the filter is frozen, update the saved state
  if (!observed && attr_saved.IsInitialized) {
    attr_saved.x.segment<2>(0) = m * attr_saved.x.segment<2>(0) + t; // Update saved position
    attr_saved.x.segment<2>(4) = m * attr_saved.x.segment<2>(4); // Update saved velocity

    attr_saved.P.block<2, 2>(0, 0) = m * attr_saved.P.block<2, 2>(0, 0)
                                     * m.transpose(); // Saved position covariance
    attr_saved.P.block<2, 2>(4, 4) = m * attr_saved.P.block<2, 2>(4, 4)
                                     * m.transpose(); // Saved velocity covariance
    attr_saved.last_measurement.segment<2>(0)
        = m * attr_saved.last_measurement.segment<2>(0) + t; // Update saved measurement
  }
}

void
KalmanFilterNew::update(Eigen::VectorXf z_)
{
  history_obs.push_back(z_);
  if (z_.size() == 0) {
    if (true == observed) {
      /* The Python code does not include this check, possibly because history_obs.size() is always >= 2*/
      if (history_obs.size() >= 2) {
        last_measurement = history_obs[history_obs.size() - 2];
      } else {
        std::runtime_error("history_obs.size() < 2, cannot set last_measurement");
      }

      freeze();
    }
    observed = false;
    z = Eigen::VectorXf::Zero(dim_z, 1);
    x_post = x;
    P_post = P;
    y = Eigen::VectorXf::Zero(dim_z, 1);
    return;
  }
  if (false == observed)
    unfreeze();
  observed = true;

  y.noalias() = z_ - H * x;
  Eigen::MatrixXf PHT;
  PHT.noalias() = P * H.transpose();
  S.noalias() = H * PHT + R;
  SI = S.inverse();
  K.noalias() = PHT * SI;
  x.noalias() = x + K * y;
  Eigen::MatrixXf I_KH;
  I_KH.noalias() = I - K * H;
  Eigen::MatrixXf P_INT;
  P_INT.noalias() = (I_KH * P);
  P.noalias() = (P_INT * I_KH.transpose()) + ((K * R) * K.transpose());

  z = z_;
  x_post = x;
  P_post = P;
}
void
KalmanFilterNew::freeze()
{
  attr_saved.IsInitialized = true;
  attr_saved.x = x;
  attr_saved.P = P;
  attr_saved.Q = Q;
  attr_saved.B = B;
  attr_saved.F = F;
  attr_saved.H = H;
  attr_saved.R = R;
  attr_saved._alpha_sq = _alpha_sq;
  attr_saved.M = M;
  attr_saved.z = z;
  attr_saved.K = K;
  attr_saved.y = y;
  attr_saved.S = S;
  attr_saved.SI = SI;
  attr_saved.x_prior = x_prior;
  attr_saved.P_prior = P_prior;
  attr_saved.x_post = x_post;
  attr_saved.P_post = P_post;
  attr_saved.history_obs = history_obs;
  attr_saved.last_measurement = last_measurement;
}
void
KalmanFilterNew::unfreeze()
{
  if (true == attr_saved.IsInitialized) {
    new_history.clear();
    for (const auto &obs : history_obs) {
      new_history.push_back(obs); // Perform a deep copy of each element
    }
    x = attr_saved.x;
    P = attr_saved.P;
    Q = attr_saved.Q;
    B = attr_saved.B;
    F = attr_saved.F;
    H = attr_saved.H;
    R = attr_saved.R;
    _alpha_sq = attr_saved._alpha_sq;
    M = attr_saved.M;
    z = attr_saved.z;
    K = attr_saved.K;
    y = attr_saved.y;
    S = attr_saved.S;
    SI = attr_saved.SI;
    x_prior = attr_saved.x_prior;
    P_prior = attr_saved.P_prior;
    x_post = attr_saved.x_post;
    P_post = attr_saved.P_post;
    history_obs = attr_saved.history_obs;
    last_measurement = attr_saved.last_measurement;
    history_obs.erase(history_obs.end() - 1);
    Eigen::VectorXf box1;
    Eigen::VectorXf box2;
    int lastNotNullIndex = -1;
    int secondLastNotNullIndex = -1;
    for (int i = new_history.size() - 1; i >= 0; i--) {
      if (new_history[i].size() > 0) {
        if (lastNotNullIndex == -1) {
          lastNotNullIndex = i;
          box2 = new_history.at(lastNotNullIndex);
        } else if (secondLastNotNullIndex == -1) {
          secondLastNotNullIndex = i;
          box1 = new_history.at(secondLastNotNullIndex);
          break;
        }
      }
    }

    box1 = last_measurement;
    if (box1.size() == 0) {
      std::runtime_error("box1.size() == 0, cannot unfreeze");
    }

    double time_gap = lastNotNullIndex - secondLastNotNullIndex;
    double x1 = box1[0];
    double x2 = box2[0];
    double y1 = box1[1];
    double y2 = box2[1];
    double w1 = std::sqrt(box1[2] * box1[3]);
    double h1 = std::sqrt(box1[2] / box1[3]);
    double w2 = std::sqrt(box2[2] * box2[3]);
    double h2 = std::sqrt(box2[2] / box2[3]);
    double dx = (x2 - x1) / time_gap;
    double dy = (y1 - y2) / time_gap;
    double dw = (w2 - w1) / time_gap;
    double dh = (h2 - h1) / time_gap;

    for (int i = 0; i < time_gap; i++) {
      double x = x1 + (i + 1) * dx;
      double y = y1 + (i + 1) * dy;
      double w = w1 + (i + 1) * dw;
      double h = h1 + (i + 1) * dh;
      double s = w * h;
      double r = w / (h * 1.0);
      Eigen::VectorXf new_box(4, 1);
      new_box << x, y, s, r;

      this->y.noalias() = new_box - this->H * this->x;
      Eigen::MatrixXf PHT;
      PHT.noalias() = this->P * this->H.transpose();
      this->S.noalias() = this->H * PHT + this->R;
      this->SI = (this->S).inverse();
      this->K.noalias() = PHT * this->SI;
      this->x.noalias() = this->x + this->K * this->y;
      Eigen::MatrixXf I_KH;
      I_KH.noalias() = this->I - this->K * this->H;
      Eigen::MatrixXf P_INT;
      P_INT.noalias() = (I_KH * this->P);
      this->P.noalias() = (P_INT * I_KH.transpose())
                          + ((this->K * this->R) * (this->K).transpose());
      this->z = new_box;
      this->x_post = this->x;
      this->P_post = this->P;
      if (i != (time_gap - 1))
        predict();
    }
  }
}
} // namespace ocsort
