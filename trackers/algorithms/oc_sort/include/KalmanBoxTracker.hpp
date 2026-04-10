// Copyright Axelera AI, 2024
#ifndef OC_SORT_CPP_KALMANBOXTRACKER_HPP
#define OC_SORT_CPP_KALMANBOXTRACKER_HPP
////////////// KalmanBoxTracker /////////////
#include <map>
#include <memory>
#include <set>
#include "../include/KalmanFilter.hpp"
#include "../include/Utilities.hpp"
#include "iostream"
/*
This class represents the internal state of individual
tracked objects observed as bbox.
*/
namespace ocsort
{

class KalmanBoxTracker
{
  public:
  /*method*/
  KalmanBoxTracker() {};
  KalmanBoxTracker(Eigen::VectorXf bbox_, const Eigen::VectorXf &emb_, int cls_,
      int det_id, int delta_t_ = 3, int frame_id = 0);
  void update(Eigen::Matrix<float, 5, 1> *bbox_, int cls_, int det_id, int frame_id);
  void update_emb(const Eigen::VectorXf &emb_, float alpha = 0.9);
  Eigen::VectorXf get_emb() const;
  void apply_affine_correction(const Eigen::Matrix<float, 2, 3> &affine);
  Eigen::RowVectorXf predict();
  Eigen::VectorXf get_state();

  int next_id();
  void release_id();

  public:
  /*variable*/
  static std::map<int, std::pair<std::set<int>, std::set<int>>> id_tables;
  static std::map<int, int> count_per_class;
  static int count;
  static int max_id;

  Eigen::VectorXf bbox; // [5,1]
  Eigen::VectorXf emb;

  std::shared_ptr<KalmanFilterNew> kf;
  int time_since_update;
  int frame_of_last_update;
  int id;
  std::vector<Eigen::VectorXf> history;
  int hits;
  int hit_streak;
  int age = 0;
  float conf;
  int cls;
  Eigen::RowVectorXf last_observation = Eigen::RowVectorXf::Zero(5);
  std::map<int, Eigen::VectorXf> observations;
  std::vector<Eigen::VectorXf> history_observations;
  Eigen::RowVectorXf velocity = Eigen::RowVectorXf::Zero(2); // [2,1]
  int delta_t;
  int latest_detection_id;
};
} // namespace ocsort

#endif // OC_SORT_CPP_KALMANBOXTRACKER_HPP
