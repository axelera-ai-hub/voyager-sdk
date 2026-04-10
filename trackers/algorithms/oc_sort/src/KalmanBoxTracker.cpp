// Copyright Axelera AI, 2024
#include "../include/KalmanBoxTracker.hpp"

#include <utility>
namespace ocsort
{
std::map<int, std::pair<std::set<int>, std::set<int>>> KalmanBoxTracker::id_tables;
std::map<int, int> KalmanBoxTracker::count_per_class;
int KalmanBoxTracker::max_id = 0;
int KalmanBoxTracker::count = 0;

KalmanBoxTracker::KalmanBoxTracker(Eigen::VectorXf bbox_,
    const Eigen::VectorXf &emb_, int cls_, int det_id, int delta_t_, int frame_id)
    : kf(std::make_shared<KalmanFilterNew>(7, 4))
{
  bbox = std::move(bbox_);
  delta_t = delta_t_;
  kf->F << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1;
  kf->H << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0;
  kf->R.block(2, 2, 2, 2) *= 10.0;
  kf->P.block(4, 4, 3, 3) *= 1000.0;
  kf->P *= 10.0;
  kf->Q.bottomRightCorner(1, 1)(0, 0) *= 0.01;
  kf->Q.block(4, 4, 3, 3) *= 0.01;
  kf->x.head<4>() = convert_bbox_to_z(bbox);
  time_since_update = 0;
  frame_of_last_update = frame_id;
  cls = cls_;
  id = next_id();
  history.clear();
  hits = 0;
  hit_streak = 0;
  age = 0;
  conf = bbox(4);

  emb = emb_;
  latest_detection_id = det_id;
  last_observation.fill(-1);
  observations.clear();
  history_observations.clear();
  velocity.fill(0);
}

void
KalmanBoxTracker::update(Eigen::Matrix<float, 5, 1> *bbox_, int cls_, int det_id, int frame_id)
{
  if (bbox_ != nullptr) {
    conf = (*bbox_)[4];
    cls = cls_;
    latest_detection_id = det_id;
    frame_of_last_update = frame_id;
    if (int(last_observation.sum()) >= 0) {
      Eigen::VectorXf previous_box_tmp;
      for (int dt = delta_t; dt > 0; --dt) {
        auto it = observations.find(age - dt);
        if (it != observations.end()) {
          previous_box_tmp = it->second;
          break;
        }
      }
      if (0 == previous_box_tmp.size()) {
        previous_box_tmp = last_observation;
      }
      const int maxSize = 300;
      if (observations.size() > maxSize) {
        observations.erase(observations.begin());
      }
      velocity = speed_direction(previous_box_tmp, *bbox_);
    }
    last_observation = *bbox_;
    observations[age] = *bbox_;
    history_observations.push_back(*bbox_);
    time_since_update = 0;
    history.clear();
    hits += 1;
    hit_streak += 1;
    Eigen::VectorXf tmp = convert_bbox_to_z(*bbox_);
    kf->update(tmp);
  } else {
    kf->update(Eigen::VectorXf());
  }
}

void
KalmanBoxTracker::update_emb(const Eigen::VectorXf &emb_, float alpha)
{
  if (emb.size() == 0) {
    emb = emb_; // Initialize if empty
  } else {
    emb = alpha * emb + (1 - alpha) * emb_;
  }
  emb.normalize(); // Normalize the embedding vector
}

Eigen::VectorXf
KalmanBoxTracker::get_emb() const
{
  return emb;
}


void
KalmanBoxTracker::apply_affine_correction(const Eigen::Matrix<float, 2, 3> &affine)
{
  Eigen::Matrix<float, 2, 2> m = affine.block<2, 2>(0, 0); // Extract rotation/scale matrix
  Eigen::Vector2f t = affine.block<2, 1>(0, 2); // Extract translation vector

  // Apply correction to last_observation if valid
  if (last_observation.sum() > 0) {
    Eigen::Matrix<float, 2, 2> ps;
    ps << last_observation(0), last_observation(2), last_observation(1),
        last_observation(3);
    ps = (m * ps).colwise() + t;
    last_observation(0) = ps(0, 0);
    last_observation(1) = ps(1, 0);
    last_observation(2) = ps(0, 1);
    last_observation(3) = ps(1, 1);
  }

  // Apply correction to observations within the velocity computation range
  for (int dt = delta_t; dt >= 0; --dt) {
    auto it = observations.find(age - dt);
    if (it != observations.end()) {
      Eigen::Matrix<float, 2, 2> ps;
      ps << it->second(0), it->second(2), it->second(1), it->second(3);
      ps = (m * ps).colwise() + t;
      it->second(0) = ps(0, 0);
      it->second(1) = ps(1, 0);
      it->second(2) = ps(0, 1);
      it->second(3) = ps(1, 1);
    }
  }

  // Apply correction to Kalman filter state
  kf->apply_affine_correction(m, t);
}

Eigen::RowVectorXf
KalmanBoxTracker::predict()
{
  if (kf->x[6] + kf->x[2] <= 0)
    kf->x[6] *= 0.0;
  kf->predict();
  age += 1;
  if (time_since_update > 0)
    hit_streak = 0;
  time_since_update += 1;
  auto vec_out = convert_x_to_bbox(kf->x);
  history.push_back(vec_out);
  return vec_out;
}
Eigen::VectorXf
KalmanBoxTracker::get_state()
{
  return convert_x_to_bbox(kf->x);
}


int
KalmanBoxTracker::next_id()
{
  /* Class-based ID assignment, if max_id > 0. Otherwise global ID assignment.
  The class-based ID assignment has prefixing the ID with the class identifier (cls)
  to ensure uniqueness across different classes; this is for compatibility with the
  tracker meta passing from C++ to Python. For C++ example, this is not necessary.
  */
  int cls_factor = 10000;

  if (max_id > 0) {
    auto &[existing_ids, available_ids] = id_tables[this->cls];
    auto &class_count = count_per_class[this->cls];

    if (!available_ids.empty()) {
      int reused_id = *available_ids.begin();
      available_ids.erase(available_ids.begin());
      existing_ids.insert(reused_id);
      return reused_id;
    }

    // Check if all possible IDs are already in use
    if (existing_ids.size() == max_id) {
      std::cerr << "Error: All IDs for class " << this->cls << " are in use."
                << " Please consider increasing max_id." << std::endl;
      exit(1);
    }

    // Find the next available ID
    do {
      class_count = (class_count % max_id) + 1;
      int combined_id = this->cls * cls_factor + class_count;
      if (existing_ids.find(combined_id) == existing_ids.end()) {
        existing_ids.insert(combined_id);
        return combined_id;
      }
    } while (true);
  } else {
    // Global count without class distinction
    return ++count;
  }
}
void
KalmanBoxTracker::release_id()
{
  if (max_id > 0) {
    auto &[existing_ids, available_ids] = id_tables[this->cls];
    if (this->id <= max_id) {
      available_ids.insert(this->id);
      existing_ids.erase(this->id);
    }
  }
}
} // namespace ocsort
