#include "../include/Association.hpp"
#include <iomanip>
#include <iostream>
#include <numeric>

namespace ocsort
{
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf>
speed_direction_batch(const Eigen::MatrixXf &dets, const Eigen::MatrixXf &tracks)
{
  Eigen::VectorXf CX1 = (dets.col(0) + dets.col(2)) / 2.0;
  Eigen::VectorXf CY1 = (dets.col(1) + dets.col(3)) / 2.f;
  Eigen::MatrixXf CX2 = (tracks.col(0) + tracks.col(2)) / 2.f;
  Eigen::MatrixXf CY2 = (tracks.col(1) + tracks.col(3)) / 2.f;
  Eigen::MatrixXf dx
      = CX1.transpose().replicate(tracks.rows(), 1) - CX2.replicate(1, dets.rows());
  Eigen::MatrixXf dy
      = CY1.transpose().replicate(tracks.rows(), 1) - CY2.replicate(1, dets.rows());
  Eigen::MatrixXf norm = (dx.array().square() + dy.array().square()).sqrt() + 1e-6f;
  dx = dx.array() / norm.array();
  dy = dy.array() / norm.array();
  return std::make_tuple(dy, dx);
}
Eigen::MatrixXf
iou_batch(const Eigen::MatrixXf &bboxes1, const Eigen::MatrixXf &bboxes2)
{
  Eigen::Matrix<float, Eigen::Dynamic, 1> a = bboxes1.col(0); // bboxes1[..., 0] (n1,1)
  Eigen::Matrix<float, 1, Eigen::Dynamic> b = bboxes2.col(0); // bboxes2[..., 0] (1,n2)
  Eigen::MatrixXf xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
  a = bboxes1.col(1); // bboxes1[..., 1]
  b = bboxes2.col(1); // bboxes2[..., 1]
  Eigen::MatrixXf yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
  a = bboxes1.col(2); // bboxes1[..., 2]
  b = bboxes2.col(2); // bboxes1[..., 2]
  Eigen::MatrixXf xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
  a = bboxes1.col(3); // bboxes1[..., 3]
  b = bboxes2.col(3); // bboxes1[..., 3]
  Eigen::MatrixXf yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
  Eigen::MatrixXf w = (xx2 - xx1).cwiseMax(0);
  Eigen::MatrixXf h = (yy2 - yy1).cwiseMax(0);
  Eigen::MatrixXf wh = w.array() * h.array();
  a = (bboxes1.col(2) - bboxes1.col(0)).array()
      * (bboxes1.col(3) - bboxes1.col(1)).array();
  b = (bboxes2.col(2) - bboxes2.col(0)).array()
      * (bboxes2.col(3) - bboxes2.col(1)).array();
  Eigen::MatrixXf part1_ = a.replicate(1, b.cols());
  Eigen::MatrixXf part2_ = b.replicate(a.rows(), 1);
  Eigen::MatrixXf Sum = part1_ + part2_ - wh;
  return wh.cwiseQuotient(Sum);
}

Eigen::MatrixXf
giou_batch(const Eigen::MatrixXf &bboxes1, const Eigen::MatrixXf &bboxes2)
{
  Eigen::Matrix<float, Eigen::Dynamic, 1> a = bboxes1.col(0); // bboxes1[..., 0] (n1,1)
  Eigen::Matrix<float, 1, Eigen::Dynamic> b = bboxes2.col(0); // bboxes2[..., 0] (1,n2)
  Eigen::MatrixXf xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
  a = bboxes1.col(1); // bboxes1[..., 1]
  b = bboxes2.col(1); // bboxes2[..., 1]
  Eigen::MatrixXf yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
  a = bboxes1.col(2); // bboxes1[..., 2]
  b = bboxes2.col(2); // bboxes1[..., 2]
  Eigen::MatrixXf xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
  a = bboxes1.col(3); // bboxes1[..., 3]
  b = bboxes2.col(3); // bboxes1[..., 3]
  Eigen::MatrixXf yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
  Eigen::MatrixXf w = (xx2 - xx1).cwiseMax(0);
  Eigen::MatrixXf h = (yy2 - yy1).cwiseMax(0);
  Eigen::MatrixXf wh = w.array() * h.array();
  a = (bboxes1.col(2) - bboxes1.col(0)).array()
      * (bboxes1.col(3) - bboxes1.col(1)).array();
  b = (bboxes2.col(2) - bboxes2.col(0)).array()
      * (bboxes2.col(3) - bboxes2.col(1)).array();
  Eigen::MatrixXf part1_ = a.replicate(1, b.cols());
  Eigen::MatrixXf part2_ = b.replicate(a.rows(), 1);
  Eigen::MatrixXf Sum = part1_ + part2_ - wh;
  Eigen::MatrixXf iou = wh.cwiseQuotient(Sum);

  a = bboxes1.col(0);
  b = bboxes2.col(0);
  Eigen::MatrixXf xxc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
  a = bboxes1.col(1); // bboxes1[..., 1]
  b = bboxes2.col(1); // bboxes2[..., 1]
  Eigen::MatrixXf yyc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
  a = bboxes1.col(2); // bboxes1[..., 2]
  b = bboxes2.col(2); // bboxes1[..., 2]
  Eigen::MatrixXf xxc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
  a = bboxes1.col(3); // bboxes1[..., 3]
  b = bboxes2.col(3); // bboxes1[..., 3]
  Eigen::MatrixXf yyc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));

  Eigen::MatrixXf wc = xxc2 - xxc1;
  Eigen::MatrixXf hc = yyc2 - yyc1;
  if ((wc.array() > 0).all() && (hc.array() > 0).all())
    return iou;
  else {
    Eigen::MatrixXf area_enclose = wc.array() * hc.array();
    Eigen::MatrixXf giou
        = iou.array() - (area_enclose.array() - wh.array()) / area_enclose.array();
    giou = (giou.array() + 1) / 2.0;
    return giou;
  }
}


Eigen::MatrixXf
compute_aw_new_metric(const Eigen::MatrixXf &emb_cost, float w_association_emb,
    float max_diff = 0.5)
{
  // Initialize w_emb with the same size as emb_cost, filled with w_association_emb
  Eigen::MatrixXf w_emb
      = Eigen::MatrixXf::Constant(emb_cost.rows(), emb_cost.cols(), w_association_emb);

  // Initialize w_emb_bonus with the same size as emb_cost, filled with 0
  Eigen::MatrixXf w_emb_bonus = Eigen::MatrixXf::Zero(emb_cost.rows(), emb_cost.cols());

  // Row-wise processing
  if (emb_cost.cols() >= 2) { // Needs at least two columns to compute row weights
    for (int idx = 0; idx < emb_cost.rows(); ++idx) {
      // Sort indices of the row in descending order of values
      std::vector<int> indices(emb_cost.cols());
      std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., cols-1
      std::sort(indices.begin(), indices.end(),
          [&](int a, int b) { return emb_cost(idx, a) > emb_cost(idx, b); });

      // Compute row weight as the difference between the top and second-top values
      float row_weight
          = std::min(emb_cost(idx, indices[0]) - emb_cost(idx, indices[1]), max_diff);

      // Add half of the row weight to the corresponding row in w_emb_bonus
      w_emb_bonus.row(idx).array() += row_weight / 2.0f;
    }
  }

  // Column-wise processing
  if (emb_cost.rows() >= 2) { // Needs at least two rows to compute column weights
    for (int idj = 0; idj < emb_cost.cols(); ++idj) {
      // Sort indices of the column in descending order of values
      std::vector<int> indices(emb_cost.rows());
      std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., rows-1
      std::sort(indices.begin(), indices.end(),
          [&](int a, int b) { return emb_cost(a, idj) > emb_cost(b, idj); });

      // Compute column weight as the difference between the top and second-top values
      float col_weight
          = std::min(emb_cost(indices[0], idj) - emb_cost(indices[1], idj), max_diff);

      // Add half of the column weight to the corresponding column in w_emb_bonus
      w_emb_bonus.col(idj).array() += col_weight / 2.0f;
    }
  }

  // Return the sum of w_emb and w_emb_bonus
  return w_emb + w_emb_bonus;
}

std::tuple<std::vector<Eigen::Matrix<int, 1, 2>>, std::vector<int>, std::vector<int>>
associate(Eigen::MatrixXf detections, Eigen::MatrixXf trackers,
    Eigen::MatrixXf dets_embs, Eigen::MatrixXf trk_embs, float iou_threshold,
    Eigen::MatrixXf velocities, Eigen::MatrixXf previous_obs_, float vdc_weight,
    float w_assoc_emb, bool aw_off, float aw_param)
{
  if (trackers.rows() == 0) {
    std::vector<int> unmatched_dets;
    for (int i = 0; i < detections.rows(); i++) {
      unmatched_dets.push_back(i);
    }
    return std::make_tuple(std::vector<Eigen::Matrix<int, 1, 2>>(),
        unmatched_dets, std::vector<int>());
  }
  Eigen::MatrixXf Y, X;
  auto result = speed_direction_batch(detections, previous_obs_);
  Y = std::get<0>(result);
  X = std::get<1>(result);
  Eigen::MatrixXf inertia_Y = velocities.col(0);
  Eigen::MatrixXf inertia_X = velocities.col(1);
  Eigen::MatrixXf inertia_Y_ = inertia_Y.replicate(1, Y.cols());
  Eigen::MatrixXf inertia_X_ = inertia_X.replicate(1, X.cols());
  Eigen::MatrixXf diff_angle_cos
      = inertia_X_.array() * X.array() + inertia_Y_.array() * Y.array();
  diff_angle_cos = (diff_angle_cos.array().min(1).max(-1)).matrix();
  Eigen::MatrixXf diff_angle = Eigen::acos(diff_angle_cos.array());
  diff_angle = (pi / 2.0 - diff_angle.array().abs()).array() / (pi);
  Eigen::Array<bool, 1, Eigen::Dynamic> valid_mask
      = Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(previous_obs_.rows());
  valid_mask = valid_mask.array()
               * ((previous_obs_.col(4).array() >= 0).transpose()).array();
  Eigen::MatrixXf iou_matrix = iou_batch(detections, trackers);
  Eigen::MatrixXf scores
      = detections.col(detections.cols() - 2).replicate(1, trackers.rows());
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask_
      = (valid_mask.transpose()).replicate(1, X.cols());
  Eigen::MatrixXf angle_diff_cost;
  auto valid_float = valid_mask_.cast<float>();
  auto intermediate_result
      = (valid_float.array() * diff_angle.array() * vdc_weight).transpose();
  angle_diff_cost.noalias() = (intermediate_result.array() * scores.array()).matrix();

  Eigen::MatrixXf emb_cost;
  if (trk_embs.size() == 0 || dets_embs.size() == 0) {
    emb_cost = Eigen::MatrixXf(); // Empty matrix
  } else {
    emb_cost = dets_embs * trk_embs.transpose();
  }

  Eigen::Matrix<float, Eigen::Dynamic, 2> matched_indices(0, 2);
  if (std::min(iou_matrix.cols(), iou_matrix.rows()) > 0) {
    Eigen::MatrixXf a = (iou_matrix.array() > iou_threshold).cast<float>();
    float sum1 = (a.rowwise().sum()).maxCoeff();
    float sum0 = (a.colwise().sum()).maxCoeff();
    if ((fabs(sum1 - 1) < 1e-12) && (fabs(sum0 - 1) < 1e-12)) {
      for (int i = 0; i < a.rows(); i++) {
        for (int j = 0; j < a.cols(); j++) {
          if (a(i, j) > 0) {
            Eigen::RowVectorXf row(2);
            row << i, j;
            matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
            matched_indices.row(matched_indices.rows() - 1) = row;
          }
        }
      }
    } else {

      Eigen::MatrixXf emb_cost_matrix
          = Eigen::MatrixXf::Zero(iou_matrix.rows(), iou_matrix.cols());
      if (emb_cost.size() > 0) {
        if (!aw_off) {
          // Adaptive Weighting, 3.4 from https://arxiv.org/pdf/2302.11813
          Eigen::MatrixXf w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param);
          emb_cost_matrix = emb_cost.cwiseProduct(w_matrix);
        } else {
          emb_cost_matrix = emb_cost * w_assoc_emb;
        }
      }

      Eigen::MatrixXf cost_matrix
          = iou_matrix.array() + angle_diff_cost.array() + emb_cost_matrix.array();

      std::vector<std::vector<float>> cost_iou_matrix(
          cost_matrix.rows(), std::vector<float>(cost_matrix.cols()));
      for (int i = 0; i < cost_matrix.rows(); i++) {
        for (int j = 0; j < cost_matrix.cols(); j++) {
          cost_iou_matrix[i][j] = -cost_matrix(i, j);
        }
      }
      std::vector<int> rowsol, colsol;
      execLapjv(cost_iou_matrix, rowsol, colsol, true, 0.01, false);
      for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol.at(i) >= 0) {
          Eigen::RowVectorXf row(2);
          row << colsol.at(rowsol.at(i)), rowsol.at(i);
          matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
          matched_indices.row(matched_indices.rows() - 1) = row;
        }
      }
    }
  } else {
    matched_indices = Eigen::MatrixXf(0, 2);
  }
  std::vector<int> unmatched_detections;
  for (int i = 0; i < detections.rows(); i++) {
    if ((matched_indices.col(0).array() == i).sum() == 0) {
      unmatched_detections.push_back(i);
    }
  }
  std::vector<int> unmatched_trackers;
  for (int i = 0; i < trackers.rows(); i++) {
    if ((matched_indices.col(1).array() == i).sum() == 0) {
      unmatched_trackers.push_back(i);
    }
  }
  std::vector<Eigen::Matrix<int, 1, 2>> matches;
  Eigen::Matrix<int, 1, 2> tmp;
  for (int i = 0; i < matched_indices.rows(); i++) {
    tmp = (matched_indices.row(i)).cast<int>();
    if (iou_matrix(tmp(0), tmp(1)) < iou_threshold) {
      unmatched_detections.push_back(tmp(0));
      unmatched_trackers.push_back(tmp(1));
    } else {
      matches.push_back(tmp);
    }
  }
  if (matches.size() == 0) {
    matches.clear();
  }
  return std::make_tuple(matches, unmatched_detections, unmatched_trackers);
}
} // namespace ocsort
