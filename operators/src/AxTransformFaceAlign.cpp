// Copyright Axelera AI, 2023
#include <AxOpUtils.hpp>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxUtils.hpp"

struct facealign_properties {
  std::string master_meta;
  std::string keypoints_submeta_key;
  int width = 0;
  int height = 0;
  float padding = 0.0;
  std::vector<float> XXn{};
  std::vector<float> YYn{};

  constexpr static std::array<float, 5> XXn_5
      = { 30.2946 / 96, 65.5318 / 96, 48.0252 / 96, 33.5493 / 96, 62.7299 / 96 };
  constexpr static std::array<float, 5> YYn_5
      = { 51.6963 / 96, 51.5014 / 96, 71.7366 / 96, 92.3655 / 96, 92.2041 / 96 };
  constexpr static std::array<float, 51> XXn_51 = {
    0.000213256,
    0.0752622,
    0.18113,
    0.29077,
    0.393397,
    0.586856,
    0.689483,
    0.799124,
    0.904991,
    0.98004,
    0.490127,
    0.490127,
    0.490127,
    0.490127,
    0.36688,
    0.426036,
    0.490127,
    0.554217,
    0.613373,
    0.121737,
    0.187122,
    0.265825,
    0.334606,
    0.260918,
    0.182743,
    0.645647,
    0.714428,
    0.793132,
    0.858516,
    0.79751,
    0.719335,
    0.254149,
    0.340985,
    0.428858,
    0.490127,
    0.551395,
    0.639268,
    0.726104,
    0.642159,
    0.556721,
    0.490127,
    0.423532,
    0.338094,
    0.290379,
    0.428096,
    0.490127,
    0.552157,
    0.689874,
    0.553364,
    0.490127,
    0.42689,
  };
  constexpr static std::array<float, 51> YYn_51 = {
    0.106454,
    0.038915,
    0.0187482,
    0.0344891,
    0.0773906,
    0.0773906,
    0.0344891,
    0.0187482,
    0.038915,
    0.106454,
    0.203352,
    0.307009,
    0.409805,
    0.515625,
    0.587326,
    0.609345,
    0.628106,
    0.609345,
    0.587326,
    0.216423,
    0.178758,
    0.179852,
    0.231733,
    0.245099,
    0.244077,
    0.231733,
    0.179852,
    0.178758,
    0.216423,
    0.244077,
    0.245099,
    0.780233,
    0.745405,
    0.727388,
    0.742578,
    0.727388,
    0.745405,
    0.780233,
    0.864805,
    0.902192,
    0.909281,
    0.902192,
    0.864805,
    0.784792,
    0.778746,
    0.785343,
    0.778746,
    0.784792,
    0.824182,
    0.831803,
    0.824182,
  };
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "master_meta",
    "keypoints_submeta_key",
    "width",
    "height",
    "padding",
    "template_keypoints_x",
    "template_keypoints_y",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<facealign_properties> prop = std::make_shared<facealign_properties>();
  prop->master_meta = Ax::get_property(
      input, "master_meta", "facealign_static_properties", prop->master_meta);
  if (prop->master_meta.empty()) {
    throw std::runtime_error("facealign: master meta key not provided");
  }
  prop->keypoints_submeta_key = Ax::get_property(input, "keypoints_submeta_key",
      "facealign_static_properties", prop->keypoints_submeta_key);
  if (prop->keypoints_submeta_key.empty()) {
    throw std::runtime_error("facealign: keypoints submeta key not provided");
  }
  prop->width = Ax::get_property(input, "width", "facealign_static_properties", prop->width);
  prop->height
      = Ax::get_property(input, "height", "facealign_static_properties", prop->height);
  prop->padding = Ax::get_property(
      input, "padding", "facealign_static_properties", prop->padding);
  prop->XXn = Ax::get_property(
      input, "template_keypoints_x", "facealign_static_properties", prop->XXn);
  prop->YYn = Ax::get_property(
      input, "template_keypoints_y", "facealign_static_properties", prop->YYn);
  if (prop->XXn.size() != prop->YYn.size()) {
    throw std::runtime_error(
        "facealign: template_keypoints_x and template_keypoints_y must have the same number of elements");
  }
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const facealign_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error("facealign works on video input only");
  }
  AxDataInterface output = interface;
  if (prop->width == 0 || prop->height == 0) {
    return output;
  }
  auto &info = std::get<AxVideoInterface>(output).info;
  info.width = prop->width;
  info.height = prop->height;
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const facealign_properties *prop, unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  auto *master_meta = ax_utils::get_meta<AxMetaBbox>(prop->master_meta, map, "facealign");
  if (master_meta->num_elements() != number_of_subframes) {
    throw std::runtime_error(
        "facealign: master meta key does not have the correct number of subframes");
  }
  auto box = master_meta->get_box_xyxy(subframe_index);
  AxMetaKptsDetection *kpts_meta
      = dynamic_cast<AxMetaKptsDetection *>(master_meta->submeta_map->get(
          prop->keypoints_submeta_key, subframe_index, number_of_subframes));
  if (kpts_meta == nullptr) {
    throw std::runtime_error("facealign: keypoints submeta key not of type AxMetaKptsDetection");
  }
  int kpts_per_box = kpts_meta->get_kpts_shape()[0];
  if (kpts_per_box != kpts_meta->num_elements()) {
    throw std::runtime_error(
        "facealign: keypoints submeta does not have the correct number of elements");
  }
  std::vector<float> XX = prop->XXn;
  std::vector<float> YY = prop->YYn;
  int kpts_offset = 0;
  if (XX.empty() && kpts_per_box == 68) {
    XX = std::vector<float>(prop->XXn_51.begin(), prop->XXn_51.end());
    YY = std::vector<float>(prop->YYn_51.begin(), prop->YYn_51.end());
    kpts_offset = 17;
  } else if (XX.empty() && kpts_per_box == 5) {
    XX = std::vector<float>(prop->XXn_5.begin(), prop->XXn_5.end());
    YY = std::vector<float>(prop->YYn_5.begin(), prop->YYn_5.end());
  } else {
    if (XX.size() != kpts_per_box || YY.size() != kpts_per_box) {
      throw std::runtime_error(
          "facealign: template_keypoints_x and template_keypoints_y must have the same number of elements as the number of keypoints in the submeta");
    }
  }

  std::vector<float> X;
  std::vector<float> Y;
  for (int i = kpts_offset; i < kpts_per_box; ++i) {
    KptXyv kpt = kpts_meta->get_kpt_xy(i);
    X.push_back(kpt.x - box.x1);
    Y.push_back(kpt.y - box.y1);
  }
  float meanX = std::accumulate(X.begin(), X.end(), 0.0) / X.size();
  float meanY = std::accumulate(Y.begin(), Y.end(), 0.0) / Y.size();
  float stdX = std::sqrt(
      std::inner_product(X.begin(), X.end(), X.begin(), 0.0) / X.size() - meanX * meanX);
  float stdY = std::sqrt(
      std::inner_product(Y.begin(), Y.end(), Y.begin(), 0.0) / Y.size() - meanY * meanY);
  std::for_each(X.begin(), X.end(), [&](float &x) { x = (x - meanX) / stdX; });
  std::for_each(Y.begin(), Y.end(), [&](float &y) { y = (y - meanY) / stdY; });

  auto &input_video = std::get<AxVideoInterface>(input);
  auto &output_video = std::get<AxVideoInterface>(output);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);
  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);

  std::for_each(XX.begin(), XX.end(),
      [&](float &x) { x = (x + prop->padding) / (2 * prop->padding + 1); });
  std::for_each(YY.begin(), YY.end(),
      [&](float &y) { y = (y + prop->padding) / (2 * prop->padding + 1); });
  std::for_each(
      XX.begin(), XX.end(), [&](float &x) { x *= output_mat.size().width; });
  std::for_each(
      YY.begin(), YY.end(), [&](float &y) { y *= output_mat.size().height; });
  float meanXX = std::accumulate(XX.begin(), XX.end(), 0.0) / XX.size();
  float meanYY = std::accumulate(YY.begin(), YY.end(), 0.0) / YY.size();
  float stdXX = std::sqrt(
      std::inner_product(XX.begin(), XX.end(), XX.begin(), 0.0) / XX.size() - meanXX * meanXX);
  float stdYY = std::sqrt(
      std::inner_product(YY.begin(), YY.end(), YY.begin(), 0.0) / YY.size() - meanYY * meanYY);
  std::for_each(XX.begin(), XX.end(), [&](float &x) { x = (x - meanXX) / stdXX; });
  std::for_each(YY.begin(), YY.end(), [&](float &y) { y = (y - meanYY) / stdYY; });

  cv::Mat_<float> A(2, 2);
  cv::Mat_<float> W(2, 2);
  cv::Mat_<float> U(2, 2);
  cv::Mat_<float> Vt(2, 2);

  A(0, 0) = std::inner_product(X.begin(), X.end(), XX.begin(), 0.0);
  A(0, 1) = std::inner_product(X.begin(), X.end(), YY.begin(), 0.0);
  A(1, 0) = std::inner_product(Y.begin(), Y.end(), XX.begin(), 0.0);
  A(1, 1) = std::inner_product(Y.begin(), Y.end(), YY.begin(), 0.0);

  cv::SVD::compute(A, W, U, Vt);

  cv::Mat_<float> R = (U * Vt).t();

  cv::Mat_<float> M(2, 3);
  M(0, 0) = R(0, 0) * (stdXX / stdX);
  M(1, 0) = R(1, 0) * (stdYY / stdY);
  M(0, 1) = R(0, 1) * (stdXX / stdX);
  M(1, 1) = R(1, 1) * (stdYY / stdY);
  M(0, 2) = meanXX - (stdXX / stdX) * (R(0, 0) * meanX + R(0, 1) * meanY);
  M(1, 2) = meanYY - (stdYY / stdY) * (R(1, 0) * meanX + R(1, 1) * meanY);

  cv::warpAffine(input_mat, output_mat, M, output_mat.size());
}
