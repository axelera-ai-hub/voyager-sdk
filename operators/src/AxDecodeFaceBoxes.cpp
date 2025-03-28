// Copyright Axelera AI, 2023
// This file is based on the following code
// https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/cpp/models/src/detection_model_faceboxes.cpp
// Copyright (C) 2020-2023 Intel Corporation
// Licensed under the Apache License, Version 2.0

#include <numeric>
#include <unordered_set>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxNms.hpp"
#include "AxUtils.hpp"

struct Anchor {
  float left;
  float top;
  float right;
  float bottom;

  float getWidth() const
  {
    return (right - left) + 1.0f;
  }
  float getHeight() const
  {
    return (bottom - top) + 1.0f;
  }
  float getXCenter() const
  {
    return left + (getWidth() - 1.0f) / 2.0f;
  }
  float getYCenter() const
  {
    return top + (getHeight() - 1.0f) / 2.0f;
  }
};

void
calculateAnchors(std::vector<Anchor> &anchors, const std::vector<float> &vx,
    const std::vector<float> &vy, const int minSize, const int step)
{
  float skx = static_cast<float>(minSize);
  float sky = static_cast<float>(minSize);

  std::vector<float> dense_cx, dense_cy;

  for (auto x : vx) {
    dense_cx.push_back(x * step);
  }

  for (auto y : vy) {
    dense_cy.push_back(y * step);
  }

  for (auto cy : dense_cy) {
    for (auto cx : dense_cx) {
      anchors.push_back({ cx - 0.5f * skx, cy - 0.5f * sky, cx + 0.5f * skx,
          cy + 0.5f * sky }); // left top right bottom
    }
  }
}

void
calculateAnchorsZeroLevel(std::vector<Anchor> &anchors, const int fx,
    const int fy, const std::vector<int> &minSizes, const int step)
{
  for (auto s : minSizes) {
    std::vector<float> vx, vy;
    if (s == 32) {
      vx.push_back(static_cast<float>(fx));
      vx.push_back(fx + 0.25f);
      vx.push_back(fx + 0.5f);
      vx.push_back(fx + 0.75f);

      vy.push_back(static_cast<float>(fy));
      vy.push_back(fy + 0.25f);
      vy.push_back(fy + 0.5f);
      vy.push_back(fy + 0.75f);
    } else if (s == 64) {
      vx.push_back(static_cast<float>(fx));
      vx.push_back(fx + 0.5f);

      vy.push_back(static_cast<float>(fy));
      vy.push_back(fy + 0.5f);
    } else {
      vx.push_back(fx + 0.5f);
      vy.push_back(fy + 0.5f);
    }
    calculateAnchors(anchors, vx, vy, s, step);
  }
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "meta_key",
    "width", "height", "confidence_threshold", "nms_threshold" };
  return allowed_properties;
}

struct faceboxes_properties {
  std::string meta_key
      = "meta_" + std::to_string(reinterpret_cast<long long unsigned int>(this));
  unsigned int width = 0;
  unsigned int height = 0;
  float confidence_threshold = 0.7;
  float nms_threshold = 0.5;
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<faceboxes_properties> prop = std::make_shared<faceboxes_properties>();

  prop->meta_key = Ax::get_property(
      input, "meta_key", "faceboxes_static_properties", prop->meta_key);

  prop->width = Ax::get_property(input, "width", "faceboxes_static_properties", prop->width);
  prop->height
      = Ax::get_property(input, "height", "faceboxes_static_properties", prop->height);
  logger(AX_DEBUG) << "prop->width is " << prop->width << std::endl;
  logger(AX_DEBUG) << "prop->height is " << prop->height << std::endl;
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    faceboxes_properties *prop, Ax::Logger &logger)
{
  prop->confidence_threshold = Ax::get_property(input, "confidence_threshold",
      "faceboxes_dynamic_properties", prop->confidence_threshold);
  prop->nms_threshold = Ax::get_property(input, "nms_threshold",
      "faceboxes_dynamic_properties", prop->nms_threshold);
  logger(AX_DEBUG) << "prop->confidence_threshold is "
                   << prop->confidence_threshold << std::endl;
  logger(AX_DEBUG) << "prop->nms_threshold is " << prop->nms_threshold << std::endl;
}

extern "C" void
decode_to_meta(const AxTensorsInterface &tensors,
    const faceboxes_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_info, Ax::Logger &logger)
{
  unsigned int width = prop->width;
  unsigned int height = prop->height;
  if (width == 0 || height == 0) {
    throw std::runtime_error(
        "faceboxes_decode_to_meta : Faceboxes dimensions not specified or zero");
  }

  unsigned int orig_width = std::get<AxVideoInterface>(video_info).info.width;
  unsigned int orig_height = std::get<AxVideoInterface>(video_info).info.height;
  if (orig_width == 0 || orig_height == 0) {
    throw std::runtime_error("faceboxes_decode_to_meta : Original dimensions are zero");
  }

  float confidence_threshold = prop->confidence_threshold;
  std::vector<float> variance{ 0.1f, 0.2f };
  std::vector<int> steps{ 32, 64, 128 };
  std::vector<std::vector<int>> minSizes{ { 32, 64, 128 }, { 256 }, { 512 } };

  std::vector<std::pair<size_t, size_t>> featureMaps;
  for (auto s : steps) {
    featureMaps.push_back({ std::ceil(static_cast<double>(height) / s),
        std::ceil(static_cast<double>(width) / s) });
  }

  if (tensors.size() != 2 * featureMaps.size()) {
    throw std::runtime_error("faceboxes_decode_to_meta : Number of tensors must be "
                             "twice the number of feature maps, which is "
                             + std::to_string(featureMaps.size()));
  }

  std::vector<BboxXyxy> boxes;
  std::vector<float> scores;

  double scale = 1.0;
  int xoffset = 0;
  int yoffset = 0;

  if (orig_width > 0 && orig_height > 0) {
    double scale_width = double(orig_width) / width;
    double scale_height = double(orig_height) / height;
    bool scale_is_width = (scale_width >= scale_height);
    scale = scale_is_width ? scale_width : scale_height;
    xoffset = scale_is_width ? 0 : (width - int(orig_width / scale)) / 2;
    yoffset = scale_is_width ? (height - int(orig_height / scale)) / 2 : 0;
  }

  auto to_orig_x = [scale, xoffset, orig_width](int x) {
    return static_cast<int>(
        std::clamp((x - xoffset) * scale, 0.0, (double) orig_width - 1));
  };

  auto to_orig_y = [scale, yoffset, orig_height](int y) {
    return static_cast<int>(
        std::clamp((y - yoffset) * scale, 0.0, (double) orig_height - 1));
  };


  for (int k = 0; k < (int) featureMaps.size(); ++k) {
    int stride = featureMaps[k].first * featureMaps[k].second;
    float *boxesPtr = (float *) tensors[k].data;
    float *confPtr = (float *) tensors[k + featureMaps.size()].data;

    for (int i = 0; i < (int) featureMaps[k].first; ++i) {
      for (int j = 0; j < (int) featureMaps[k].second; ++j) {
        std::vector<Anchor> anchors;
        if (k == 0) {
          calculateAnchorsZeroLevel(anchors, j, i, minSizes[k], steps[k]);
        } else {
          calculateAnchors(anchors, { j + 0.5f }, { i + 0.5f }, minSizes[k][0], steps[k]);
        }

        int I = i * featureMaps[k].second + j;

        for (int a = 0; a < (int) anchors.size(); ++a) {
          int index_to_tvm = I + a * 4 * stride;
          int index_to_tvm_conf = I + a * 2 * stride;

          auto curr_conf_background = confPtr[index_to_tvm_conf];
          auto curr_conf = confPtr[index_to_tvm_conf + 1 * stride];
          curr_conf = std::exp(curr_conf)
                      / (std::exp(curr_conf) + std::exp(curr_conf_background));

          if (curr_conf > confidence_threshold) {
            scores.push_back(curr_conf);

            auto dx = boxesPtr[index_to_tvm + 0 * stride];
            auto dy = boxesPtr[index_to_tvm + 1 * stride];
            auto dw = boxesPtr[index_to_tvm + 2 * stride];
            auto dh = boxesPtr[index_to_tvm + 3 * stride];

            auto predCtrX = dx * variance[0] * anchors[a].getWidth()
                            + anchors[a].getXCenter();
            auto predCtrY = dy * variance[0] * anchors[a].getHeight()
                            + anchors[a].getYCenter();
            auto predW = exp(dw * variance[1]) * anchors[a].getWidth();
            auto predH = exp(dh * variance[1]) * anchors[a].getHeight();

            auto box = BboxXyxy{
              to_orig_x(predCtrX - 0.5f * predW),
              to_orig_y(predCtrY - 0.5f * predH),
              to_orig_x(predCtrX + 0.5f * predW),
              to_orig_y(predCtrY + 0.5f * predH),
            };
            boxes.push_back(box);
          }
        }
      }
    }
  }

  map[prop->meta_key] = std::make_unique<AxMetaObjDetection>(
      std::move(boxes), std::move(scores), std::vector<int>());
}
