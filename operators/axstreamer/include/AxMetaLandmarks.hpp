// Copyright Axelera AI, 2025

#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "AxMeta.hpp"
#include "AxUtils.hpp"

class AxMetaLandmarks : public AxMetaBase
{
  public:
  struct point {
    float x;
    float y;
  };
  static inline constexpr int total_landmarks = 68;

  using landmark = std::array<point, total_landmarks>;
  std::vector<landmark> all_landmarks;

  AxMetaLandmarks(int num_landmarks) : all_landmarks(num_landmarks)
  {
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    return { { "landmarks", "facial_landmarks", int(all_landmarks.size() * sizeof(landmark)),
        reinterpret_cast<const char *>(all_landmarks.data()) } };
  }

  enum class landmark_type {
    face,
    eyebrow_1,
    eyebrow_2,
    nose,
    nostril,
    eye_1,
    eye_2,
    lips,
    teeth
  };

  struct landmark_info {
    landmark_type type;
    int start_point;
    int end_point;
  };

  static constexpr inline landmark_info landmark_details[]{
    { landmark_type::face, 0, 17 },
    { landmark_type::eyebrow_1, 17, 22 },
    { landmark_type::eyebrow_2, 22, 27 },
    { landmark_type::nose, 27, 31 },
    { landmark_type::nostril, 31, 36 },
    { landmark_type::eye_1, 36, 42 },
    { landmark_type::eye_2, 42, 48 },
    { landmark_type::lips, 48, 60 },
    { landmark_type::teeth, 60, 68 },
  };


  void draw_lines(cv::Mat &canvas, const landmark &landmarks,
      const landmark_info &info, const cv::Scalar &color)
  {
    for (int i = info.start_point; i != info.end_point - 1; ++i) {
      cv::line(canvas, cv::Point(landmarks[i].x, landmarks[i].y),
          cv::Point(landmarks[i + 1].x, landmarks[i + 1].y), color);
    }
  }

  void draw_connecting_line(cv::Mat &canvas, const landmark &landmarks,
      const landmark_info &info, const cv::Scalar &color)
  {
    cv::line(canvas,
        cv::Point(landmarks[info.end_point - 1].x, landmarks[info.end_point - 1].y),
        cv::Point(landmarks[info.start_point].x, landmarks[info.start_point].y), color);
  }

  void draw_points(cv::Mat &canvas, const landmark &landmarks,
      const landmark_info &info, const cv::Scalar &color)
  {
    constexpr int radius = 3;
    for (int i = info.start_point; i != info.end_point; ++i) {
      cv::circle(canvas, cv::Point(landmarks[i].x, landmarks[i].y), radius, color);
    }
  }

  static inline const auto fuchsia = cv::Scalar(255, 0, 255);
  static inline const auto lightskyblue = cv::Scalar(135, 206, 250);
  static inline const auto lightblue = cv::Scalar(173, 216, 230);
  static inline const auto mediumblue = cv::Scalar(0, 0, 205);

  static constexpr landmark_type circles[] = {
    landmark_type::eye_1,
    landmark_type::eye_2,
    landmark_type::teeth,
    landmark_type::lips,
  };

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    if (video.info.format != AxVideoFormat::RGB && video.info.format != AxVideoFormat::RGBA) {
      throw std::runtime_error("Landmarks can only be drawn on RGB or RGBA");
    }
    cv::Mat mat(cv::Size(video.info.width, video.info.height),
        Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);

    for (const auto &landmarks : all_landmarks) {
      for (const auto &info : landmark_details) {
        draw_lines(mat, landmarks, info, fuchsia);
        if (std::find(std::begin(circles), std::end(circles), info.type)
            != std::end(circles)) {
          draw_connecting_line(mat, landmarks, info, fuchsia);
        }
        draw_points(mat, landmarks, info, mediumblue);
      }
    }
  }
};
