// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>
#include <time.h>

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

class AxMetaStreamId : public AxMetaBase
{
  public:
  int stream_id = 0;
  timespec timestamp{};
  AxMetaStreamId(int stream_id) : stream_id{ stream_id }
  {
    (void) ::clock_gettime(CLOCK_REALTIME, &timestamp);
    // printf("time is %d\n", timestamp.tv_sec, timestamp.tv_nsec);
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    if (video.info.format != AxVideoFormat::RGB && video.info.format != AxVideoFormat::RGBA) {
      throw std::runtime_error("Stream ID can only be drawn on RGB or RGBA");
    }
    cv::Mat mat(cv::Size(video.info.width, video.info.height),
        Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);

    const auto red = cv::Scalar(255, 0, 0);
    cv::putText(mat, std::to_string(stream_id).c_str(),
        cv::Point(video.info.width / 10, video.info.height / 10),
        cv::FONT_HERSHEY_SIMPLEX, 10.0, red);
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    const char *class_meta = "stream_meta";
    auto results = std::vector<extern_meta>{
      { class_meta, "stream_id", int(sizeof(stream_id)),
          reinterpret_cast<const char *>(&stream_id) },
      { class_meta, "timestamp", int(sizeof(timestamp)),
          reinterpret_cast<const char *>(&timestamp) },
    };
    return results;
  }
};
