// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

enum class box_format { xyxy = 0, xywh = 1, ltxywh = 2 };

struct box_xyxy {
  int x1;
  int y1;
  int x2;
  int y2;
};

struct box_xywh {
  int x;
  int y;
  int w;
  int h;
};

struct box_ltxywh {
  int x;
  int y;
  int w;
  int h;
};

using BboxXyxy = box_xyxy;
using BboxXyxyVector = std::vector<BboxXyxy>;

class AxMetaBbox : public virtual AxMetaBase
{
  public:
  AxMetaBbox(BboxXyxyVector boxes) : bboxvec(std::move(boxes))
  {
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    if (video.info.format != AxVideoFormat::RGB && video.info.format != AxVideoFormat::RGBA) {
      throw std::runtime_error("Boxes can only be drawn on RGB or RGBA");
    }
    cv::Mat mat(cv::Size(video.info.width, video.info.height),
        Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);
    for (auto i = size_t{}; i < bboxvec.size(); ++i) {
      cv::rectangle(mat,
          cv::Rect(cv::Point(bboxvec[i].x1, bboxvec[i].y1),
              cv::Point(bboxvec[i].x2, bboxvec[i].y2)),
          cv::Scalar(0, 0, 0));
    }
  }

  ///
  /// @brief Get the number of boxes in the metadata
  /// @return The number of boxes
  ///
  size_t num_elements() const
  {
    return bboxvec.size();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    return { { "bbox", "bbox", int(bboxvec.size() * sizeof(BboxXyxy)),
        reinterpret_cast<const char *>(bboxvec.data()) } };
  }

  ///
  /// @brief Get the box at the given index in xyxy format. No bounds checking
  /// is performed, if idx is out of bounds, the behaviour is undefined.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  BboxXyxy get_box_xyxy(size_t idx) const
  {
    return bboxvec[idx];
  }

  ///
  /// @brief Get the box at the given index in ltxyxy format. No bounds checking
  /// is performed, if idx is out of bounds, the behaviour is undefined.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_ltxywh get_box_ltxywh(size_t idx) const
  {
    const box_xyxy &box = bboxvec[idx];
    return { box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1 };
  }

  ///
  /// @brief Get the box at the given index in xywh format. No bounds checking
  /// is performed, if idx is out of bounds, the behaviour is undefined.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_xywh get_box_xywh(size_t idx) const
  {
    const box_xyxy &box = bboxvec[idx];
    return { (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, box.x2 - box.x1,
      box.y2 - box.y1 };
  }

  ///
  /// @brief Get the box at the given index in xyxy format.Bounds checking
  /// is performed, if idx is out of bounds, std::out_of_range is thrown.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  BboxXyxy get_box_xyxy_at(size_t idx) const
  {
    return bboxvec.at(idx);
  }

  ///
  /// @brief Get the box at the given index in ltxyxy format.Bounds checking
  /// is performed, if idx is out of bounds, std::out_of_range is thrown.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_ltxywh get_box_ltxywh_at(size_t idx) const
  {
    const box_xyxy &box = bboxvec.at(idx);
    return { box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1 };
  }

  ///
  /// @brief Get the box at the given index in xywh format.Bounds checking
  /// is performed, if idx is out of bounds, std::out_of_range is thrown.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_xywh get_box_xywh_at(size_t idx) const
  {
    const box_xyxy &box = bboxvec.at(idx);
    return { (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, box.x2 - box.x1,
      box.y2 - box.y1 };
  }

  const BboxXyxy *get_boxes_data() const
  {
    return bboxvec.data();
  }

  ///
  /// @brief Get the number of boxes in the metadata
  /// @return Get the number of subframes (number of boxes in the metadata)
  ///
  size_t get_number_of_subframes() const override
  {
    return bboxvec.size();
  }

  protected:
  BboxXyxyVector bboxvec;
};
