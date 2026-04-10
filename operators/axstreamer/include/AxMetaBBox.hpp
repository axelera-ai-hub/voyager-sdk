// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

enum class box_format {
  xyxy = 0,
  xywh = 1,
  ltxywh = 2,
  xyxyxyxy = 3,
  xywhr = 4
};

struct box_xyxy {
  int x1;
  int y1;
  int x2;
  int y2;
};

struct box_xyxyxyxy {
  int x11;
  int y11;
  int x12;
  int y12;
  int x21;
  int y21;
  int x22;
  int y22;
};

struct __attribute__((packed)) box_xywhr {
  int x;
  int y;
  int w;
  int h;
  float r;
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

using BboxXyxyxyxy = box_xyxyxyxy;
using BboxXyxyxyxyVector = std::vector<BboxXyxyxyxy>;

using BboxXywhr = box_xywhr;
using BboxXywhrVector = std::vector<BboxXywhr>;

class AxMetaBbox : public virtual AxMetaBase
{
  public:
  AxMetaBbox() = default;

  explicit AxMetaBbox(BboxXyxyVector boxes, std::vector<float> scores,
      std::vector<int> classes, std::vector<int> ids)
      : bboxvec(std::move(boxes)),
        scores_(std::move(scores)),
        classes_(std::move(classes)),
        ids(std::move(ids))
  {
    if (num_elements() != classes_.size() && !classes_.empty()) {
      throw std::logic_error(
          "AxMetaObjDetection: scores and classes must have the same size as boxes");
    }

    if (!ids.empty() && ids.size() != bboxvec.size()) {
      throw std::runtime_error(
          "When constructing AxMetaBbox with ids, the number of ids must match the number of boxes");
    }
    if (num_elements() != scores_.size()) {
      throw std::logic_error(
          "AxMetaObjDetection: scores and classes must have the same size as boxes");
    }
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
  /// @brief Get the elements score at the given index. No bounds checking is
  /// performed. The behaviour is undefined if idx is out of range
  /// @param idx - Index of the requested element
  /// @return The score of the given element
  ///
  float score(size_t idx) const
  {
    return scores_[idx];
  }
  ///
  /// @brief As score, but if idx is out of bounds std::out_of_range is thrown.
  /// @param idx - Index of the requested element
  /// @return The score of the given element
  ///
  float score_at(size_t idx) const
  {
    return scores_.at(idx);
  }

  void set_score(size_t idx, float score)
  {
    scores_[idx] = score;
  }


  ///
  /// @brief Get the class id of the given element. No bounds checking is
  /// performed. The behaviour is undefined if idx is out of range
  /// @param idx - Index of the requested element
  /// @return The class id of the given element
  ///
  int class_id(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_[idx];
  }
  ///
  /// @brief As class_id, but if idx is out of bounds std::out_of_range is thrown.
  /// @param idx - Index of the requested element
  /// @return The class id of the given element
  ///
  int class_id_at(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_.at(idx);
  }
  // true if the metadata has class ids
  bool has_class_id() const
  {
    return !classes_.empty();
  }

  // true if the metadata has multiple class ids,
  // this is currenty an alias for has_class_id but this may change in the future
  bool is_multi_class() const
  {
    return has_class_id();
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

  void set_box_xyxy(size_t idx, const BboxXyxy &box)
  {
    bboxvec[idx] = box;
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
  /// @brief Set box id
  /// @param idx - Index of the box
  /// @param id - Id to set
  ///
  void set_id(size_t idx, int id)
  {
    if (ids.size() < bboxvec.size()) {
      ids.resize(bboxvec.size(), -1);
    }
    ids[idx] = id;
  }

  ///
  /// @brief Get box id
  /// @param idx - Index of the box
  /// @return - Id of the box
  ///
  int get_id(size_t idx) const
  {
    if (idx >= bboxvec.size()) {
      throw std::out_of_range("Index out of range");
    }
    if (idx >= ids.size()) {
      return -1;
    }
    return ids[idx];
  }

  ///
  /// @brief Get the number of boxes in the metadata
  /// @return Get the number of subframes (number of boxes in the metadata)
  ///
  size_t get_number_of_subframes() const override
  {
    return bboxvec.size();
  }

  void extend(const AxMetaBbox &other)
  {
    if (other.bboxvec.size() != other.scores_.size()) {
      throw std::runtime_error(
          "Other bbox and scores must have the same size in extend of AxMetaBbox");
    }
    bboxvec.insert(bboxvec.end(), other.bboxvec.begin(), other.bboxvec.end());
    scores_.insert(scores_.end(), other.scores_.begin(), other.scores_.end());
    classes_.insert(classes_.end(), other.classes_.begin(), other.classes_.end());
    ids.insert(ids.end(), other.ids.begin(), other.ids.end());
  }

  void extend(AxMetaBbox &&other)
  {
    if (other.bboxvec.size() != other.scores_.size()) {
      throw std::runtime_error(
          "Other bbox and scores must have the same size in extend of AxMetaBbox");
    }
    bboxvec.insert(bboxvec.end(), std::make_move_iterator(other.bboxvec.begin()),
        std::make_move_iterator(other.bboxvec.end()));
    scores_.insert(scores_.end(), std::make_move_iterator(other.scores_.begin()),
        std::make_move_iterator(other.scores_.end()));
    classes_.insert(classes_.end(), std::make_move_iterator(other.classes_.begin()),
        std::make_move_iterator(other.classes_.end()));
    ids.insert(ids.end(), std::make_move_iterator(other.ids.begin()),
        std::make_move_iterator(other.ids.end()));
  }

  protected:
  // Protected accessors for derived classes
  size_t scores_size() const
  {
    return scores_.size();
  }
  size_t classes_size() const
  {
    return classes_.size();
  }
  bool classes_empty() const
  {
    return classes_.empty();
  }
  const float *scores_data() const
  {
    return scores_.data();
  }
  const int *classes_data() const
  {
    return classes_.data();
  }

  BboxXyxyVector bboxvec;
  std::vector<float> scores_;
  std::vector<int> classes_;
  std::vector<int> ids;
};

class AxMetaBboxOBBBase : public virtual AxMetaBase
{
  public:
  AxMetaBboxOBBBase() = default;

  virtual ~AxMetaBboxOBBBase() = default;

  protected:
  AxMetaBboxOBBBase(std::vector<float> scores, std::vector<int> classes, std::vector<int> ids)
      : scores_(std::move(scores)),
        classes_(std::move(classes)),
        ids(std::move(ids))
  {
  }

  // Protected accessors for derived classes
  size_t scores_size() const
  {
    return scores_.size();
  }
  size_t classes_size() const
  {
    return classes_.size();
  }
  size_t ids_size() const
  {
    return ids.size();
  }
  bool classes_empty() const
  {
    return classes_.empty();
  }
  bool ids_empty() const
  {
    return ids.empty();
  }
  const float *scores_data() const
  {
    return scores_.data();
  }
  const int *classes_data() const
  {
    return classes_.data();
  }
  const int *ids_data() const
  {
    return ids.data();
  }

  public:
  virtual size_t num_elements() const = 0;

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
  }

  float score(size_t idx) const
  {
    return scores_[idx];
  }

  float score_at(size_t idx) const
  {
    return scores_.at(idx);
  }

  void set_score(size_t idx, float score)
  {
    scores_[idx] = score;
  }

  int class_id(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_[idx];
  }

  int class_id_at(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_.at(idx);
  }

  bool has_class_id() const
  {
    return !classes_.empty();
  }

  bool is_multi_class() const
  {
    return has_class_id();
  }

  void set_id(size_t idx, int id)
  {
    if (ids.size() < num_elements()) {
      ids.resize(num_elements(), -1);
    }
    ids[idx] = id;
  }

  int get_id(size_t idx) const
  {
    if (idx >= num_elements()) {
      throw std::out_of_range("Index out of range");
    }
    if (idx >= ids.size()) {
      return -1;
    }
    return ids[idx];
  }

  size_t get_number_of_subframes() const override
  {
    return num_elements();
  }

  private:
  std::vector<float> scores_;
  std::vector<int> classes_;
  std::vector<int> ids;
};

class AxMetaBboxOBBXYXYXYXY : public AxMetaBboxOBBBase
{
  public:
  AxMetaBboxOBBXYXYXYXY() = default;

  explicit AxMetaBboxOBBXYXYXYXY(BboxXyxyxyxyVector boxes,
      std::vector<float> scores, std::vector<int> classes, std::vector<int> ids)
      : AxMetaBboxOBBBase(std::move(scores), std::move(classes), std::move(ids)),
        bboxvec_xyxyxyxy(std::move(boxes))
  {

    if (num_elements() != classes_size() && !classes_empty()) {
      throw std::logic_error("AxMetaBboxOBBXYXYXYXY: scores and classes must have the same size as boxes. num_elements: "
                             + std::to_string(num_elements())
                             + " classes: " + std::to_string(classes_size()));
    }

    if (!ids_empty() && ids_size() != bboxvec_xyxyxyxy.size()) {
      throw std::runtime_error(
          "When constructing AxMetaBboxOBBXYXYXYXY with ids, the number of ids must match the number of boxes");
    }
    if (num_elements() != scores_size()) {
      throw std::logic_error(
          "AxMetaBboxOBBXYXYXYXY: scores and classes must have the same size as boxes");
    }
  }


  size_t num_elements() const override
  {
    return bboxvec_xyxyxyxy.size();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    return { { "bbox_obb", "bbox_obb",
                 static_cast<int>(bboxvec_xyxyxyxy.size() * sizeof(BboxXyxyxyxy)),
                 reinterpret_cast<const char *>(bboxvec_xyxyxyxy.data()) },
      { "is_xywhr", "is_xywhr", static_cast<int>(sizeof(is_xywhr)),
          reinterpret_cast<const char *>(&is_xywhr) } };
  }

  BboxXyxyxyxy get_box_xyxyxyxy(size_t idx) const
  {
    return bboxvec_xyxyxyxy[idx];
  }

  void set_box_xyxyxyxy(size_t idx, const BboxXyxyxyxy &box)
  {
    bboxvec_xyxyxyxy[idx] = box;
  }

  BboxXyxyxyxy get_box_xyxyxyxy_at(size_t idx) const
  {
    return bboxvec_xyxyxyxy.at(idx);
  }

  BboxXywhr to_xywhr(const BboxXyxyxyxy &box) const
  {
    std::vector<cv::Point2f> points
        = { cv::Point2f(box.x11, box.y11), cv::Point2f(box.x12, box.y12),
            cv::Point2f(box.x21, box.y21), cv::Point2f(box.x22, box.y22) };
    cv::RotatedRect rect = cv::minAreaRect(points);
    return { static_cast<int>(rect.center.x), static_cast<int>(rect.center.y),
      static_cast<int>(rect.size.width), static_cast<int>(rect.size.height),
      static_cast<float>(rect.angle * CV_PI / 180.0) };
  }

  const BboxXyxyxyxy *get_boxes_xyxyxyxy_data() const
  {
    return bboxvec_xyxyxyxy.data();
  }

  protected:
  BboxXyxyxyxyVector bboxvec_xyxyxyxy;
  static constexpr bool is_xywhr = false;
};

class AxMetaBboxXYWHR : public AxMetaBboxOBBBase
{
  public:
  AxMetaBboxXYWHR() = default;

  explicit AxMetaBboxXYWHR(BboxXywhrVector boxes, std::vector<float> scores,
      std::vector<int> classes, std::vector<int> ids)
      : AxMetaBboxOBBBase(std::move(scores), std::move(classes), std::move(ids)),
        bboxvec_xywhr(std::move(boxes))
  {

    if (num_elements() != classes_size() && !classes_empty()) {
      throw std::logic_error("AxMetaBboxXYWHR: scores and classes must have the same size as boxes. num_elements: "
                             + std::to_string(num_elements())
                             + " classes: " + std::to_string(classes_size()));
    }

    if (!ids_empty() && ids_size() != bboxvec_xywhr.size()) {
      throw std::runtime_error(
          "When constructing AxMetaBboxXYWHR with ids, the number of ids must match the number of boxes");
    }
    if (num_elements() != scores_size()) {
      throw std::logic_error(
          "AxMetaBboxXYWHR: scores and classes must have the same size as boxes");
    }
  }


  size_t num_elements() const override
  {
    return bboxvec_xywhr.size();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    return { { "bbox_obb", "bbox_obb",
                 static_cast<int>(bboxvec_xywhr.size() * sizeof(BboxXywhr)),
                 reinterpret_cast<const char *>(bboxvec_xywhr.data()) },
      { "is_xywhr", "is_xywhr", static_cast<int>(sizeof(is_xywhr)),
          reinterpret_cast<const char *>(&is_xywhr) } };
  }

  BboxXywhr get_box_xywhr(size_t idx) const
  {
    return bboxvec_xywhr[idx];
  }

  void set_box_xywhr(size_t idx, const BboxXywhr &box)
  {
    bboxvec_xywhr[idx] = box;
  }

  BboxXywhr get_box_xywhr_at(size_t idx) const
  {
    return bboxvec_xywhr.at(idx);
  }

  const BboxXywhr *get_boxes_xywhr_data() const
  {
    return bboxvec_xywhr.data();
  }

  protected:
  BboxXywhrVector bboxvec_xywhr;
  static constexpr bool is_xywhr = true;
};
