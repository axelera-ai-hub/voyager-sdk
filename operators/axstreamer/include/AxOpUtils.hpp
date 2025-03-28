// Copyright Axelera AI, 2025
#pragma once

#include <array>
#include <cstdint>
#include <future>
#include <vector>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaKpts.hpp"

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace ax_utils
{
struct buffer_details {
  cl_int width{};
  cl_int height{};
  cl_int crop_x;
  cl_int crop_y;
  cl_int channels{};
  cl_int stride{};
  std::variant<void *, int> data{};
  std::vector<size_t> offsets;
  std::vector<size_t> strides;
  AxVideoFormat format{};
};
std::vector<buffer_details> extract_buffer_details(const AxDataInterface &input);

int determine_size(const buffer_details &info, int which_channel);

int determine_buffer_size(const buffer_details &info);

using lookups = std::array<float, 256>;

float to_sigmoid(float value);

float dequantize(int value, float scale, int32_t zero_point);

struct fbox {
  float x1;
  float y1;
  float x2;
  float y2;
};
struct fkpt {
  float x;
  float y;
  float visibility;
};

struct segment {
  int x1;
  int y1;
  int x2;
  int y2;
  std::vector<float> map;
};

using segment_func
    = std::function<ax_utils::segment(const std::vector<float> &, size_t, size_t)>;
struct inferences {
  std::vector<fbox> boxes;
  std::vector<fkpt> kpts;
  std::vector<segment> segments;
  std::vector<segment_func> seg_funcs;
  std::vector<float> scores;
  std::vector<int> class_ids;
  std::vector<int> kpts_shape;

  int prototype_width;
  int prototype_height;
  int prototype_depth;
  std::vector<float> prototype_coefs;
  void set_prototype_dims(int width, int height, int depth)
  {
    prototype_width = width;
    prototype_height = height;
    prototype_depth = depth;
  }

  inferences(int amount, int amount_kpts = 0)
  {
    boxes.reserve(amount);
    scores.reserve(amount);
    class_ids.reserve(amount);
    kpts.reserve(amount_kpts);
  }
};

std::vector<int> indices_for_topk(const std::vector<float> &scores, int topk);
std::vector<int> indices_for_topk_area(const std::vector<box_xyxy> &boxes, int topk);
std::vector<int> indices_for_topk_center(
    const std::vector<box_xyxy> &boxes, int topk, int width, int height);

inferences topk(const inferences &predictions, int topk);

template <typename F = std::identity>
std::vector<lookups>
build_general_dequantization_tables(const std::vector<float> &zero_points,
    const std::vector<float> &scales, F &&f = std::identity())
{
  std::vector<lookups> dequant_tables;
  for (size_t i = 0; i < scales.size(); ++i) {
    lookups dequant_table;
    // The zero points do not matter if we have ratios
    // In this case for numerical stability the largest value is the best
    float zero_point = 127.0;
    if (!zero_points.empty()) {
      zero_point = zero_points.at(i);
    }
    for (int j = 0; j != static_cast<int>(dequant_table.size()); ++j) {
      // Assuming signed int8, i.e. the range of the table is from -128 to 127
      dequant_table[j] = f(ax_utils::dequantize(j - 128, scales[i], zero_point));
    }
    dequant_tables.push_back(dequant_table);
  }
  return dequant_tables;
}

std::vector<lookups> build_sigmoid_tables(
    const std::vector<float> &zero_points, const std::vector<float> &scales);

std::vector<lookups> build_exponential_tables(
    const std::vector<float> &zero_points, const std::vector<float> &scales);

std::vector<lookups> build_exponential_tables_with_zero_point(
    const std::vector<float> &zero_points, const std::vector<float> &scales);

std::vector<lookups> build_dequantization_tables(
    const std::vector<float> &zero_points, const std::vector<float> &scales);

struct tensor_dims {
  int width;
  int height;
  int depth;
};

tensor_dims get_dims(const AxTensorsInterface &tensors, int level, bool transpose);

void softmax(const int8_t *input, int num_elems, size_t stride,
    const float *lookups, float *output);

template <typename T> class stride_iterator
{
  public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;

  stride_iterator(T *start, int offset, size_t step)
      : ptr(start + step * offset), step(step)
  {
  }

  reference operator*() const
  {
    return *ptr;
  }

  stride_iterator &operator++()
  {
    ptr += step;
    return *this;
  }

  stride_iterator operator++(int)
  {
    stride_iterator tmp = *this;
    ++(*this);
    return tmp;
  }

  stride_iterator &operator--()
  {
    ptr -= step;
    return *this;
  }

  stride_iterator operator--(int)
  {
    stride_iterator tmp = *this;
    --(*this);
    return tmp;
  }

  stride_iterator &operator+=(difference_type n)
  {
    ptr += n * step;
    return *this;
  }

  stride_iterator &operator-=(difference_type n)
  {
    ptr -= n * step;
    return *this;
  }

  difference_type operator-(const stride_iterator &other) const
  {
    return (ptr - other.ptr) / step;
  }

  stride_iterator operator+(difference_type n) const
  {
    return stride_iterator(ptr + n * step, step);
  }

  stride_iterator operator-(difference_type n) const
  {
    return stride_iterator(ptr - n * step, step);
  }

  bool operator<(const stride_iterator &other) const
  {
    return ptr < other.ptr;
  }
  bool operator>(const stride_iterator &other) const
  {
    return ptr > other.ptr;
  }
  bool operator<=(const stride_iterator &other) const
  {
    return ptr <= other.ptr;
  }
  bool operator>=(const stride_iterator &other) const
  {
    return ptr >= other.ptr;
  }

  bool operator!=(const stride_iterator &other) const
  {
    return ptr != other.ptr;
  }
  bool operator==(const stride_iterator &other) const
  {
    return ptr == other.ptr;
  }

  reference operator[](difference_type n) const
  {
    return *(ptr + n * step);
  }

  private:
  T *ptr;
  size_t step;
};

template <typename T>
stride_iterator<T>
make_stride_iterator(T *it, int offset, size_t n)
{
  return stride_iterator<T>(it, offset, n);
}

inline float
sigmoid(int8_t value, const float *sigmoids)
{
  int index = value + 128;
  return sigmoids[index];
}

inline float
sigmoid(float value, const float * /*unused*/)
{
  return ax_utils::to_sigmoid(value);
}

///
/// Dequantize, decode and filter classes according to score
/// confidence.
/// @param data - pointer to the raw tensor data
/// @param sigmoids - lookup table dequantizing values and applying sigmoid
/// @param confidence - minimum confidence score to keep a box
/// @param z_stride - working along z axis, stride to next element
/// @param props - properties of the model
/// @param outputs - output inferences
/// @return number of boxes added to outputs
///
template <bool multiclass, typename input_type>
int
decode_scores(const input_type *first, const float *sigmoids, int z_stride,
    const std::vector<int> &filter, float confidence, float object_score,
    inferences &outputs, float objectness_score = 1.0F)
{

  if (multiclass) {
    for (auto i : filter) {
      auto score = sigmoid(first[i * z_stride], sigmoids) * object_score * objectness_score;
      if (confidence <= score) {
        outputs.scores.push_back(score);
        outputs.class_ids.push_back(i);
      }
    }
  } else {
    auto highest_class = filter.front();
    auto highest_score = first[highest_class * z_stride];
    for (auto i : filter) {
      if (first[i * z_stride] > highest_score) {
        highest_score = first[i * z_stride];
        highest_class = i;
      }
    }
    auto score = sigmoid(highest_score, sigmoids) * object_score * objectness_score;
    if (confidence <= score) {
      outputs.scores.push_back(score);
      outputs.class_ids.push_back(highest_class);
    }
  }

  return outputs.scores.size() - outputs.boxes.size();
}

template <typename input_type>
int
decode_scores(const input_type *data, const float *lookups, int z_stride,
    const std::vector<int> &filter, float confidence, bool multiclass,
    inferences &outputs, float objectness_score = 1.0F)
{
  return multiclass ? decode_scores<true>(data, lookups, z_stride, filter,
             confidence, 1.0F, outputs, objectness_score) :
                      decode_scores<false>(data, lookups, z_stride, filter,
                          confidence, 1.0F, outputs, objectness_score);
}

std::vector<BboxXyxy> scale_boxes(const std::vector<fbox> &norm_boxes, int video_width,
    int video_height, int tensor_width, int tensor_height, bool scale_up, bool letterbox);

std::vector<BboxXyxy> scale_boxes(const std::vector<ax_utils::fbox> &norm_boxes,
    const AxVideoInterface &vinfo, int model_width, int model_height,
    bool scale_up, bool letterbox);

std::vector<BboxXyxy> scale_shift_boxes(const std::vector<ax_utils::fbox> &norm_boxes,
    BboxXyxy master_box, int tensor_width, int tensor_height, bool scale_up, bool letterbox);

std::vector<KptXyv> scale_kpts(const std::vector<fkpt> &norm_kpts, int video_width,
    int video_height, int tensor_width, int tensor_height, bool scale_up, bool letterbox);

std::vector<KptXyv> scale_shift_kpts(const std::vector<ax_utils::fkpt> &norm_kpts,
    BboxXyxy master_box, int tensor_width, int tensor_height, bool scale_up, bool letterbox);

std::vector<std::string> read_class_labels(
    const std::string &filename, const std::string &src, Ax::Logger &logger);

void validate_classes(const std::vector<std::string> &class_labels,
    int num_classes, const std::string &src, Ax::Logger &logger);

std::string_view trim(std::string_view s);

template <typename T, typename U>
std::tuple<std::vector<BboxXyxy>, std::vector<T>, std::vector<U>>
remove_empty_boxes(std::vector<BboxXyxy> in_boxes, std::vector<T> in_T,
    std::vector<U> in_U, int stride_T = 1, int stride_U = 1)
{
  auto size = in_boxes.size();
  if (size * stride_T != in_T.size()) {
    throw std::runtime_error("remove_empty_boxes : in_boxes and in_T size mismatch");
  }
  if (size * stride_U != in_U.size()) {
    throw std::runtime_error("remove_empty_boxes : in_boxes and in_U size mismatch");
  }
  auto out_boxes = std::vector<BboxXyxy>{};
  auto out_T = std::vector<T>{};
  auto out_U = std::vector<U>{};

  for (int i = 0; i < size; ++i) {
    auto &box = in_boxes[i];
    if (box.x1 != box.x2 && box.y1 != box.y2) {
      out_boxes.push_back(box);
      for (int j = 0; j < stride_T; ++j) {
        out_T.push_back(in_T[i * stride_T + j]);
      }
      for (int j = 0; j < stride_U; ++j) {
        out_U.push_back(in_U[i * stride_U + j]);
      }
    }
  }
  return { std::move(out_boxes), std::move(out_T), std::move(out_U) };
}

template <typename T>
T *
get_meta(const std::string &meta_name,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    const std::string &src)
{
  if (meta_name.empty()) {
    if (!src.empty()) {
      throw std::runtime_error(src + " : No meta key given");
    }
    throw std::runtime_error("No meta key given");
  }
  auto meta_itr = meta_map.find(meta_name);
  if (meta_itr == meta_map.end()) {
    std::string error_msg = meta_name + " not found in meta map";
    if (!src.empty()) {
      throw std::runtime_error(src + " : " + error_msg);
    }
    throw std::runtime_error(error_msg);
  }
  T *meta = dynamic_cast<T *>(meta_itr->second.get());
  if (!meta) {
    auto desired_type = typeid(T).name();
    auto &rmeta = *meta_itr->second;
    auto actual_type = typeid(rmeta).name();
    std::string error_msg = "Meta key " + meta_name + " is not of type "
                            + desired_type + " but " + actual_type;
    if (!src.empty()) {
      throw std::runtime_error(src + " : " + error_msg);
    }
    throw std::runtime_error(error_msg);
  }
  return meta;
}

template <typename T, typename... Args>
void
insert_meta(std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const std::string &key, const std::string &master_key, int subframe_index,
    int number_of_subframes, Args &&...args)
{
  if (master_key.empty()) {
    map[key] = std::make_unique<T>(std::forward<Args>(args)...);
  } else {
    auto master_meta = get_meta<AxMetaBase>(master_key, map, "insert_meta");
    master_meta->submeta_map->insert(key, subframe_index, number_of_subframes,
        std::make_shared<T>(std::forward<Args>(args)...));
  }
}

} // namespace ax_utils
