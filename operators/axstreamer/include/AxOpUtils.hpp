// Copyright Axelera AI, 2023
#pragma once

#include <array>
#include <cstdint>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <future>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaBBox.hpp"
#include "AxMetaKpts.hpp"
#include "AxMetaTracker.hpp"

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace ax_utils
{
enum DistanceMetric {
  EUCLIDEAN_DISTANCE = 1,
  SQUARED_EUCLIDEAN_DISTANCE,
  COSINE_DISTANCE,
  COSINE_SIMILARITY
};

int parse_metric_type(const std::unordered_map<std::string, std::string> &input,
    int default_value, const std::string &error_type);

// Only declarations in header
std::vector<float> embeddings_cosine_similarity(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise = true);

std::vector<float> embeddings_euclidean_distance(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise = false);

std::vector<float> embeddings_squared_euclidean_distance(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise = false);

std::vector<float> embeddings_cosine_distance(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise = true);

void add_vec_to_matrix(const std::vector<float> &vec, Eigen::MatrixXf &matrix);

std::pair<Eigen::MatrixXf, std::vector<std::string>> read_embedding_json(
    const std::string &filename, bool normalise, Ax::Logger &logger);

void write_embedding_json(const Eigen::MatrixXf &embeddings,
    const std::vector<std::string> &labels, const std::string &filename,
    Ax::Logger &logger);


typedef enum : int {
  RGBA_OUTPUT = 0,
  BGRA_OUTPUT = 1,
  RGB_OUTPUT = 3,
  BGR_OUTPUT = 4,
  GRAY_OUTPUT = 5
} output_format;

struct buffer_details {
  cl_int width{};
  cl_int height{};
  cl_int crop_x;
  cl_int crop_y;
  cl_int channels{};
  cl_int stride{};
  std::variant<void *, int, VASurfaceID_proxy *, opencl_buffer *> data{};
  std::vector<size_t> offsets;
  std::vector<size_t> strides;
  AxVideoFormat format{};
  cl_int actual_height{};
};

struct transfer_info {
  bool is_crop = false;
  std::vector<int> in_sizes{};
  std::vector<int> out_sizes{};
  std::vector<cv::Range> ranges{};
};
size_t get_bytes_per_pixel(AxVideoFormat format);
transfer_info get_transfer_info(
    const std::vector<int> &sizes, const std::vector<int> &padding);

std::string sizes_to_string(const std::vector<int> &sizes);
std::vector<buffer_details> extract_buffer_details(const AxDataInterface &input);
bool validate_shape(const std::vector<int> &new_shape, const std::vector<int> &original);

int determine_size(const buffer_details &info, int which_channel);

int determine_buffer_size(const buffer_details &info);

void remove_cropinfo(AxDataInterface &out);

using lookups = std::array<float, 256>;
using sin_cos_lookups = std::array<float, 512>;

float to_sigmoid(float value);

float dequantize(int value, float scale, int32_t zero_point);

struct fobox {
  float x;
  float y;
  float w;
  float h;
  float angle;
};

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
  int base_box_x1;
  int base_box_y1;
  int base_box_x2;
  int base_box_y2;
  int x1;
  int y1;
  int x2;
  int y2;
  std::vector<uint8_t> map;
};

struct segment_details {
  float x1;
  float y1;
  float x2;
  float y2;
  float scale;
  float zero;
  std::vector<int8_t> mask_data;
};

struct prototype_details {
  int width;
  int height;
  int depth;
  float scale;
  float zero;
  std::unique_ptr<uint8_t[]> coefs;
  size_t coefs_size;
};

struct inferences {
  std::vector<fbox> boxes;
  std::vector<fobox> obb;
  std::vector<fkpt> kpts;
  std::vector<segment> segments;
  std::vector<segment_details> seg_info;
  std::vector<float> scores;
  std::vector<int> class_ids;
  std::vector<int> kpts_shape;

  prototype_details prototype;

  void set_prototype(prototype_details proto)
  {
    prototype = std::move(proto);
  }

  inferences(int amount, int amount_kpts = 0)
  {
    boxes.reserve(amount);
    obb.reserve(amount);
    kpts.reserve(amount_kpts);
    segments.reserve(amount);
    seg_info.reserve(amount);
    scores.reserve(amount);
    class_ids.reserve(amount);
  }

  void extend(inferences other)
  {
    boxes.insert(boxes.end(), std::make_move_iterator(other.boxes.begin()),
        std::make_move_iterator(other.boxes.end()));
    obb.insert(obb.end(), std::make_move_iterator(other.obb.begin()),
        std::make_move_iterator(other.obb.end()));
    scores.insert(scores.end(), std::make_move_iterator(other.scores.begin()),
        std::make_move_iterator(other.scores.end()));
    class_ids.insert(class_ids.end(), std::make_move_iterator(other.class_ids.begin()),
        std::make_move_iterator(other.class_ids.end()));
    seg_info.insert(seg_info.end(), std::make_move_iterator(other.seg_info.begin()),
        std::make_move_iterator(other.seg_info.end()));
    segments.insert(segments.end(), std::make_move_iterator(other.segments.begin()),
        std::make_move_iterator(other.segments.end()));
    kpts.insert(kpts.end(), std::make_move_iterator(other.kpts.begin()),
        std::make_move_iterator(other.kpts.end()));
  }
};

std::vector<int> indices_for_topk(const std::vector<float> &scores, int topk);
std::vector<int> indices_for_topk_area(const std::vector<box_xyxy> &boxes, int topk);
std::vector<int> indices_for_topk_center(
    const std::vector<box_xyxy> &boxes, int topk, int width, int height);

inferences topk(inferences predictions, int topk);

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

std::vector<sin_cos_lookups> build_trigonometric_tables(
    const std::vector<float> &zero_points, const std::vector<float> &scales);

std::vector<sin_cos_lookups> build_sigmoid_trigonometric_tables(
    const std::vector<float> &zero_points, const std::vector<float> &scales,
    float add, float mul);

struct tensor_dims {
  int width;
  int height;
  int depth;
};

tensor_dims get_dims(const AxTensorsInterface &tensors, int level, bool transpose);

void softmax(const int8_t *input, int num_elems, const float *lookups, float *output);

template <typename T> class stride_iterator
{
  public:
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;

  stride_iterator(T *start, int offset, size_t step)
      : ptr(start + step * offset),
        step(step)
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

#if defined(__AVX2__)
#define USE_AVX2
#include <immintrin.h>

///
/// AVX2-optimized function to find maximum score with filter for int8_t
/// @param data - pointer to int8_t scores
/// @param filter - byte array where 0xff means keep, 0x00 means skip
/// @param size - number of elements
/// @return pair of (max_value, max_index) or (-128, -1) if no valid element
///
inline std::pair<int8_t, int>
find_max_filtered_avx2(const int8_t *data, const uint8_t *filter, int size)
{
  constexpr int8_t MIN_VAL = std::numeric_limits<int8_t>::min();
  __m256i max_vec = _mm256_set1_epi8(MIN_VAL);
  __m256i idx_vec = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
  __m256i max_idx_vec = _mm256_set1_epi8(-1);
  __m256i increment = _mm256_set1_epi8(32);

  int i = 0;
  // Process 32 elements at a time
  for (; i + 31 < size; i += 32) {
    __m256i data_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i));
    __m256i filter_vec
        = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(filter + i));

    // Apply filter: set filtered-out values to MIN_VAL
    __m256i mask
        = _mm256_cmpeq_epi8(filter_vec, _mm256_set1_epi8(static_cast<int8_t>(0xff)));
    __m256i filtered_data = _mm256_blendv_epi8(_mm256_set1_epi8(MIN_VAL), data_vec, mask);

    // Update max values and indices
    __m256i cmp = _mm256_cmpgt_epi8(filtered_data, max_vec);
    max_vec = _mm256_max_epi8(max_vec, filtered_data);
    max_idx_vec = _mm256_blendv_epi8(max_idx_vec, idx_vec, cmp);

    idx_vec = _mm256_add_epi8(idx_vec, increment);
  }

  // Horizontal reduction to find the maximum
  alignas(32) int8_t max_arr[32];
  alignas(32) int8_t idx_arr[32];
  _mm256_store_si256(reinterpret_cast<__m256i *>(max_arr), max_vec);
  _mm256_store_si256(reinterpret_cast<__m256i *>(idx_arr), max_idx_vec);

  int8_t max_val = MIN_VAL;
  int max_idx = -1;
  for (int j = 0; j < 32; ++j) {
    if (max_arr[j] > max_val) {
      max_val = max_arr[j];
      max_idx = idx_arr[j];
    }
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    if (filter[i] && data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }

  return { max_val, max_idx };
}

///
/// AVX2-optimized function to find maximum score with filter for float
/// @param data - pointer to float scores
/// @param filter - byte array where non-zero means keep, 0x00 means skip
/// @param size - number of elements
/// @return pair of (max_value, max_index) or (-inf, -1) if no valid element
///
inline std::pair<float, int>
find_max_filtered_avx2(const float *data, const uint8_t *filter, int size)
{
  const float MIN_VAL = -std::numeric_limits<float>::infinity();
  __m256 max_vec = _mm256_set1_ps(MIN_VAL);
  __m256i idx_vec = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  __m256i max_idx_vec = _mm256_set1_epi32(-1);
  __m256i increment = _mm256_set1_epi32(8);

  int i = 0;
  // Process 8 elements at a time
  for (; i + 7 < size; i += 8) {
    __m256 data_vec = _mm256_loadu_ps(data + i);

    // Load filter as bytes, convert to 32-bit mask
    __m128i filter_bytes
        = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(filter + i));
    __m256i filter_32 = _mm256_cvtepu8_epi32(filter_bytes);
    __m256i mask_i = _mm256_cmpgt_epi32(filter_32, _mm256_setzero_si256());
    __m256 mask = _mm256_castsi256_ps(mask_i);

    // Apply filter: set filtered-out values to MIN_VAL
    __m256 filtered_data = _mm256_blendv_ps(_mm256_set1_ps(MIN_VAL), data_vec, mask);

    // Update max values and indices
    __m256 cmp = _mm256_cmp_ps(filtered_data, max_vec, _CMP_GT_OQ);
    max_vec = _mm256_max_ps(max_vec, filtered_data);
    max_idx_vec = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(max_idx_vec), _mm256_castsi256_ps(idx_vec), cmp));

    idx_vec = _mm256_add_epi32(idx_vec, increment);
  }

  // Horizontal reduction to find the maximum
  alignas(32) float max_arr[8];
  alignas(32) int32_t idx_arr[8];
  _mm256_store_ps(max_arr, max_vec);
  _mm256_store_si256(reinterpret_cast<__m256i *>(idx_arr), max_idx_vec);

  float max_val = MIN_VAL;
  int max_idx = -1;
  for (int j = 0; j < 8; ++j) {
    if (max_arr[j] > max_val) {
      max_val = max_arr[j];
      max_idx = idx_arr[j];
    }
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    if (filter[i] && data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }

  return { max_val, max_idx };
}
#endif // __AVX2__

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
decode_scores(const input_type *first, const float *sigmoids,
    const std::vector<uint8_t> &filter, float confidence, float object_score,
    inferences &outputs)
{
  const auto initial_size = outputs.scores.size();
  if (multiclass) {
    for (int i = 0; i != static_cast<int>(filter.size()); ++i) {
      if (filter[i] != 0) {
        auto score = sigmoid(first[i], sigmoids) * object_score;
        if (confidence <= score) {
          outputs.scores.push_back(score);
          outputs.class_ids.push_back(i);
        }
      }
    }
  } else {
#if defined(USE_AVX2)
    // Use AVX2-optimized path when available
    auto [highest_score, highest_class] = find_max_filtered_avx2(
        first, filter.data(), static_cast<int>(filter.size()));
#else
    // Scalar fallback
    auto highest_score = std::numeric_limits<input_type>::min();
    auto highest_class = -1;
    for (int i = 0; i != static_cast<int>(filter.size()); ++i) {
      if (filter[i] && first[i] > highest_score) {
        highest_score = first[i];
        highest_class = i;
      }
    }
#endif
    auto score = sigmoid(highest_score, sigmoids) * object_score;
    if (confidence <= score) {
      outputs.scores.push_back(score);
      outputs.class_ids.push_back(highest_class);
    }
  }

  return outputs.scores.size() - initial_size;
}

template <typename input_type>
int
decode_scores(const input_type *data, const float *lookups,
    const std::vector<uint8_t> &filter, float confidence, bool multiclass,
    inferences &outputs, float objectness_score = 1.0F)
{
  return multiclass ? decode_scores<true>(data, lookups, filter, confidence,
                          objectness_score, outputs) :
                      decode_scores<false>(data, lookups, filter, confidence,
                          objectness_score, outputs);
}

std::vector<BboxXyxy> scale_shift_boxes(const std::vector<ax_utils::fbox> &norm_boxes,
    BboxXyxy master_box, int tensor_width, int tensor_height, bool scale_up, bool letterbox);

std::vector<BboxXywhr> scale_shift_boxes(const std::vector<ax_utils::fobox> &norm_boxes,
    BboxXyxy master_box, int tensor_width, int tensor_height, bool scale_up, bool letterbox);

std::vector<KptXyv> scale_shift_kpts(const std::vector<ax_utils::fkpt> &norm_kpts,
    BboxXyxy master_box, int tensor_width, int tensor_height, bool scale_up, bool letterbox);

std::vector<std::string> read_class_labels(const std::string &filename,
    const std::string &src, Ax::Logger &logger, bool trimmed = true);

void validate_classes(const std::vector<std::string> &class_labels,
    int num_classes, const std::string &src, Ax::Logger &logger);

std::string_view trim(std::string_view s);

template <typename T>
T *
get_meta(const std::string &meta_name,
    const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    const std::string &src = "")
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
  AxMetaBase *base = meta_itr->second.get();
  if (!base) {
    std::string error_msg = meta_name + " is nullptr";
    if (!src.empty()) {
      throw std::runtime_error(src + " : " + error_msg);
    }
    throw std::runtime_error(error_msg);
  }
  T *meta = dynamic_cast<T *>(base);
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
T *
insert_meta(std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const std::string &key, const std::string &master_key, int subframe_index,
    int number_of_subframes, Args &&...args)
{
  if (master_key.empty()) {
    auto res = map.try_emplace(key, std::make_unique<T>(std::forward<Args>(args)...));
    if (!res.second) {
      throw std::runtime_error("insert_meta : key already exists: " + key);
    }
    return dynamic_cast<T *>(res.first->second.get());
  } else {
    auto *master_meta = get_meta<AxMetaBase>(master_key, map, "insert_meta");
    if (number_of_subframes != master_meta->get_number_of_subframes()) {
      throw std::runtime_error(
          "insert_meta : number_of_subframes mismatch " + std::to_string(number_of_subframes)
          + " vs " + std::to_string(master_meta->get_number_of_subframes()));
    }
    auto submeta = std::make_shared<T>(std::forward<Args>(args)...);
    T *submeta_ptr = submeta.get();
    master_meta->insert_submeta(
        key, subframe_index, number_of_subframes, std::move(submeta));
    return submeta_ptr;
  }
}

template <typename T, typename... Args>
T *
insert_and_associate_meta(std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const std::string &key, const std::string &master_key, int subframe_index,
    int number_of_subframes, const std::string &associate_key, Args &&...args)
{
  if (associate_key.empty() || associate_key == master_key) {
    return insert_meta<T>(map, key, master_key, subframe_index,
        number_of_subframes, std::forward<Args>(args)...);
  }
  auto *associate_meta
      = get_meta<AxMetaBbox>(associate_key, map, "insert_and_associate_meta");
  if (number_of_subframes != associate_meta->get_number_of_subframes()) {
    throw std::runtime_error("insert_and_associate_meta : number_of_subframes mismatch");
  }
  int unfiltered_subframe_index = associate_meta->get_id(subframe_index);
  if (unfiltered_subframe_index == -1) {
    throw std::runtime_error("insert_and_associate_meta : id not found");
  }
  auto *master_meta = get_meta<AxMetaBase>(master_key, map, "insert_and_associate_meta");
  if (auto *tracker_meta = dynamic_cast<AxMetaTracker *>(master_meta)) {
    auto submeta = std::make_unique<T>(std::forward<Args>(args)...);
    T *submeta_ptr = submeta.get();
    auto &tracking_descriptor
        = tracker_meta->track_id_to_tracking_descriptor.at(unfiltered_subframe_index);
    tracking_descriptor.collection->set_frame_data_map(
        tracking_descriptor.frame_id, key, std::move(submeta));
    return submeta_ptr;
  }
  int unfiltered_number_of_subframes = master_meta->get_number_of_subframes();
  if (unfiltered_subframe_index >= unfiltered_number_of_subframes) {
    throw std::runtime_error("insert_and_associate_meta : subframe_index out of bounds");
  }
  auto submeta = std::make_shared<T>(std::forward<Args>(args)...);
  T *submeta_ptr = submeta.get();
  master_meta->insert_submeta(key, unfiltered_subframe_index,
      unfiltered_number_of_subframes, std::move(submeta));
  return submeta_ptr;
}

std::vector<uint8_t> build_filter(const std::vector<int> &input_filter, int num_classes);

BboxXyxy get_master_box(std::string master, std::string associated,
    const AxDataInterface &video_interface, unsigned int subframe_index,
    const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const std::string &decoder);

} // namespace ax_utils
