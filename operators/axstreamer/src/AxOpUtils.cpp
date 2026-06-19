// Copyright Axelera AI, 2023
#include "AxOpUtils.hpp"
#include "AxOpenClExtensions.hpp"
#include "AxStreamerUtils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string_view>
namespace ax_utils
{
int
parse_metric_type(const std::unordered_map<std::string, std::string> &input,
    int default_value, const std::string &error_type)
{
  auto found = input.find("metric_type");
  if (found == input.end()) {
    return default_value;
  }

  const auto &raw = found->second;
  if (raw == "euclidean_distance") {
    return EUCLIDEAN_DISTANCE;
  }
  if (raw == "squared_euclidean_distance") {
    return SQUARED_EUCLIDEAN_DISTANCE;
  }
  if (raw == "cosine_distance") {
    return COSINE_DISTANCE;
  }
  if (raw == "cosine_similarity") {
    return COSINE_SIMILARITY;
  }

  throw std::runtime_error(error_type
                           + " : metric_type must be one of euclidean_distance, "
                             "squared_euclidean_distance, cosine_distance, cosine_similarity; got '"
                           + raw + "'");
}

std::string
sizes_to_string(const std::vector<int> &sizes)
{
  return std::string("(") + Ax::Internal::join(sizes, ",") + ")";
}
bool
validate_shape(const std::vector<int> &new_shape, const std::vector<int> &original)
{
  if (new_shape.empty()) {
    return true;
  }
  auto new_size = std::accumulate(
      new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
  auto original_size
      = std::accumulate(original.begin(), original.end(), 1, std::multiplies<int>());
  return new_size == original_size;
}

transfer_info
get_transfer_info(const std::vector<int> &sizes, const std::vector<int> &padding)
{
  // strip leading ones until the padding is the right length. This is a bit of
  // a hack to avoid having an explicit reshape. we can implicity do this if
  // e.g. (0, 0, 0, 24) but shape is(1, 1, 1, 1024)
  const auto padding_ndims = padding.size() / 2;
  if (sizes.size() < padding_ndims
      || !std::all_of(sizes.cbegin(), std::prev(sizes.cend(), padding_ndims),
          [](int x) { return x == 1; })) {
    throw std::runtime_error(
        "transform_padding: " + ax_utils::sizes_to_string(padding) + " too short for input shape "
        + ax_utils::sizes_to_string(sizes) + " after any implicit squeezing");
  }

  const auto first_size = std::prev(sizes.cend(), padding_ndims);
  const auto crop
      = std::any_of(padding.begin(), padding.end(), [](int x) { return x < 0; });
  const auto grow
      = std::any_of(padding.begin(), padding.end(), [](int x) { return x > 0; });
  if (crop && grow) {
    throw std::runtime_error("transform_padding: can remove or add padding, but not both in different dimensions:"
                             + ax_utils::sizes_to_string(padding));
  }

  transfer_info info;
  info.is_crop = crop;
  info.in_sizes.assign(first_size, sizes.end());
  info.out_sizes.reserve(padding_ndims);
  info.ranges.reserve(padding_ndims);
  for (size_t i = 0; i < padding_ndims; ++i) {
    info.out_sizes.push_back(padding[2 * i] + info.in_sizes[i] + padding[2 * i + 1]);
    info.ranges.emplace_back(std::abs(padding[2 * i]),
        std::abs(padding[2 * i]) + std::min(info.in_sizes[i], info.out_sizes[i]));
    if (info.out_sizes.back() <= 0) {
      throw std::runtime_error(
          "transform_padding: negative padding " + ax_utils::sizes_to_string(padding)
          + " greater than tensor size in dimension " + std::to_string(i)
          + " with input tensor " + ax_utils::sizes_to_string(sizes));
    }
  }
  return info;
}

float
dequantize(int value, float scale, int32_t zero_point)
{
  return scale * (value - zero_point);
}

float
to_sigmoid(float value)
{
  return 1.0f / (1.0f + std::exp(-value));
}

std::vector<int>
indices_for_topk(const std::vector<float> &scores, int topk)
{
  std::vector<int> indices(scores.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  if (topk >= static_cast<int>(indices.size())) {
    std::sort(std::begin(indices), std::end(indices),
        [&scores](int i, int j) { return scores[i] > scores[j]; });
    return indices;
  }
  std::partial_sort(std::begin(indices), std::next(std::begin(indices), topk),
      std::end(indices),
      [&scores](int i, int j) { return scores[i] > scores[j]; });
  indices.resize(topk);
  return indices;
}

std::vector<int>
indices_for_topk_area(const std::vector<box_xyxy> &boxes, int topk)
{
  std::vector<int> indices(boxes.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  std::vector<int> area;
  area.reserve(boxes.size());
  for (const auto &box : boxes) {
    area.push_back((box.x2 - box.x1) * (box.y2 - box.y1));
  }
  if (topk >= static_cast<int>(indices.size())) {
    std::sort(std::begin(indices), std::end(indices),
        [&area](int i, int j) { return area[i] > area[j]; });
    return indices;
  }
  std::partial_sort(std::begin(indices), std::next(std::begin(indices), topk),
      std::end(indices), [&area](int i, int j) { return area[i] > area[j]; });
  indices.resize(topk);
  return indices;
}

std::vector<int>
indices_for_topk_center(const std::vector<box_xyxy> &boxes, int topk, int width, int height)
{
  std::vector<int> indices(boxes.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  std::vector<int> sq_dist_from_center;
  for (const auto &box : boxes) {
    int dist_x = (box.x1 + box.x2 - width) / 2;
    int dist_y = (box.y1 + box.y2 - height) / 2;
    sq_dist_from_center.push_back(dist_x * dist_x + dist_y * dist_y);
  }
  if (topk >= static_cast<int>(indices.size())) {
    std::sort(std::begin(indices), std::end(indices), [&sq_dist_from_center](int i, int j) {
      return sq_dist_from_center[i] < sq_dist_from_center[j];
    });
    return indices;
  }

  std::partial_sort(std::begin(indices), std::next(std::begin(indices), topk),
      std::end(indices), [&sq_dist_from_center](int i, int j) {
        return sq_dist_from_center[i] < sq_dist_from_center[j];
      });
  indices.resize(topk);
  return indices;
}

inferences
topk(inferences predictions, int topk)
{
  std::vector<int> indices = indices_for_topk(predictions.scores, topk);
  inferences result(topk);
  for (auto idx : indices) {
    result.scores.push_back(std::move(predictions.scores[idx]));
    result.class_ids.push_back(std::move(predictions.class_ids[idx]));

    if (!predictions.obb.empty()) {
      result.obb.push_back(std::move(predictions.obb[idx]));
    } else {
      if (!predictions.boxes.empty())
        result.boxes.push_back(std::move(predictions.boxes[idx]));

      if (!predictions.seg_info.empty())
        result.seg_info.push_back(std::move(predictions.seg_info[idx]));
      if (!predictions.segments.empty())
        result.segments.push_back(std::move(predictions.segments[idx]));

      if (!predictions.kpts_shape.empty() && predictions.kpts_shape[0] != 0) {
        auto start
            = std::next(predictions.kpts.begin(), idx * predictions.kpts_shape[0]);
        auto end = std::next(start, predictions.kpts_shape[0]);
        result.kpts.insert(result.kpts.end(), start, end);
      }
    }
  }
  result.prototype = std::move(predictions.prototype);
  result.kpts_shape = std::move(predictions.kpts_shape);
  return result;
}

std::vector<lookups>
build_dequantization_tables(
    const std::vector<float> &zero_points, const std::vector<float> &scales)
{
  return build_general_dequantization_tables(zero_points, scales);
}

//
//  Given that we only have 256 possible values, we can precompute
//  the dequantized values and apply the sigmoid/exp function for all possible
//  values and store them in a table.  This will allow us to avoid the expensive
//  exp() call at runtime.
//
std::vector<lookups>
build_sigmoid_tables(const std::vector<float> &zero_points, const std::vector<float> &scales)
{
  return build_general_dequantization_tables(zero_points, scales, ax_utils::to_sigmoid);
}

//
//  The usage for this table is dequantize and calculate exp(x) for all possible
//  values and store them in a table.  This will allow us to avoid the expensive
//  exp() call at runtime.
//  It is used in softmax where we have to calculate exp(x) where x is y - m, where
//  y is the output of the last layer and m is the maximum value of y. This means
//  that all values will be in the range of 0, -255.
//  Because all of the values we will lookup will be the difference of two values
//  we can ignore the zero point as it will cancel out.
//
std::vector<lookups>
build_exponential_tables(
    const std::vector<float> & /*zero_points*/, const std::vector<float> &scales)
{
  return build_general_dequantization_tables(
      {}, scales, [](float x) { return std::exp(x); });
}

std::vector<lookups>
build_exponential_tables_with_zero_point(
    const std::vector<float> &zero_points, const std::vector<float> &scales)
{
  return build_general_dequantization_tables(
      zero_points, scales, [](float x) { return std::exp(x); });
}

std::vector<sin_cos_lookups>
build_general_trigonometric_tables(const std::vector<float> &zero_points,
    const std::vector<float> &scales, auto &&f1, auto &&f2)
{
  std::vector<sin_cos_lookups> trig_tables;
  for (size_t i = 0; i < scales.size(); ++i) {
    sin_cos_lookups trig_table;
    // The zero points do not matter if we have ratios
    // In this case for numerical stability the largest value is the best
    float zero_point = 127.0;
    if (!zero_points.empty()) {
      zero_point = zero_points.at(i);
    }
    for (int j = 0; j < trig_table.size(); ++j) {
      // Assuming signed int8, i.e. the range of the table is from -128 to 127
      if (j < 256) {
        trig_table[j] = f1(ax_utils::dequantize(j - 128, scales[i], zero_point));
      } else {
        trig_table[j] = f2(ax_utils::dequantize(j - 384, scales[i], zero_point));
      }
    }

    trig_tables.push_back(trig_table);
  }
  return trig_tables;
}

std::vector<sin_cos_lookups>
build_trigonometric_tables(
    const std::vector<float> &zero_points, const std::vector<float> &scales)
{
  auto sin_cos_tables = build_general_trigonometric_tables(
      zero_points, scales, [](float x) { return std::sin(x); },
      [](float x) { return std::cos(x); });

  return sin_cos_tables;
}

std::vector<sin_cos_lookups>
build_sigmoid_trigonometric_tables(const std::vector<float> &zero_points,
    const std::vector<float> &scales, float add, float mul)
{
  auto sin_cos_tables = build_general_trigonometric_tables(
      zero_points, scales,
      [add, mul](
          float x) { return std::sin((ax_utils::to_sigmoid(x) + add) * mul); },
      [add, mul](
          float x) { return std::cos((ax_utils::to_sigmoid(x) + add) * mul); });

  return sin_cos_tables;
}

tensor_dims
get_dims(const AxTensorsInterface &tensors, int level, bool transpose)
{
  if (level == -1) {
    return { 0, 0, 0 };
  }
  auto width = tensors[level].sizes[2];
  auto height = tensors[level].sizes[1];
  auto depth = tensors[level].sizes[3];
  if (!transpose) {
    //  NCHW format
    width = tensors[level].sizes[3];
    height = tensors[level].sizes[2];
    depth = tensors[level].sizes[1];
  }
  return { width, height, depth };
}

float
exponential(int value, const float *lookups)
{
  int index = value + 255;
  return lookups[index];
}

void
softmax(const int8_t *input, int num_elems, const float *lookups, float *output)
{
  auto first = input;
  auto last = input + num_elems;
  auto largest = *std::max_element(first, last);
  auto total = 0.0F;

  for (auto out = output; first != last; ++first, ++out) {
    auto val = exponential(*first - largest, lookups);
    total += val;
    *out = val;
  }
  auto recip_sum = 1.0F / total;
  std::transform(output, std::next(output, num_elems), output,
      [recip_sum](float val) { return val * recip_sum; });
}

struct scale_details {
  float x_scale;
  float y_scale;
  float x_adjust;
  float y_adjust;
};

scale_details
determine_scale(int video_width, int video_height, int tensor_width,
    int tensor_height, bool scale_up, bool letterbox)
{
  if (!letterbox) {
    auto x_scale = static_cast<float>(video_width);
    auto y_scale = static_cast<float>(video_height);
    return { x_scale, y_scale, 0.0F, 0.0F };
  }
  bool scale_to_height = static_cast<double>(tensor_width) / tensor_height
                         > static_cast<double>(video_width) / video_height;

  auto adjusted_width = scale_to_height ? video_height * tensor_width
                                              / static_cast<float>(tensor_height) :
                                          video_width;
  auto adjusted_height = scale_to_height ? video_height :
                                           video_width * tensor_height
                                               / static_cast<float>(tensor_width);

  auto x_adjust = (adjusted_width - video_width) / 2.0F;
  auto y_adjust = (adjusted_height - video_height) / 2.0F;

  auto scale_factor = static_cast<float>(std::max(adjusted_height, adjusted_width));
  if (!scale_up && video_width < tensor_width && video_height < tensor_height) {
    x_adjust = (tensor_width - video_width) / 2;
    y_adjust = (tensor_height - video_height) / 2;
    scale_factor = std::max(tensor_width, tensor_height);
  }
  return { scale_factor, scale_factor, x_adjust, y_adjust };
}

struct scale_to_original {
  std::function<int(float)> to_orig_x;
  std::function<int(float)> to_orig_y;
  float scale_x;
  float scale_y;
};
scale_to_original
scale(int video_width, int video_height, int tensor_width, int tensor_height,
    bool scale_up, bool letterbox)
{
  auto [x_scale, y_scale, x_adjust, y_adjust] = determine_scale(video_width,
      video_height, tensor_width, tensor_height, scale_up, letterbox);

  auto to_orig_x = [x_scale, x_adjust, max_width = video_width - 1](float x) {
    const int adjusted = x * x_scale - x_adjust + 0.5F;
    return std::clamp(adjusted, 0, max_width);
  };

  auto to_orig_y = [y_scale, y_adjust, max_height = video_height - 1](float y) {
    const int adjusted = y * y_scale - y_adjust + 0.5F;
    return std::clamp(adjusted, 0, max_height);
  };

  return { to_orig_x, to_orig_y, x_scale, y_scale };
}

std::vector<KptXyv>
scale_shift_kpts(const std::vector<ax_utils::fkpt> &norm_kpts, BboxXyxy master_box,
    int tensor_width, int tensor_height, bool scale_up, bool letterbox)
{
  auto box_width = 1 + master_box.x2 - master_box.x1;
  auto box_height = 1 + master_box.y2 - master_box.y1;
  const auto [to_orig_x, to_orig_y, unused_scale_x, unused_scale_y]
      = scale(box_width, box_height, tensor_width, tensor_height, scale_up, letterbox);
  (void) unused_scale_x;
  (void) unused_scale_y;
  std::vector<KptXyv> keypoints;
  keypoints.reserve(norm_kpts.size());

  for (const auto &k : norm_kpts) {
    KptXyv kpt{
      to_orig_x(k.x) + master_box.x1,
      to_orig_y(k.y) + master_box.y1,
      k.visibility,
    };
    keypoints.push_back(kpt);
  }
  return keypoints;
}

std::vector<BboxXyxy>
scale_shift_boxes(const std::vector<ax_utils::fbox> &norm_boxes, BboxXyxy master_box,
    int tensor_width, int tensor_height, bool scale_up, bool letterbox)
{
  auto box_width = 1 + master_box.x2 - master_box.x1;
  auto box_height = 1 + master_box.y2 - master_box.y1;

  const auto [to_orig_x, to_orig_y, unused_scale_x, unused_scale_y]
      = scale(box_width, box_height, tensor_width, tensor_height, scale_up, letterbox);
  (void) unused_scale_x;
  (void) unused_scale_y;
  std::vector<BboxXyxy> boxes;
  boxes.reserve(norm_boxes.size());

  for (const auto &b : norm_boxes) {
    BboxXyxy box{
      to_orig_x(b.x1) + master_box.x1,
      to_orig_y(b.y1) + master_box.y1,
      to_orig_x(b.x2) + master_box.x1,
      to_orig_y(b.y2) + master_box.y1,
    };
    boxes.push_back(box);
  }
  return boxes;
}

std::vector<BboxXywhr>
scale_shift_boxes(const std::vector<ax_utils::fobox> &norm_boxes, BboxXyxy master_box,
    int tensor_width, int tensor_height, bool scale_up, bool letterbox)
{
  auto box_width = 1 + master_box.x2 - master_box.x1;
  auto box_height = 1 + master_box.y2 - master_box.y1;

  const auto [to_orig_x, to_orig_y, scale_x, scale_y]
      = scale(box_width, box_height, tensor_width, tensor_height, scale_up, letterbox);
  std::vector<BboxXywhr> boxes;
  boxes.reserve(norm_boxes.size());

  for (const auto &obb : norm_boxes) {
    BboxXywhr transformed_box{
      to_orig_x(obb.x) + master_box.x1,
      to_orig_y(obb.y) + master_box.y1,
      static_cast<int>(scale_x * obb.w),
      static_cast<int>(scale_y * obb.h),
      obb.angle,
    };
    boxes.push_back(transformed_box);
  }
  return boxes;
}

std::string_view
trim(std::string_view s)
{
  auto first
      = std::find_if(s.begin(), s.end(), [](char c) { return !std::isspace(c); });
  auto last = std::find_if(s.rbegin(), std::make_reverse_iterator(first), [](char c) {
    return !std::isspace(c);
  }).base();
  return first != last ? std::string_view(&*first, std::distance(first, last)) :
                         std::string_view();
}

std::vector<std::string>
read_class_labels(const std::string &filename, const std::string &src,
    Ax::Logger &logger, bool trimmed)
{
  std::vector<std::string> class_labels;
  auto file = std::ifstream(filename);
  if (!file) {
    std::stringstream ss;
    logger(AX_ERROR) << src << " : classlabels_file '" << filename << "' not found";
    return {};
  }
  std::string s;
  int line_num = 0;
  bool bom_checked = false;
  while (getline(file, s)) {
    line_num++;

    // Check for and remove UTF-8 BOM only on the very first line read
    if (!bom_checked && line_num == 1 && s.length() >= 3
        && static_cast<unsigned char>(s[0]) == 0xEF
        && static_cast<unsigned char>(s[1]) == 0xBB
        && static_cast<unsigned char>(s[2]) == 0xBF) {
      logger(AX_DEBUG) << src << " : Detected and removed UTF-8 BOM from " << filename;
      s = s.substr(3);
    }
    bom_checked = true; // Don't check again

    if (trimmed == true) {
      s = trim(s);
      // Add only if not empty after trimming
      if (!s.empty()) {
        class_labels.push_back(s);
      }
    } else {
      // Add the line as is (potentially empty, potentially after BOM removal)
      class_labels.push_back(s);
    }
  }

  // Log only the final count, not every line
  logger(AX_DEBUG) << src << " : Finished reading file " << filename
                   << ". Total labels added: " << class_labels.size();

  return class_labels;
}


void
validate_classes(const std::vector<std::string> &class_labels, int num_classes,
    const std::string &src, Ax::Logger &logger)
{
  if (num_classes == 0 && class_labels.empty()) {
    throw std::runtime_error(src + " : you must either provide classes or classlabels_file");
  }
  if (num_classes != 0 && !class_labels.empty() && num_classes != class_labels.size()) {

    std::stringstream ss;
    logger(AX_WARN) << src << " : number of classes (" << num_classes
                    << ") and classlabels_file size (" << class_labels.size()
                    << ") differ. Assuming num_classes is correct." << std::endl;
  }
}

buffer_details
get_buffer_details(const AxTensorInterface &input)
{
  buffer_details details;
  details.width = input.sizes[2];
  //  If we have a batch, then treat as a single tensor of height * batch
  details.height = input.sizes[1] * input.sizes[0];
  details.channels = input.sizes[3];
  details.stride = details.width * details.channels;
  if (input.fd != -1) {
    details.data = input.fd;
  } else if (input.ocl_buffer) {
    details.data = input.ocl_buffer;
  } else {
    details.data = input.data;
  }
  details.offsets = { 0 };
  details.strides = { static_cast<size_t>(details.stride) };
  details.format = AxVideoFormat::UNDEFINED;
  details.crop_x = 0;
  details.crop_y = 0;
  details.actual_height = details.height;
  return details;
}

std::vector<buffer_details>
get_buffer_details(const AxTensorsInterface &input)
{
  std::vector<buffer_details> details(input.size());
  std::transform(input.begin(), input.end(), details.begin(),
      [](const auto &t) { return get_buffer_details(t); });

  return details;
}

std::vector<buffer_details>
get_buffer_details(const AxVideoInterface &input)
{
  buffer_details details;
  details.width = input.info.width;
  details.height = input.info.height;
  details.channels = AxVideoFormatNumChannels(input.info.format);
  details.stride = input.info.stride;
  if (input.fd != -1) {
    details.data = input.fd;
  } else if (input.vaapi && input.vaapi->type == VASurfaceID_proxy::opencl) {
    details.data = static_cast<opencl_planes *>(input.vaapi);
  } else if (input.vaapi) {
    details.data = input.vaapi;
  } else if (input.ocl_buffer) {
    details.data = input.ocl_buffer;
  } else {
    details.data = input.data;
  }
  details.strides = input.strides;
  details.offsets = input.offsets;
  details.format = input.info.format;
  if (details.strides.empty()) {
    details.strides = { static_cast<size_t>(details.stride) };
  }
  if (details.offsets.empty()) {
    details.offsets = { 0 };
  }
  details.crop_x = input.info.cropped ? input.info.x_offset : 0;
  details.crop_y = input.info.cropped ? input.info.y_offset : 0;
  // If the actual height is not set, use the height
  auto actual_height = input.info.actual_height != 0 ? input.info.actual_height :
                                                       input.info.height;
  //  If we have a cropped image use the actual height not the cropped height
  details.actual_height = input.info.cropped ? actual_height : input.info.height;
  return { details };
}

std::vector<buffer_details>
get_buffer_details(const std::monostate &input)
{
  return {};
}

std::vector<buffer_details>
extract_buffer_details(const AxDataInterface &input)
{
  return std::visit([](auto &&arg) { return get_buffer_details(arg); }, input);
}

size_t
determine_height(const buffer_details &info, int which_channel)
{
  switch (info.format) {
    case AxVideoFormat::BGRA:
    case AxVideoFormat::RGBA:
    case AxVideoFormat::RGB:
    case AxVideoFormat::BGR:
    case AxVideoFormat::GRAY8:
      return info.actual_height;
    case AxVideoFormat::YUY2:
      return info.actual_height;
    case AxVideoFormat::I420:
    case AxVideoFormat::NV12:
      return which_channel == 0 ? info.actual_height : info.actual_height / 2;
    case AxVideoFormat::NV16:
      return info.actual_height; // Both Y and UV planes have full height
    case AxVideoFormat::UNDEFINED:
      if (which_channel == 0) {
        return info.actual_height;
      } else {
        //  TODO: handle multiple channels
        throw std::runtime_error("Tensors should not have multiple channels");
      }
    default:
      throw std::runtime_error(
          "AxOpUtils unsupported format: " + AxVideoFormatToString(info.format));
  }
}

int
determine_size(const buffer_details &info, int which_channel)
{
  auto height = determine_height(info, which_channel);
  auto stride = info.strides.empty() ? info.stride : info.strides[which_channel];
  return stride * height;
}

int
determine_buffer_size(const buffer_details &info)
{
  auto last_channel = info.strides.size() - 1;
  return info.offsets[last_channel] + determine_size(info, last_channel);
}

void
remove_cropinfo(AxDataInterface &out)
{
  if (auto *video = std::get_if<AxVideoInterface>(&out)) {
    video->info.stride
        = video->info.width * AxVideoFormatNumChannels(video->info.format);
    video->strides = { size_t(video->info.stride) };
    video->info.cropped = false;
    video->info.x_offset = 0;
    video->info.y_offset = 0;
    video->info.actual_height = video->info.height;
  }
}


namespace
{
constexpr float kEmbeddingNormEpsilon = 1e-6f;
} // namespace

std::vector<float>
embeddings_cosine_similarity(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise)
{
  Eigen::VectorXf desc_vec = Eigen::Map<const Eigen::VectorXf>(desc.data(), desc.size());
  Eigen::VectorXf similarity_vec = embeddings * desc_vec;

  if (normalise) {
    const float desc_norm = desc_vec.norm();
    if (desc_norm > kEmbeddingNormEpsilon) {
      similarity_vec /= desc_norm;
    } else {
      similarity_vec.setZero();
    }

    const Eigen::VectorXf row_norms = embeddings.rowwise().norm();
    for (int i = 0; i < similarity_vec.size(); ++i) {
      const float row_norm = row_norms(i);
      if (row_norm > kEmbeddingNormEpsilon) {
        similarity_vec(i) /= row_norm;
      } else {
        similarity_vec(i) = 0.0f;
      }
    }
  }

  return std::vector<float>(
      similarity_vec.data(), similarity_vec.data() + similarity_vec.size());
}

std::vector<float>
embeddings_euclidean_distance(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise)
{
  Eigen::VectorXf desc_vec = Eigen::Map<const Eigen::VectorXf>(desc.data(), desc.size());
  if (normalise) {
    Eigen::VectorXf desc_normed = desc_vec;
    const float desc_norm = desc_normed.norm();
    if (desc_norm > kEmbeddingNormEpsilon) {
      desc_normed /= desc_norm;
    } else {
      desc_normed.setZero();
    }

    const float desc_normed_norm = desc_normed.norm();
    const Eigen::VectorXf row_norms = embeddings.rowwise().norm();
    Eigen::VectorXf distances(embeddings.rows());
    for (int i = 0; i < embeddings.rows(); ++i) {
      const float row_norm = row_norms(i);
      if (row_norm > kEmbeddingNormEpsilon) {
        distances(i) = (embeddings.row(i) / row_norm - desc_normed.transpose()).norm();
      } else {
        distances(i) = desc_normed_norm;
      }
    }
    return std::vector<float>(distances.data(), distances.data() + distances.size());
  }

  Eigen::VectorXf distances
      = (embeddings.rowwise() - desc_vec.transpose()).rowwise().norm();

  return std::vector<float>(distances.data(), distances.data() + distances.size());
}

std::vector<float>
embeddings_squared_euclidean_distance(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise)
{
  Eigen::VectorXf desc_vec = Eigen::Map<const Eigen::VectorXf>(desc.data(), desc.size());
  if (normalise) {
    Eigen::VectorXf desc_normed = desc_vec;
    const float desc_norm = desc_normed.norm();
    if (desc_norm > kEmbeddingNormEpsilon) {
      desc_normed /= desc_norm;
    } else {
      desc_normed.setZero();
    }

    const float desc_normed_sqnorm = desc_normed.squaredNorm();
    const Eigen::VectorXf row_norms = embeddings.rowwise().norm();
    Eigen::VectorXf distances(embeddings.rows());
    for (int i = 0; i < embeddings.rows(); ++i) {
      const float row_norm = row_norms(i);
      if (row_norm > kEmbeddingNormEpsilon) {
        distances(i) = (embeddings.row(i) / row_norm - desc_normed.transpose()).squaredNorm();
      } else {
        distances(i) = desc_normed_sqnorm;
      }
    }
    return std::vector<float>(distances.data(), distances.data() + distances.size());
  }

  Eigen::VectorXf distances
      = (embeddings.rowwise() - desc_vec.transpose()).rowwise().squaredNorm();

  return std::vector<float>(distances.data(), distances.data() + distances.size());
}

std::vector<float>
embeddings_cosine_distance(const std::vector<float> &desc,
    const Eigen::MatrixXf &embeddings, bool normalise)
{
  std::vector<float> distances;
  auto similarities = embeddings_cosine_similarity(desc, embeddings, normalise);
  for (const auto &sim : similarities) {
    float clamped_sim = std::clamp(sim, -1.0f, 1.0f);
    if (!std::isfinite(clamped_sim)) {
      clamped_sim = 0.0f;
    }
    distances.push_back(std::acos(clamped_sim) / static_cast<float>(M_PI));
  }
  return distances;
}

void
add_vec_to_matrix(const std::vector<float> &vec, Eigen::MatrixXf &matrix)
{
  // Check if this is the first row and set the matrix columns accordingly
  if (matrix.rows() == 0) {
    matrix.resize(0, vec.size());
  }

  // Check if the vector size matches the matrix column size
  if (vec.size() != matrix.cols()) {
    throw std::runtime_error("Invalid format of embeddings file");
  }

  // Add a new row
  int new_rows = matrix.rows() + 1;
  matrix.conservativeResize(new_rows, Eigen::NoChange);

  // Copy the vector data to the new row
  for (int i = 0; i < vec.size(); ++i) {
    matrix(new_rows - 1, i) = vec[i];
  }
}

std::pair<Eigen::MatrixXf, std::vector<std::string>>
read_embedding_json(const std::string &filename, bool normalise, Ax::Logger &logger)
{
  std::ifstream file(filename);

  if (!file.is_open()) {
    logger(AX_ERROR) << "Failed to open '" << filename << "'";
    return {};
  }
  nlohmann::ordered_json json_data;
  file >> json_data;

  std::vector<std::vector<float>> embeddings;
  Eigen::MatrixXf matrix;

  std::vector<std::string> labels;
  for (auto &[key, value] : json_data.items()) {
    labels.push_back(key); // The key itself is the label

    if (false == normalise) {
      embeddings.push_back(value.get<std::vector<float>>());
    } else {
      std::vector<float> desc = value.get<std::vector<float>>();
      auto len2 = std::accumulate(desc.begin(), desc.end(), 0.0,
          [](float sum, float val) { return sum + std::pow(val, 2); });
      std::transform(desc.begin(), desc.end(), desc.begin(),
          [length = std::sqrt(len2)](float val) { return val / length; });
      embeddings.push_back(std::move(desc));
    }
  }

  for (const auto &vec : embeddings) {
    add_vec_to_matrix(vec, matrix);
  }

  // TODO: Sanity check of vector sizes eg invalid file
  return { matrix, labels };
}

void
write_embedding_json(const Eigen::MatrixXf &embeddings,
    const std::vector<std::string> &labels, const std::string &filename, Ax::Logger &logger)
{
  if (embeddings.rows() != labels.size()) {
    logger(AX_ERROR) << "Number of embeddings (" << embeddings.rows()
                     << ") does not match number of labels (" << labels.size() << ")";
    return;
  }

  nlohmann::ordered_json json_data;
  for (int i = 0; i < embeddings.rows(); ++i) {
    std::vector<float> vec(embeddings.cols());
    for (int j = 0; j < vec.size(); ++j) {
      vec[j] = embeddings(i, j);
    }
    json_data[labels[i]] = vec;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    logger(AX_ERROR) << "Failed to open '" << filename << "'";
    return;
  }
  file << json_data.dump(2); // Pretty print with indentation of 2 spaces
}


size_t
get_bytes_per_pixel(AxVideoFormat format)
{
  switch (format) {
    case AxVideoFormat::BGRA:
    case AxVideoFormat::RGBA:
      return 4;
    case AxVideoFormat::RGB:
    case AxVideoFormat::BGR:
      return 3;
    case AxVideoFormat::YUY2:
      return 2;
    case AxVideoFormat::I420:
    case AxVideoFormat::NV12:
    case AxVideoFormat::NV16:
    case AxVideoFormat::GRAY8:
      return 1; // Y channel
    case AxVideoFormat::UNDEFINED:
      return 0; // Undefined format
    default:
      throw std::runtime_error("Unsupported video format: " + AxVideoFormatToString(format));
  }
}

std::vector<uint8_t>
build_filter(const std::vector<int> &input_filter, int num_classes)
{
  if (input_filter.empty()) {
    //  If the input filter is empty we allow all classes
    return std::vector<uint8_t>(num_classes, 0xFF);
  }
  std::vector<uint8_t> filter(num_classes, 0x00);
  for (auto i : input_filter) {
    if (0 <= i && i < num_classes) {
      filter[i] = 0xFF;
    } else {
      throw std::runtime_error("Filter index " + std::to_string(i) + " out of range for number of classes "
                               + std::to_string(num_classes));
    }
  }
  return filter;
}

BboxXyxy
get_master_box(std::string master, std::string associated,
    const AxDataInterface &video_interface, unsigned int subframe_index,
    const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const std::string &decoder)
{
  if (master.empty()) {
    auto vinfo = std::get<AxVideoInterface>(video_interface);
    return { 0, 0, vinfo.info.width - 1, vinfo.info.height - 1 };
  }
  const auto &box_key = associated.empty() ? master : associated;
  auto master_meta = ax_utils::get_meta<AxMetaBbox>(box_key, map, decoder);
  return master_meta->get_box_xyxy(subframe_index);
}

} // namespace ax_utils
