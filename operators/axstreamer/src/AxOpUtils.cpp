// Copyright Axelera AI, 2025
#include "AxOpUtils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

#include <iostream>
#include <string_view>
namespace ax_utils
{
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
    return indices;
  }
  //  Note: We use nth_element here rather than partial_sort because we do not
  //  care about order of the elements that are in the topk and K will typically
  //  be quite large and generally, nth_element is the more efficient in this
  //  case. When K is small, partial_sort is usually more efficient.
  std::nth_element(std::begin(indices), std::next(std::begin(indices), topk),
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
  if (topk >= static_cast<int>(indices.size())) {
    return indices;
  }

  std::vector<int> area;
  for (const auto &box : boxes) {
    area.push_back((box.x2 - box.x1) * (box.y2 - box.y1));
  }
  std::nth_element(std::begin(indices), std::next(std::begin(indices), topk),
      std::end(indices), [&area](int i, int j) { return area[i] > area[j]; });
  indices.resize(topk);
  return indices;
}

std::vector<int>
indices_for_topk_center(const std::vector<box_xyxy> &boxes, int topk, int width, int height)
{
  std::vector<int> indices(boxes.size());
  std::iota(std::begin(indices), std::end(indices), 0);
  if (topk >= static_cast<int>(indices.size())) {
    return indices;
  }

  std::vector<int> sq_dist_from_center;
  for (const auto &box : boxes) {
    int dist_x = (box.x1 + box.x2 - width) / 2;
    int dist_y = (box.y1 + box.y2 - height) / 2;
    sq_dist_from_center.push_back(dist_x * dist_x + dist_y * dist_y);
  }

  std::nth_element(std::begin(indices), std::next(std::begin(indices), topk),
      std::end(indices), [&sq_dist_from_center](int i, int j) {
        return sq_dist_from_center[i] < sq_dist_from_center[j];
      });
  indices.resize(topk);
  return indices;
}

inferences
topk(const inferences &predictions, int topk)
{
  if (predictions.boxes.size() <= topk) {
    return predictions;
  }
  std::vector<int> indices = indices_for_topk(predictions.scores, topk);
  inferences result(topk);
  result.set_prototype_dims(predictions.prototype_width,
      predictions.prototype_height, predictions.prototype_depth);
  result.prototype_coefs = std::move(predictions.prototype_coefs);
  for (auto idx : indices) {
    result.boxes.push_back(predictions.boxes[idx]);
    result.scores.push_back(predictions.scores[idx]);
    result.class_ids.push_back(predictions.class_ids[idx]);
    if (!predictions.seg_funcs.empty())
      result.seg_funcs.emplace_back(std::move(predictions.seg_funcs[idx]));
    if (!predictions.segments.empty())
      result.segments.emplace_back(std::move(predictions.segments[idx]));
    if (predictions.kpts_shape.empty()) {
      continue;
    }
    if (0 != predictions.kpts_shape[0]) {
      auto start = std::next(predictions.kpts.begin(), idx * predictions.kpts_shape[0]);
      auto end = std::next(start, predictions.kpts_shape[0]);
      result.kpts.insert(result.kpts.end(), start, end);
    }
  }
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
softmax(const int8_t *input, int num_elems, size_t stride, const float *lookups, float *output)
{
  auto first = make_stride_iterator(input, 0, stride);
  auto last = make_stride_iterator(input, num_elems, stride);
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

  int adjusted_width
      = scale_to_height ? video_height * tensor_width / tensor_height : video_width;
  int adjusted_height
      = scale_to_height ? video_height : video_width * tensor_height / tensor_width;

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

  return { to_orig_x, to_orig_y };
}

std::vector<KptXyv>
scale_kpts(const std::vector<ax_utils::fkpt> &norm_kpts, int video_width,
    int video_height, int tensor_width, int tensor_height, bool scale_up, bool letterbox)
{
  const auto [to_orig_x, to_orig_y] = scale(video_width, video_height,
      tensor_width, tensor_height, scale_up, letterbox);
  std::vector<KptXyv> keypoints;
  keypoints.reserve(norm_kpts.size());

  for (const auto &k : norm_kpts) {
    KptXyv kpt{
      to_orig_x(k.x),
      to_orig_y(k.y),
      k.visibility,
    };
    keypoints.push_back(kpt);
  }
  return keypoints;
}

std::vector<KptXyv>
scale_shift_kpts(const std::vector<ax_utils::fkpt> &norm_kpts, BboxXyxy master_box,
    int tensor_width, int tensor_height, bool scale_up, bool letterbox)
{
  auto box_width = master_box.x2 - master_box.x1;
  auto box_height = master_box.y2 - master_box.y1;
  auto kpts = scale_kpts(norm_kpts, box_width, box_height, tensor_width,
      tensor_height, scale_up, letterbox);
  for (auto &kpt : kpts) {
    kpt.x += master_box.x1;
    kpt.y += master_box.y1;
  }
  return kpts;
}

std::vector<BboxXyxy>
scale_boxes(const std::vector<ax_utils::fbox> &norm_boxes, int video_width,
    int video_height, int tensor_width, int tensor_height, bool scale_up, bool letterbox)
{
  const auto [to_orig_x, to_orig_y] = scale(video_width, video_height,
      tensor_width, tensor_height, scale_up, letterbox);
  std::vector<BboxXyxy> boxes;
  boxes.reserve(norm_boxes.size());

  for (const auto &b : norm_boxes) {
    BboxXyxy box{
      to_orig_x(b.x1),
      to_orig_y(b.y1),
      to_orig_x(b.x2),
      to_orig_y(b.y2),
    };
    boxes.push_back(box);
  }
  return boxes;
}

std::vector<BboxXyxy>
scale_boxes(const std::vector<ax_utils::fbox> &norm_boxes, const AxVideoInterface &vinfo,
    int model_width, int model_height, bool scale_up, bool letterbox)
{
  return scale_boxes(norm_boxes, vinfo.info.width, vinfo.info.height,
      model_width, model_height, scale_up, letterbox);
}

std::vector<BboxXyxy>
scale_shift_boxes(const std::vector<ax_utils::fbox> &norm_boxes, BboxXyxy master_box,
    int tensor_width, int tensor_height, bool scale_up, bool letterbox)
{
  auto box_width = master_box.x2 - master_box.x1;
  auto box_height = master_box.y2 - master_box.y1;
  auto boxes = scale_boxes(norm_boxes, box_width, box_height, tensor_width,
      tensor_height, scale_up, letterbox);
  for (auto &box : boxes) {
    box.x1 += master_box.x1;
    box.y1 += master_box.y1;
    box.x2 += master_box.x1;
    box.y2 += master_box.y1;
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
read_class_labels(const std::string &filename, const std::string &src, Ax::Logger &logger)
{
  std::vector<std::string> class_labels;
  auto file = std::ifstream(filename);
  if (!file) {
    std::stringstream ss;
    logger(AX_ERROR) << src << " : classlabels_file '" << filename << "' not found";
    return {};
  }
  std::string s;
  while (getline(file, s)) {
    s = trim(s);
    if (!s.empty()) {
      class_labels.push_back(s);
    }
  }

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
  details.data = input.data;
  details.offsets = { 0 };
  details.strides = { static_cast<size_t>(details.stride) };
  details.format = AxVideoFormat::UNDEFINED;
  details.crop_x = 0;
  details.crop_y = 0;
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
  details.crop_x = input.info.x_offset;
  details.crop_y = input.info.y_offset;
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
      return info.height;
    case AxVideoFormat::YUY2:
      return info.height;
    case AxVideoFormat::I420:
    case AxVideoFormat::NV12:
      return which_channel == 0 ? info.height : info.height / 2;
    case AxVideoFormat::UNDEFINED:
      if (which_channel == 0) {
        return info.height;
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
  auto height = determine_height(info, which_channel) + info.crop_y;
  auto stride = info.strides.empty() ? info.stride : info.strides[which_channel];
  return stride * height;
}

int
determine_buffer_size(const buffer_details &info)
{
  auto last_channel = info.strides.size() - 1;
  return info.offsets[last_channel] + determine_size(info, last_channel);
}


} // namespace ax_utils
