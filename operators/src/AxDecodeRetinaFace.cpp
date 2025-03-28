// Copyright Axelera AI, 2023
#include <unordered_set>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
    "width",
    "height",
    "padding",
    "zero_points",
    "scales",
    "transpose",
    "confidence_threshold",
    "steps",
    "min_sizes",
    "variances",
    "clip",
    "decoder_name",
    "scale_up",
    "letterbox",
  };
  return allowed_properties;
}

using lookups = std::array<float, 256>;

struct Prior {
  float cx;
  float cy;
  float s_kx;
  float s_ky;
};

std::vector<std::vector<Prior>>
generate_priors(float width, float height, std::vector<std::vector<int>> min_sizes,
    std::vector<int> steps, bool clip = false)
{
  std::vector<std::vector<Prior>> priors(steps.size());
  for (int k = 0; k < steps.size(); k++) {
    for (int i = 0; i < ceil(height / steps[k]); i++) {
      for (int j = 0; j < ceil(width / steps[k]); j++) {
        for (int min_size : min_sizes[k]) {
          float cx = (j + 0.5) * steps[k] / width;
          float cy = (i + 0.5) * steps[k] / height;
          float s_kx = min_size / width;
          float s_ky = min_size / height;
          if (clip) {
            cx = std::clamp(cx, 0.0f, 1.0f);
            cy = std::clamp(cy, 0.0f, 1.0f);
            s_kx = std::clamp(s_kx, 0.0f, 1.0f);
            s_ky = std::clamp(s_ky, 0.0f, 1.0f);
          }
          priors[k].push_back({ cx, cy, s_kx, s_ky });
        }
      }
    }
  }
  return priors;
}

struct retinaface_properties {
  std::string meta_key
      = "meta_" + std::to_string(reinterpret_cast<long long unsigned int>(this));
  unsigned int width{ 0 };
  unsigned int height{ 0 };
  std::vector<std::vector<int>> padding{};
  std::vector<lookups> dequantize_tables{};
  std::vector<lookups> dequantize_variance_tables{};
  std::vector<lookups> exponential_variance_tables{};
  bool transpose = true;
  float confidence_threshold = { 0.25 };
  std::vector<int> steps{};
  std::string decoder_name{};
  std::vector<std::vector<int>> min_sizes{};
  bool clip = false;
  std::vector<std::vector<Prior>> priors{};
  bool scale_up{ true };
  bool letterbox{ true };
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<retinaface_properties>();

  prop->meta_key = Ax::get_property(
      input, "meta_key", "retinaface_static_properties", prop->meta_key);
  prop->width = Ax::get_property(input, "width", "retinaface_static_properties", prop->width);
  prop->height = Ax::get_property(
      input, "height", "retinaface_static_properties", prop->height);
  prop->padding = Ax::get_property(
      input, "padding", "retinaface_static_properties", prop->padding);
  std::vector<float> zero_points{};
  std::vector<float> scales{};
  zero_points = Ax::get_property(
      input, "zero_points", "retinaface_static_properties", zero_points);
  scales = Ax::get_property(input, "scales", "retinaface_static_properties", scales);
  if (scales.size() != zero_points.size()) {
    logger(AX_ERROR) << "Scales and zero_points must have the same size" << std::endl;
    throw std::runtime_error("Scales and zero_points must have the same size");
  }
  prop->transpose = Ax::get_property(
      input, "transpose", "retinaface_static_properties", prop->transpose);
  if (!prop->transpose) {
    logger(AX_ERROR) << "decode_retinaface only implemented for transpose = true"
                     << std::endl;
    throw std::runtime_error("decode_retinaface only implemented for transpose = true");
  }
  prop->confidence_threshold = Ax::get_property(input, "confidence_threshold",
      "retinaface_static_properties", prop->confidence_threshold);
  prop->steps = Ax::get_property(input, "steps", "retinaface_static_properties", prop->steps);
  if (zero_points.size() != prop->steps.size() * 3) {
    logger(AX_ERROR) << "zero_points must have size 3 * steps" << std::endl;
    throw std::runtime_error("zero_points must have size 3 * steps");
  }
  prop->min_sizes = Ax::get_property(
      input, "min_sizes", "retinaface_static_properties", prop->min_sizes);
  if (prop->min_sizes.size() != prop->steps.size()) {
    logger(AX_ERROR) << "min_sizes and steps must have the same size" << std::endl;
    throw std::runtime_error("min_sizes and steps must have the same size");
  }
  std::vector<float> variances{};
  variances = Ax::get_property(input, "variances", "retinaface_static_properties", variances);
  if (variances.size() != 2) {
    logger(AX_ERROR) << "variances must have size 2" << std::endl;
    throw std::runtime_error("variances must have size 2");
  }
  prop->clip = Ax::get_property(input, "clip", "retinaface_static_properties", prop->clip);
  prop->decoder_name = Ax::get_property(
      input, "decoder_name", "retinaface_static_properties", prop->decoder_name);
  prop->scale_up = Ax::get_property(
      input, "scale_up", "retinaface_static_properties", prop->scale_up);

  prop->dequantize_tables = ax_utils::build_exponential_tables(zero_points, scales);
  prop->dequantize_variance_tables = ax_utils::build_general_dequantization_tables(
      zero_points, scales, [v = variances[0]](float x) { return v * x; });
  prop->exponential_variance_tables
      = ax_utils::build_general_dequantization_tables(zero_points, scales,
          [v = variances[1]](float x) { return std::exp(v * x); });

  prop->priors = generate_priors(
      prop->width, prop->height, prop->min_sizes, prop->steps, prop->clip);
  prop->letterbox = Ax::get_property(
      input, "letterbox", "retinaface_static_properties", prop->letterbox);
  return prop;
}

float
dequantize_using_table(int8_t value, const float *the_table)
{
  int index = value + 128;
  return the_table[index];
}

std::pair<std::vector<float>, std::vector<int>>
decode_scores(const AxTensorInterface &tensor, int left_channel_padding,
    int right_channel_padding, const float *dequantize_table, float confidence)
{
  static constexpr int element_stride = 2;
  std::vector<float> scores;
  std::vector<int> keep;
  int8_t *data = static_cast<int8_t *>(tensor.data);
  int ind_detection = -1;
  for (int k = 0; k < tensor.sizes[1]; k++) {
    for (int j = 0; j < tensor.sizes[2]; j++) {
      for (int i = left_channel_padding;
           i < tensor.sizes[3] - right_channel_padding; i = i + element_stride) {
        int ind_tensor = k * tensor.sizes[2] * tensor.sizes[3] + j * tensor.sizes[3] + i;
        ++ind_detection;
        float background = dequantize_using_table(data[ind_tensor++], dequantize_table);
        float face = dequantize_using_table(data[ind_tensor++], dequantize_table);
        float score = face / (face + background);
        if (score > confidence) {
          scores.push_back(score);
          keep.push_back(ind_detection);
        }
      }
    }
  }
  return std::make_pair(std::move(scores), std::move(keep));
}

template <typename T, int element_stride, typename F, typename... Args>
std::vector<T>
decode_selection(const AxTensorInterface &tensor, int left_channel_padding,
    int right_channel_padding, const std::vector<int> &keep,
    const std::vector<Prior> &prior, F &&f_decode, Args... args)
{
  std::vector<T> result;
  if (keep.empty()) {
    return result;
  }
  int ind_detection = -1;
  auto keep_itr = keep.begin();
  for (int k = 0; k < tensor.sizes[1]; k++) {
    for (int j = 0; j < tensor.sizes[2]; j++) {
      for (int i = left_channel_padding;
           i < tensor.sizes[3] - right_channel_padding; i = i + element_stride) {
        int ind_tensor = k * tensor.sizes[2] * tensor.sizes[3] + j * tensor.sizes[3] + i;
        const int8_t *data = static_cast<int8_t *>(tensor.data) + ind_tensor;
        ++ind_detection;
        if (*keep_itr != ind_detection) {
          continue;
        }
        f_decode(data, prior[ind_detection], result, args...);
        if (++keep_itr == keep.end()) {
          return result;
        }
      }
    }
  }
  return result;
}

std::vector<ax_utils::fbox>
decode_loc(const AxTensorInterface &tensor, int left_channel_padding, int right_channel_padding,
    const std::vector<int> &keep, const std::vector<Prior> &prior,
    const float *dequantize_variance_table, const float *exponential_variance_table)
{
  return decode_selection<ax_utils::fbox, 4>(
      tensor, left_channel_padding, right_channel_padding, keep, prior,
      [](const int8_t *data, const Prior &prior, std::vector<ax_utils::fbox> &result,
          const float *dequantize_variance_table, const float *exponential_variance_table) {
        float x = dequantize_using_table(*data++, dequantize_variance_table);
        float y = dequantize_using_table(*data++, dequantize_variance_table);
        float w = dequantize_using_table(*data++, exponential_variance_table);
        float h = dequantize_using_table(*data++, exponential_variance_table);
        x = x * prior.s_kx + prior.cx;
        y = y * prior.s_ky + prior.cy;
        w = w * prior.s_kx;
        h = h * prior.s_ky;
        result.push_back({ x - w / 2, y - h / 2, x + w / 2, y + h / 2 });
      },
      dequantize_variance_table, exponential_variance_table);
}

std::vector<ax_utils::fkpt>
decode_landm(const AxTensorInterface &tensor, int left_channel_padding,
    int right_channel_padding, const std::vector<int> &keep,
    const std::vector<Prior> &prior, const float *dequantize_variance_table)
{
  return decode_selection<ax_utils::fkpt, 10>(
      tensor, left_channel_padding, right_channel_padding, keep, prior,
      [](const int8_t *data, const Prior &prior, std::vector<ax_utils::fkpt> &result,
          const float *dequantize_variance_table) {
        for (int h = 0; h < 5; h++) {
          float x = dequantize_using_table(*data++, dequantize_variance_table);
          float y = dequantize_using_table(*data++, dequantize_variance_table);
          x = x * prior.s_kx + prior.cx;
          y = y * prior.s_ky + prior.cy;
          result.push_back({ x, y, 1 });
        }
      },
      dequantize_variance_table);
}

extern "C" void
decode_to_meta(const AxTensorsInterface &tensors,
    const retinaface_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  if (tensors.size() != 3 * prop->steps.size()) {
    logger(AX_ERROR) << "Expected 3 tensors for each step but got "
                     << tensors.size() << std::endl;
    throw std::runtime_error("Expected 3 tensors for each step but got "
                             + std::to_string(tensors.size()));
  }

  std::vector<float> scores;
  std::vector<BboxXyxy> boxes;
  std::vector<KptXyv> kpts;
  int num_steps = prop->steps.size();
  for (int i = 0; i < num_steps; i++) {
    auto [scores_i, keep] = decode_scores(tensors[i + num_steps],
        prop->padding[i + num_steps][6], prop->padding[i + num_steps][7],
        prop->dequantize_tables[i + num_steps].data(), prop->confidence_threshold);

    auto fboxes = decode_loc(tensors[i], prop->padding[i][6], prop->padding[i][7],
        keep, prop->priors[i], prop->dequantize_variance_tables[i].data(),
        prop->exponential_variance_tables[i].data());
    auto boxes_i = ax_utils::scale_boxes(fboxes, std::get<AxVideoInterface>(video_interface),
        prop->width, prop->height, prop->scale_up, prop->letterbox);

    auto fkpts = decode_landm(tensors[i + 2 * num_steps],
        prop->padding[i + 2 * num_steps][6], prop->padding[i + 2 * num_steps][7], keep,
        prop->priors[i], prop->dequantize_variance_tables[i + 2 * num_steps].data());
    auto &vinfo = std::get<AxVideoInterface>(video_interface);
    auto kpts_i = ax_utils::scale_kpts(fkpts, vinfo.info.width, vinfo.info.height,
        prop->width, prop->height, prop->scale_up, prop->letterbox);

    scores.insert(scores.end(), scores_i.begin(), scores_i.end());
    boxes.insert(boxes.end(), boxes_i.begin(), boxes_i.end());
    kpts.insert(kpts.end(), kpts_i.begin(), kpts_i.end());
  }

  map[prop->meta_key] = std::make_unique<AxMetaKptsDetection>(std::move(boxes),
      std::move(kpts), std::move(scores), std::vector<int>{ 5, 3 }, prop->decoder_name);
}
