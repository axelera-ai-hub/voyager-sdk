// Copyright Axelera AI, 2023
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

#include <optional>

namespace
{
struct padding_properties {
  std::vector<int> padding;
  std::optional<int8_t> fill{};
  std::vector<int> in_shape{};
  std::vector<int> out_shape{};
};

struct transfer_info {
  bool is_crop = false;
  std::vector<int> in_sizes{};
  std::vector<int> out_sizes{};
  std::vector<cv::Range> ranges{};
};

std::string
sizes_to_string(const std::vector<int> &sizes)
{
  std::string s;
  for (auto sz : sizes) {
    if (!s.empty())
      s += ",";
    s += std::to_string(sz);
  }
  return "(" + s + ")";
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
    throw std::runtime_error("transform_padding: " + sizes_to_string(padding) + " too short for input shape "
                             + sizes_to_string(sizes) + " after any implicit squeezing");
  }

  const auto first_size = std::prev(sizes.cend(), padding_ndims);
  const auto crop
      = std::any_of(padding.begin(), padding.end(), [](int x) { return x < 0; });
  const auto grow
      = std::any_of(padding.begin(), padding.end(), [](int x) { return x > 0; });
  if (crop && grow) {
    throw std::runtime_error("transform_padding: can remove or add padding, but not both in different dimensions:"
                             + sizes_to_string(padding));
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
          "transform_padding: negative padding " + sizes_to_string(padding) + " greater than tensor size in dimension "
          + std::to_string(i) + " with input tensor " + sizes_to_string(sizes));
    }
  }
  return info;
}
} // namespace

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "padding",
    "fill", "input_shape", "output_shape" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<padding_properties> prop = std::make_shared<padding_properties>();
  prop->padding = Ax::get_property(input, "padding", "padding_properties", prop->padding);
  prop->fill = Ax::get_property(input, "fill", "padding_properties", prop->fill);
  prop->in_shape
      = Ax::get_property(input, "input_shape", "padding_properties", prop->in_shape);
  prop->out_shape
      = Ax::get_property(input, "output_shape", "padding_properties", prop->out_shape);
  return prop;
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

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const padding_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("transform_padding requires tensor input");
  }
  auto input = std::get<AxTensorsInterface>(interface);
  if (input.size() != 1 || input[0].bytes != 1) {
    throw std::runtime_error("transform_padding requires single int8 tensor input");
  }
  if ((prop->padding.size() % 2) != 0) {
    throw std::runtime_error("transform_padding: padding must be a multiple of 2:"
                             + sizes_to_string(prop->padding));
  }
  if (prop->padding.size() / 2 > input[0].sizes.size()) {
    throw std::runtime_error(
        "transform_padding: padding " + sizes_to_string(prop->padding)
        + " too long for input tensor " + sizes_to_string(input[0].sizes));
  }
  if (!validate_shape(prop->in_shape, input[0].sizes)) {
    throw std::runtime_error(
        "transform_padding: input_shape " + sizes_to_string(prop->in_shape)
        + " does not match input tensor " + sizes_to_string(input[0].sizes));
  }
  auto in_sizes = prop->in_shape.empty() ? input[0].sizes : prop->in_shape;
  const auto info = get_transfer_info(in_sizes, prop->padding);
  auto output = input;
  if (!validate_shape(prop->out_shape, info.out_sizes)) {
    throw std::runtime_error(
        "transform_padding: output_shape " + sizes_to_string(prop->in_shape)
        + " does not match output tensor " + sizes_to_string(input[0].sizes));
  }

  output[0].sizes = prop->out_shape.empty() ? info.out_sizes : prop->out_shape;
  return { output };
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const padding_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto &input_tensor = std::get<AxTensorsInterface>(input)[0];
  auto &output_tensor = std::get<AxTensorsInterface>(output)[0];
  auto in_shape = prop->in_shape.empty() ? input_tensor.sizes : prop->in_shape;
  const auto info = get_transfer_info(in_shape, prop->padding);

  cv::Mat input_mat(info.in_sizes, CV_8UC1, input_tensor.data);
  cv::Mat output_mat(info.out_sizes, CV_8UC1, output_tensor.data);

  if (info.is_crop) {
    input_mat(info.ranges).copyTo(output_mat);
  } else {
    if (prop->fill) {
      output_mat.setTo(*prop->fill);
    }
    input_mat.copyTo(output_mat(info.ranges));
  }
}
