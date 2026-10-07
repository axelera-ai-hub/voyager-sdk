// Copyright Axelera AI, 2023
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"

#include <optional>

#include <opencv2/core/ocl.hpp>

namespace
{
struct padding_properties {
  std::vector<std::vector<int>> paddings;
  std::optional<int8_t> fill{};
  std::vector<int> in_shape{};
  std::vector<int> out_shape{};
};


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
  prop->paddings = Ax::get_property(
      input, "padding", "padding_properties", std::vector<std::vector<int>>{});
  // For backward compatibility - if no paddings were specified but there's a single padding vector, convert it
  if (prop->paddings.empty()) {
    std::vector<int> single_padding = Ax::get_property(
        input, "padding", "padding_properties", std::vector<int>{});
    if (!single_padding.empty()) {
      prop->paddings.push_back(single_padding);
    }
  }
  auto tmp_fill = Ax::get_property(
      input, "fill", "padding_properties", std::optional<int16_t>{});
  if (tmp_fill.has_value()) {
    prop->fill = std::clamp(*tmp_fill,
        static_cast<int16_t>(std::numeric_limits<int8_t>::min()),
        static_cast<int16_t>(std::numeric_limits<int8_t>::max()));
  }
  prop->in_shape
      = Ax::get_property(input, "input_shape", "padding_properties", prop->in_shape);
  prop->out_shape
      = Ax::get_property(input, "output_shape", "padding_properties", prop->out_shape);
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const padding_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("transform_padding requires tensor input");
  }
  auto input = std::get<AxTensorsInterface>(interface);


  // Make sure we have at least one tensor
  if (input.empty() || input[0].bytes != 1) {
    throw std::runtime_error("transform_padding requires at least one int8 tensor input");
  }

  // Make sure we have paddings for each tensor or at least one default padding
  if (prop->paddings.empty()) {
    throw std::runtime_error("transform_padding: no padding configurations provided");
  }

  if (prop->paddings.size() < input.size()) {
    throw std::runtime_error("transform_padding: fewer padding configurations than tensors, expected "
                             + std::to_string(input.size()) + " but got "
                             + std::to_string(prop->paddings.size()));
  }

  // Validate each padding configuration
  for (size_t i = 0; i < std::min(prop->paddings.size(), input.size()); ++i) {
    const auto &padding = prop->paddings[i];
    if ((padding.size() % 2) != 0) {
      throw std::runtime_error("transform_padding: padding must be a multiple of 2:"
                               + ax_utils::sizes_to_string(padding));
    }
    if (padding.size() / 2 > input[i].sizes.size()) {
      throw std::runtime_error("transform_padding: padding "
                               + ax_utils::sizes_to_string(padding) + " too long for input tensor "
                               + ax_utils::sizes_to_string(input[i].sizes));
    }
  }

  if (!ax_utils::validate_shape(prop->in_shape, input[0].sizes)) {
    throw std::runtime_error("transform_padding: input_shape "
                             + ax_utils::sizes_to_string(prop->in_shape) + " does not match input tensor "
                             + ax_utils::sizes_to_string(input[0].sizes));
  }

  auto output = input;

  // Calculate output sizes for each tensor based on its padding
  for (size_t i = 0; i < input.size(); ++i) {
    // Use the appropriate padding for this tensor (or the last one if we have fewer paddings than tensors)
    const auto &padding
        = i < prop->paddings.size() ? prop->paddings[i] : prop->paddings.back();

    auto in_sizes = prop->in_shape.empty() ? input[i].sizes : prop->in_shape;
    const auto info = ax_utils::get_transfer_info(in_sizes, padding);

    if (i == 0 && !ax_utils::validate_shape(prop->out_shape, info.out_sizes)) {
      throw std::runtime_error("transform_padding: output_shape "
                               + ax_utils::sizes_to_string(prop->out_shape) + " does not match calculated output tensor "
                               + ax_utils::sizes_to_string(info.out_sizes));
    }

    output[i].sizes = prop->out_shape.empty() ? info.out_sizes : prop->out_shape;
  }


  return { output };
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const padding_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  cv::ocl::setUseOpenCL(false);

  auto input_tensors = std::get<AxTensorsInterface>(input);
  auto output_tensors = std::get<AxTensorsInterface>(output);

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    // Use the appropriate padding for this tensor (or the last one if we have fewer paddings than tensors)
    const auto &padding
        = i < prop->paddings.size() ? prop->paddings[i] : prop->paddings.back();

    auto in_shape = prop->in_shape.empty() ? input_tensors[i].sizes : prop->in_shape;
    const auto info = ax_utils::get_transfer_info(in_shape, padding);

    cv::Mat input_mat(info.in_sizes, CV_8UC1, input_tensors[i].data);
    cv::Mat output_mat(info.out_sizes, CV_8UC1, output_tensors[i].data);

    if (info.is_crop) {
      input_mat(info.ranges).copyTo(output_mat);
    } else {
      if (prop->fill) {
        output_mat.setTo(*prop->fill);
      }
      input_mat.copyTo(output_mat(info.ranges));
    }
  }
}
