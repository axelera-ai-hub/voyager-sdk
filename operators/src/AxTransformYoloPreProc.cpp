// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxMeta.hpp"

#include "AxLog.hpp"
#include "AxUtils.hpp"

#include <unordered_map>
#include <unordered_set>

namespace
{
struct padding_properties {
  std::vector<int> padding;
  std::optional<int8_t> fill{};
};

static constexpr int output_channels = 12;

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

} // namespace


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "padding", "fill" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<padding_properties> prop = std::make_shared<padding_properties>();
  prop->padding = Ax::get_property(input, "padding", "padding_properties", prop->padding);
  if (prop->padding.empty()) {
    prop->padding = { 0, 0, 0, 0, 0, 0, 0, 0 };
  }
  prop->fill = Ax::get_property(input, "fill", "padding_properties", prop->fill);
  return prop;
}

std::vector<int>
determine_output_sizes(std::vector<int> input, const std::vector<int> &padding)
{
  std::vector<int> outsizes(4);
  outsizes[0] = padding[0] + padding[1] + input[0];
  outsizes[1] = padding[2] + padding[3] + input[1] / 2;
  outsizes[2] = padding[4] + padding[5] + input[2] / 2;
  outsizes[3] = output_channels + padding[6] + padding[7];
  return outsizes;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const padding_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("yolo preprocess layer works on tensor input only");
  }
  if (std::get<AxTensorsInterface>(interface).size() != 1) {
    throw std::runtime_error("yolo preprocess layer works on single tensor input only");
  }
  if (std::get<AxTensorsInterface>(interface)[0].bytes != 1) {
    throw std::runtime_error("yolo preprocess layer works on int8 tensor input only");
  }
  auto output = std::get<AxTensorsInterface>(interface);
  auto outsizes = output[0].sizes;

  if (outsizes.size() != 4) {
    throw std::runtime_error("yolo preprocess layer works on 4D tensor input only");
  }
  if (outsizes[1] % 2) {
    throw std::runtime_error("yolo preprocess layer works on even height input only");
  }
  if (outsizes[2] % 2) {
    throw std::runtime_error("yolo preprocess layer works on even width input only");
  }
  if (outsizes[3] != 3 && outsizes[3] != 4) {
    throw std::runtime_error(
        "yolo preprocess layer works on RGB or RGBA input with channels last");
  }
  if ((prop->padding.size() % 2) != 0) {
    throw std::runtime_error("transform_padding: padding must be a multiple of 2:"
                             + sizes_to_string(prop->padding));
  }
  if (prop->padding.size() / 2 > output[0].sizes.size()) {
    throw std::runtime_error(
        "transform_padding: padding " + sizes_to_string(prop->padding)
        + " too long for input tensor " + sizes_to_string(output[0].sizes));
  }

  outsizes = determine_output_sizes(outsizes, prop->padding);

  output[0].sizes = outsizes;
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const padding_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{

  const auto &input_tensor = std::get<AxTensorsInterface>(input)[0];
  const auto &output_tensor = std::get<AxTensorsInterface>(output)[0];
  int8_t *input_ptr = static_cast<int8_t *>(input_tensor.data);
  int8_t *output_ptr = static_cast<int8_t *>(output_tensor.data);

  const int batches = input_tensor.sizes[0];
  const int height = input_tensor.sizes[1];
  const int width = input_tensor.sizes[2];
  const int channels = input_tensor.sizes[3];
  const int input_stride = width * channels;

  const auto in_sizes = input_tensor.sizes;
  const auto outsizes = determine_output_sizes(in_sizes, prop->padding);

  const auto channel_stride = output_tensor.sizes[3];
  const auto x_stride = channel_stride * output_tensor.sizes[2];
  const auto y_stride = x_stride * output_tensor.sizes[1];

  for (int batch = 0; batch != batches; ++batch) {
    auto b_offset = batch * y_stride;
    if (prop->fill) {
      std::memset(output_ptr + b_offset, *prop->fill, prop->padding[2] * x_stride);
    }
    b_offset += prop->padding[2] * x_stride;
    auto w_offset = b_offset;
    for (int y = 0; y != height; y += 2) {
      auto c_offset = w_offset;
      if (prop->fill) {
        std::memset(output_ptr + c_offset, *prop->fill, prop->padding[4] * channel_stride);
      }
      c_offset += prop->padding[4] * channel_stride;
      for (int x = 0; x != width; x += 2) {
        if (prop->fill) {
          std::memset(output_ptr + c_offset, *prop->fill, prop->padding[6]);
        }
        auto offset = c_offset + prop->padding[6];
        output_ptr[offset + 0]
            = input_ptr[(y + 0) * input_stride + (x + 0) * channels + 0];
        output_ptr[offset + 1]
            = input_ptr[(y + 0) * input_stride + (x + 0) * channels + 1];
        output_ptr[offset + 2]
            = input_ptr[(y + 0) * input_stride + (x + 0) * channels + 2];
        output_ptr[offset + 3]
            = input_ptr[(y + 1) * input_stride + (x + 0) * channels + 0];
        output_ptr[offset + 4]
            = input_ptr[(y + 1) * input_stride + (x + 0) * channels + 1];
        output_ptr[offset + 5]
            = input_ptr[(y + 1) * input_stride + (x + 0) * channels + 2];
        output_ptr[offset + 6]
            = input_ptr[(y + 0) * input_stride + (x + 1) * channels + 0];
        output_ptr[offset + 7]
            = input_ptr[(y + 0) * input_stride + (x + 1) * channels + 1];
        output_ptr[offset + 8]
            = input_ptr[(y + 0) * input_stride + (x + 1) * channels + 2];
        output_ptr[offset + 9]
            = input_ptr[(y + 1) * input_stride + (x + 1) * channels + 0];
        output_ptr[offset + 10]
            = input_ptr[(y + 1) * input_stride + (x + 1) * channels + 1];
        output_ptr[offset + 11]
            = input_ptr[(y + 1) * input_stride + (x + 1) * channels + 2];
        if (prop->fill) {
          auto end_fill_size = prop->padding[7];
          std::memset(output_ptr + offset + output_channels, *prop->fill, end_fill_size);
        }
        c_offset += channel_stride;
      }
      if (prop->fill) {
        std::memset(output_ptr + c_offset, *prop->fill, prop->padding[5] * channel_stride);
      }
      w_offset += x_stride;
    }
    if (prop->fill) {
      std::memset(output_ptr + w_offset, *prop->fill, prop->padding[3] * x_stride);
    }
    input_ptr += input_tensor.total() / batches;
  }
}
