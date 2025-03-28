// Copyright Axelera AI, 2023
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

#include <PillowResize/PillowResize.hpp>

struct resizercrope_properties {
  int resize_size = 0;
  int extra_crop = 0;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "resize_size",
    "final_size_after_crop" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<resizercrope_properties> prop
      = std::make_shared<resizercrope_properties>();
  prop->resize_size = Ax::get_property(input, "resize_size",
      "resizeratiocropexcess_properties", prop->resize_size);
  int final_size_after_crop = prop->resize_size;
  final_size_after_crop = Ax::get_property(input, "final_size_after_crop",
      "resizeratiocropexcess_properties", final_size_after_crop);
  if (!prop->resize_size && final_size_after_crop) {
    throw std::runtime_error(
        "resizeratiocropexcess_properties: final_size_after_crop is specified but resize_size is not");
  }
  prop->extra_crop = prop->resize_size - final_size_after_crop;
  if (prop->extra_crop < 0) {
    throw std::runtime_error(
        "resizeratiocropexcess_properties: final_size_after_crop must be smaller than resize_size");
  }
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const resizercrope_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error("resizeratiocropexcess works on video only");
  }

  AxDataInterface output = interface;
  auto &info = std::get<AxVideoInterface>(output).info;
  if (info.format != AxVideoFormat::RGBA && info.format != AxVideoFormat::BGRA) {
    throw std::runtime_error("resizeratiocropexcess expects RGBA/BGRA but got "
                             + AxVideoFormatToString(info.format));
  }

  info.width = prop->resize_size - prop->extra_crop;
  info.height = prop->resize_size - prop->extra_crop;

  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const resizercrope_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)
      || !std::holds_alternative<AxVideoInterface>(output)) {
    throw std::runtime_error("resizeratiocropexcess works on video only");
  }

  auto &input_video = std::get<AxVideoInterface>(input);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);

  auto &output_video = std::get<AxVideoInterface>(output);
  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);

  if ((input_video.info.format != AxVideoFormat::RGBA
          && input_video.info.format != AxVideoFormat::BGRA)
      || (input_video.info.format != output_video.info.format)) {
    throw std::runtime_error(
        "resizeratiocropexcess expects and provides RGBA/BGRA only but got "
        + AxVideoFormatToString(input_video.info.format) + " input and "
        + AxVideoFormatToString(output_video.info.format) + " output");
  }

  auto short_side = std::min(input_video.info.width, input_video.info.height);
  auto left = (input_video.info.width - short_side) / 2;
  auto top = (input_video.info.height - short_side) / 2;
  cv::Rect crop_rect(left, top, short_side, short_side);
  auto cropped_mat = input_mat(crop_rect);

  cv::Mat resized_mat = PillowResize::resize(cropped_mat,
      cv::Size(prop->resize_size, prop->resize_size),
      PillowResize::InterpolationMethods::INTERPOLATION_BILINEAR);

  auto rect = cv::Rect(prop->extra_crop / 2, prop->extra_crop / 2,
      prop->resize_size - prop->extra_crop, prop->resize_size - prop->extra_crop);
  resized_mat(rect).copyTo(output_mat);
}
