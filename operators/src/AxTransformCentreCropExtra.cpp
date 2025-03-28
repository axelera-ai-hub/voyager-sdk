// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

#include <iostream>
#include <unordered_set>

struct crop_properties {
  int scale_size = 0;
  int crop_size = 0;
};

struct crop_box {
  int x1;
  int y1;
  int x2;
  int y2;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "scalesize",
    "cropsize",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<crop_properties> prop = std::make_shared<crop_properties>();
  prop->scale_size = Ax::get_property(
      input, "scalesize", "centercropextra_static_properties", prop->scale_size);
  prop->crop_size = Ax::get_property(
      input, "cropsize", "centercropextra_static_properties", prop->crop_size);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    crop_properties *prop, Ax::Logger &logger)
{
}

crop_box
determine_crop_box(int img_width, int img_height, int scale_size, int crop_size)
{
  //  First we determine the parts to crop to make square
  auto shortest = std::min(img_width, img_height);
  auto crop_left = (img_width - shortest) / 2;
  auto crop_top = (img_height - shortest) / 2;

  // Compute cropping region's size
  auto cropped_size = static_cast<int>(std::round((shortest * crop_size) / scale_size));

  // Compute the cropping coordinates; always cropping from the center
  auto x1 = crop_left + (shortest - cropped_size) / 2;
  auto y1 = crop_top + (shortest - cropped_size) / 2;
  auto x2 = x1 + cropped_size;
  auto y2 = y1 + cropped_size;
  return { x1, y1, x2, y2 };
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const crop_properties *prop, Ax::Logger &logger)
{
  if (prop->scale_size == 0 || prop->crop_size == 0) {
    logger(AX_ERROR) << "both scalesize and cropsize must be specified\n";
    throw std::runtime_error("both scalesize and cropsize must be specified in centercropextra "
                             + std::to_string(prop->crop_size));
  }

  if (prop->scale_size < prop->crop_size) {
    logger(AX_ERROR) << "scalesize must be larger than cropsize\n";
    throw std::runtime_error("scalesize must be larger than cropsize in centercropextra");
  }

  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    logger(AX_ERROR) << "centercropextra works on video only\n";
    throw std::runtime_error("centercropextra works on video only");
  }

  auto &input_video = std::get<AxVideoInterface>(interface);

  crop_box box = determine_crop_box(input_video.info.width,
      input_video.info.height, prop->scale_size, prop->crop_size);

  AxDataInterface output_data = input_video;
  AxVideoInterface &output_video = std::get<AxVideoInterface>(output_data);
  auto &info = output_video.info;

  info.width = box.x2 - box.x1;
  info.height = box.y2 - box.y1;
  info.x_offset = box.x1 + input_video.info.x_offset;
  info.y_offset = box.y1 + input_video.info.y_offset;
  info.cropped = true;
  return output_data;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const crop_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  auto &input_video = std::get<AxVideoInterface>(input);
  auto &output_video = std::get<AxVideoInterface>(output);

  if (input_video.info.format != output_video.info.format) {
    logger(AX_ERROR) << "input and output formats must match\n";
    throw std::runtime_error("input and output formats must match");
  }
  if (input_video.info.format != AxVideoFormat::RGBA
      && input_video.info.format != AxVideoFormat::BGRA) {
    //  TODO: Add support for other formats
    logger(AX_ERROR) << "centre crop only works on RGBA or BGRA\n";
    throw std::runtime_error("centre crop only works on RGBA or BGRA");
  }

  cv::Mat in_mat(cv::Size(input_video.info.width + input_video.info.x_offset,
                     input_video.info.height + input_video.info.y_offset),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.strides[0]);

  // Now add in the crop offsets
  cv::Rect in_crop_rect(input_video.info.x_offset, input_video.info.y_offset,
      input_video.info.width, input_video.info.height);
  cv::Mat input_mat = in_mat(in_crop_rect);


  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.strides[0]);

  auto out = set_output_interface(input, prop, logger);
  auto &out_cropped = std::get<AxVideoInterface>(out).info;
  cv::Rect crop_rect(out_cropped.x_offset, out_cropped.y_offset,
      out_cropped.width, out_cropped.height);

  cv::Mat cropped_mat = input_mat(crop_rect);
  cropped_mat.copyTo(output_mat);
}

extern "C" int
handles_crop_meta()
{
  return 1;
}
