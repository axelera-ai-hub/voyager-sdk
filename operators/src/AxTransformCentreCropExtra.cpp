// Copyright Axelera AI, 2025
#include <iostream>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

#include <opencv2/core/ocl.hpp>

struct crop_properties {
  int scale_size = 0;
  int crop_size = 0;
  int crop_width = 0;
  int crop_height = 0;
  int scale_width = 0;
  int scale_height = 0;
  bool downstream_supports_opencl{ false };
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
    "crop_width",
    "crop_height",
    "scale_width",
    "scale_height",
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
  prop->crop_width = Ax::get_property(input, "crop_width",
      "centercropextra_static_properties", prop->crop_width);
  prop->crop_height = Ax::get_property(input, "crop_height",
      "centercropextra_static_properties", prop->crop_height);
  prop->scale_width = Ax::get_property(input, "scale_width",
      "centercropextra_static_properties", prop->scale_width);
  prop->scale_height = Ax::get_property(input, "scale_height",
      "centercropextra_static_properties", prop->scale_height);

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    crop_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "centercropextra_static_properties", prop->downstream_supports_opencl);
}

struct width_height {
  int width;
  int height;
};

/// @brief  This determines the crop region in the input image that would correspond the specified
///         crop region in the image after it has been resized.
/// @param in_width - The width of the input rectangle
/// @param in_height - The height of the input rectangle
/// @param size - If size is provided, it is the size that the shorter side of the input rectangle
///               will be scaled to, the longer side will be scaled proportionally to maintain aspect ratio.
/// @param width - If width and height are provided the input image would be scaled
///                 to these sizes
/// @param height - If width and height are provided the input image would be scaled to these sizes
/// @param crop_w - crop_w is the width of the cropped region in the resized image
/// @param crop_h - crop_h is the height of the cropped region in the resized image
///
/// @return - Width and height of the cropped region in the input image that would
///           correspond to the width and height of the cropped region after resize
width_height
determine_scaled_crop(int in_width, int in_height, int size, int width,
    int height, int crop_w, int crop_h)
{
  if (size != 0) {
    auto shortest = std::min(in_width, in_height);
    auto scale = static_cast<double>(shortest) / size;
    return {
      static_cast<int>(std::round(crop_w * scale)),
      static_cast<int>(std::round(crop_h * scale)),
    };
  } else if (width != 0) {
    auto scale_x = in_width / width;
    auto scale_y = in_height / height;
    return {
      static_cast<int>(std::round(crop_w * scale_x)),
      static_cast<int>(std::round(crop_h * scale_y)),
    };
  }
  //  If no scale size or width is specified, do not resize
  return { crop_w, crop_h };
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const crop_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    logger(AX_ERROR) << "centercropextra works on video only\n";
    throw std::runtime_error("centercropextra works on video only");
  }
  if ((prop->crop_size != 0 && (prop->crop_width != 0 || prop->crop_height != 0))
      || (prop->crop_size == 0 && (prop->crop_width == 0 || prop->crop_height == 0))) {
    throw std::runtime_error(
        "centercropextra: you must specify either cropsize or crop_width/crop_height");
  }
  if (prop->scale_size != 0 && (prop->scale_width != 0 || prop->scale_height != 0)) {
    throw std::runtime_error(
        "centercropextra: you cannot pecify scale_width/scale_height with scalesize");
  }
  if (prop->scale_size == 0) {
    if ((prop->scale_width != 0 && prop->scale_height == 0)
        || (prop->scale_width == 0 && prop->scale_height != 0)) {
      throw std::runtime_error(
          "centercropextra: you must specify both scale_width and scale_height or scalesize or none of them");
    }
  }

  auto &input_video = std::get<AxVideoInterface>(interface);
  auto crop_w = prop->crop_size != 0 ? prop->crop_size : prop->crop_width;
  auto crop_h = prop->crop_size != 0 ? prop->crop_size : prop->crop_height;

  auto [cropped_width, cropped_height]
      = determine_scaled_crop(input_video.info.width, input_video.info.height,
          prop->scale_size, prop->scale_width, prop->scale_height, crop_w, crop_h);

  if (input_video.info.width < cropped_width) {
    logger(AX_ERROR) << "centercropextra: input width is smaller than cropped width, using input width"
                     << std::endl;
    cropped_width = input_video.info.width;
  }
  if (input_video.info.height < cropped_height) {
    logger(AX_ERROR) << "centercropextra: input height is smaller than cropped height, using input height"
                     << std::endl;
    cropped_height = input_video.info.height;
  }
  auto crop_x = (input_video.info.width - cropped_width) / 2;
  auto crop_y = (input_video.info.height - cropped_height) / 2;

  AxDataInterface output_data = input_video;
  return Ax::create_roi(input_video, crop_x, crop_y, cropped_width, cropped_height);
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
      && input_video.info.format != AxVideoFormat::BGRA
      && input_video.info.format != AxVideoFormat::GRAY8) {
    //  TODO: Add support for other formats
    logger(AX_ERROR) << "centre crop only works on RGBA or BGRA\n";
    throw std::runtime_error("centre crop only works on RGBA or BGRA");
  }
  cv::ocl::setUseOpenCL(false);
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

extern "C" bool
query_supports(Ax::PluginFeature feature,
    const crop_properties *crop_properties, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return crop_properties && crop_properties->downstream_supports_opencl;
  }
  if (feature == Ax::PluginFeature::crop_meta) {
    return true;
  }
  return false;
}
