// Copyright Axelera AI, 2023
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpenCl.hpp"
#include "AxUtils.hpp"

#include <opencv2/opencv.hpp>

struct resize_properties {
  int size = 0;
  int width = 0;
  int height = 0;
  int padding = 114;
  bool to_tensor = false;
  bool letterbox = false;
  bool scale_up = true;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "size",
    "width",
    "height",
    "padding",
    "to_tensor",
    "letterbox",
    "scale_up",

  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<resize_properties> prop = std::make_shared<resize_properties>();
  prop->padding = Ax::get_property(input, "padding", "resize_properties", prop->padding);
  prop->width = Ax::get_property(input, "width", "resize_properties", prop->width);
  prop->height = Ax::get_property(input, "height", "resize_properties", prop->height);
  prop->to_tensor
      = Ax::get_property(input, "to_tensor", "resize_properties", prop->to_tensor);
  prop->letterbox
      = Ax::get_property(input, "letterbox", "resize_properties", prop->letterbox);
  prop->size = Ax::get_property(input, "size", "resize_properties", prop->size);
  prop->scale_up = Ax::get_property(input, "scale_up", "resize_properties", prop->scale_up);

  //  If only one of width or height is set we assume square output
  if (prop->size != 0) {
    if (prop->width != 0 || prop->height != 0) {
      logger(AX_ERROR) << "You must provide only one of width/height or size" << std::endl;
      throw std::runtime_error("You must provide only one of width/height or size");
    }
    if (prop->letterbox) {
      prop->width = prop->size;
      prop->height = prop->size;
    }
  }
  if (prop->width == 0) {
    prop->width = prop->height;
  } else if (prop->height == 0) {
    prop->height = prop->width;
  }
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const resize_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output = interface;
  auto &info = std::get<AxVideoInterface>(output).info;
  if (prop->letterbox || prop->size == 0) {
    if (prop->width != 0 || prop->height != 0) {
      info.width = prop->width;
      info.height = prop->height;
    }
  } else {
    auto shortest = std::min(info.width, info.height);
    info.width = info.width * prop->size / shortest;
    info.height = info.height * prop->size / shortest;
  }
  if (prop->to_tensor) {
    AxTensorsInterface output = { { { 1, info.height, info.width, 4 }, 1, nullptr } };
    return AxDataInterface(output);
  }

  return output;
}

cv::Mat
get_output_mat(const AxDataInterface &input, const AxDataInterface &output,
    const resize_properties *prop)
{
  if (prop->to_tensor) {
    auto &input_video = std::get<AxVideoInterface>(input);
    auto &output_tensor = std::get<AxTensorsInterface>(output)[0];
    return cv::Mat(cv::Size(output_tensor.sizes[2], output_tensor.sizes[1]),
        Ax::opencv_type_u8(input_video.info.format), output_tensor.data,
        output_tensor.sizes[2] * output_tensor.sizes[3]);
  }
  auto &output_video = std::get<AxVideoInterface>(output);
  return cv::Mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const resize_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)) {
    logger(AX_ERROR) << "resize works on video only" << std::endl;
    throw std::runtime_error("resize works on video only");
  }

  auto &input_video = std::get<AxVideoInterface>(input);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);

  auto output_mat = get_output_mat(input, output, prop);

  cv::Scalar fill_color;
  switch (AxVideoFormatNumChannels(input_video.info.format)) {
    case 1:
      fill_color = cv::Scalar(prop->padding);
      break;
    case 3:
      fill_color = cv::Scalar(prop->padding, prop->padding, prop->padding);
      break;
    case 4:
      fill_color = cv::Scalar(prop->padding, prop->padding, prop->padding, 255);
      break;
    default:
      logger(AX_ERROR) << "resize does not support video format: "
                       << AxVideoFormatToString(input_video.info.format) << std::endl;
      throw std::runtime_error("resize does not support this format");
  }

  if (!prop->letterbox) {
    cv::resize(input_mat, output_mat,
        cv::Size(output_mat.size().width, output_mat.size().height), cv::INTER_LINEAR);
    return;
  }
  bool scale_to_height
      = static_cast<double>(prop->width) / prop->height
        > static_cast<double>(input_video.info.width) / input_video.info.height;

  auto scale_factor = scale_to_height ?
                          static_cast<double>(prop->height) / input_mat.size().height :
                          static_cast<double>(prop->width) / input_mat.size().width;

  auto height = std::lround(input_mat.size().height * scale_factor);
  auto width = std::lround(input_mat.size().width * scale_factor);

  auto padding_top = std::lround((prop->height - height) / 2.);
  auto padding_left = std::lround((prop->width - width) / 2.);

  if (!prop->scale_up) {
    if (input_video.info.width < prop->width && input_video.info.height < prop->height) {
      width = input_video.info.width;
      height = input_video.info.height;
      padding_top = (prop->height - height) / 2;
      padding_left = (prop->width - width) / 2;
      auto whole_image = cv::Rect(0, 0, prop->width, prop->height);
      cv::Mat output_window(output_mat, cv::Rect(padding_left, padding_top, width, height));
      cv::rectangle(output_mat, whole_image, fill_color, cv::FILLED);
      cv::resize(input_mat, output_window, cv::Size(width, height), cv::INTER_LINEAR);
      return;
    }
  }

  cv::Mat output_window(output_mat, cv::Rect(padding_left, padding_top, width, height));

  auto top_left = scale_to_height ? cv::Rect(0, 0, padding_left, prop->height) :
                                    cv::Rect(0, 0, prop->width, padding_top);

  auto bottom_right = scale_to_height ? cv::Rect(padding_left + width, 0,
                          prop->width - padding_left - width, prop->height) :
                                        cv::Rect(0, padding_top + height, prop->width,
                                            prop->height - padding_top - height);

  cv::rectangle(output_mat, top_left, fill_color, cv::FILLED);
  cv::rectangle(output_mat, bottom_right, fill_color, cv::FILLED);
  cv::resize(input_mat, output_window, cv::Size(width, height), cv::INTER_LINEAR);
}
