// Copyright Axelera AI, 2023
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaBBox.hpp"
#include "AxUtils.hpp"

struct cropresize_properties {
  std::string meta_key;
  int width = 0;
  int height = 0;
  int respect_aspectratio = 0;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "meta_key",
    "width", "height", "respect_aspectratio" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<cropresize_properties> prop
      = std::make_shared<cropresize_properties>();
  prop->meta_key
      = Ax::get_property(input, "meta_key", "cropresize_properties", prop->meta_key);
  prop->width = Ax::get_property(input, "width", "cropresize_properties", prop->width);
  prop->height = Ax::get_property(input, "height", "cropresize_properties", prop->height);
  prop->respect_aspectratio = Ax::get_property(input, "respect_aspectratio",
      "cropresize_properties", prop->respect_aspectratio);
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const cropresize_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error("cropresize works on video input only");
  }
  AxDataInterface output = interface;
  auto &out_info = std::get<AxVideoInterface>(output).info;

  out_info.width = prop->width;
  out_info.height = prop->height;

  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const cropresize_properties *prop, unsigned int subframe_index, unsigned int subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  auto &input_video = std::get<AxVideoInterface>(input);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);

  cv::Mat input_cropped;
  if (prop->meta_key.empty()) {
    input_cropped = input_mat;
  } else {
    AxMetaBbox *box_meta = dynamic_cast<AxMetaBbox *>(map.at(prop->meta_key).get());
    if (box_meta == nullptr) {
      logger(AX_ERROR) << "cropresize has not been provided with AxMetaBbox\n";
      throw std::runtime_error("cropresize has not been provided with AxMetaBbox");
    }
    if (subframe_number == 0) {
      logger(AX_ERROR) << "cropresize must not get subframe number 0\n";
      throw std::runtime_error("cropresize must not get subframe number 0");
    }

    const auto &[x1, y1, x2, y2] = box_meta->get_box_xyxy(subframe_index);
    const auto w = x2 - x1;
    const auto h = y2 - y1;
    const float out_aspect_ratio = (float) prop->width / prop->height;
    const float in_aspect_ratio = (float) w / h;
    if (in_aspect_ratio == out_aspect_ratio || prop->respect_aspectratio == 0) {
      input_cropped = input_mat(cv::Range(y1, y2), cv::Range(x1, x2));
    } else if (in_aspect_ratio > out_aspect_ratio) {
      const auto xoffset = (w - (h * out_aspect_ratio)) / 2;
      input_cropped = input_mat(cv::Range(y1, y2), cv::Range(x1 + xoffset, x2 - xoffset));
    } else {
      const auto yoffset = (h - (w / out_aspect_ratio)) / 2;
      input_cropped
          = input_mat(cv::Range(y1 + yoffset, y2 - yoffset), cv::Range(x1, x2));
    }
  }

  auto &output_video = std::get<AxVideoInterface>(output);
  if (input_video.info.format != output_video.info.format)
    throw std::runtime_error("cropresize cannot do video format conversions");
  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);

  cv::InterpolationFlags interpolation_method = cv::INTER_LINEAR;
  cv::resize(input_cropped, output_mat,
      cv::Size(output_mat.size().width, output_mat.size().height), 0.0, 0.0,
      interpolation_method);
}
