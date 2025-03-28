// Copyright Axelera AI, 2023
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaBBox.hpp"
#include "AxUtils.hpp"

struct croproi_properties {
  std::string meta_key;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "meta_key",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<croproi_properties>();
  prop->meta_key = Ax::get_property(input, "meta_key", "croproi_properties", prop->meta_key);
  return prop;
}

extern "C" AxDataInterface
set_output_interface_from_meta(const AxDataInterface &interface,
    const croproi_properties *prop, unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error("croproi works on video input only");
  }
  AxDataInterface output = interface;
  auto &out_info = std::get<AxVideoInterface>(output).info;

  auto input = std::get<AxVideoInterface>(interface);

  if (!prop->meta_key.empty()) {
    AxMetaBbox *box_meta
        = dynamic_cast<AxMetaBbox *>(meta_map.at(prop->meta_key).get());
    if (box_meta == nullptr) {
      logger(AX_ERROR) << "croproi has not been provided with AxMetaBbox" << std::endl;
      throw std::runtime_error("croproi has not been provided with AxMetaBbox");
    }
    if (number_of_subframes <= subframe_index) {
      logger(AX_ERROR)
          << "Subframe index must be less than number of subframes" << std::endl;
      throw std::runtime_error("Subframe index must be less than number of subframes");
    }

    const auto &[x1, y1, x2, y2] = box_meta->get_box_xyxy(subframe_index);
    out_info.width = 1 + x2 - x1;
    out_info.height = 1 + y2 - y1;
    out_info.x_offset = x1;
    out_info.y_offset = y1;
    out_info.cropped = true;
  }
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const croproi_properties *prop, unsigned int subframe_index, unsigned int subframe_number,
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
    auto output = set_output_interface_from_meta(
        input, prop, subframe_index, subframe_number, map, logger);
    auto &out_info = std::get<AxVideoInterface>(output).info;

    cv::Rect crop_rect(
        out_info.x_offset, out_info.y_offset, out_info.width, out_info.height);
    input_cropped = input_mat(crop_rect);
  }

  auto &output_video = std::get<AxVideoInterface>(output);
  if (input_video.info.format != output_video.info.format)
    throw std::runtime_error("croproi cannot do video format conversions");

  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      Ax::opencv_type_u8(output_video.info.format), output_video.data,
      output_video.info.stride);

  input_cropped.copyTo(output_mat);
}
