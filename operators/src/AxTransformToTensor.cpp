// Copyright Axelera AI, 2023
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

struct totensor_properties {
  std::string type{};
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "type" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<totensor_properties> prop = std::make_shared<totensor_properties>();
  prop->type = Ax::get_property(input, "type", "totensor_properties", prop->type);
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const totensor_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    throw std::runtime_error("totensor works on video input only");
  }
  auto &info = std::get<AxVideoInterface>(interface).info;
  AxDataInterface output = AxTensorsInterface(1);
  auto &tensor = std::get<AxTensorsInterface>(output)[0];
  if (prop->type == "int8") {
    tensor.bytes = 1;
    tensor.sizes = std::vector<int>{ 1, info.height, info.width,
      AxVideoFormatNumChannels(info.format) };
  }
  if (prop->type == "float32") {
    tensor.bytes = 4;
    tensor.sizes = std::vector<int>{ 1, AxVideoFormatNumChannels(info.format),
      info.height, info.width };
  }
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const totensor_properties *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto &input_video = std::get<AxVideoInterface>(input);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);

  auto &output_tensor = std::get<AxTensorsInterface>(output)[0];

  if (output_tensor.bytes == 1) {
    cv::Mat output_mat(cv::Size(output_tensor.sizes[2], output_tensor.sizes[1]),
        CV_MAKETYPE(CV_8U, output_tensor.sizes[3]), output_tensor.data);

    input_mat.copyTo(output_mat);
    return;
  }

  if (output_tensor.bytes == 4) {
    cv::Mat output_mat(output_tensor.sizes, CV_32FC1, output_tensor.data);

    cv::dnn::blobFromImage(input_mat, output_mat, 1.0 / 255.0);
    return;
  }

  throw std::runtime_error("Nonexisting datatype chosen in totensor.");
}
