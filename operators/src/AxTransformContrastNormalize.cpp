// Copyright Axelera AI, 2023
#include <unordered_map>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const void *, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)
      || !std::holds_alternative<AxTensorsInterface>(output)) {
    throw std::runtime_error("contrastnormalize works on video input and tensor output only");
  }

  auto &input_video = std::get<AxVideoInterface>(input);
  cv::Mat input_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_u8(input_video.info.format), input_video.data,
      input_video.info.stride);

  auto &output_tensors = std::get<AxTensorsInterface>(output);
  if (output_tensors.size() != 1) {
    logger(AX_ERROR) << "contrastnormalize returns single tensor only\n";
    throw std::runtime_error("contrastnormalize returns single tensor only");
  }

  auto &output_tensor = output_tensors[0];
  if (output_tensor.bytes != 4) {
    logger(AX_ERROR) << "contrastnormalize must return float32\n";
    throw std::runtime_error("contrastnormalize must return float32");
  }
  if (output_tensor.sizes[0] != 1) {
    logger(AX_ERROR) << "contrastnormalize does not return batched tensors\n";
    throw std::runtime_error("contrastnormalize does not return batched tensors");
  }
  if (output_tensor.sizes[1] != input_video.info.height) {
    logger(AX_ERROR) << "contrastnormalize height error\n";
    throw std::runtime_error("contrastnormalize height error");
  }
  if (output_tensor.sizes[2] != input_video.info.width) {
    logger(AX_ERROR) << "contrastnormalize width error\n";
    throw std::runtime_error("contrastnormalize width error");
  }
  if (output_tensor.sizes[3] != AxVideoFormatNumChannels(input_video.info.format)) {
    logger(AX_ERROR) << "contrastnormalize must return the same number of channels as the input format in NHWC format\n";
    throw std::runtime_error(
        "contrastnormalize must return the same number of channels as the input format in NHWC format");
  }
  cv::Mat output_mat(cv::Size(input_video.info.width, input_video.info.height),
      Ax::opencv_type_f32(input_video.info.format), output_tensor.data);

  double minVal;
  double maxVal;
  cv::minMaxLoc(input_mat.reshape(1, int(input_mat.total())), &minVal, &maxVal);

  double alpha = 2.0 / (maxVal - minVal);
  double beta = -(maxVal + minVal) / (maxVal - minVal);
  input_mat.convertTo(output_mat, CV_32F, alpha, beta);
}
