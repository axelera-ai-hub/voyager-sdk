// Copyright Axelera AI, 2024
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxRoiCropCommon.h"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

#include <opencv2/core/ocl.hpp>

struct roicrop_properties : RoiCropParams {
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  return roicrop_allowed_properties();
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<roicrop_properties>();
  parse_roicrop_params(input, *prop, "roicrop", logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    roicrop_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input,
      "downstream_supports_opencl", "roicrop", prop->downstream_supports_opencl);
}

extern "C" AxDataInterface
set_output_interface_from_meta(const AxDataInterface &interface,
    const roicrop_properties *prop, unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  return roicrop_set_output_interface(interface, *prop, subframe_index,
      number_of_subframes, meta_map, "roicrop", logger);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const roicrop_properties *prop, unsigned int subframe_index, unsigned int subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  cv::ocl::setUseOpenCL(false);

  auto &input_video = std::get<AxVideoInterface>(input);
  auto &output_video = std::get<AxVideoInterface>(output);

  if (input_video.info.format != output_video.info.format)
    throw std::runtime_error("roicrop cannot do video format conversions");

  auto [x1, y1, x2, y2] = roicrop_get_roi_with_margin(
      *prop, subframe_index, subframe_number, map, "roicrop", logger);

  int in_width = input_video.info.width;
  int in_height = input_video.info.height;
  int cv_type = Ax::opencv_type_u8(input_video.info.format);

  cv::Mat input_mat(cv::Size(in_width, in_height), cv_type, input_video.data,
      input_video.info.stride);
  cv::Mat output_mat(cv::Size(output_video.info.width, output_video.info.height),
      cv_type, output_video.data, output_video.info.stride);

  // Clamp ROI to input image bounds
  int src_x1 = std::max(x1, 0);
  int src_y1 = std::max(y1, 0);
  int src_x2 = std::min(x2, in_width - 1);
  int src_y2 = std::min(y2, in_height - 1);

  if (x1 == src_x1 && y1 == src_y1 && x2 == src_x2 && y2 == src_y2) {
    // Fast path: entire ROI is within input bounds
    input_mat(cv::Rect(x1, y1, output_video.info.width, output_video.info.height))
        .copyTo(output_mat);
    return;
  }

  // Partial overlap: fill output with black then copy the valid intersection
  output_mat.setTo(cv::Scalar::all(0));

  if (src_x1 <= src_x2 && src_y1 <= src_y2) {
    int copy_w = src_x2 - src_x1 + 1;
    int copy_h = src_y2 - src_y1 + 1;
    input_mat(cv::Rect(src_x1, src_y1, copy_w, copy_h))
        .copyTo(output_mat(cv::Rect(src_x1 - x1, src_y1 - y1, copy_w, copy_h)));
  }
}

extern "C" bool
query_supports(Ax::PluginFeature feature, const roicrop_properties *prop, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return prop && prop->downstream_supports_opencl;
  }
  return false;
}
