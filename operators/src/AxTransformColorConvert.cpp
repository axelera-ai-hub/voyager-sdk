// Copyright Axelera AI, 2024
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

#include <opencv2/core/ocl.hpp>

struct cc_ocv_properties {
  AxVideoFormat format{ AxVideoFormat::UNDEFINED };
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "format",
  };
  return allowed_properties;
}

struct {
  std::string color;
  AxVideoFormat format;
} supported_output_formats[] = {
  { "rgba", AxVideoFormat::RGBA },
  { "bgra", AxVideoFormat::BGRA },
  { "rgb", AxVideoFormat::RGB },
  { "bgr", AxVideoFormat::BGR },
  { "gray", AxVideoFormat::GRAY8 },
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<cc_ocv_properties>();
  std::string fmt_string
      = Ax::get_property(input, "format", "ColorConvertProperties", std::string{});
  auto fmt_itr = std::find_if(std::begin(supported_output_formats),
      std::end(supported_output_formats),
      [&](auto f) { return f.color == fmt_string; });
  if (fmt_itr == std::end(supported_output_formats)) {
    logger(AX_ERROR) << "Invalid output format given in color conversion: " << fmt_string
                     << std::endl;
    throw std::runtime_error("Invalid output format given in color conversion: " + fmt_string);
  }
  prop->format = fmt_itr->format;
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const cc_ocv_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    logger(AX_ERROR) << "color_convert works on video input only" << std::endl;
    throw std::runtime_error("color_convert works on video input only");
  }
  auto out = std::get<AxVideoInterface>(interface);
  out.info.format = prop->format;
  return out;
}

/// @brief  Check if the plugin has any work to do
/// @param input
/// @param output
/// @param logger
/// @return true if the plugin can pass through the input to output
extern "C" bool
can_passthrough(const AxDataInterface &input, const AxDataInterface &output,
    const cc_ocv_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)) {
    throw std::runtime_error("color_convert works on video input only");
  }

  if (!std::holds_alternative<AxVideoInterface>(output)) {
    throw std::runtime_error("color_convert works on video input only");
  }
  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("color_convert works on single video (possibly batched) input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("color_convert works on single video (possibly batched) output only");
  }
  return (input_details[0].format == output_details[0].format);
}

static cv::Mat
pack_y42b_to_yuyv(const AxVideoInterface &in_video)
{
  const int w = in_video.info.width;
  const int h = in_video.info.height;
  const int ystride = in_video.strides.size() > 0 ? in_video.strides[0] : w;
  const int ustride = in_video.strides.size() > 1 ? in_video.strides[1] : w / 2;
  const int vstride = in_video.strides.size() > 2 ? in_video.strides[2] : w / 2;
  const auto *base = static_cast<const uint8_t *>(in_video.data);
  const auto *y_plane = base + (in_video.offsets.size() > 0 ? in_video.offsets[0] : 0);
  const auto *u_plane
      = base + (in_video.offsets.size() > 1 ? in_video.offsets[1] : ystride * h);
  const auto *v_plane
      = base + (in_video.offsets.size() > 2 ? in_video.offsets[2] : ystride * h + ustride * h);
  cv::Mat yuyv(h, w, CV_8UC2);
  for (int row = 0; row < h; ++row) {
    const uint8_t *y = y_plane + row * ystride;
    const uint8_t *u = u_plane + row * ustride;
    const uint8_t *v = v_plane + row * vstride;
    auto *dst = yuyv.ptr<uint8_t>(row);
    for (int col = 0; col < w; col += 2) {
      dst[2 * col] = y[col];
      dst[2 * col + 1] = u[col / 2];
      dst[2 * col + 2] = y[col + 1];
      dst[2 * col + 3] = v[col / 2];
    }
  }
  return yuyv;
}

cv::Mat
make_contiguous_if_needed(const AxVideoInterface &in_video, Ax::Logger &logger)
{
  if (in_video.info.format == AxVideoFormat::NV12) {
    if (in_video.offsets.size() != 2 || in_video.strides.size() != 2) {
      throw std::runtime_error("NV12 input has unrecognised number of offsets or strides (not 2)");
    }
    if (in_video.offsets[1] != in_video.strides[0] * in_video.info.height) {
      //  Here we need to make the image contiguous
      std::vector<cv::Mat> planes{
        { cv::Size(in_video.info.width, in_video.info.height), CV_8UC1,
            (uint8_t *) in_video.data + in_video.offsets[0], in_video.strides[0] },
        { cv::Size(in_video.info.width, in_video.info.height / 2), CV_8UC1,
            (uint8_t *) in_video.data + in_video.offsets[1], in_video.strides[1] },
      };
      cv::Mat contiguous;
      logger(AX_WARN) << "Performance issue: Concatenating non contiguous NV12 planes"
                      << std::endl;
      cv::vconcat(planes, contiguous);
      return contiguous;
    } else {
      return cv::Mat(cv::Size(in_video.info.width, in_video.info.height * 3 / 2),
          CV_8UC1, in_video.data, in_video.strides[0]);
    }
  } else if (in_video.info.format == AxVideoFormat::I420) {
    if (in_video.offsets.size() != 3 || in_video.strides.size() != 3) {
      throw std::runtime_error("I420 input has unrecognised number of offsets or strides (not 3)");
    }
    if (in_video.strides[0] != in_video.info.stride
        || in_video.strides[1] != in_video.strides[0] / 2
        || in_video.strides[2] != in_video.strides[0] / 2) {
      throw std::runtime_error(
          "OpenCV color conversion does not support non-standard I420 strides");
    }
    if (in_video.offsets[1] != in_video.strides[0] * in_video.info.height
        || in_video.offsets[2] != in_video.strides[0] * in_video.info.height * 5 / 4) {
      //  Here we need to make the image contiguous
      std::vector<cv::Mat> planes{
        { cv::Size(in_video.info.width, in_video.info.height), CV_8UC1,
            (uint8_t *) in_video.data + in_video.offsets[0], in_video.strides[0] },
        { cv::Size(in_video.info.width / 2, in_video.info.height / 2), CV_8UC1,
            (uint8_t *) in_video.data + in_video.offsets[1], in_video.strides[1] },
        { cv::Size(in_video.info.width / 2, in_video.info.height / 2), CV_8UC1,
            (uint8_t *) in_video.data + in_video.offsets[2], in_video.strides[2] },
      };
      cv::Mat contiguous;
      logger(AX_WARN) << "Performance issue: Concatenating non contiguous I420 planes"
                      << std::endl;
      cv::vconcat(planes, contiguous);
      return contiguous;
    } else {
      return cv::Mat(cv::Size(in_video.info.width, in_video.info.height * 3 / 2),
          CV_8UC1, in_video.data, in_video.strides[0]);
    }
  } else {
    if (in_video.strides.size() != 1) {
      throw std::runtime_error("OpenCV color conversion does not support multiple strides");
    }
    if (in_video.strides[0] != in_video.info.stride) {
      throw std::runtime_error(
          "OpenCV color conversion does not support inconsistent input stride");
    }
  }
  int input_opencv_type = CV_MAKETYPE(CV_8U, 1);
  return cv::Mat(cv::Size(in_video.info.width, in_video.info.height),
      input_opencv_type, in_video.data, in_video.strides[0]);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const cc_ocv_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  auto in_video = std::get<AxVideoInterface>(input);
  auto out_video = std::get<AxVideoInterface>(output);
  if (in_video.info.cropped) {
    throw std::runtime_error("OpenCV color conversion does not support cropped input");
  }
  if (in_video.info.format == AxVideoFormat::NV16) {
    throw std::runtime_error(
        "OpenCV color conversion does not support NV16 input format. Use colorconvert_cl (OpenCL) operator instead.");
  }
  if (in_video.info.format == AxVideoFormat::NV12) {
    if (in_video.offsets.size() != 2 || in_video.strides.size() != 2) {
      throw std::runtime_error("NV12 input has unrecognised number of offsets or strides (not 2)");
    }

  } else if (in_video.info.format == AxVideoFormat::I420) {
    if (in_video.offsets.size() != 3 || in_video.strides.size() != 3) {
      throw std::runtime_error("I420 input has unrecognised number of offsets or strides (not 3)");
    }
  } else if (in_video.info.format == AxVideoFormat::Y42B) {
    if (in_video.offsets.size() != 3 || in_video.strides.size() != 3) {
      throw std::runtime_error("Y42B input has unrecognised number of offsets or strides (not 3)");
    }
  } else {
    if (in_video.strides.size() != 1) {
      throw std::runtime_error("OpenCV color conversion does not support multiple strides");
    }
    if (in_video.strides[0] != in_video.info.stride) {
      throw std::runtime_error(
          "OpenCV color conversion does not support inconsistent input stride");
    }
    for (size_t i = 0; i < in_video.offsets.size(); i++) {
      if (in_video.offsets[i] != 0) {
        throw std::runtime_error("OpenCV color conversion does not support non-zero offset "
                                 + std::to_string(in_video.offsets[i])
                                 + " at plane " + std::to_string(i));
      }
    }
  }
  cv::ocl::setUseOpenCL(false);
  AxVideoFormat in_format = in_video.info.format;
  int input_opencv_type = Ax::opencv_type_u8(in_video.info.format);
  int height = in_video.info.height;
  int stride = in_video.info.stride;
  cv::Mat input_mat;
  if (in_video.info.format == AxVideoFormat::YUY2) {
    input_opencv_type = CV_MAKETYPE(CV_8U, 2);
    input_mat = cv::Mat(cv::Size(in_video.info.width, height),
        input_opencv_type, in_video.data, stride);
  } else if (in_video.info.format == AxVideoFormat::NV12
             || in_video.info.format == AxVideoFormat::I420) {

    input_mat = make_contiguous_if_needed(in_video, logger);
  } else if (in_video.info.format == AxVideoFormat::Y42B) {
    input_mat = pack_y42b_to_yuyv(in_video);
  } else {
    input_mat = cv::Mat(cv::Size(in_video.info.width, height),
        input_opencv_type, in_video.data, stride);
  }

  AxVideoFormat out_format = out_video.info.format;
  cv::Mat output_mat(cv::Size(out_video.info.width, out_video.info.height),
      Ax::opencv_type_u8(out_format), out_video.data, out_video.info.stride);

  cv::cvtColor(input_mat, output_mat, Ax::Internal::format2format(in_format, out_format));
}

extern "C" bool
query_supports(Ax::PluginFeature feature, const void *resize_properties, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::video_meta) {
    return false;
  }
  return Ax::PluginFeatureDefaults(feature);
}
