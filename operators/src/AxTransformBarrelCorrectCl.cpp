// Copyright Axelera AI, 2026
#include <array>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxUtils.hpp"


class CLBarrelCorrect;
struct barrelcorrect_properties {
  int width{};
  int height{};
  int size{};
  std::vector<cl_float> camera_props;
  std::vector<cl_float> distort_coefs;
  bool normalised{ true };
  std::string out_format{};
  std::unique_ptr<CLBarrelCorrect> barrelcorrect;
};

const char *kernel_cl = R"##(



float2 barrel_distortion_correction(
    float x, float y, const float fx, const float fy, const float cx, const float cy,
    __constant const float *new_camera_props, __constant const float *coeffs)
{
    const float k1 = coeffs[0];
    const float k2 = coeffs[1];
    const float p1 = coeffs[2];
    const float p2 = coeffs[3];
    const float k3 = coeffs[4];
    const float fx_new = new_camera_props[0];
    const float fy_new = new_camera_props[1];
    const float cx_new = new_camera_props[2];
    const float cy_new = new_camera_props[3];
    // Convert pixel coordinates to normalized coordinates (x_n, y_n)
    const float x_n = (x - cx_new) / fx_new;
    const float y_n = (y - cy_new) / fy_new;

    const float r2 = x_n * x_n + y_n * y_n;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;

    // Radial and tangential distortion
    const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
    const float x_tangential = 2.0f * p1 * x_n * y_n + p2 * (r2 + 2.0f * x_n * x_n);
    const float y_tangential = p1 * (r2 + 2.0f * y_n * y_n) + 2.0f * p2 * x_n * y_n;
    const float x_distorted = mad(x_n, radial, x_tangential);
    const float y_distorted = mad(y_n, radial, y_tangential);

    // Map back to pixel coordinates in the original image
    const float u = mad(fx, x_distorted, cx);
    const float v = mad(fy, y_distorted, cy);
    return (float2)(u, v);
}

__kernel void rgb_barrel_correct_bl(__global const uchar *in, __global uchar *out, int in_width, int in_height, int out_width, int out_height,
                        int strideIn, int strideOut, const float x_scale, const float y_scale, const float fx, const float fy, const float cx, const float cy,
                        __constant const float *new_camera_props, __constant const float *coeffs, output_format format, uchar is_input_bgr) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float x = (col + 0.5F) * x_scale;
    float y = (row + 0.5F) * y_scale;

    float2 corrected = barrel_distortion_correction(x, y, fx, fy, cx, cy, new_camera_props, coeffs);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    __global uchar *p_in = advance_uchar_ptr(in, row * strideIn);

    rgb_image img = {in_width, in_height, strideIn, 0, 0};
    uchar4 pixel = rgb_sampler_bl(in, corrected.x, corrected.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(is_input_bgr ? pixel.zyx : pixel.xyz);
      return;
    }
    is_input_bgr ? vstore3((format == BGR_OUTPUT) ? pixel.xyz : pixel.zyx, col, prgb) : vstore3((format == BGR_OUTPUT) ? pixel.zyx : pixel.xyz, col, prgb);
}


__kernel void nv12_barrel_correct_bl(__global const uchar *in_y, __global uchar *out, int uv_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInUV, int strideOut,
                            const float x_scale, const float y_scale, const float fx, const float fy, const float cx, const float cy,
                            __constant const float *new_camera_props, __constant const float *coeffs, output_format format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float x = (col + 0.5F) * x_scale;
    float y = (row + 0.5F) * y_scale;

    float2 corrected = barrel_distortion_correction(x, y, fx, fy, cx, cy, new_camera_props, coeffs);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }
    __global uchar2 *in_uv = (__global uchar2 *)(in_y + uv_offset);
    int uvStrideI = strideInUV / sizeof(uchar2);
    nv12_image img = {in_width, in_height, strideInY, uvStrideI, 0, 0};
    uchar4 pixel = nv12_sampler(in_y, in_uv,  corrected.x, corrected.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
}

__kernel void i420_barrel_correct_bl(__global const uchar *in_y, __global uchar *out, int u_offset, int v_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInU, int strideInV, int strideOut,
                            const float x_scale, const float y_scale, const float fx, const float fy, const float cx, const float cy,
                            __constant const float *new_camera_props, __constant const float *coeffs, output_format format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float x = (col + 0.5F) * x_scale;
    float y = (row + 0.5F) * y_scale;

    float2 corrected = barrel_distortion_correction(x, y, fx, fy, cx, cy, new_camera_props, coeffs);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    __global const uchar *in_u = in_y + u_offset;
    __global const uchar *in_v = in_y + v_offset;

    i420_image img = {in_width, in_height, strideInY, strideInU, strideInV, 0, 0};
    uchar4 pixel = i420_sampler(in_y, in_u, in_v, corrected.x, corrected.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
  }

__kernel void yuyv_barrel_correct_bl(__global const uchar4 *in_y, __global uchar *out, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideOut, const float x_scale, const float y_scale,
                            const float fx, const float fy, const float cx, const float cy,
                            __constant const float *new_camera_props, __constant const float *coeffs, output_format format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float x = (col + 0.5F) * x_scale;
    float y = (row + 0.5F) * y_scale;
    int strideI = strideInY / sizeof(uchar4);

    float2 corrected = barrel_distortion_correction(x, y, fx, fy, cx, cy, new_camera_props, coeffs);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    yuyv_image img = {in_width, in_height, strideI, 0, 0};
    uchar4 pixel = yuyv_sampler(in_y, corrected.x, corrected.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
  }

__kernel void gray_barrel_correct_bl(__global const uchar *in, __global uchar *out, int in_width, int in_height, int out_width, int out_height,
                        int strideIn, int strideOut, const float x_scale, const float y_scale, const float fx, const float fy, const float cx, const float cy,
                        __constant const float *new_camera_props, __constant const float *coeffs) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }
    float x = (col + 0.5F) * x_scale;
    float y = (row + 0.5F) * y_scale;

    float2 corrected = barrel_distortion_correction(x, y, fx, fy, cx, cy, new_camera_props, coeffs);
    __global uchar* pgray = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      pgray[col] = 0;
      return;
    }

    gray8_image img = {in_width, in_height, strideIn, 0, 0};
    uchar pixel = gray8_sampler_bl(in, corrected.x, corrected.y, &img);
    pgray[col] = pixel;
  }

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;
class CLBarrelCorrect
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLBarrelCorrect(std::string source, opencl_details *context, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils() + source, context, logger),
        rgb_barrel_correct{ program.get_kernel("rgb_barrel_correct_bl") },
        nv12_barrel_correct{ program.get_kernel("nv12_barrel_correct_bl") },
        i420_barrel_correct{ program.get_kernel("i420_barrel_correct_bl") },
        yuyv_barrel_correct{ program.get_kernel("yuyv_barrel_correct_bl") }, gray_barrel_correct{
          program.get_kernel("gray_barrel_correct_bl")
        }
  {
  }

  int run_kernel(cl_kernel k, const buffer_details &out, buffer &outbuf)
  {
    size_t global_work_size[3] = { 1, 1, 1 };
    const int numpix_per_kernel = 1;
    global_work_size[0] = out.width;
    global_work_size[1] = out.height;
    error = program.execute_kernel(k, 2, global_work_size);
    if (auto *p = std::get_if<opencl_buffer *>(&out.data)) {
    } else {
      program.flush_output_buffer(outbuf, out.stride * out.height);
    }
    return error;
  }

  int run_kernel(cl_kernel k, const buffer_details &out, buffer &inbuf, buffer &outbuf)
  {
    return run_kernel(k, out, outbuf);
  }

  cv::Mat determine_optimal_matrix(const std::vector<cl_float> &distort_coefs,
      std::span<float> camera_props, const buffer_details &in)
  {
    //  Create the input matrix
    auto input_matrix = cv::Mat({ 3, 3 }, {
                                              camera_props[0],
                                              0.0F,
                                              camera_props[2],
                                              0.0F,
                                              camera_props[1],
                                              camera_props[3],
                                              0.0F,
                                              0.0F,
                                              1.0F,
                                          });
    // [TODO]
    // return cv::getOptimalNewCameraMatrix(input_matrix, distort_coefs,
    //     cv::Size(in.width, in.height), 0, cv::Size(in.width, in.height));
    return input_matrix;
  }

  int run(const buffer_details &in, const buffer_details &out,
      const barrelcorrect_properties &prop)
  {
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto width = prop.normalised ? in.width : 1.0F;
    auto height = prop.normalised ? in.height : 1.0F;
    auto original_camera_props = std::array<cl_float, 4>{
      prop.camera_props[0] * width,
      prop.camera_props[1] * height,
      prop.camera_props[2] * width,
      prop.camera_props[3] * height,
    };

    auto x_scale = static_cast<float>(in.width) / out.width;
    auto y_scale = static_cast<float>(in.height) / out.height;

    if (!distort_coeffs) {
      distort_coeffs = program.create_buffer(1,
          prop.distort_coefs.size() * sizeof(prop.distort_coefs[0]),
          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          const_cast<float *>(prop.distort_coefs.data()), 1);
    }

    if (!camera_props) {
      auto new_matrix
          = determine_optimal_matrix(prop.distort_coefs, original_camera_props, in);
      auto new_camera_props = std::array<cl_float, 4>{
        new_matrix.at<float>(0, 0),
        new_matrix.at<float>(1, 1),
        new_matrix.at<float>(0, 2),
        new_matrix.at<float>(1, 2),
      };

      camera_props = program.create_buffer(1,
          new_camera_props.size() * sizeof(new_camera_props[0]),
          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          const_cast<float *>(new_camera_props.data()), 1);
    }

    if (in.format == AxVideoFormat::RGB || in.format == AxVideoFormat::BGR) {
      auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int out_format = ax_utils::get_output_format(out.format);
      cl_uchar is_input_bgr = in.format == AxVideoFormat::BGR ? 1 : 0;
      program.set_kernel_args(*rgb_barrel_correct, 0, *inbuf, *outbuf, in.width,
          in.height, out.width, out.height, in.stride, out.stride, x_scale,
          y_scale, original_camera_props[0], original_camera_props[1],
          original_camera_props[2], original_camera_props[3], *camera_props,
          *distort_coeffs, out_format, is_input_bgr);
      return run_kernel(*rgb_barrel_correct, out, inbuf, outbuf);

    } else if (in.format == AxVideoFormat::NV12) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int uv_offset = in.offsets[1];
      cl_int uv_stride = in.strides[1];
      cl_int out_format = ax_utils::get_output_format(out.format);
      program.set_kernel_args(*nv12_barrel_correct, 0, *inbuf_y, *outbuf,
          uv_offset, in.width, in.height, out.width, out.height, in.stride,
          uv_stride, out.stride, x_scale, y_scale, original_camera_props[0],
          original_camera_props[1], original_camera_props[2],
          original_camera_props[3], *camera_props, *distort_coeffs, out_format);
      return run_kernel(*nv12_barrel_correct, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::I420) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int u_offset = in.offsets[1];
      cl_int u_stride = in.strides[1];
      cl_int v_offset = in.offsets[2];
      cl_int v_stride = in.strides[2];
      cl_int out_format = ax_utils::get_output_format(out.format);
      program.set_kernel_args(*i420_barrel_correct, 0, *inbuf_y, *outbuf, u_offset,
          v_offset, in.width, in.height, out.width, out.height, in.stride, u_stride,
          v_stride, out.stride, x_scale, y_scale, original_camera_props[0],
          original_camera_props[1], original_camera_props[2],
          original_camera_props[3], *camera_props, *distort_coeffs, out_format);

      return run_kernel(*i420_barrel_correct, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::YUY2) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
      cl_int out_format = ax_utils::get_output_format(out.format);
      cl_uchar is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(*yuyv_barrel_correct, 0, *inbuf_y, *outbuf, in.width,
          in.height, out.width, out.height, in.stride, out.stride, x_scale, y_scale,
          original_camera_props[0], original_camera_props[1], original_camera_props[2],
          original_camera_props[3], *camera_props, *distort_coeffs, out_format);

      return run_kernel(*yuyv_barrel_correct, out, inbuf_y, outbuf);
    } else if (in.format == AxVideoFormat::GRAY8) { // When the input is gray8 output could be only gray8
      auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
      program.set_kernel_args(*gray_barrel_correct, 0, *inbuf, *outbuf, in.width,
          in.height, out.width, out.height, in.stride, out.stride, x_scale, y_scale,
          original_camera_props[0], original_camera_props[1], original_camera_props[2],
          original_camera_props[3], *camera_props, *distort_coeffs);
      return run_kernel(*gray_barrel_correct, out, inbuf, outbuf);
    } else {
      throw std::runtime_error("Unsupported format: " + AxVideoFormatToString(in.format));
    }
    return {};
  }

  private:
  CLProgram program;
  int error{};
  kernel rgb_barrel_correct;
  kernel nv12_barrel_correct;
  kernel i420_barrel_correct;
  kernel yuyv_barrel_correct;
  kernel gray_barrel_correct;
  CLProgram::ax_buffer camera_props{ nullptr };
  CLProgram::ax_buffer distort_coeffs{ nullptr };
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "camera_props",
    "distort_coefs",
    "out_format",
    "format",
    "normalized_properties",
    "width",
    "height",
    "size",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<barrelcorrect_properties>();

  prop->camera_props = Ax::get_property(input, "camera_props",
      "barrelcorrect_static_properties", prop->camera_props);
  prop->distort_coefs = Ax::get_property(input, "distort_coefs",
      "barrelcorrect_static_properties", prop->distort_coefs);
  prop->out_format = Ax::get_property(
      input, "out_format", "barrelcorrect_dynamic_properties", prop->out_format);
  prop->out_format = Ax::get_property(
      input, "format", "barrelcorrect_dynamic_properties", prop->out_format);

  prop->normalised = Ax::get_property(input, "normalized_properties",
      "barrelcorrect_static_properties", prop->normalised);
  prop->size = Ax::get_property(input, "size", "barrelcorrect_static_properties", prop->size);
  prop->width = Ax::get_property(
      input, "width", "barrelcorrect_static_properties", prop->width);
  prop->height = Ax::get_property(
      input, "height", "barrelcorrect_static_properties", prop->height);

  constexpr auto camera_props_size = 4;
  if (prop->camera_props.size() != camera_props_size) {
    throw std::runtime_error("camera_props must have 4 values");
  }
  constexpr auto distort_coefs_size = 5;
  if (prop->distort_coefs.size() != distort_coefs_size) {
    throw std::runtime_error("distort_coefs must have 5 values");
  }
  prop->barrelcorrect = std::make_unique<CLBarrelCorrect>(
      kernel_cl, static_cast<opencl_details *>(context), logger);
  if (prop->size > 0 && (prop->width > 0 || prop->height > 0)) {
    throw std::runtime_error("You must provide only one of width/height or size");
  }

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    barrelcorrect_properties * /*prop*/, Ax::Logger & /*logger*/)
{
}

std::pair<int, int>
determine_width_height(const AxVideoInterface &in_info, int size)
{
  auto width = in_info.info.width;
  auto height = in_info.info.height;
  auto height_is_shortest = height < width;
  auto scale = height_is_shortest ? static_cast<double>(size) / height :
                                    static_cast<double>(size) / width;
  return { static_cast<int>(std::round(width * scale)),
    static_cast<int>(std::round(height * scale)) };
}

std::array valid_formats = {
  AxVideoFormat::RGB,
  AxVideoFormat::BGR,
  AxVideoFormat::GRAY8,
};

const char *name = "Barrel Correction";

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const barrelcorrect_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    auto width = in_info.info.width;
    int height = in_info.info.height;
    if (prop->size) {
      std::tie(width, height) = determine_width_height(in_info, prop->size);
    } else if (prop->width != 0 && prop->height != 0) {
      width = prop->width;
      height = prop->height;
    }
    out_info.info.width = width;
    out_info.info.height = height;
    out_info.info.actual_height = height;

    auto format = prop->out_format.empty() ? out_info.info.format :
                                             AxVideoFormatFromString(prop->out_format);
    out_info.info.format = format;
    Ax::validate_output_format(out_info.info.format, prop->out_format, name, valid_formats);
    output = out_info;
  }
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const barrelcorrect_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto in_info = std::get<AxVideoInterface>(input);
  auto out_info = std::get<AxVideoInterface>(output);

  //  Validate input and output formats

  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("resize works on single video input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("resize works on single video output only");
  }
  auto valid_formats = std::array{
    AxVideoFormat::RGB,
    AxVideoFormat::BGR,
    AxVideoFormat::NV12,
    AxVideoFormat::I420,
    AxVideoFormat::YUY2,
    AxVideoFormat::GRAY8,
  };
  if (std::none_of(valid_formats.begin(), valid_formats.end(), [input_details](auto format) {
        return format == input_details[0].format;
      })) {
    throw std::runtime_error("Barrel Correction does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  Ax::validate_output_format(output_details[0].format, prop->out_format, name, valid_formats);
  prop->barrelcorrect->run(input_details[0], output_details[0], *prop);
}

extern "C" int
query_supports(Ax::PluginFeature feature, const barrelcorrect_properties *prop,
    Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return 1;
  }
  return Ax::PluginFeatureDefaults(feature);
}
