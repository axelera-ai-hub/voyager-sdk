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


class CLPolarTransform;
struct polar_properties {
  int width{};
  int height{};
  int size{};
  float center_x{ 0.5f };
  float center_y{ 0.5f };
  float start_angle{ M_PI / 2.0f };
  bool rotate180{ true };
  float max_radius{};
  bool inverse{ false };
  bool linear_polar{ true };
  std::string format{};
  std::unique_ptr<CLPolarTransform> polar_transform;
};

const char *kernel_cl = R"##(

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

float2 polar_to_cartesian(float start_angle, float rho, float theta, float center_x, float center_y, float max_radius, int in_width, int in_height, int linear_polar)
{
    if (linear_polar) {
        // Linear polar mapping
        rho = rho * max_radius;
    } else {
        // Semi-log polar mapping
        rho = exp(rho * log(max_radius + 1.0f)) - 1.0f;
    }

    float x = center_x + rho * cos(theta-start_angle);
    float y = center_y + rho * sin(theta-start_angle);

    return (float2)(x, y);
}

float2 cartesian_to_polar(float start_angle, float x, float y, float center_x, float center_y, float max_radius, int out_width, int out_height, int linear_polar)
{
    float dx = x - center_x;
    float dy = y - center_y;

    float rho = sqrt(dx * dx + dy * dy);
    float theta = atan2(dy, dx);
    theta -= start_angle;

    // Normalize theta to [0, 2*PI]
    if (theta < 0.0f) theta += 2.0f * (float)M_PI;

    if (linear_polar) {
        // Linear polar mapping
        rho = rho / max_radius;
    } else {
        // Semi-log polar mapping
        rho = log(rho + 1.0f) / log(max_radius + 1.0f);
    }

    return (float2)(rho, theta);
}


float2 transform_coordinates(int row, int col, int in_width, int in_height, int out_width, int out_height,
                           float center_x, float center_y, float max_radius, int inverse, int linear_polar,
                           float start_angle, int rotate180) {
    float2 src_coords;
    if (!inverse) {
        // Polar to Cartesian (unwrap polar image back to cartesian)
        float rho = (float)row / (float)(out_height - 1);
        float theta = (float)col / (float)(out_width - 1) * 2.0f * (float)M_PI;
        if (rotate180) {
          rho = 1.0f - rho;
          theta += (float)M_PI;
          if (theta >= 2.0f * (float)M_PI) theta -= 2.0f * (float)M_PI;
        }
        src_coords = polar_to_cartesian(start_angle, rho, theta, center_x * in_width, center_y * in_height, max_radius, in_width, in_height, linear_polar);
    } else {
        // Cartesian to Polar (wrap cartesian image to polar)
        float x = (float)col;
        float y = (float)row;
        float2 polar = cartesian_to_polar(start_angle, x, y, center_x * out_width, center_y * out_height, max_radius, out_width, out_height, linear_polar);

        // Map polar coordinates to output space
        float rho_norm = polar.x;
        float theta_norm = polar.y / (2.0f * (float)M_PI);

        src_coords.x = theta_norm * (in_width - 1);
        src_coords.y = rho_norm * (in_height - 1);

        if (rotate180) {
            // Rotate input coordinates by 180 degrees
            src_coords.x = in_width - 1 - src_coords.x;
            src_coords.y = in_height - 1 - src_coords.y;
        }
    }
    return src_coords;
}

__kernel void rgb_polar_transform_bl(__global const uchar *in, __global uchar *out, int in_width, int in_height, int out_width, int out_height,
                        int strideIn, int strideOut, float center_x, float center_y, float max_radius, int inverse, int linear_polar, float start_angle, int rotate180,
                        output_format format, uchar is_input_bgr) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    float2 src_coords = transform_coordinates(row, col, in_width, in_height, out_width, out_height,
                           center_x, center_y, max_radius, inverse, linear_polar,
                           start_angle, rotate180);


    if (src_coords.x < 0 || src_coords.x >= in_width || src_coords.y < 0 || src_coords.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    rgb_image img = {in_width, in_height, strideIn, 0, 0};
    uchar4 pixel = rgb_sampler_bl(in, src_coords.x, src_coords.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(is_input_bgr ? pixel.zyx : pixel.xyz);
      return;
    }
    is_input_bgr ? vstore3((format == BGR_OUTPUT) ? pixel.xyz : pixel.zyx, col, prgb) : vstore3((format == BGR_OUTPUT) ? pixel.zyx : pixel.xyz, col, prgb);
}

__kernel void nv12_polar_transform_bl(__global const uchar *in_y, __global uchar *out, int uv_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInUV, int strideOut,
                            float center_x, float center_y, float max_radius, int inverse, int linear_polar, float start_angle, int rotate180, output_format format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    float2 src_coords = transform_coordinates(row, col, in_width, in_height, out_width, out_height,
                           center_x, center_y, max_radius, inverse, linear_polar,
                           start_angle, rotate180);


    if (src_coords.x < 0 || src_coords.x >= in_width || src_coords.y < 0 || src_coords.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0,0,0), col, prgb);
      return;
    }

    __global uchar2 *in_uv = (__global uchar2 *)(in_y + uv_offset);
    int uvStrideI = strideInUV / sizeof(uchar2);
    nv12_image img = {in_width, in_height, strideInY, uvStrideI, 0, 0};
    uchar4 pixel = nv12_sampler(in_y, in_uv, src_coords.x, src_coords.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
}

__kernel void i420_polar_transform_bl(__global const uchar *in_y, __global uchar *out, int u_offset, int v_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInU, int strideInV, int strideOut,
                            float center_x, float center_y, float max_radius, int inverse, int linear_polar, float start_angle, int rotate180, output_format format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    float2 src_coords = transform_coordinates(row, col, in_width, in_height, out_width, out_height,
                           center_x, center_y, max_radius, inverse, linear_polar,
                           start_angle, rotate180);

    if (src_coords.x < 0 || src_coords.x >= in_width || src_coords.y < 0 || src_coords.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    __global const uchar *in_u = in_y + u_offset;
    __global const uchar *in_v = in_y + v_offset;

    i420_image img = {in_width, in_height, strideInY, strideInU, strideInV, 0, 0};
    uchar4 pixel = i420_sampler(in_y, in_u, in_v, src_coords.x, src_coords.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
  }

__kernel void yuyv_polar_transform_bl(__global const uchar4 *in_y, __global uchar *out, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideOut,
                            float center_x, float center_y, float max_radius, int inverse, int linear_polar, float start_angle, int rotate180, output_format format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    int strideI = strideInY / sizeof(uchar4);

    float2 src_coords = transform_coordinates(row, col, in_width, in_height, out_width, out_height,
                           center_x, center_y, max_radius, inverse, linear_polar,
                           start_angle, rotate180);

    if (src_coords.x < 0 || src_coords.x >= in_width || src_coords.y < 0 || src_coords.y >= in_height) {
      format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    yuyv_image img = {in_width, in_height, strideI, 0, 0};
    uchar4 pixel = yuyv_sampler(in_y, src_coords.x, src_coords.y, &img);
    if (format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
  }

__kernel void gray_polar_transform_bl(__global const uchar *in, __global uchar *out, int in_width, int in_height, int out_width, int out_height,
                        int strideIn, int strideOut, float center_x, float center_y, float max_radius, int inverse, int linear_polar, float start_angle, int rotate180) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= out_height || col >= out_width) {
      return;
    }

    __global uchar* pgray = advance_uchar_ptr(out, row * strideOut);
    float2 src_coords = transform_coordinates(row, col, in_width, in_height, out_width, out_height,
                           center_x, center_y, max_radius, inverse, linear_polar,
                           start_angle, rotate180);

    if (src_coords.x < 0 || src_coords.x >= in_width || src_coords.y < 0 || src_coords.y >= in_height) {
      pgray[col] = 0;
      return;
    }

    gray8_image img = {in_width, in_height, strideIn, 0, 0};
    uchar pixel = gray8_sampler_bl(in, src_coords.x, src_coords.y, &img);
    pgray[col] = pixel;
  }

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;
class CLPolarTransform
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLPolarTransform(std::string source, opencl_details *context, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils() + source, context, logger),
        rgb_polar_transform{ program.get_kernel("rgb_polar_transform_bl") },
        nv12_polar_transform{ program.get_kernel("nv12_polar_transform_bl") },
        i420_polar_transform{ program.get_kernel("i420_polar_transform_bl") },
        yuyv_polar_transform{ program.get_kernel("yuyv_polar_transform_bl") }, gray_polar_transform{
          program.get_kernel("gray_polar_transform_bl")
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

  float calculate_max_radius(const buffer_details &in, float center_x, float center_y)
  {
    float cx = center_x * in.width;
    float cy = center_y * in.height;

    // Calculate distance to corners
    float d1 = sqrt(cx * cx + cy * cy);
    float d2 = sqrt((in.width - cx) * (in.width - cx) + cy * cy);
    float d3 = sqrt(cx * cx + (in.height - cy) * (in.height - cy));
    float d4 = sqrt(
        (in.width - cx) * (in.width - cx) + (in.height - cy) * (in.height - cy));

    return fmax(fmax(d1, d2), fmax(d3, d4));
  }

  int run(const buffer_details &in, const buffer_details &out, const polar_properties &prop)
  {
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_float max_radius = prop.max_radius;
    if (max_radius == 0.0f) {
      max_radius = calculate_max_radius(in, prop.center_x, prop.center_y);
    }

    if (in.format == AxVideoFormat::RGB || in.format == AxVideoFormat::BGR) {
      auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int out_format = ax_utils::get_output_format(out.format);
      cl_uchar is_input_bgr = in.format == AxVideoFormat::BGR ? 1 : 0;
      program.set_kernel_args(*rgb_polar_transform, 0, *inbuf, *outbuf,
          in.width, in.height, out.width, out.height, in.stride, out.stride,
          static_cast<cl_float>(prop.center_x), static_cast<cl_float>(prop.center_y),
          max_radius, static_cast<cl_int>(prop.inverse),
          static_cast<cl_int>(prop.linear_polar), static_cast<cl_float>(prop.start_angle),
          static_cast<cl_int>(prop.rotate180), out_format, is_input_bgr);
      return run_kernel(*rgb_polar_transform, out, inbuf, outbuf);

    } else if (in.format == AxVideoFormat::NV12) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int uv_offset = in.offsets[1];
      cl_int uv_stride = in.strides[1];
      cl_int out_format = ax_utils::get_output_format(out.format);
      program.set_kernel_args(*nv12_polar_transform, 0, *inbuf_y, *outbuf,
          uv_offset, in.width, in.height, out.width, out.height, in.stride,
          uv_stride, out.stride, static_cast<cl_float>(prop.center_x),
          static_cast<cl_float>(prop.center_y), max_radius,
          static_cast<cl_int>(prop.inverse), static_cast<cl_int>(prop.linear_polar),
          static_cast<cl_float>(prop.start_angle),
          static_cast<cl_int>(prop.rotate180), out_format);
      return run_kernel(*nv12_polar_transform, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::I420) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int u_offset = in.offsets[1];
      cl_int u_stride = in.strides[1];
      cl_int v_offset = in.offsets[2];
      cl_int v_stride = in.strides[2];
      cl_int out_format = ax_utils::get_output_format(out.format);
      program.set_kernel_args(*i420_polar_transform, 0, *inbuf_y, *outbuf,
          u_offset, v_offset, in.width, in.height, out.width, out.height, in.stride,
          u_stride, v_stride, out.stride, static_cast<cl_float>(prop.center_x),
          static_cast<cl_float>(prop.center_y), max_radius,
          static_cast<cl_int>(prop.inverse), static_cast<cl_int>(prop.linear_polar),
          static_cast<cl_float>(prop.start_angle),
          static_cast<cl_int>(prop.rotate180), out_format);

      return run_kernel(*i420_polar_transform, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::YUY2) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
      cl_int out_format = ax_utils::get_output_format(out.format);
      cl_uchar is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(*yuyv_polar_transform, 0, *inbuf_y, *outbuf,
          in.width, in.height, out.width, out.height, in.stride, out.stride,
          static_cast<cl_float>(prop.center_x), static_cast<cl_float>(prop.center_y),
          max_radius, static_cast<cl_int>(prop.inverse),
          static_cast<cl_int>(prop.linear_polar), static_cast<cl_float>(prop.start_angle),
          static_cast<cl_int>(prop.rotate180), out_format);

      return run_kernel(*yuyv_polar_transform, out, inbuf_y, outbuf);
    } else if (in.format == AxVideoFormat::GRAY8) {
      auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
      program.set_kernel_args(*gray_polar_transform, 0, *inbuf, *outbuf,
          in.width, in.height, out.width, out.height, in.stride, out.stride,
          static_cast<cl_float>(prop.center_x),
          static_cast<cl_float>(prop.center_y), max_radius,
          static_cast<cl_int>(prop.inverse), static_cast<cl_int>(prop.linear_polar),
          static_cast<cl_float>(prop.start_angle), static_cast<cl_int>(prop.rotate180));
      return run_kernel(*gray_polar_transform, out, inbuf, outbuf);

    } else {
      throw std::runtime_error("Unsupported format: " + AxVideoFormatToString(in.format));
    }
    return {};
  }

  private:
  CLProgram program;
  int error{};
  kernel rgb_polar_transform;
  kernel nv12_polar_transform;
  kernel i420_polar_transform;
  kernel yuyv_polar_transform;
  kernel gray_polar_transform;
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "center_x",
    "center_y", "max_radius", "inverse", "linear_polar", "format", "width",
    "height", "size", "start_angle", "rotate180" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<polar_properties>();

  prop->center_x
      = Ax::get_property(input, "center_x", "polar_static_properties", prop->center_x);
  prop->center_y
      = Ax::get_property(input, "center_y", "polar_static_properties", prop->center_y);
  prop->max_radius = Ax::get_property(
      input, "max_radius", "polar_static_properties", prop->max_radius);
  prop->inverse
      = Ax::get_property(input, "inverse", "polar_static_properties", prop->inverse);
  prop->linear_polar = Ax::get_property(
      input, "linear_polar", "polar_static_properties", prop->linear_polar);
  prop->format = Ax::get_property(input, "format", "polar_dynamic_properties", prop->format);
  prop->size = Ax::get_property(input, "size", "polar_static_properties", prop->size);
  prop->width = Ax::get_property(input, "width", "polar_static_properties", prop->width);
  prop->height = Ax::get_property(input, "height", "polar_static_properties", prop->height);
  prop->start_angle = Ax::get_property(
      input, "start_angle", "polar_static_properties", prop->start_angle);
  prop->rotate180 = Ax::get_property(
      input, "rotate180", "polar_static_properties", prop->rotate180);

  prop->polar_transform = std::make_unique<CLPolarTransform>(
      kernel_cl, static_cast<opencl_details *>(context), logger);
  if (prop->size > 0 && (prop->width > 0 || prop->height > 0)) {
    throw std::runtime_error("You must provide only one of width/height or size");
  }

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    polar_properties * /*prop*/, Ax::Logger & /*logger*/)
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

const char *name = "Polar Transform";

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const polar_properties *prop, Ax::Logger &logger)
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

    auto format = prop->format.empty() ? out_info.info.format :
                                         AxVideoFormatFromString(prop->format);
    out_info.info.format = format;
    Ax::validate_output_format(out_info.info.format, prop->format, name, valid_formats);
    output = out_info;
  }
  return output;
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const polar_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto in_info = std::get<AxVideoInterface>(input);
  auto out_info = std::get<AxVideoInterface>(output);

  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("polar transform works on single video input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("polar transform works on single video output only");
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
    throw std::runtime_error("Polar Transform does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  Ax::validate_output_format(output_details[0].format, prop->format, name, valid_formats);
  prop->polar_transform->run(input_details[0], output_details[0], *prop);
}

extern "C" int
query_supports(Ax::PluginFeature feature, const polar_properties *prop, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return 1;
  }
  return 0;
}
