// Copyright Axelera AI, 2023
#include <array>
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
  std::vector<cl_float> camera_props;
  std::vector<cl_float> distort_coefs;
  int bgra_out{ 1 };
  std::unique_ptr<CLBarrelCorrect> barrelcorrect;
};

const char *kernel_cl = R"##(

float2 barrel_distortion_correction(
    int x, int y, const float fx, const float fy, const float cx, const float cy,
    const float k1, const float k2, const float k3, const float p1, const float p2)
{
    x += 0.5F;
    y += 0.5F;
    float u = (x - cx) / fx;
    float v = (y - cy) / fy;

    float r2 = u * u + v * v;
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    float radial_distortion = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

    float x_tangential = 2.0f * p1 * u * v + p2 * (r2 + 2.0f * u * u);
    float y_tangential = p1 * (r2 + 2.0f * v * v) + 2.0f * p2 * u * v;

    float u_distorted = mad(u, radial_distortion, x_tangential);
    float v_distorted = mad(v, radial_distortion, y_tangential);

    float x_distorted = mad(u_distorted, fx, cx);
    float y_distorted = mad(v_distorted, fy, cy);

    return (float2)(x_distorted, y_distorted);

}

__kernel void rgba_barrel_correct_bl(__global const uchar4 *in, __global uchar4 *out, int in_width, int in_height, int out_width, int out_height,
                        int strideIn, int strideOut, const float fx, const float fy, const float cx, const float cy,
                            const float k1, const float k2, const float k3, const float p1, const float p2, uchar is_bgr) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int strideO = strideOut / sizeof(uchar4);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = barrel_distortion_correction(col, row, fx, fy, cx, cy, k1, k2, k3, p1, p2);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out[row * strideO + col] = (uchar4)(0, 0, 0, 255);
      return;
    }

    int strideI = strideIn / sizeof(uchar4);
    rgb_image img = {in_width, in_height, strideI, 0, 0};
    uchar4 pixel = rgb_sampler_bl(in, corrected.x, corrected.y, &img);
    out[row * strideO + col] = is_bgr ? pixel.zyxw : pixel;
}

__kernel void nv12_barrel_correct_bl(__global const uchar *in_y, __global uchar4 *out, int uv_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInUV, int strideOut,
                            const float fx, const float fy, const float cx, const float cy,
                            const float k1, const float k2, const float k3, const float p1, const float p2, uchar is_bgr) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int strideO = strideOut / sizeof(uchar4);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = barrel_distortion_correction(col, row, fx, fy, cx, cy, k1, k2, k3, p1, p2);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out[row * strideO + col] = (uchar4)(0, 0, 0, 255);
      return;
    }
    __global uchar2 *in_uv = (__global uchar2 *)(in_y + uv_offset);
    int uvStrideI = strideInUV / sizeof(uchar2);
    nv12_image img = {in_width, in_height, strideInY, uvStrideI, 0, 0};
    uchar4 pixel = nv12_sampler(in_y, in_uv,  corrected.x, corrected.y, &img);

    out[row * strideO + col] = is_bgr ? pixel.zyxw : pixel;
}

__kernel void i420_barrel_correct_bl(__global const uchar *in_y, __global uchar4 *out, int u_offset, int v_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInU, int strideInV, int strideOut,
                            const float fx, const float fy, const float cx, const float cy,
                            const float k1, const float k2, const float k3, const float p1, const float p2, uchar is_bgr) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int strideO = strideOut / sizeof(uchar4);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = barrel_distortion_correction(col, row, fx, fy, cx, cy, k1, k2, k3, p1, p2);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out[row * strideO + col] = (uchar4)(0, 0, 0, 255);
      return;
    }

    __global const uchar *in_u = in_y + u_offset;
    __global const uchar *in_v = in_y + v_offset;

    i420_image img = {in_width, in_height, strideInY, strideInU, strideInV, 0, 0};
    uchar4 pixel = i420_sampler(in_y, in_u, in_v, corrected.x, corrected.y, &img);
    out[row * strideO + col] = is_bgr ? pixel.zyxw : pixel;
}

__kernel void yuyv_barrel_correct_bl(__global const uchar4 *in_y, __global uchar4 *out, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideOut,
                            const float fx, const float fy, const float cx, const float cy,
                            const float k1, const float k2, const float k3, const float p1, const float p2, uchar is_bgr) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int strideO = strideOut / sizeof(uchar4);
    int strideI = strideInY / sizeof(uchar4);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = barrel_distortion_correction(col, row, fx, fy, cx, cy, k1, k2, k3, p1, p2);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out[row * strideO + col] = (uchar4)(0, 0, 0, 255);
      return;
    }

    yuyv_image img = {in_width, in_height, strideI, 0, 0};
    uchar4 pixel = yuyv_sampler(in_y, corrected.x, corrected.y, &img);
    out[row * strideO + col] = is_bgr ? pixel.zyxw : pixel;
}

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
class CLBarrelCorrect
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLBarrelCorrect(std::string source, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils() + source, logger),
        rgba_barrel_correct{ program.get_kernel("rgba_barrel_correct_bl") },
        nv12_barrel_correct{ program.get_kernel("nv12_barrel_correct_bl") },
        i420_barrel_correct{ program.get_kernel("i420_barrel_correct_bl") }, yuyv_barrel_correct{
          program.get_kernel("yuyv_barrel_correct_bl")
        }
  {
  }

  CLProgram::flush_details run_kernel(
      const kernel &kernel, const buffer_details &out, const buffer &outbuf)
  {
    size_t global_work_size[3] = { 1, 1, 1 };
    global_work_size[0] = out.width;
    global_work_size[1] = out.height;
    error = program.execute_kernel(kernel, 2, global_work_size);
    if (error != CL_SUCCESS) {
      throw std::runtime_error(
          "Unable to execute kernel. Error code: " + std::to_string(error));
    }
    return program.flush_output_buffer_async(outbuf, ax_utils::determine_buffer_size(out));
  }

  std::function<void()> run_kernel(
      kernel &k, const buffer_details &out, buffer &inbuf, buffer &outbuf)
  {
    auto [error, event, mapped] = run_kernel(k, out, outbuf);
    if (error != CL_SUCCESS) {
      throw std::runtime_error(
          "Unable to map output buffer, error = " + std::to_string(error));
    }
    return [this, event, mapped, inbuf, outbuf]() {
      program.unmap_buffer(event, outbuf, mapped);
    };
  }


  std::function<void()> run(const buffer_details &in, const buffer_details &out,
      const barrelcorrect_properties &prop)
  {
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    auto camera_props = std::array<cl_float, 4>{
      prop.camera_props[0] * in.width,
      prop.camera_props[1] * in.height,
      prop.camera_props[2] * in.width,
      prop.camera_props[3] * in.height,
    };

    if (in.format == AxVideoFormat::RGBA || in.format == AxVideoFormat::BGRA) {
      auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_uchar is_bgr = in.format != out.format;
      program.set_kernel_args(rgba_barrel_correct, 0, *inbuf, *outbuf, in.width,
          in.height, out.width, out.height, in.stride, out.stride,
          camera_props[0], camera_props[1], camera_props[2], camera_props[3],
          prop.distort_coefs[0], prop.distort_coefs[1], prop.distort_coefs[2],
          prop.distort_coefs[3], prop.distort_coefs[4], is_bgr);

      return run_kernel(rgba_barrel_correct, out, inbuf, outbuf);

    } else if (in.format == AxVideoFormat::NV12) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int uv_offset = in.offsets[1];
      cl_int uv_stride = in.strides[1];
      cl_uchar is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(nv12_barrel_correct, 0, *inbuf_y, *outbuf, uv_offset,
          in.width, in.height, out.width, out.height, in.stride, uv_stride, out.stride,
          camera_props[0], camera_props[1], camera_props[2], camera_props[3],
          prop.distort_coefs[0], prop.distort_coefs[1], prop.distort_coefs[2],
          prop.distort_coefs[3], prop.distort_coefs[4], is_bgr);
      return run_kernel(nv12_barrel_correct, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::I420) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int u_offset = in.offsets[1];
      cl_int u_stride = in.strides[1];
      cl_int v_offset = in.offsets[2];
      cl_int v_stride = in.strides[2];
      cl_uchar is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(i420_barrel_correct, 0, *inbuf_y, *outbuf, u_offset,
          v_offset, in.width, in.height, out.width, out.height, in.stride, u_stride,
          v_stride, out.stride, camera_props[0], camera_props[1], camera_props[2],
          camera_props[3], prop.distort_coefs[0], prop.distort_coefs[1],
          prop.distort_coefs[2], prop.distort_coefs[3], prop.distort_coefs[4], is_bgr);

      return run_kernel(i420_barrel_correct, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::YUY2) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_uchar is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(yuyv_barrel_correct, 0, *inbuf_y, *outbuf,
          in.width, in.height, out.width, out.height, in.stride, out.stride,
          camera_props[0], camera_props[1], camera_props[2], camera_props[3],
          prop.distort_coefs[0], prop.distort_coefs[1], prop.distort_coefs[2],
          prop.distort_coefs[3], prop.distort_coefs[4], is_bgr);

      return run_kernel(yuyv_barrel_correct, out, inbuf_y, outbuf);

    } else {
      throw std::runtime_error("Unsupported format: " + AxVideoFormatToString(in.format));
    }
    return {};
  }

  private:
  CLProgram program;
  int error{};
  kernel rgba_barrel_correct;
  kernel nv12_barrel_correct;
  kernel i420_barrel_correct;
  kernel yuyv_barrel_correct;
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "camera_props",
    "distort_coefs",
    "bgra_out",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<barrelcorrect_properties>();

  prop->camera_props = Ax::get_property(input, "camera_props",
      "barrelcorrect_static_properties", prop->camera_props);
  prop->distort_coefs = Ax::get_property(input, "distort_coefs",
      "barrelcorrect_static_properties", prop->distort_coefs);
  prop->bgra_out = Ax::get_property(
      input, "bgra_out", "barrelcorrect_dynamic_properties", prop->bgra_out);

  constexpr auto camera_props_size = 4;
  if (prop->camera_props.size() != camera_props_size) {
    throw std::runtime_error("camera_props must have 4 values");
  }
  constexpr auto distort_coefs_size = 5;
  if (prop->distort_coefs.size() != distort_coefs_size) {
    throw std::runtime_error("distort_coefs must have 5 values");
  }
  prop->barrelcorrect = std::make_unique<CLBarrelCorrect>(kernel_cl, logger);

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    barrelcorrect_properties * /*prop*/, Ax::Logger & /*logger*/)
{
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const barrelcorrect_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    out_info.info.format = prop->bgra_out ? AxVideoFormat::BGRA : AxVideoFormat::RGBA;
    output = out_info;
  }
  return output;
}


extern "C" std::function<void()>
transform_async(const AxDataInterface &input, const AxDataInterface &output,
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
    AxVideoFormat::RGBA,
    AxVideoFormat::BGRA,
    AxVideoFormat::NV12,
    AxVideoFormat::I420,
    AxVideoFormat::YUY2,
  };
  if (std::none_of(valid_formats.begin(), valid_formats.end(), [input_details](auto format) {
        return format == input_details[0].format;
      })) {
    throw std::runtime_error("Barrel Correction does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  if (output_details[0].format != AxVideoFormat::RGBA
      && output_details[0].format != AxVideoFormat::BGRA) {
    throw std::runtime_error("Barrel Correction does not work with the output format: "
                             + AxVideoFormatToString(output_details[0].format));
  }
  return prop->barrelcorrect->run(input_details[0], output_details[0], *prop);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const barrelcorrect_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta, Ax::Logger &logger)
{
  logger(AX_WARN) << "Running in synchronous mode, possible performance degradation"
                  << std::endl;
  auto completer = transform_async(input, output, prop, 0, 0, meta, logger);
  completer();
}
