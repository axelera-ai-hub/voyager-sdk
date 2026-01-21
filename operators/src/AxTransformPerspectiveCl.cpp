// Copyright Axelera AI, 2026
#include <array>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxUtils.hpp"

/**
 * This file implements perspective transformation using OpenCL.
 *
 * The transformation uses a 3x3 homography matrix provided in row-major order:
 * [ m00 m01 m02 ]
 * [ m10 m11 m12 ]
 * [ m20 m21 m22 ]
 *
 * For optimization and alignment purposes, we convert this to a 4x4 matrix:
 * [ m00 m01 m02 0 ]
 * [ m10 m11 m12 0 ]
 * [ m20 m21 m22 0 ]
 * [ 0   0   0   1 ]
 *
 * This allows us to use float4 operations in the kernel for better performance.
 */


class CLPerspective;
struct perspective_properties {
  std::vector<cl_float> matrix; // Original 3x3 matrix (row-major, 9 elements)
  std::vector<cl_float> matrix_4x4; // Converted 4x4 matrix (row-major, 16 elements)
  std::string out_format{};
  std::unique_ptr<CLPerspective> perspective;
};

const char *kernel_cl = R"##(

float2
perspective_transform(int2 coord, __constant float4 *perspective_matrix)
{
  const float4 coord_f = (float4)(coord.x + 0.5F, coord.y + 0.5F, 1.0F, 0.0F);
  const float4 x_row = perspective_matrix[0];
  const float4 y_row = perspective_matrix[1];
  const float4 z_row = perspective_matrix[2];
  const float w = 1.0F / dot(coord_f, z_row);
  const float new_x = dot(coord_f, x_row) * w;
  const float new_y = dot(coord_f, y_row) * w;
  return (float2) (new_x, new_y);
}

__kernel void rgb_perspective_bl(__global const uchar *in, __global uchar *out, int in_width, int in_height, int out_width, int out_height,
                        int strideIn, int strideOut, __constant float4 *perspective_matrix, output_format out_format, uchar is_input_bgr) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = perspective_transform((int2)(col, row), perspective_matrix);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out_format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    __global uchar *p_in = advance_uchar_ptr(in, row * strideIn);
    rgb_image img = {in_width, in_height, strideIn, 0, 0};
    uchar4 pixel = rgb_sampler_bl(in, corrected.x, corrected.y, &img);
    if (out_format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(is_input_bgr ? pixel.zyx : pixel.xyz);
      return;
    }
    is_input_bgr ? vstore3((out_format == BGR_OUTPUT) ? pixel.xyz : pixel.zyx, col, prgb) : vstore3((out_format == BGR_OUTPUT) ? pixel.zyx : pixel.xyz, col, prgb);
}

__kernel void nv12_perspective_bl(__global const uchar *in_y, __global uchar *out, int uv_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInUV, int strideOut,
                            __constant float4 *perspective_matrix, output_format out_format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = perspective_transform((int2)(col, row), perspective_matrix);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out_format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }
    __global uchar2 *in_uv = (__global uchar2 *)(in_y + uv_offset);
    int uvStrideI = strideInUV / sizeof(uchar2);
    nv12_image img = {in_width, in_height, strideInY, uvStrideI, 0, 0};
    uchar4 pixel = nv12_sampler(in_y, in_uv,  corrected.x, corrected.y, &img);
    if (out_format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY( pixel.xyz);
      return;
    }
    vstore3(out_format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
}

__kernel void i420_perspective_bl(__global const uchar *in_y, __global uchar *out, int u_offset, int v_offset, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideInU, int strideInV, int strideOut,
                            __constant float4 *perspective_matrix, output_format out_format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = perspective_transform((int2)(col, row), perspective_matrix);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out_format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    __global const uchar *in_u = in_y + u_offset;
    __global const uchar *in_v = in_y + v_offset;

    i420_image img = {in_width, in_height, strideInY, strideInU, strideInV, 0, 0};
    uchar4 pixel = i420_sampler(in_y, in_u, in_v, corrected.x, corrected.y, &img);
    if (out_format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(out_format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
  }

__kernel void yuyv_perspective_bl(__global const uchar *in_y, __global uchar *out, int in_width, int in_height,
                            int out_width, int out_height, int strideInY, int strideOut,
                            __constant float4 *perspective_matrix, output_format out_format) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int strideI = strideInY / sizeof(uchar4);
    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = perspective_transform((int2)(col, row), perspective_matrix);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out_format == GRAY_OUTPUT ?  out[row * strideOut + col] = 0 : vstore3((uchar3)(0, 0, 0), col, prgb);
      return;
    }

    yuyv_image img = {in_width, in_height, strideI, 0, 0};
    uchar4 pixel = yuyv_sampler(in_y, corrected.x, corrected.y, &img);
    if (out_format == GRAY_OUTPUT) {
      out[row * strideOut + col] = RGB_to_GRAY(pixel.xyz);
      return;
    }
    vstore3(out_format == BGR_OUTPUT ? pixel.zyx : pixel.xyz, col, prgb);
}

__kernel void gray_perspective_bl(__global const uchar *in, __global uchar *out, int in_width, int in_height, int out_width, int out_height,
                        int strideIn, int strideOut, __constant float4 *perspective_matrix) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= out_height || col >= out_width) {
      return;
    }

    float2 corrected = perspective_transform((int2)(col, row), perspective_matrix);
    __global uchar* prgb = advance_uchar_ptr(out, row * strideOut);
    if (corrected.x < 0 || corrected.x >= in_width || corrected.y < 0 || corrected.y >= in_height) {
      out[row * strideOut + col] = 0;
      return;
    }

    __global uchar *p_in = advance_uchar_ptr(in, row * strideIn);
    rgb_image img = {in_width, in_height, strideIn, 0, 0};
    uchar pixel = gray8_sampler_bl(in, corrected.x, corrected.y, &img);
    out[row * strideOut + col] = pixel;

}

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;
class CLPerspective
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLPerspective(std::string source, opencl_details *ocl, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils() + source, ocl, logger),
        // rgba_perspective{ program.get_kernel("rgba_perspective_bl") },
        rgb_perspective(program.get_kernel("rgb_perspective_bl")),
        nv12_perspective{ program.get_kernel("nv12_perspective_bl") },
        i420_perspective{ program.get_kernel("i420_perspective_bl") },
        yuyv_perspective{ program.get_kernel("yuyv_perspective_bl") }, gray_perspective{
          program.get_kernel("gray_perspective_bl")
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


  int run(const buffer_details &in, const buffer_details &out,
      const perspective_properties &prop)
  {
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    auto perspective_matrix = program.create_buffer(1,
        prop.matrix_4x4.size() * sizeof(prop.matrix_4x4[0]), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        const_cast<float *>(prop.matrix_4x4.data()), 1);

    if (in.format == AxVideoFormat::RGB || in.format == AxVideoFormat::BGR) {
      auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int out_format = ax_utils::get_output_format(out.format);
      cl_uchar is_input_bgr = in.format == AxVideoFormat::BGR ? 1 : 0;
      program.set_kernel_args(*rgb_perspective, 0, *inbuf, *outbuf, in.width,
          in.height, out.width, out.height, in.stride, out.stride,
          *perspective_matrix, out_format, is_input_bgr);

      return run_kernel(*rgb_perspective, out, inbuf, outbuf);
    } else if (in.format == AxVideoFormat::NV12) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int uv_offset = in.offsets[1];
      cl_int uv_stride = in.strides[1];

      cl_int out_format = ax_utils::get_output_format(out.format);
      program.set_kernel_args(*nv12_perspective, 0, *inbuf_y, *outbuf,
          uv_offset, in.width, in.height, out.width, out.height, in.stride,
          uv_stride, out.stride, *perspective_matrix, out_format);
      return run_kernel(*nv12_perspective, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::I420) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int u_offset = in.offsets[1];
      cl_int u_stride = in.strides[1];
      cl_int v_offset = in.offsets[2];
      cl_int v_stride = in.strides[2];
      cl_int out_format = ax_utils::get_output_format(out.format);
      program.set_kernel_args(*i420_perspective, 0, *inbuf_y, *outbuf, u_offset,
          v_offset, in.width, in.height, out.width, out.height, in.stride,
          u_stride, v_stride, out.stride, *perspective_matrix, out_format);

      return run_kernel(*i420_perspective, out, inbuf_y, outbuf);
    } else if (in.format == AxVideoFormat::YUY2) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int out_format = ax_utils::get_output_format(out.format);
      program.set_kernel_args(*yuyv_perspective, 0, *inbuf_y, *outbuf, in.width,
          in.height, out.width, out.height, in.stride, out.stride,
          *perspective_matrix, out_format);

      return run_kernel(*yuyv_perspective, out, inbuf_y, outbuf);
    } else if (in.format == AxVideoFormat::GRAY8) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      program.set_kernel_args(*gray_perspective, 0, *inbuf_y, *outbuf, in.width,
          in.height, out.width, out.height, in.stride, out.stride, *perspective_matrix);

      return run_kernel(*gray_perspective, out, inbuf_y, outbuf);

    } else {
      throw std::runtime_error("Unsupported format: " + AxVideoFormatToString(in.format));
    }
    return {};
  }

  private:
  CLProgram program;
  int error{};
  kernel rgb_perspective;
  kernel nv12_perspective;
  kernel i420_perspective;
  kernel yuyv_perspective;
  kernel gray_perspective;
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "matrix",
    "out_format",
    "format",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<perspective_properties>();

  prop->matrix = Ax::get_property(
      input, "matrix", "perspective_static_properties", prop->matrix);
  prop->out_format = Ax::get_property(
      input, "out_format", "barrelcorrect_static_properties", prop->out_format);
  prop->out_format = Ax::get_property(
      input, "format", "barrelcorrect_static_properties", prop->out_format);

  constexpr auto matrix_size = 9;
  if (prop->matrix.size() != matrix_size) {
    throw std::runtime_error("Matrix size should be 9");
  }

  // Convert 3x3 row-major matrix to 4x4 row-major matrix
  prop->matrix_4x4.resize(16, 0.0f);

  // Copy the 3x3 matrix into the 4x4 matrix
  // [0 1 2]    [0 1 2 0]
  // [3 4 5] -> [3 4 5 0]
  // [6 7 8]    [6 7 8 0]
  //            [0 0 0 1]

  // First row
  prop->matrix_4x4[0] = prop->matrix[0]; // m00
  prop->matrix_4x4[1] = prop->matrix[1]; // m01
  prop->matrix_4x4[2] = prop->matrix[2]; // m02
  prop->matrix_4x4[3] = 0.0f; // m03

  // Second row
  prop->matrix_4x4[4] = prop->matrix[3]; // m10
  prop->matrix_4x4[5] = prop->matrix[4]; // m11
  prop->matrix_4x4[6] = prop->matrix[5]; // m12
  prop->matrix_4x4[7] = 0.0f; // m13

  // Third row
  prop->matrix_4x4[8] = prop->matrix[6]; // m20
  prop->matrix_4x4[9] = prop->matrix[7]; // m21
  prop->matrix_4x4[10] = prop->matrix[8]; // m22
  prop->matrix_4x4[11] = 0.0f; // m23

  // Fourth row
  prop->matrix_4x4[12] = 0.0f; // m30
  prop->matrix_4x4[13] = 0.0f; // m31
  prop->matrix_4x4[14] = 0.0f; // m32
  prop->matrix_4x4[15] = 1.0f; // m33

  logger(AX_INFO) << "Converted 3x3 perspective matrix to 4x4 matrix" << std::endl;

  prop->perspective = std::make_unique<CLPerspective>(
      kernel_cl, static_cast<opencl_details *>(context), logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    perspective_properties * /*prop*/, Ax::Logger & /*logger*/)
{
}

std::array valid_formats = {
  AxVideoFormat::RGB,
  AxVideoFormat::BGR,
  AxVideoFormat::GRAY8,
};

const char *name = "Perspective";

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const perspective_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
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
    const perspective_properties *prop, unsigned int, unsigned int,
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
    throw std::runtime_error("Perspective does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  Ax::validate_output_format(output_details[0].format, prop->out_format, name, valid_formats);

  prop->perspective->run(input_details[0], output_details[0], *prop);
}

extern "C" bool
query_supports(Ax::PluginFeature feature, const perspective_properties *prop, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return true;
  }
  return Ax::PluginFeatureDefaults(feature);
}
