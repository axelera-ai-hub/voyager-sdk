// Copyright Axelera AI, 2024
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
  bool downstream_supports_opencl{ false };
  std::unique_ptr<CLPerspective> perspective;
};

const char *perspective_kernel = R"##(
float2
perspective_transform(int2 coord, float16 perspective_matrix)
{
  const float4 coord_f = (float4)(coord.x + 0.5F, coord.y + 0.5F, 1.0F, 0.0F);
  const float4 x_row = perspective_matrix.s0123;
  const float4 y_row = perspective_matrix.s4567;
  const float4 z_row = perspective_matrix.s89ab;
  const float w = 1.0F / dot(coord_f, z_row);
  const float new_x = dot(coord_f, x_row) * w;
  const float new_y = dot(coord_f, y_row) * w;
  return (float2) (new_x, new_y);
}

__kernel void perspective(%s__global uchar *out, int4 image_dims,
                        int4 strides, int4 offsets, float16 perspective_matrix, float16 color_matrix) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= image_dims.w || col >= image_dims.z) {
      return;
    }
    float2 corrected = perspective_transform((int2)(col, row), perspective_matrix);
    image_description img = {image_dims, strides, offsets, (int4)(0, 0, image_dims.z, image_dims.w), (int4)(0,0,0,0)};
    uchar fill = 0;

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;

class CLPerspective
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLPerspective(opencl_details *ocl, Ax::Logger &logger)
      : program("", ocl, logger)
  {
  }

  ax_utils::CLProgram::ax_kernel build_kernel(ax_utils::CLProgram &program,
      AxVideoFormat in_format, AxVideoFormat out_format, int flip_type, int num_planes)
  {
    std::string kernel_code = perspective_kernel;

    auto input_details = ax_utils::get_input_details(
        in_format, ax_utils::Interpolation::bilinear, num_planes);
    auto output_details = ax_utils::get_output_details(in_format, out_format);

    const auto &output_code = output_details.sampler;
    const auto &sampler_code = input_details.sampler;

    auto n = snprintf(
        nullptr, 0, kernel_code.c_str(), input_details.input_params.c_str());
    std::vector<char> buffer(n + 1);
    snprintf(buffer.data(), buffer.size(), kernel_code.c_str(),
        input_details.input_params.c_str());
    auto final_kernel = std::string(buffer.data());

    final_kernel += sampler_code;
    final_kernel += output_code;
    final_kernel = ax_utils::get_kernel_utils(flip_type, program.has_fp16()) + final_kernel;

    return program.build_kernel_from_source(final_kernel, "perspective");
  }

  int run(const buffer_details &in, const buffer_details &out,
      const perspective_properties &prop)
  {
    auto num_planes = ax_utils::get_num_planes(in);
    if (!converter) {
      converter = build_kernel(program, in.format, out.format, 0, num_planes);
    }

    bool start_flush = !prop.downstream_supports_opencl;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    std::array<cl_int, 4> image_dims = { in.width, in.height, out.width, out.height };
    auto strides = ax_utils::build_strides(in, out);
    auto offsets = ax_utils::build_offsets(in, out, num_planes);
    auto in_bufs = program.create_buffers(1, ax_utils::determine_buffer_size(in),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, in.data, in.offsets.size());
    auto matrix = ax_utils::get_color_conversion_matrix(in.format, out.format);

    program.set_kernel_args(*converter, 0, in_bufs, *outbuf, image_dims,
        strides, offsets, prop.matrix_4x4, matrix);
    return run_kernel(program, *converter, in, out, in_bufs[0], outbuf, start_flush);
  }

  private:
  CLProgram program;
  int error{};
  kernel converter{ nullptr };
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
      input, "out_format", "perspective_static_properties", prop->out_format);
  prop->out_format = Ax::get_property(
      input, "format", "perspective_static_properties", prop->out_format);

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
      static_cast<opencl_details *>(context), logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    perspective_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "perspective_dynamic_properties", prop->downstream_supports_opencl);
}

std::array valid_formats = {
  AxVideoFormat::RGB,
  AxVideoFormat::BGR,
  AxVideoFormat::RGBA,
  AxVideoFormat::BGRA,
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
    AxVideoFormat::RGBA,
    AxVideoFormat::BGRA,
    AxVideoFormat::NV12,
    AxVideoFormat::NV16,
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
