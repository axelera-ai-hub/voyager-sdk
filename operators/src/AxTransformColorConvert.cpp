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


const char *kernel_cl = R"##(

//  Convert YUV values to RGBA
uchar4 convert_YUV2RGBA_c(uchar y, uchar u, uchar v, char bgr) {
    uchar4 result = convert_YUV2RGBA((float3)(y, u, v));
    return bgr ? result.zyxw : result;
}

__kernel void nv12_to_rgba(int width, int height, int strideInY, int strideUV, int strideOut,
    int uv_offset, char is_bgr, __global const uchar *iny, __global uchar4 *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }

    int strideO = strideOut;
    int top_row = 2 * row * strideO;
    int bottom_row = top_row + strideO;
    int left_pixel = 2 * col;
    int right_pixel = left_pixel + 1;

    int in_top_row = 2 * row * strideInY;
    int in_bottom_row = in_top_row + strideInY;

    __global const uchar *inuv = iny + uv_offset;
    uchar u = inuv[row * strideUV + col * 2];
    uchar v = inuv[row * strideUV + col * 2 + 1];

    uchar y = iny[in_top_row + left_pixel];
    rgb[top_row + left_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

    y = iny[in_top_row + right_pixel];
    rgb[top_row + right_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

    y = iny[in_bottom_row + left_pixel];
    rgb[bottom_row + left_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

    y = iny[in_bottom_row + right_pixel];
    rgb[bottom_row + right_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

}

__kernel void i420_to_rgba(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    int u_offset, int v_offset, char is_bgr, __global const uchar *iny, __global uchar4 *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
    __global const uchar *inu = iny + u_offset;
    __global const uchar *inv = iny + v_offset;

    int strideO = strideOut / 4;
    int top_row = 2 * row * strideO;
    int bottom_row = (2 * row + 1) * strideO;
    int left_pixel = 2 * col;
    int right_pixel = 2 * col + 1;

    uchar u = inu[row * strideU + col];
    uchar v = inv[row * strideV + col];

    uchar y = iny[2 * row * strideInY + col * 2];
    rgb[top_row + left_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

    y = iny[2 * row * strideInY + col * 2 + 1];
    rgb[top_row + right_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

    y = iny[(2 * row + 1) * strideInY + col * 2];
    rgb[bottom_row + left_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

    y = iny[(2 * row + 1) * strideInY + col * 2 + 1];
    rgb[bottom_row + right_pixel] = convert_YUV2RGBA_c(y, u, v, is_bgr);

}

__kernel void YUYV_to_rgba(int width, int height, int strideIn, int strideOut,
    char is_bgr, __global const uchar4 *in, __global uchar4 *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }
    int strideO = strideOut / 4;
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4

    row *= 2;

    int left = row * strideO + col * 2;
    int right = left + 1;

    uchar4 i = in[row * strideI + col];
    uchar y1 = i.x;
    uchar u = i.y;
    uchar y2 = i.z;
    uchar v = i.w;

    rgb[left] = convert_YUV2RGBA_c(y1, u, v, is_bgr);
    rgb[right] = convert_YUV2RGBA_c(y2, u, v, is_bgr);

    left += strideO;
    right = left + 1;

    i = in[(row + 1) * strideI + col];
    y1 = i.x;
    u = i.y;
    y2 = i.z;
    v = i.w;

    rgb[left] = convert_YUV2RGBA_c(y1, u, v, is_bgr);
    rgb[right] = convert_YUV2RGBA_c(y2, u, v, is_bgr);


}

__kernel void bgra_to_rgba(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar4 *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }
    int strideO = strideOut / 4;
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    row *=2;
    col *=2;
    uchar4 i = in[row * strideI + col];
    rgb[row * strideO + col] = i.zyxw;

    i = in[row * strideI + col + 1];
    rgb[row * strideO + col + 1] = i.zyxw;

    i = in[(row + 1) * strideI + col];
    rgb[(row + 1) * strideO + col] = i.zyxw;

    i = in[(row + 1) * strideI + col + 1];
    rgb[(row + 1) * strideO + col + 1] = i.zyxw;
}

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;

class CLColorConvert
{
  public:
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  CLColorConvert(std::string source, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils() + source, logger),
        nv12_to_rgba{ program.get_kernel("nv12_to_rgba") },
        i420_to_rgba{ program.get_kernel("i420_to_rgba") },
        YUYV_to_rgba{ program.get_kernel("YUYV_to_rgba") }, bgra_to_rgba{
          program.get_kernel("bgra_to_rgba")
        }
  {
  }

  CLProgram::flush_details run_kernel(kernel &k, const buffer_details &out, buffer &outbuf)
  {
    size_t global_work_size[3] = { 1, 1, 1 };
    const int numpix_per_kernel = 1;
    global_work_size[0] = out.width / 2;
    global_work_size[1] = out.height / 2;
    error = program.execute_kernel(k, 2, global_work_size);
    if (error != CL_SUCCESS) {
      return {};
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

  std::function<void()> run_nv12_to_rgba(
      const buffer_details &in, const buffer_details &out, cl_char is_bgr)
  {
    const int rgba_size = 4;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    cl_int uv_stride = in.strides[1];
    cl_int uv_offset = in.offsets[1];
    //  Set the kernel arguments
    program.set_kernel_args(nv12_to_rgba, 0, in.width / 2, in.height / 2, y_stride,
        uv_stride, out.stride / rgba_size, uv_offset, is_bgr, *inpbuf, *outbuf);
    return run_kernel(nv12_to_rgba, out, inpbuf, outbuf);
  }

  std::function<void()> run_i420_to_rgba(
      const buffer_details &in, const buffer_details &out, cl_char is_bgr)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    cl_int u_stride = in.strides[1];
    cl_int v_stride = in.strides[2];
    cl_int u_offset = in.offsets[1];
    cl_int v_offset = in.offsets[2];
    //  Set the kernel arguments
    program.set_kernel_args(i420_to_rgba, 0, in.width / 2, in.height / 2, y_stride,
        u_stride, v_stride, out.stride, u_offset, v_offset, is_bgr, *inpbuf, *outbuf);
    return run_kernel(i420_to_rgba, out, inpbuf, outbuf);
  }

  std::function<void()> run_YUYV_to_rgba(
      const buffer_details &in, const buffer_details &out, cl_char is_bgr)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    //  Set the kernel arguments
    program.set_kernel_args(YUYV_to_rgba, 0, in.width / 2, in.height / 2,
        y_stride, out.stride, is_bgr, *inpbuf, *outbuf);
    return run_kernel(YUYV_to_rgba, out, inpbuf, outbuf);
  }

  std::function<void()> run_bgra_to_rgba(const buffer_details &in, const buffer_details &out)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.stride;
    //  Set the kernel arguments
    program.set_kernel_args(bgra_to_rgba, 0, in.width / 2, in.height / 2,
        y_stride, out.stride, *inpbuf, *outbuf);
    return run_kernel(bgra_to_rgba, out, inpbuf, outbuf);
  }

  std::function<void()> run(const buffer_details &in, const buffer_details &out,
      const std::string &format)
  {
    cl_char is_bgr = format == "bgra";
    if (in.format == AxVideoFormat::NV12) {
      return run_nv12_to_rgba(in, out, is_bgr);
    } else if (in.format == AxVideoFormat::I420) {
      return run_i420_to_rgba(in, out, is_bgr);
    } else if (in.format == AxVideoFormat::YUY2) {
      return run_YUYV_to_rgba(in, out, is_bgr);
    } else if (in.format == AxVideoFormat::RGBA || in.format == AxVideoFormat::BGRA) {
      return run_bgra_to_rgba(in, out);
    }
    return {};
  }

  bool can_use_dmabuf() const
  {
    return program.can_use_dmabuf();
  }

  private:
  CLProgram program;
  int error{};
  kernel nv12_to_rgba;
  kernel i420_to_rgba;
  kernel YUYV_to_rgba;
  kernel bgra_to_rgba;
};

struct cc_properties {
  std::string format{ "rgba" };
  float quant_scale;
  float quant_zeropoint;
  std::vector<cl_float> add;
  std::vector<cl_float> mul;
  mutable std::unique_ptr<CLColorConvert> color_convert;
  bool to_tensor{};
  mutable int total_time{};
  mutable int num_calls{};
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "format" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<cc_properties>();
  prop->format = Ax::get_property(input, "format", "ColorConvertProperties", prop->format);
  prop->color_convert = std::make_unique<CLColorConvert>(kernel_cl, logger);

  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    cc_properties * /*prop*/, Ax::Logger & /*logger*/)
{
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const cc_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    logger(AX_INFO) << "Setting output format to " << prop->format << std::endl;
    out_info.info.format
        = prop->format == "rgba" ? AxVideoFormat::RGBA : AxVideoFormat::BGRA;
    output = out_info;
  }
  return output;
}

/// @brief  Check if the plugin has any work to do
/// @param input
/// @param output
/// @param logger
/// @return true if the plugin can pass through the input to output
extern "C" bool
can_passthrough(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, Ax::Logger &logger)
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

  return input_details[0].format == output_details[0].format;
}


extern "C" std::function<void()>
transform_async(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  //  These must be video interfaces as we have already checked in can_passthrough
  auto in_info = std::get<AxVideoInterface>(input);
  auto out_info = std::get<AxVideoInterface>(output);

  //  Validate input and output formats

  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("color_convert works on single tensor (possibly batched) input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error(
        "color_convert works on single tensor (possibly batched) output only");
  }

  return prop->color_convert->run(input_details[0], output_details[0], prop->format);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  logger(AX_WARN) << "Running in synchronous mode, possible performance degradation"
                  << std::endl;
  auto completer = transform_async(input, output, prop, 0, 0, meta_map, logger);
  completer();
}

extern "C" bool
can_use_dmabuf(const cc_properties *prop, Ax::Logger &logger)
{
  return prop->color_convert->can_use_dmabuf();
}
