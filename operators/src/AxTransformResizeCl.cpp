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


class CLResize;
struct resize_properties {
  int width{};
  int height{};
  int size{};
  bool letterbox{};
  bool scale_up{ true };
  int fill{ 114 };
  AxVideoFormat format{ AxVideoFormat::UNDEFINED };

  bool to_tensor{};
  float quant_scale{ 1.0F / 255.0F };
  float quant_zeropoint{};
  std::vector<cl_float> add{ 0.0F, 0.0F, 0.0F, 0.0F };
  std::vector<cl_float> mul{ 1.0F, 1.0F, 1.0F, 1.0F };
  std::unique_ptr<CLResize> resize;
};

const char *kernel_cl = R"##(


__kernel void rgba_resize_bl(__global const uchar4 *in, __global uchar4 *out, int in_width, int in_height, int crop_x, int crop_y,
                            int out_width, int out_height, int strideIn, int strideOut, float xscale, float yscale, int scaled_width,
                            int scaled_height, uchar fill, uchar is_bgr, float4 mul, float4 add) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int xoffset = (out_width - scaled_width) / 2;
    int yoffset = (out_height - scaled_height) / 2;

    int strideO = strideOut / sizeof(uchar4);
    if (col < xoffset || row < yoffset || col >= scaled_width + xoffset || row >= scaled_height + yoffset) {
      if (row < out_height && col < out_width) {
        uchar4 pixel = (uchar4)(fill, fill, fill, 255);
        if (mul.x != 0.0F) {
          char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
          pixel = convert_uchar4(pix);
        }
        out[row * strideO + col] = pixel;
      }
      return;
    }

    int strideI = strideIn / sizeof(uchar4);
    rgb_image img = {in_width, in_height, strideI, crop_x, crop_y};
    uchar4 pixel = rgb_sampler_bl(in, (0.5F + col - xoffset) * xscale, (0.5F + row - yoffset) * yscale, &img);
    pixel = is_bgr ? pixel.zyxw : pixel;
    if (mul.x != 0.0F) {
      char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
      pixel = convert_uchar4(pix);
    }
    out[row * strideO + col] = pixel;
}


__kernel void nv12_resize_bl(__global const uchar *in_y, __global uchar4 *out, int uv_offset, int in_width, int in_height,
                            int crop_x, int crop_y, int out_width, int out_height, int strideInY, int strideInUV, int strideOut,
                            float xscale, float yscale, int scaled_width, int scaled_height, uchar fill, uchar is_bgr, float4 mul, float4 add) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int xoffset = (out_width - scaled_width) / 2;
    int yoffset = (out_height - scaled_height) / 2;

    int strideO = strideOut / sizeof(uchar4);
    if (col < xoffset || row < yoffset || col >= scaled_width + xoffset || row >= scaled_height + yoffset) {
      if (row < out_height && col < out_width) {
        uchar4 pixel = (uchar4)(fill, fill, fill, 255);
        if (mul.x != 0.0F) {
          char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
          pixel = convert_uchar4(pix);
        }
        out[row * strideO + col] = pixel;
      }
      return;
    }

    __global uchar2 *in_uv = (__global uchar2 *)(in_y + uv_offset);
    int uvStrideI = strideInUV / sizeof(uchar2);
    nv12_image img = {in_width, in_height, strideInY, uvStrideI, crop_x, crop_y};
    uchar4 pixel = nv12_sampler(in_y, in_uv, (0.5F + col - xoffset) * xscale, (0.5F + row - yoffset) * yscale, &img);
    pixel = is_bgr ? pixel.zyxw : pixel;
    if (mul.x != 0.0F) {
      char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
      pixel = convert_uchar4(pix);
    }
    out[row * strideO + col] = pixel;
}

__kernel void i420_resize_bl(__global const uchar *in_y, __global uchar4 *out, int u_offset, int v_offset, int in_width, int in_height,
                            int crop_x, int crop_y, int out_width, int out_height, int strideInY, int strideInU, int strideInV, int strideOut,
                            float xscale, float yscale, int scaled_width, int scaled_height, uchar fill, uchar is_bgr, float4 mul, float4 add) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int xoffset = (out_width - scaled_width) / 2;
    int yoffset = (out_height - scaled_height) / 2;

    int strideO = strideOut / sizeof(uchar4);
    if (col < xoffset || row < yoffset || col >= scaled_width + xoffset || row >= scaled_height + yoffset) {
      if (row < out_height && col < out_width) {
        uchar4 pixel = (uchar4)(fill, fill, fill, 255);
        if (mul.x != 0.0F) {
          char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
          pixel = convert_uchar4(pix);
        }
        out[row * strideO + col] = pixel;
      }
      return;
    }

    __global const uchar *in_u = in_y + u_offset;
    __global const uchar *in_v = in_y + v_offset;

    i420_image img = {in_width, in_height, strideInY, strideInU, strideInV, crop_x, crop_y};
    uchar4 pixel = i420_sampler(in_y, in_u, in_v, (0.5F + col - xoffset) * xscale, (0.5F + row - yoffset) * yscale, &img);
    pixel = is_bgr ? pixel.zyxw : pixel;
    if (mul.x != 0.0F) {
      char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
      pixel = convert_uchar4(pix);
    }
    out[row * strideO + col] = pixel;
}

__kernel void yuyv_resize_bl(__global const uchar4 *in_y, __global uchar4 *out, int in_width, int in_height,
                            int crop_x, int crop_y, int out_width, int out_height, int strideInY, int strideOut,
                            float xscale, float yscale, int scaled_width, int scaled_height, uchar fill, uchar is_bgr, float4 mul, float4 add) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int xoffset = (out_width - scaled_width) / 2;
    int yoffset = (out_height - scaled_height) / 2;

    int strideO = strideOut / sizeof(uchar4);
    int strideI = strideInY / sizeof(uchar4);
    if (col < xoffset || row < yoffset || col >= scaled_width + xoffset || row >= scaled_height + yoffset) {
      if (row < out_height && col < out_width) {
        uchar4 pixel = (uchar4)(fill, fill, fill, 255);
        if (mul.x != 0.0F) {
          char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
          pixel = convert_uchar4(pix);
        }
        out[row * strideO + col] = pixel;
      }
      return;
    }

    yuyv_image img = {in_width, in_height, strideI, crop_x, crop_y};
    uchar4 pixel = yuyv_sampler(in_y, (0.5F + col - xoffset) * xscale, (0.5F + row - yoffset) * yscale, &img);
    pixel = is_bgr ? pixel.zyxw : pixel;
    if (mul.x != 0.0F) {
      char4 pix = convert_char4_sat_rte(mad(convert_float4(pixel), mul, add));
      pixel = convert_uchar4(pix);
    }
    out[row * strideO + col] = pixel;
}

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;
class CLResize
{
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  public:
  CLResize(std::string source, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils() + source, logger),
        rgba_resize{ program.get_kernel("rgba_resize_bl") },
        nv12_resize{ program.get_kernel("nv12_resize_bl") },
        i420_resize{ program.get_kernel("i420_resize_bl") }, yuyv_resize{
          program.get_kernel("yuyv_resize_bl")
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
      const resize_properties &prop)
  {
    cl_float xscale = (float) in.width / out.width;
    cl_float yscale = (float) in.height / out.height;
    cl_int scaled_width = out.width;
    cl_int scaled_height = out.height;
    if (prop.letterbox) {
      bool scale_to_height = static_cast<double>(prop.width) / prop.height
                             > static_cast<double>(in.width) / in.height;

      auto scale_factor = scale_to_height ?
                              static_cast<double>(prop.height) / in.height :
                              static_cast<double>(prop.width) / in.width;

      auto height = std::lround(in.height * scale_factor);
      auto width = std::lround(in.width * scale_factor);

      xscale = 1.0F / scale_factor;
      yscale = 1.0F / scale_factor;
      scaled_width = width;
      scaled_height = height;
    }

    if (in.width < out.width && in.height < out.height && !prop.scale_up) {
      xscale = 1.0F;
      yscale = 1.0F;
      scaled_width = in.width;
      scaled_height = in.height;
    }
    cl_uchar fill = prop.fill;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    if (in.format == AxVideoFormat::RGBA || in.format == AxVideoFormat::BGRA) {
      auto inbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_char is_bgr = in.format != out.format;
      program.set_kernel_args(rgba_resize, 0, *inbuf, *outbuf, in.width, in.height,
          in.crop_x, in.crop_y, out.width, out.height, in.stride, out.stride, xscale,
          yscale, scaled_width, scaled_height, fill, is_bgr, prop.mul, prop.add);
      return run_kernel(rgba_resize, out, inbuf, outbuf);

    } else if (in.format == AxVideoFormat::NV12) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int uv_offset = in.offsets[1];
      cl_int uv_stride = in.strides[1];
      cl_char is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(nv12_resize, 0, *inbuf_y, *outbuf, uv_offset,
          in.width, in.height, in.crop_x, in.crop_y, out.width, out.height,
          in.stride, uv_stride, out.stride, xscale, yscale, scaled_width,
          scaled_height, fill, is_bgr, prop.mul, prop.add);
      return run_kernel(nv12_resize, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::I420) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_int u_offset = in.offsets[1];
      cl_int u_stride = in.strides[1];
      cl_int v_offset = in.offsets[2];
      cl_int v_stride = in.strides[2];
      cl_char is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(i420_resize, 0, *inbuf_y, *outbuf, u_offset,
          v_offset, in.width, in.height, in.crop_x, in.crop_y, out.width,
          out.height, in.stride, u_stride, v_stride, out.stride, xscale, yscale,
          scaled_width, scaled_height, fill, is_bgr, prop.mul, prop.add);

      return run_kernel(i420_resize, out, inbuf_y, outbuf);

    } else if (in.format == AxVideoFormat::YUY2) {
      auto inbuf_y = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

      cl_char is_bgr = out.format == AxVideoFormat::BGRA;
      program.set_kernel_args(yuyv_resize, 0, *inbuf_y, *outbuf, in.width, in.height,
          in.crop_x, in.crop_y, out.width, out.height, in.stride, out.stride, xscale,
          yscale, scaled_width, scaled_height, fill, is_bgr, prop.mul, prop.add);

      return run_kernel(yuyv_resize, out, inbuf_y, outbuf);

    } else {
      throw std::runtime_error(
          "ResizeCl unsupported input format: " + AxVideoFormatToString(in.format));
    }
    return {};
  }

  private:
  CLProgram program;
  int error{};
  kernel rgba_resize;
  kernel nv12_resize;
  kernel i420_resize;
  kernel yuyv_resize;
};


extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "width",
    "height",
    "size",
    "letterbox",
    "padding",
    "format",
    "scale_up",
    //  For normalisation
    "mean",
    "std",
    "quant_scale",
    "quant_zeropoint",
    "to_tensor",

  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<resize_properties>();
  prop->resize = std::make_unique<CLResize>(kernel_cl, logger);
  prop->size = Ax::get_property(input, "size", "resize_cl_static_properties", prop->size);
  prop->width = Ax::get_property(input, "width", "resize_cl_static_properties", prop->width);
  prop->height
      = Ax::get_property(input, "height", "resize_cl_static_properties", prop->height);
  prop->letterbox = Ax::get_property(
      input, "letterbox", "resize_cl_static_properties", prop->letterbox);

  prop->scale_up = Ax::get_property(
      input, "scale_up", "resize_cl_static_properties", prop->scale_up);
  auto format = Ax::get_property(
      input, "format", "resize_cl_static_properties", std::string{});
  if (format == "rgba") {
    prop->format = AxVideoFormat::RGBA;
  } else if (format == "bgra") {
    prop->format = AxVideoFormat::BGRA;
  } else if (format == "") {
    prop->format = AxVideoFormat::UNDEFINED;
  } else {
    throw std::runtime_error(
        "Resize with color convert only outputs RGBA or BGRA, given: " + format);
  }
  prop->fill = Ax::get_property(input, "padding", "resize_cl_static_properties", prop->fill);
  if (prop->letterbox) {
    if (prop->width == 0) {
      prop->width = prop->height;
    } else if (prop->height == 0) {
      prop->height = prop->width;
    }
  }
  if (prop->size > 0 && (prop->width > 0 || prop->height > 0)) {
    throw std::runtime_error("You must provide only one of width/height or size");
  }
  if (prop->size == 0 && (prop->width <= 0 || prop->height <= 0)) {
    throw std::runtime_error("Invalid width or height");
  }

  prop->to_tensor = Ax::get_property(
      input, "to_tensor", "resize_cl_static_properties", prop->to_tensor);

  prop->quant_scale = Ax::get_property(
      input, "quant_scale", "resize_cl_static_properties", prop->quant_scale);
  prop->quant_zeropoint = Ax::get_property(input, "quant_zeropoint",
      "resize_cl_static_properties", prop->quant_zeropoint);
  auto mean = Ax::get_property(
      input, "mean", "resize_cl_static_properties", std::vector<cl_float>());
  auto std = Ax::get_property(
      input, "std", "resize_cl_static_properties", std::vector<cl_float>());
  if (mean.empty()) {
    mean = std::vector<float>(std.size(), 0.0);
  }
  if (std.empty()) {
    std = std::vector<float>(mean.size(), 1.0);
  }
  if (mean.size() != std.size()) {
    throw std::runtime_error("mean and std must have equal lengths in resize_cl");
  }
  if (mean.empty() && std.empty()) {
    prop->mul = { 0.0F, 0.0F, 0.0F, 0.0F };
    prop->add = { 0.0F, 0.0F, 0.0F, 0.0F };
  } else {
    const auto max_size = 4;
    auto size = std::max(max_size, static_cast<int>(mean.size()));
    prop->add.resize(size, 0.0F);
    prop->mul.resize(size, 1.0F);
    for (int i = 0; i < mean.size(); ++i) {
      prop->mul[i] = 1.0 / (255.0 * prop->quant_scale * std[i]);
      prop->add[i] = 255 * prop->quant_zeropoint * std[i] * prop->quant_scale
                     - (255 * mean[i]);
      prop->add[i] *= prop->mul[i];
    }
  }
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    resize_properties * /*prop*/, Ax::Logger & /*logger*/)
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

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const resize_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    auto [width, height] = prop->size ? determine_width_height(in_info, prop->size) :
                                        std::make_pair(prop->width, prop->height);
    out_info.info.width = width;
    out_info.info.height = height;
    out_info.info.format
        = prop->format == AxVideoFormat::UNDEFINED ? in_info.info.format : prop->format;
    output = out_info;
  }
  if (prop->to_tensor) {
    auto &info = std::get<AxVideoInterface>(output).info;
    AxTensorsInterface output = { { { 1, info.height, info.width, 4 }, 1, nullptr } };
    return output;
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
    const resize_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)) {
    logger(AX_WARN) << "Resize works on video input only" << std::endl;
    return false;
  }

  auto input_details = ax_utils::extract_buffer_details(input);
  auto output_details = ax_utils::extract_buffer_details(output);

  return (input_details[0].width == output_details[0].width
          && input_details[0].height == output_details[0].height
          && (input_details[0].format == output_details[0].format
              || output_details[0].format == AxVideoFormat::UNDEFINED)
          && prop->mul[0] == 0.0F);
}

extern "C" std::function<void()>
transform_async(const AxDataInterface &input, const AxDataInterface &output,
    const resize_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{

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
    throw std::runtime_error("Resize does not work with the input format: "
                             + AxVideoFormatToString(input_details[0].format));
  }
  if (output_details[0].format == AxVideoFormat::UNDEFINED) {
    output_details[0].format = prop->format == AxVideoFormat::UNDEFINED ?
                                   input_details[0].format :
                                   prop->format;
  }
  if (output_details[0].format != AxVideoFormat::RGBA
      && output_details[0].format != AxVideoFormat::BGRA) {
    throw std::runtime_error("Resize does not work with the output format: "
                             + AxVideoFormatToString(output_details[0].format));
  }
  return prop->resize->run(input_details[0], output_details[0], *prop);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const resize_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta, Ax::Logger &logger)
{
  logger(AX_WARN) << "Running in synchronous mode, possible performance degradation"
                  << std::endl;
  auto completer = transform_async(input, output, prop, 0, 0, meta, logger);
  completer();
}
extern "C" int
handles_crop_meta()
{
  return 1;
}
