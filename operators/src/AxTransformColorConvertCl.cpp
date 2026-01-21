// Copyright Axelera AI, 2026
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

const char *kernel_cl = R"##(

uchar3 yuv_to_rgb(uchar y, uchar u, uchar v, char bgr) {
    uchar3 result = YUV_to_RGB(y, u, v);
    return bgr ? result.zyx : result;
}

uchar4 yuv_to_rgba(uchar y, uchar u, uchar v, char bgr) {
    return (uchar4)(yuv_to_rgb(y, u, v, bgr), 255);
}

uchar4 convert_to_rgba(float y, float u, float v, char bgr) {
    float r = y + 1.402f * (v - 0.5f);
    float g = y - 0.344f * (u - 0.5f) - 0.714f * (v - 0.5f);
    float b = y + 1.772f * (u - 0.5f);
    return bgr ?  (uchar4)(convert_uchar_sat(b * 255.0f),
                    convert_uchar_sat(g * 255.0f),
                    convert_uchar_sat(r * 255.0f),

                    255) :
                (uchar4)(convert_uchar_sat(r * 255.0f),
                    convert_uchar_sat(g * 255.0f),
                    convert_uchar_sat(b * 255.0f),
                    255);
}

__kernel void nv12_planes_to_rgba(int width, int height, int strideInY, int strideUV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inuv, __global uchar4 *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    int uv_idx = top_left.y / 2 * strideUV + (top_left.x & ~1); ;
    uchar u = inuv[uv_idx];
    uchar v = inuv[uv_idx + 1];
    __global uchar4* prgb = advance_uchar4_ptr(rgb, row * strideOut);
    prgb[col] = yuv_to_rgba(y, u, v, is_bgr);
}

__kernel void nv12_planes_to_rgb(int width, int height, int strideInY, int strideUV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inuv, __global uchar *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    int uv_idx = top_left.y / 2 * strideUV + (top_left.x & ~1); ;
    uchar u = inuv[uv_idx];
    uchar v = inuv[uv_idx + 1];
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    vstore3(yuv_to_rgb(y, u, v, is_bgr), col, prgb);
}

__kernel void nv12_to_rgba(int width, int height, int strideInY, int strideUV, int strideOut,
    int uv_offset, char is_bgr, __global const uchar *iny, __global uchar4 *rgb) {

  return nv12_planes_to_rgba(width, height, strideInY, strideUV, strideOut, is_bgr, iny, iny + uv_offset, rgb);
}

__kernel void nv12_to_rgb(int width, int height, int strideInY, int strideUV, int strideOut,
    int uv_offset, char is_bgr, __global const uchar *iny, __global uchar *rgb) {

  return nv12_planes_to_rgb(width, height, strideInY, strideUV, strideOut, is_bgr, iny, iny + uv_offset, rgb);
}


__kernel void i420_planes_to_rgba(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inu , __global const uchar *inv, __global uchar4 *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar4* prgb = advance_uchar4_ptr(rgb, strideOut * row);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    uchar u = inu[top_left.y / 2 * strideU + top_left.x / 2];
    uchar v = inv[top_left.y / 2 * strideV + top_left.x / 2];
    prgb[col] = yuv_to_rgba(y, u, v, is_bgr);
}

__kernel void i420_planes_to_rgb(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inu , __global const uchar *inv, __global uchar *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar* prgb = advance_uchar_ptr(rgb, strideOut * row);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    uchar u = inu[top_left.y / 2 * strideU + top_left.x / 2];
    uchar v = inv[top_left.y / 2 * strideV + top_left.x / 2];
    vstore3(yuv_to_rgb(y, u, v, is_bgr), col, prgb);
}


__kernel void i420_to_rgba(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    int u_offset, int v_offset, char is_bgr, __global const uchar *iny, __global uchar4 *rgb) {
  return i420_planes_to_rgba(width, height, strideInY, strideU, strideV, strideOut, is_bgr, iny, iny + u_offset, iny + v_offset, rgb);
}

__kernel void i420_to_rgb(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    int u_offset, int v_offset, char is_bgr, __global const uchar *iny, __global uchar *rgb) {
  return i420_planes_to_rgb(width, height, strideInY, strideU, strideV, strideOut, is_bgr, iny, iny + u_offset, iny + v_offset, rgb);
}

__kernel void YUYV_to_rgba(int width, int height, int strideIn, int strideOut,
    char is_bgr, __global const uchar4 *in, __global uchar4 *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }
    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar4 *p_in = advance_uchar4_ptr(in, top_left.y * strideIn);
    __global uchar4* prgb = advance_uchar4_ptr(rgb, strideOut * row);

    uchar4 i = p_in[top_left.x / 2];
    uchar y =  top_left.x % 2 == 0 ? i.x : i.z;
    uchar u = i.y;
    uchar v = i.w;
    prgb[col] = yuv_to_rgba(y, u, v, is_bgr);
}

__kernel void YUYV_to_rgb(int width, int height, int strideIn, int strideOut,
    char is_bgr, __global const uchar4 *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);

    __global uchar4 *p_in = advance_uchar4_ptr(in, top_left.y * strideIn);
    __global uchar* prgb = advance_uchar_ptr(rgb, strideOut * row);

    uchar4 i = p_in[top_left.x / 2];
    uchar y =  top_left.x % 2 == 0 ? i.x : i.z;
    uchar u = i.y;
    uchar v = i.w;
    vstore3(yuv_to_rgb(y, u, v, is_bgr), col, prgb);
}

__kernel void yuyv_to_gray(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar *gray) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar4 *p_in = advance_uchar4_ptr(in, top_left.y * strideIn);
    __global uchar* p_gray = advance_uchar_ptr(gray, row * strideOut);

    // YUY2/YUYV format stores: Y0 U0 Y1 V0
    uchar4 i = p_in[top_left.x / 2];
    // Extract Y value based on even/odd position
    uchar y = (top_left.x % 2 == 0) ? i.x : i.z;
    p_gray[col] = y;
}

__kernel void bgra_to_rgba(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar4 *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar4* prgb = advance_uchar4_ptr(rgb, row * strideOut);
    uchar4 i = in[top_left.y * strideI + top_left.x];
    prgb[col] = i.zyxw;
}

__kernel void rgba_to_rgba(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar4 *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar4* prgb = advance_uchar4_ptr(rgb, row * strideOut);
    uchar4 i = in[top_left.y * strideI + top_left.x];
    prgb[col] = i;
}

__kernel void bgr_to_rgb(int width, int height, int strideIn, int strideOut,
    __global const uchar *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar *p_in = advance_uchar_ptr(in, top_left.y * strideIn);
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    uchar3 i = vload3(top_left.x, p_in);
    vstore3(i.zyx, col, prgb);
}

__kernel void rgb_to_rgb(int width, int height, int strideIn, int strideOut,
    __global const uchar *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar *p_in = advance_uchar_ptr(in, top_left.y * strideIn);
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    uchar3 i = vload3(top_left.x, p_in);
    vstore3(i, col, prgb);
}


__kernel void rgba_to_bgr(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    uchar4 i = in[top_left.y * strideI + top_left.x];
    vstore3(i.zyx, col, prgb);
}

__kernel void rgba_to_rgb(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    uchar4 i = in[top_left.y * strideI + top_left.x];
    vstore3(i.xyz, col, prgb);
}

__kernel void rgb_to_gray(int width, int height, int strideIn, int strideOut,
    __global const uchar *in, __global uchar *gray) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global const uchar *p_in = advance_uchar_ptr(in, top_left.y * strideIn);
    __global uchar* p_gray = advance_uchar_ptr(gray, row * strideOut);

    uchar3 rgb = vload3(top_left.x, p_in);
    // Convert RGB to grayscale using luminance formula: Y = 0.299R + 0.587G + 0.114B
    uchar gray_value = (uchar)(0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
    p_gray[col] = gray_value;
}

__kernel void bgr_to_gray(int width, int height, int strideIn, int strideOut,
    __global const uchar *in, __global uchar *gray) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global const uchar *p_in = advance_uchar_ptr(in, top_left.y * strideIn);
    __global uchar* p_gray = advance_uchar_ptr(gray, row * strideOut);

    uchar3 rgb = vload3(top_left.x, p_in);
    uchar gray_value = (uchar)(0.114f * rgb.x + 0.587f * rgb.y + 0.299f * rgb.z);
    p_gray[col] = gray_value;
}

__kernel void rgba_to_gray(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar *gray) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar* p_gray = advance_uchar_ptr(gray, row * strideOut);

    uchar4 rgba = in[top_left.y * strideI + top_left.x];
    // Convert RGB to grayscale using luminance formula: Y = 0.299R + 0.587G + 0.114B
    uchar gray_value = (uchar)(0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z);
    p_gray[col] = gray_value;
}

__kernel void bgra_to_gray(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar *gray) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar* p_gray = advance_uchar_ptr(gray, row * strideOut);

    uchar4 bgra = in[top_left.y * strideI + top_left.x];
    uchar gray_value = (uchar)(0.114f * bgra.x + 0.587f * bgra.y + 0.299f * bgra.z);
    p_gray[col] = gray_value;
}



)##";

class CLColorConvert;
struct cc_properties {
  std::string format{ "rgba" };
  std::string flip_method{ "none" };
  mutable std::unique_ptr<CLColorConvert> color_convert;
  mutable int total_time{};
  mutable int num_calls{};
  bool downstream_supports_opencl{};
};

using ax_utils::buffer_details;
using ax_utils::CLProgram;
using ax_utils::opencl_details;
class CLColorConvert
{
  public:
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  CLColorConvert(std::string source, int flip_type, opencl_details *display, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils(flip_type) + source, display, logger),
        nv12_to_rgba{ program.get_kernel("nv12_to_rgba") },
        i420_to_rgba{ program.get_kernel("i420_to_rgba") },
        YUYV_to_rgba{ program.get_kernel("YUYV_to_rgba") }, //
        nv12_to_rgb{ program.get_kernel("nv12_to_rgb") }, //
        i420_to_rgb{ program.get_kernel("i420_to_rgb") }, //
        YUYV_to_rgb{ program.get_kernel("YUYV_to_rgb") }, //
        bgra_to_rgba{ program.get_kernel("bgra_to_rgba") }, //
        rgba_to_rgb{ program.get_kernel("rgba_to_rgb") }, //
        rgba_to_bgr{ program.get_kernel("rgba_to_bgr") }, //
        bgr_to_rgb{ program.get_kernel("bgr_to_rgb") }, //
        rgb_to_gray{ program.get_kernel("rgb_to_gray") }, //
        rgba_to_gray{ program.get_kernel("rgba_to_gray") }, //
        bgr_to_gray{ program.get_kernel("bgr_to_gray") }, //
        bgra_to_gray{ program.get_kernel("bgra_to_gray") }, //
        yuyv_to_gray{ program.get_kernel("yuyv_to_gray") }, //
        rgb_to_rgb{ program.get_kernel("rgb_to_rgb") }, //
        rgba_to_rgba{ program.get_kernel("rgba_to_rgba") }
  {
  }

  ax_utils::CLProgram::flush_details run_kernel(
      cl_kernel k, const buffer_details &out, buffer &outbuf, bool start_flush)
  {
    size_t global_work_size[3] = { 1, 1, 1 };
    const int numpix_per_kernel = 1;
    global_work_size[0] = out.width;
    global_work_size[1] = out.height;
    error = program.execute_kernel(k, 2, global_work_size);
    if (error != CL_SUCCESS) {
      throw std::runtime_error("Unable to execute kernel. Error code: "
                               + ax_utils::cl_error_to_string(error));
    }
    if (start_flush) {
      //  Here the downstream does not support OpenCL buffers, so start the
      //  mapping now.
      return program.start_flush_output_buffer(outbuf, out.stride * out.height);
    }
    return {};
  }

  int run_kernel(cl_kernel k, const buffer_details &out, buffer &inbuf,
      buffer &outbuf, bool start_flush)
  {
    auto details = run_kernel(k, out, outbuf, start_flush);
    if (details.event) {
      // The downstream does not support OpenCL buffers so the buffer has begun
      //  mapping to system memory. The event will be signalled when complete.
      //  Store this away so that when the buffer is mapped we just wait on the
      //  event.
      if (auto *p = std::get_if<opencl_buffer *>(&out.data)) {
        (*p)->event = std::move(details.event);
        (*p)->mapped = details.mapped;
      } else {
        clWaitForEvents(1, &*details.event);
        details.event.reset();
      }
    }
    return 0;
  }


  int run_nv12_to_rgba(const buffer_details &in, const buffer_details &out,
      cl_char is_bgr, const cl_extensions &extensions, bool start_flush)
  {
    const int rgba_size = 4;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto inpbuf = program.create_buffers(1, ax_utils::determine_buffer_size(in),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, in.data, in.offsets.size());

    cl_int y_stride = in.strides[0];
    cl_int uv_stride = in.strides[1];
    cl_int uv_offset = in.offsets[1];
    //  Set the kernel arguments
    auto kernel = AxVideoFormatNumChannels(out.format) == 3 ? *nv12_to_rgb : *nv12_to_rgba;
    program.set_kernel_args(kernel, 0, out.width, out.height, y_stride,
        uv_stride, out.stride, uv_offset, is_bgr, *inpbuf[0], *outbuf);
    return run_kernel(kernel, out, inpbuf[0], outbuf, start_flush);
  }

  int run_i420_to_rgba(const buffer_details &in, const buffer_details &out,
      cl_char is_bgr, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    cl_int u_stride = in.strides[1];
    cl_int v_stride = in.strides[2];
    cl_int u_offset = in.offsets[1];
    cl_int v_offset = in.offsets[2];
    //  Set the kernel arguments
    auto kernel = AxVideoFormatNumChannels(out.format) == 3 ? *i420_to_rgb : *i420_to_rgba;
    program.set_kernel_args(kernel, 0, out.width, out.height, y_stride, u_stride,
        v_stride, out.stride, u_offset, v_offset, is_bgr, *inpbuf, *outbuf);
    return run_kernel(kernel, out, inpbuf, outbuf, start_flush);
  }

  int run_YUYV_to_rgba(const buffer_details &in, const buffer_details &out,
      cl_char is_bgr, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    //  Set the kernel arguments
    auto kernel = AxVideoFormatNumChannels(out.format) == 3 ? *YUYV_to_rgb : *YUYV_to_rgba;
    program.set_kernel_args(kernel, 0, out.width, out.height, y_stride,
        out.stride, is_bgr, *inpbuf, *outbuf);
    return run_kernel(kernel, out, inpbuf, outbuf, start_flush);
  }

  int run_bgra_to_rgba(const buffer_details &in, const buffer_details &out, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    auto kernel = *bgra_to_rgba;
    if ((in.format == AxVideoFormat::RGB && out.format == AxVideoFormat::RGB)
        || (in.format == AxVideoFormat::BGR && out.format == AxVideoFormat::BGR)) {
      kernel = *rgb_to_rgb;
    } else if ((in.format == AxVideoFormat::RGBA && out.format == AxVideoFormat::RGBA)
               || (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::BGRA)) {
      kernel = *rgba_to_rgba;
    } else if ((in.format == AxVideoFormat::RGBA && out.format == AxVideoFormat::RGB)
               || (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::BGR)) {
      kernel = *rgba_to_rgb;
    } else if ((in.format == AxVideoFormat::RGBA && out.format == AxVideoFormat::BGR)
               || (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::RGB)) {
      kernel = *rgba_to_bgr;
    } else if ((in.format == AxVideoFormat::RGB && out.format == AxVideoFormat::BGR)
               || (in.format == AxVideoFormat::BGR && out.format == AxVideoFormat::RGB)) {
      kernel = *bgr_to_rgb;
    } else if ((in.format == AxVideoFormat::RGBA && out.format == AxVideoFormat::BGRA)
               || (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::RGBA)) {
      kernel = *bgra_to_rgba;
    }
    cl_int y_stride = in.stride;
    //  Set the kernel arguments
    program.set_kernel_args(kernel, 0, out.width, out.height, y_stride,
        out.stride, *inpbuf, *outbuf);
    return run_kernel(kernel, out, inpbuf, outbuf, start_flush);
  }


  int run_rgb_to_gray(const buffer_details &in, const buffer_details &out, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    program.set_kernel_args(*rgb_to_gray, 0, out.width, out.height, in.stride,
        out.stride, *inpbuf, *outbuf);
    return run_kernel(*rgb_to_gray, out, inpbuf, outbuf, start_flush);
  }

  int run_rgba_to_gray(const buffer_details &in, const buffer_details &out, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    program.set_kernel_args(*rgba_to_gray, 0, out.width, out.height, in.stride,
        out.stride, *inpbuf, *outbuf);
    return run_kernel(*rgba_to_gray, out, inpbuf, outbuf, start_flush);
  }


  int run_bgr_to_gray(const buffer_details &in, const buffer_details &out, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    program.set_kernel_args(*bgr_to_gray, 0, out.width, out.height, in.stride,
        out.stride, *inpbuf, *outbuf);
    return run_kernel(*bgr_to_gray, out, inpbuf, outbuf, start_flush);
  }

  int run_bgra_to_gray(const buffer_details &in, const buffer_details &out, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    program.set_kernel_args(*bgra_to_gray, 0, out.width, out.height, in.stride,
        out.stride, *inpbuf, *outbuf);
    return run_kernel(*bgra_to_gray, out, inpbuf, outbuf, start_flush);
  }

  int run_yuyv_to_gray(const buffer_details &in, const buffer_details &out, bool start_flush)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    //  Set the kernel arguments
    program.set_kernel_args(*yuyv_to_gray, 0, out.width, out.height, y_stride,
        out.stride, *inpbuf, *outbuf);
    return run_kernel(*yuyv_to_gray, out, inpbuf, outbuf, start_flush);
  }

  int run(const buffer_details &in, const buffer_details &out,
      const std::string &format, const cc_properties *prop)
  {
    bool start_flush = prop && prop->downstream_supports_opencl == 0;
    cl_char is_bgr = format == "bgra" || format == "bgr";

    if (out.format == AxVideoFormat::GRAY8) {

      if (in.format == AxVideoFormat::I420 || in.format == AxVideoFormat::NV12) {
        throw std::runtime_error("I420 or NV12 to gray, should pass through instead");
      } else if (in.format == AxVideoFormat::RGB) {
        return run_rgb_to_gray(in, out, start_flush);
      } else if (in.format == AxVideoFormat::RGBA) {
        return run_rgba_to_gray(in, out, start_flush);
      } else if (in.format == AxVideoFormat::BGR) {
        return run_bgr_to_gray(in, out, start_flush);
      } else if (in.format == AxVideoFormat::BGRA) {
        return run_bgra_to_gray(in, out, start_flush);
      } else if (in.format == AxVideoFormat::YUY2) {
        return run_yuyv_to_gray(in, out, start_flush);
      }
      throw std::runtime_error("Unsupported format");
    } else {
      if (in.format == AxVideoFormat::NV12) {
        return run_nv12_to_rgba(in, out, is_bgr, program.cl_details.extensions, start_flush);
      } else if (in.format == AxVideoFormat::I420) {
        return run_i420_to_rgba(in, out, is_bgr, start_flush);
      } else if (in.format == AxVideoFormat::YUY2) {
        return run_YUYV_to_rgba(in, out, is_bgr, start_flush);
      } else if (in.format == AxVideoFormat::RGB && out.format == AxVideoFormat::RGB) {
        return run_bgra_to_rgba(in, out, start_flush);
      } else if (in.format == AxVideoFormat::BGR && out.format == AxVideoFormat::BGR) {
        return run_bgra_to_rgba(in, out, start_flush);
      } else if (in.format == AxVideoFormat::RGB && out.format == AxVideoFormat::BGR) {
        return run_bgra_to_rgba(in, out, start_flush);
      } else if (in.format == AxVideoFormat::BGR && out.format == AxVideoFormat::RGB) {
        return run_bgra_to_rgba(in, out, start_flush);
      } else if (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::BGRA) {
        return run_bgra_to_rgba(in, out, start_flush);
      } else if (in.format == AxVideoFormat::RGBA && out.format == AxVideoFormat::RGBA) {
        return run_bgra_to_rgba(in, out, start_flush);
      } else if (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::RGBA) {
        return run_bgra_to_rgba(in, out, start_flush);
      } else if (in.format == AxVideoFormat::RGBA || in.format == AxVideoFormat::BGRA) {
        return run_bgra_to_rgba(in, out, start_flush);
      }
      auto error = "Unsupported formats in color conversion: "s
                   + AxVideoFormatToString(in.format) + " to "s
                   + AxVideoFormatToString(out.format);
      throw std::runtime_error(error);
    }
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
  kernel nv12_to_rgb;
  kernel i420_to_rgb;
  kernel YUYV_to_rgb;
  kernel bgra_to_rgba;
  kernel rgba_to_rgb;
  kernel rgba_to_bgr;
  kernel bgr_to_rgb;
  kernel rgb_to_gray;
  kernel rgba_to_gray;
  kernel bgr_to_gray;
  kernel bgra_to_gray;
  kernel yuyv_to_gray;
  kernel rgb_to_rgb;
  kernel rgba_to_rgba;
};

std::string_view flips[] = {
  "none",
  "clockwise",
  "rotate-180",
  "counterclockwise",
  "horizontal-flip",
  "vertical-flip",
  "upper-left-diagonal",
  "upper-right-diagonal",
};

int
determine_flip_type(std::string_view flip)
{
  auto it = std::find(std::begin(flips), std::end(flips), flip);
  return it != std::end(flips) ? std::distance(std::begin(flips), it) : -1;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "format",
    "flip_method",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties_with_context(
    const std::unordered_map<std::string, std::string> &input, void *context, Ax::Logger &logger)
{
  auto prop = std::make_shared<cc_properties>();
  prop->format = Ax::get_property(input, "format", "ColorConvertProperties", prop->format);
  prop->flip_method = Ax::get_property(
      input, "flip_method", "ColorConvertProperties", prop->flip_method);
  auto flip_type = determine_flip_type(prop->flip_method);
  if (flip_type == -1) {
    logger(AX_ERROR) << "Invalid flip_method type: " << prop->flip_method
                     << " defaulting to none" << std::endl;
    flip_type = 0;
  }
  prop->color_convert = std::make_unique<CLColorConvert>(kernel_cl, flip_type,
      static_cast<ax_utils::opencl_details *>(context), logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    cc_properties *prop, Ax::Logger & /*logger*/)
{
  prop->downstream_supports_opencl = Ax::get_property(input, "downstream_supports_opencl",
      "ColorConvertProperties", prop->downstream_supports_opencl);
}

bool
is_a_rotate(std::string_view flip_type)
{
  static std::string_view rotates[] = {
    "clockwise",
    "counterclockwise",
    "upper-left-diagonal",
    "upper-right-diagonal",
  };
  return std::find(std::begin(rotates), std::end(rotates), flip_type) != std::end(rotates);
}

struct {
  std::string color;
  AxVideoFormat format;
} valid_formats[] = {
  { "rgba", AxVideoFormat::RGBA },
  { "bgra", AxVideoFormat::BGRA },
  { "rgb", AxVideoFormat::RGB },
  { "bgr", AxVideoFormat::BGR },
  { "gray", AxVideoFormat::GRAY8 },
};

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const cc_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    auto stride_factor = (prop->format == "gray") ? 1 : 4;
    out_info.info.stride = out_info.info.width * stride_factor;
    if (is_a_rotate(prop->flip_method)) {
      std::swap(out_info.info.width, out_info.info.height);
      out_info.info.stride = out_info.info.width * stride_factor;
      out_info.strides = { size_t(out_info.info.stride) };
    };
    auto fmt_found = std::find_if(std::begin(valid_formats), std::end(valid_formats),
        [fmt = prop->format](auto f) { return f.color == fmt; });
    if (fmt_found == std::end(valid_formats)) {
      logger(AX_ERROR)
          << "Invalid output format given in color conversion: " << prop->format
          << std::endl;
      throw std::runtime_error(
          "Invalid output format given in color conversion: " + prop->format);
    }
    logger(AX_INFO) << "Setting output format to " << prop->format << std::endl;
    out_info.info.format = fmt_found->format;
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
  // When output is GRAY and input is NV12 or I420, we can pass through, as the yuv image already has the gray image as luminance (Y) component in the beginning of the buffer
  bool gray_out_bypass = (input_details[0].format == AxVideoFormat::I420
                             || input_details[0].format == AxVideoFormat::NV12)
                         && (output_details[0].format == AxVideoFormat::GRAY8
                             && input_details[0].width == output_details[0].width
                             && input_details[0].height == output_details[0].height);

  auto flip_type = determine_flip_type(prop->flip_method);
  if (flip_type == -1) {
    flip_type = 0;
  }

  return (flip_type == 0 && input_details[0].format == output_details[0].format
             && input_details[0].width == output_details[0].width
             && input_details[0].height == output_details[0].height)
         || gray_out_bypass;
}


extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
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
  if (std::holds_alternative<void *>(input_details[0].data)) {
    const int pagesize = 4096;
    auto ptr = std::get<void *>(input_details[0].data);
    if ((reinterpret_cast<uintptr_t>(ptr) & (pagesize - 1)) != 0) {
      logger(AX_DEBUG) << "Input buffer is not page aligned" << std::endl;
    }
  }
  prop->color_convert->run(input_details[0], output_details[0], prop->format, prop);
}

extern "C" bool
query_supports(Ax::PluginFeature feature, const cc_properties *prop, Ax::Logger &logger)
{
  if (feature == Ax::PluginFeature::opencl_buffers) {
    return true;
  }
  if (feature == Ax::PluginFeature::dmabuf_buffers) {
    return prop->color_convert->can_use_dmabuf();
  }
  return Ax::PluginFeatureDefaults(feature);
}
