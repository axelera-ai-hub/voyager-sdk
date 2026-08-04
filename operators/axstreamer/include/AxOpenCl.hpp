// Copyright Axelera AI, 2023
#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include <algorithm>
#include <array>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenClExtensions.hpp"

//  AxAllocationContext and AX_ALLOCATION_CONTEXT_VERSION are now defined in
//  AxOpenClExtensions.hpp (included above) so they are available to both the
//  operators build and axelera_runtime2 without duplication.

namespace ax_utils
{

// Type trait: true for any std::vector<U> specialisation.
template <typename T> struct is_std_vector : std::false_type {
};
template <typename U> struct is_std_vector<std::vector<U>> : std::true_type {
};

//  opencl_details and copy_context_and_retain are declared in AxOpenClExtensions.hpp.

opencl_details build_cl_details(Ax::Logger &logger, const char *which_cl, void *display);

AxAllocationContextHandle clone_context(AxAllocationContext *context);

std::string cl_error_to_string(cl_int code);

class CLProgram
{
  public:
  explicit CLProgram(const std::string &source, opencl_details *display, Ax::Logger &logger);

  explicit CLProgram(const std::string &source, Ax::Logger &logger)
      : CLProgram(source, nullptr, logger)
  {
  }


  using ax_kernel = cl_object<cl_kernel>;
  using ax_buffer = cl_object<cl_mem>;
  using ax_event = cl_object<cl_event>;

  using buffer_initializer
      = std::variant<void *, int, opencl_planes *, opencl_buffer *, VASurfaceID_proxy *>;
  // The class is not copyable
  CLProgram(const CLProgram &) = delete;
  CLProgram &operator=(const CLProgram &) = delete;

  ax_kernel build_kernel_from_source(const std::string &source, const std::string &kernel_name);

  /// @brief Get a handle to the requested kernel
  /// @param kernel_name - The name of the kernel
  /// @return The kernel handle, if null error holds the status code
  /// @throw std::runtime_error if the kernel is not found
  ax_kernel get_kernel(const std::string &kernel_name) const;

  /// @brief Create a buffer on the device
  /// @param elem_size - The size of the elements in the buffer
  /// @param num_elemes - The number of elements in the buffer
  /// @param flags - Whether R/W/RW
  /// @param ptr - The data to copy into the buffer (or nullptr if it will be written later)
  ///            or a file descriptor if the buffer is to be created from a dma_buf
  /// @return A handle to the buffer
  std::vector<ax_buffer> create_buffers(int elem_size, int num_elemes,
      int flags, const buffer_initializer &ptr, int num_planes) const;

  ax_buffer create_buffer(int elem_size, int num_elemes, int flags,
      const buffer_initializer &ptr, int num_planes) const;

  /// @brief Create a buffer on the device from the description
  /// @param details - The buffer details that decide the size and type of the buffer
  /// @param flags - Whether R/W/RW
  /// @return A handle to the buffer
  ax_buffer create_buffer(const buffer_details &details, int flags);

  /// @brief  Set a kernel argument of tyoe T
  /// @param kernel - The kernel handle
  /// @param arg_index - The index of the argument
  /// @param arg - The actual argument
  /// @return - Any status code

  /// @brief  Single-argument kernel arg setter.  Uses if constexpr to dispatch:
  ///   - std::vector<ax_buffer>  → expands each cl_mem as a separate argument
  ///   - std::vector<U>          → sets the entire contiguous buffer as one argument
  ///   - anything else           → sets sizeof(arg) bytes from &arg
  template <typename T>
  int set_kernel_args(cl_kernel kernel, int arg_index, T &&arg)
  {
    using D = std::decay_t<T>;
    if constexpr (std::is_same_v<D, std::vector<ax_buffer>>) {
      for (const auto &buf : arg) {
        arg_index = set_kernel_args(kernel, arg_index, *buf);
      }
      return arg_index;
    } else if constexpr (is_std_vector<D>::value) {
      if (auto error = clSetKernelArg(
              kernel, arg_index, sizeof(arg[0]) * arg.size(), arg.data());
          error != CL_SUCCESS) {
        throw std::runtime_error("Failed to set kernel argument " + std::to_string(arg_index)
                                 + ", error: " + ax_utils::cl_error_to_string(error));
      }
      return arg_index + 1;
    } else {
      if (auto error = clSetKernelArg(kernel, arg_index, sizeof arg, &arg); error != CL_SUCCESS) {
        throw std::runtime_error("Failed to set kernel argument " + std::to_string(arg_index)
                                 + ", error: " + ax_utils::cl_error_to_string(error));
      }
      return arg_index + 1;
    }
  }


  /// @brief  Sets multiple kernel arguments of varying types
  /// @param kernel - The kernel handle
  /// @param arg_index - The index of the first argument
  /// @param arg - The first argument
  /// @param rest - The rest of the arguments
  /// @return - Any status code
  template <typename T, typename... Rest>
  int set_kernel_args(cl_kernel kernel, int arg_index, T &&arg, Rest &&...rest)
  {
    arg_index = set_kernel_args(kernel, arg_index, std::forward<T>(arg));
    return set_kernel_args(kernel, arg_index, std::forward<Rest>(rest)...);
  }

  /// @brief Execute a kernel
  /// @param kernel - The kernel handle
  /// @param num_dims - The number of dimensions
  /// @param global_work_size - The actual dimensions
  ax_event execute_kernel(cl_kernel kernel, int num_dims,
      size_t global_work_size[3], ax_event event);

  bool can_use_dmabuf() const
  {
    return can_import_dmabuf(cl_details.extensions);
  }

  // True when DMA-bufs are imported via cl_khr_external_memory and therefore
  // require explicit acquire/release around kernel use.  False when the ARM
  // import path is used (which handles synchronisation internally).
  bool uses_khr_dmabuf_import() const
  {
    return cl_details.extensions.hasKhrDmaBufImport
           && cl_details.extensions.clEnqueueAcquireExternalMemObjectsKHR
           && cl_details.extensions.clEnqueueReleaseExternalMemObjectsKHR
           && !cl_details.extensions.clImportMemoryARM_dmabuf;
  }

  bool has_fp16() const
  {
    return cl_details.extensions.has_fp16;
  }

  int acquireva(std::span<cl_mem> input_buffers);

  int releaseva(std::span<cl_mem> input_buffers);

  ax_event acquire_dmabuf_khr(std::span<cl_mem> buffers, ax_event wait_event);

  ax_event release_dmabuf_khr(std::span<cl_mem> buffers, ax_event wait_event);

  bool can_use_va() const
  {
    return can_import_va(cl_details.extensions);
  }


  struct flush_details {
    int result{};
    ax_event event{ nullptr };
    void *mapped{};
  };

  flush_details flush_output_buffer_async(const ax_buffer &out, int size, ax_event ev);

  flush_details start_flush_output_buffer(const ax_buffer &out, int size, ax_event ev);

  int unmap_buffer(ax_event event, const ax_buffer &out, void *mapped);

  ~CLProgram();

  Ax::Logger &logger;
  opencl_details cl_details;

  private:
  bool has_host_arm_import{};
  bool has_dma_buf_arm_import{};

  cl_program program{};
  static std::mutex cl_mutex;
  size_t max_work_group_size{};
  //  This is to workaround and issue on Rusticl on Raspberry Pi
  //  We get an INVALID_GROUP_SIZE error if we pass a local groupsize
  //  of 16x16. This is despite cl reporting max_work_group_size as
  //  256. If the first call to execute kernel fails, we set this which
  //  makes all subsequent calls force OpenCL to choose the size.
  bool RPi_Hack{};
};

// Converts a float32 to float16 with round-to-nearest-even.
// Assumes: f is a normal finite value in the half-precision normal range
// [-65504, 65504]. Passing subnormals, NaN, or infinity gives wrong results.
//
// float32: [s:1][exp:8 bias=127][mantissa:23]
// float16: [s:1][exp:5 bias=15 ][mantissa:10]
inline cl_half
float_to_half(float f)
{
  // 23 - 10: bits dropped when truncating f32 mantissa to f16 mantissa.
  constexpr int kMantissaShift = 13;
  constexpr uint32_t kRoundMask = (1u << kMantissaShift) - 1u;
  constexpr uint32_t kRoundHalf = 1u << (kMantissaShift - 1);
  constexpr int kF16MaxExp = 31; // 5-bit exp field: 31 = inf/NaN
  constexpr uint16_t kF16SignBit = 0x8000u;
  constexpr uint16_t kF16Inf = 0x7C00u;

  uint32_t x{};
  memcpy(&x, &f, sizeof(x));
  uint16_t sign = static_cast<uint16_t>(x >> 16) & kF16SignBit;
  int exp = ((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mantissa = x & 0x7FFFFFu;
  if (exp <= 0)
    return sign;
  if (exp >= kF16MaxExp)
    return static_cast<cl_half>(sign | kF16Inf);
  uint16_t half = sign | (static_cast<uint16_t>(exp) << 10)
                  | static_cast<uint16_t>(mantissa >> kMantissaShift);
  uint32_t remainder = mantissa & kRoundMask;
  if (remainder > kRoundHalf || (remainder == kRoundHalf && (half & 1u)))
    half += 1u; // carries correctly into exp on mantissa overflow
  return static_cast<cl_half>(half);
}

inline cl_half4
to_half4(const std::array<float, 4> &f)
{
  return { float_to_half(f[0]), float_to_half(f[1]), float_to_half(f[2]),
    float_to_half(f[3]) };
}

std::string get_kernel_utils(int rotate_type = 0, bool use_fp16 = false);

std::string get_rotation(int rotate_type);

int run_kernel(CLProgram &program, cl_kernel k, const buffer_details &in,
    const buffer_details &out, CLProgram::ax_buffer &inbuf,
    CLProgram::ax_buffer &outbuf, bool start_flush);


cl_float16 get_color_conversion_matrix(AxVideoFormat in_format, AxVideoFormat out_format);

cl_float16 get_color_conversion_matrix_with_norm(AxVideoFormat in_format,
    AxVideoFormat out_format, const std::vector<cl_float> &mul,
    const std::vector<cl_float> &add);

std::array<cl_int, 4> build_strides(const buffer_details &in, const buffer_details &out);

std::array<cl_int, 4> build_offsets(
    const buffer_details &in, const buffer_details &out, int num_planes = 1);

struct kernel_arg_details {
  AxVideoFormat in_format;
  std::string out_type;
  std::array<std::string, 3> samplers; // [0]=single-plane, [1]=two-plane, [2]=three-plane; "" → use [0]
};

struct kernel_args {
  std::string out_type;
  std::string sampler;
  std::string input_params;
};

enum class Interpolation { nearest, bilinear, pillow_bilinear };

kernel_args get_input_details(
    AxVideoFormat format, Interpolation interp, int num_planes = 1);

inline int
get_num_planes(const buffer_details &in)
{
  if (auto *p = std::get_if<opencl_planes *>(&in.data); p && *p) {
    auto n = (*p)->planes.size();
    if (0 < n && n <= 3) {
      return n;
    }
    throw std::runtime_error(
        "Invalid number of planes, should 1,2 or 3 but given " + std::to_string(n));
  }
  return 1;
}

// Returns the extra kernel parameter declaration(s) for multi-plane input.
// 1 → ""
// 2 → "__global const uchar *in_uv, "
// 3 → "__global const uchar *in_u, __global const uchar *in_v, "
inline const char *
uv_kernel_params(int num_planes)
{
  switch (num_planes) {
    case 2:
      return "__global const uchar *in_uv, ";
    case 3:
      return "__global const uchar *in_u, __global const uchar *in_v, ";
    default:
      return "";
  }
}

kernel_args get_output_details(AxVideoFormat in_format, AxVideoFormat out_format);
kernel_args get_output_norm_details(AxVideoFormat in_format, AxVideoFormat out_format);


} // namespace ax_utils
