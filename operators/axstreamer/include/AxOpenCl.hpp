// Copyright Axelera AI, 2023
#pragma once

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
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

constexpr int AX_ALLOCATION_CONTEXT_VERSION = 2;
struct AxAllocationContext {
  int version{ 0 };
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;
  cl_extensions extensions;
  std::exception_ptr exception{ nullptr };
  cl_command_queue map_commands;
};

namespace ax_utils
{

// Type trait: true for any std::vector<U> specialisation.
template <typename T> struct is_std_vector : std::false_type {
};
template <typename U> struct is_std_vector<std::vector<U>> : std::true_type {
};

using opencl_details = AxAllocationContext;


opencl_details build_cl_details(Ax::Logger &logger, const char *which_cl, void *display);

AxAllocationContextHandle clone_context(AxAllocationContext *context);
opencl_details copy_context_and_retain(opencl_details *context);

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

  int acquireva(std::span<cl_mem> input_buffers);

  int releaseva(std::span<cl_mem> input_buffers);

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

std::string get_kernel_utils(int rotate_type = 0);

std::string get_rotation(int rotate_type);

int run_kernel(CLProgram &program, cl_kernel k, const buffer_details &in,
    const buffer_details &out, CLProgram::ax_buffer &inbuf,
    CLProgram::ax_buffer &outbuf, bool start_flush);


std::array<float, 16> get_color_conversion_matrix(
    AxVideoFormat in_format, AxVideoFormat out_format);

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

enum class Interpolation { nearest, bilinear };

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
