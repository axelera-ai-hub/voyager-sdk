// Copyright Axelera AI, 2025
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
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenClExtensions.hpp"

namespace ax_utils
{

inline void
release_clobject(cl_mem obj)
{
  if (obj)
    clReleaseMemObject(obj);
}

inline void
release_clobject(cl_kernel obj)
{
  if (obj)
    clReleaseKernel(obj);
}

inline void
retain_clobject(cl_mem obj)
{
  if (obj)
    clRetainMemObject(obj);
}

inline void
retain_clobject(cl_kernel obj)
{
  if (obj)
    clRetainKernel(obj);
}
template <typename T> class cl_object
{
  public:
  explicit cl_object(T obj) : object(obj)
  {
  }

  cl_object(const cl_object &rhs)
  {
    object = rhs.object;
    if (object) {
      retain_clobject(object);
    }
  }

  cl_object &operator=(const cl_object &rhs)
  {
    auto o = rhs.object;
    if (o) {
      retain_clobject(o);
    }
    if (object) {
      release_clobject(object);
    }
    object = o;
    return *this;
  }

  T &operator*()
  {
    return object;
  }

  const T &operator*() const
  {
    return object;
  }

  ~cl_object()
  {
    if (object)
      release_clobject(object);
  }

  private:
  T object;
};

class CLProgram
{
  public:
  explicit CLProgram(const std::string &source, Ax::Logger &logger);

  using ax_kernel = cl_object<cl_kernel>;
  using ax_buffer = cl_object<cl_mem>;

  // The class is not copyable
  CLProgram(const CLProgram &) = delete;
  CLProgram &operator=(const CLProgram &) = delete;

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
  ax_buffer create_buffer(int elem_size, int num_elemes, int flags,
      const std::variant<void *, int> &ptr) const;

  /// @brief Create a buffer on the device from the description
  /// @param details - The buffer details that decide the size and type of the buffer
  /// @param flags - Whether R/W/RW
  /// @return A handle to the buffer
  ax_buffer create_buffer(const buffer_details &details, int flags);

  /// @brief Wrires data into an OpenCL buffer
  /// @param buffer - The handle to the buffer
  /// @param elem_size - The size of the elements in the buffer
  /// @param num_elems - The number of elements in the buffer
  /// @param data - Pointer to the data to write
  /// @return -
  int write_buffer(const ax_buffer &buffer, int elem_size, int num_elems, const void *data);

  /// @brief Read data from an OpenCL buffer
  /// @param buffer - The handle to the buffer
  /// @param elem_size - The size of the elements in the buffer
  /// @param num_elems - The number of elements in the buffer
  /// @param data - Pointer to the buffer to store the data
  /// @return - Any status code
  int read_buffer(const ax_buffer &buffer, int elem_size, int num_elems, void *data);

  /// @brief  Set a kernel argument of tyoe T
  /// @param kernel - The kernel handle
  /// @param arg_index - The index of the argument
  /// @param arg - The actual argument
  /// @return - Any status code

  template <typename T>
  void set_kernel_args(const ax_kernel &kernel, int arg_index, const std::vector<T> &arg)
  {
    if (auto error = clSetKernelArg(
            *kernel, arg_index, sizeof(arg[0]) * arg.size(), arg.data());
        error != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel argument " + std::to_string(arg_index)
                               + ", error = " + std::to_string(error));
    }
  }

  template <typename T>
  void set_kernel_args(const ax_kernel &kernel, int arg_index, T arg)
  {
    if (auto error = clSetKernelArg(*kernel, arg_index, sizeof arg, &arg); error != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel argument " + std::to_string(arg_index)
                               + ", error = " + std::to_string(error));
    }
  }

  /// @brief  Sets multiple kernel arguments of varying tyoes
  /// @param kernel - The kernel handle
  /// @param arg_index - The index of the first argument
  /// @param arg - The first argument
  /// @param rest - The rest of the arguments
  /// @return - Any status code
  template <typename T, typename... Rest>
  void set_kernel_args(const ax_kernel &kernel, int arg_index, T arg, Rest... rest)
  {
    set_kernel_args(kernel, arg_index, arg);
    set_kernel_args(kernel, arg_index + 1, rest...);
  }

  /// @brief Execute a kernel
  /// @param kernel - The kernel handle
  /// @param num_dims - The number of dimensions
  /// @param global_work_size - The actual dimensions
  int execute_kernel(const ax_kernel &kernel, int num_dims, size_t global_work_size[3]);

  /// @brief Ensures the output buffer is mapped to the host
  /// @param out - The buffer to map
  /// @param size - The size of the buffer
  /// @return - Any status code
  int flush_output_buffer(const ax_buffer &out, int size);

  bool can_use_dmabuf() const
  {
    return extensions.clImportMemoryARM_dmabuf != nullptr;
  }

  struct flush_details {
    int result{};
    cl_event event{};
    void *mapped{};
  };

  flush_details flush_output_buffer_async(const ax_buffer &out, int size);

  int unmap_buffer(cl_event event, const ax_buffer &out, void *mapped);

  ~CLProgram();

  cl_extensions extensions;


  protected:
  private:
  int unmap_buffer(const ax_buffer &out, void *mapped);

  Ax::Logger &logger;
  cl_device_id device_id{};
  cl_context context{};
  cl_command_queue commands{};
  cl_program program{};
  size_t max_work_group_size{};
  static std::mutex cl_mutex;
};

const char *get_kernel_utils();

} // namespace ax_utils
