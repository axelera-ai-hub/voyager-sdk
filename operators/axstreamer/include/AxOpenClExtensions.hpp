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

#ifdef HAVE_LIBVA
#include <va/va.h>
#if __has_include("CL//cl_va_api_media_sharing_intel.h")
#include "CL/cl_va_api_media_sharing_intel.h"
#define HAS_VAAPI_MEDIA_SHARING
#endif
#endif

#include "AxDataInterface.h"

#include <memory>
#include <span>
#include <string>
#include <utility>
#include <variant>

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
release_clobject(cl_event obj)
{
  if (obj)
    clReleaseEvent(obj);
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

inline void
retain_clobject(cl_event obj)
{
  if (obj)
    clRetainEvent(obj);
}

template <typename T> class cl_object
{
  public:
  explicit cl_object(T obj)
      : object(obj)
  {
  }

  cl_object(cl_object &&rhs) noexcept
      : object(std::exchange(rhs.object, nullptr))
  {
  }

  cl_object &operator=(cl_object &&rhs) noexcept
  {
    auto o = std::exchange(rhs.object, nullptr);
    release_clobject(std::exchange(object, o));
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

  operator bool() const
  {
    return object != nullptr;
  }

  T release()
  {
    return std::exchange(object, nullptr);
  }

  void reset()
  {
    release_clobject(object);
    object = nullptr;
  }

  ~cl_object()
  {
    release_clobject(object);
  }

  // private:
  T object;
};
} // namespace ax_utils


extern "C" {
using clImportMemoryARM_fn = cl_mem (*)(cl_context context, cl_mem_flags flags,
    const cl_import_properties_arm *properties, void *memory, size_t size,
    cl_int *errorcode_ret);
#if defined(HAS_VAAPI_MEDIA_SHARING)
using clGetDeviceIDsFromVA_APIMediaINTEL_fn
    = cl_int (*)(cl_platform_id, cl_va_api_device_source_intel, void *,
        cl_va_api_device_set_intel, cl_uint, cl_device_id *, cl_uint *);

using clCreateFromVA_fn = cl_mem (*)(cl_context context, cl_mem_flags flags,
    VASurfaceID_proxy *surface, cl_uint plane, cl_int *errcode_ret);

using clEnqueueAcquireVA_fn = cl_int (*)(cl_command_queue command_queue,
    cl_uint num_objects, const cl_mem *mem_objects, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);

using clEnqueueReleaseVA_fn = cl_int (*)(cl_command_queue command_queue,
    cl_uint num_objects, const cl_mem *mem_objects, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);
#endif
}

struct cl_extensions {
  clImportMemoryARM_fn clImportMemoryARM_host{};
  clImportMemoryARM_fn clImportMemoryARM_dmabuf{};
  void *display{};
#if defined(HAS_VAAPI_MEDIA_SHARING)
  clGetDeviceIDsFromVA_APIMediaINTEL_fn clGetDeviceIDsFromVA{};
  clCreateFromVA_fn clCreateFromVA{};
  clEnqueueAcquireVA_fn clEnqueueAcquireVA{};
  clEnqueueReleaseVA_fn clEnqueueReleaseVA{};
#endif
  bool unified_memory{ false };
};

cl_extensions init_extensions(cl_platform_id platform, void *display);

std::vector<cl_mem> create_optimal_buffer(cl_context ctx,
    const cl_extensions extensions, int elem_size, int num_elems, int flags,
    const std::variant<void *, int, VASurfaceID_proxy *, opencl_buffer *> &ptr,
    int plane, cl_int &error);

cl_context create_context(cl_platform_id platform, cl_device_id device,
    const cl_extensions &extensions);

bool can_import_dmabuf(const cl_extensions &extensions);

bool can_import_va(const cl_extensions &extensions);

int get_device_id(cl_platform_id platform, cl_device_id *device_id, cl_uint *num_devices,
    const cl_extensions &extensions, const std::string &which_cl);

cl_int acquire_va(cl_command_queue commands, const cl_extensions &extensions,
    std::span<cl_mem> buffers);

cl_int release_va(cl_command_queue commands, const cl_extensions &extensions,
    std::span<cl_mem> buffers);

struct opencl_buffer {
  cl_mem buffer{ nullptr };
  ax_utils::cl_object<cl_event> event{ nullptr };
  std::span<uint8_t> data{};
  //  This is all of the GstMemory that the buffer depends on. i.e they
  //  must be around until the kernel that creates this buffer has finished
  //  executing.
  void *mapped{ nullptr };
  std::vector<void *> gst_memories{};
};
