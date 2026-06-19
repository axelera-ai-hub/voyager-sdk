// Copyright Axelera AI, 2025
#include "AxOpenClExtensions.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <vector>

using namespace std::string_literals;

namespace ax_utils
{
std::string cl_error_to_string(cl_int code);
}

#ifndef CL_VERSION_3_0
cl_mem
clCreateBufferWithProperties(cl_context, const cl_mem_properties *,
    cl_mem_flags, size_t, void *, cl_int *errcode_ret)
{
  if (errcode_ret)
    *errcode_ret = CL_INVALID_OPERATION;
  return nullptr;
}
#endif

bool
has_extension(cl_platform_id platform, const std::string &name)
{
  size_t extSize;
  clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, NULL, &extSize);

  auto extensions = std::vector<char>(extSize);
  clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, extSize, extensions.data(), NULL);
  return std::search(extensions.begin(), extensions.end(), name.begin(), name.end())
         != extensions.end();
}

/// @brief Load an OpenCL extension
/// @param name - The name of the extension
/// @return Pointer to the extension function
void *
load_extension(cl_platform_id platform, const std::string &feature_name,
    const std::string &function_name)
{
  return has_extension(platform, feature_name) ?
             clGetExtensionFunctionAddressForPlatform(platform, function_name.c_str()) :
             nullptr;
}

bool
is_aligned(void *ptr, size_t alignment)
{
  return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

cl_mem
arm_host_import(cl_context ctx, cl_extensions extensions, int flags, void *ptr, int size)
{
#ifdef CL_IMPORT_TYPE_HOST_ARM
  constexpr size_t cache_size = 64;
  if (extensions.clImportMemoryARM_host && is_aligned(ptr, cache_size)) {
    if ((flags & CL_MEM_WRITE_ONLY) == 0) {
      cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_HOST_ARM,
        0,
      };
      cl_int error{};
      auto buffer = extensions.clImportMemoryARM_host(
          ctx, CL_MEM_READ_ONLY, properties, ptr, size, &error);
      if (error == CL_SUCCESS) {
        return buffer;
      }
    }
  }
#endif
  return {};
}

std::vector<cl_mem>
dmabuf_import(cl_context ctx, cl_extensions extensions, int flags,
    std::variant<void *, int, opencl_planes *, opencl_buffer *, VASurfaceID_proxy *> ptr,
    int size)
{
  if (auto *pfd = std::get_if<int>(&ptr)) {
    if (extensions.clImportMemoryARM_dmabuf) {
#ifdef CL_IMPORT_TYPE_DMA_BUF_ARM

      cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_DMA_BUF_ARM,
        0,
      };
      cl_int error{};
      auto buffer = extensions.clImportMemoryARM_dmabuf(
          ctx, flags, properties, pfd, size, &error);
      if (error != CL_SUCCESS) {
        throw std::runtime_error("Failed to create buffer, error: "
                                 + ax_utils::cl_error_to_string(error));
      }
      return { buffer };
#endif
    } else if (extensions.hasKhrDmaBufImport) {
      cl_mem_properties properties[] = {
        CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
        static_cast<cl_mem_properties>(*pfd),
        0,
      };
      cl_int error{};
      auto stripped_flags
          = flags & ~(CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR);
      auto buffer = clCreateBufferWithProperties(
          ctx, properties, stripped_flags, size, nullptr, &error);
      if (error != CL_SUCCESS) {
        throw std::runtime_error("Failed to create buffer with properties, error: "
                                 + ax_utils::cl_error_to_string(error));
      }
      return { buffer };
    } else {
      throw std::runtime_error("Import of dmabuf is not supported");
    }
  }
  return {};
}

int
get_device_id(cl_platform_id platform, cl_device_id *device_id, cl_uint *num_devices,
    const cl_extensions &extensions, const std::string &which_cl)
{
  if (extensions.display) {
#if defined(HAS_VAAPI_MEDIA_SHARING)
    return extensions.clGetDeviceIDsFromVA(platform, CL_VA_API_DISPLAY_INTEL,
        extensions.display, CL_PREFERRED_DEVICES_FOR_VA_API_INTEL, 1, device_id, num_devices);
#endif
  }

  auto device_types = which_cl == "cpu" ?
                          std::array<cl_device_type, 3>{
                            CL_DEVICE_TYPE_CPU,
                            CL_DEVICE_TYPE_GPU,
                            CL_DEVICE_TYPE_ALL,
                          } :
                          std::array<cl_device_type, 3>{
                            CL_DEVICE_TYPE_GPU,
                            CL_DEVICE_TYPE_CPU,
                            CL_DEVICE_TYPE_ALL,
                          };

  cl_int result = CL_DEVICE_NOT_AVAILABLE;
  for (const auto &device_type : device_types) {
    result = clGetDeviceIDs(platform, device_type, 1, device_id, num_devices);
    if (result == CL_SUCCESS) {
      break;
    }
  }
  return result;
}

bool
can_import_dmabuf(const cl_extensions &extensions)
{
  return extensions.hasKhrDmaBufImport || extensions.clImportMemoryARM_dmabuf != nullptr;
}

bool
can_import_va(const cl_extensions &extensions)
{
  return extensions.display != nullptr;
}

int
acquire_va(cl_command_queue commands, const cl_extensions &extensions, std::span<cl_mem> buffers)
{
#if defined(HAS_VAAPI_MEDIA_SHARING)
  if (extensions.clEnqueueAcquireVA) {
    return extensions.clEnqueueAcquireVA(
        commands, buffers.size(), buffers.data(), 0, NULL, NULL);
  }
#endif
  return CL_SUCCESS;
}

int
release_va(cl_command_queue commands, const cl_extensions &extensions, std::span<cl_mem> buffers)
{
#if defined(HAS_VAAPI_MEDIA_SHARING)
  if (extensions.clEnqueueReleaseVA) {
    return extensions.clEnqueueReleaseVA(
        commands, buffers.size(), buffers.data(), 0, NULL, NULL);
  }
#endif
  return CL_SUCCESS;
}

cl_context
create_context(cl_platform_id platform, cl_device_id device, const cl_extensions &extensions)
{
  cl_int error;
  std::vector<cl_context_properties> props{
    CL_CONTEXT_PLATFORM,
    (cl_context_properties) platform,
  };
#if defined(HAS_VAAPI_MEDIA_SHARING)
  if (extensions.display != nullptr) {
    props.push_back(CL_CONTEXT_VA_API_DISPLAY_INTEL);
    props.push_back((cl_context_properties) extensions.display);
    props.push_back(CL_CONTEXT_INTEROP_USER_SYNC);
    props.push_back(CL_FALSE);
  }
#endif
  props.push_back(0);
  auto context = clCreateContext(props.data(), 1, &device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL context, error: "
                             + ax_utils::cl_error_to_string(error));
  }
  return context;
}


cl_extensions
init_extensions(cl_platform_id platform, void *display)
{
  cl_extensions extensions{
    .clImportMemoryARM_host = reinterpret_cast<clImportMemoryARM_fn>(
        load_extension(platform, "cl_arm_import_memory_host", "clImportMemoryARM")),
    .clImportMemoryARM_dmabuf = reinterpret_cast<clImportMemoryARM_fn>(
        load_extension(platform, "cl_arm_import_memory_dma_buf", "clImportMemoryARM")),
    .hasKhrDmaBufImport = has_extension(platform, "cl_khr_external_memory_dma_buf")
                          && has_extension(platform, "cl_khr_external_memory"),
    .clEnqueueAcquireExternalMemObjectsKHR
    = reinterpret_cast<clEnqueueAcquireExternalMemObjectsKHR_fn>(load_extension(
        platform, "cl_khr_external_memory", "clEnqueueAcquireExternalMemObjectsKHR")),
    .clEnqueueReleaseExternalMemObjectsKHR
    = reinterpret_cast<clEnqueueReleaseExternalMemObjectsKHR_fn>(load_extension(
        platform, "cl_khr_external_memory", "clEnqueueReleaseExternalMemObjectsKHR")),
#if defined(HAS_VAAPI_MEDIA_SHARING)
    .display = display,
    .clGetDeviceIDsFromVA = reinterpret_cast<clGetDeviceIDsFromVA_APIMediaINTEL_fn>(
        load_extension(platform, "cl_intel_va_api_media_sharing",
            "clGetDeviceIDsFromVA_APIMediaAdapterINTEL")),
    .clCreateFromVA = reinterpret_cast<clCreateFromVA_fn>(load_extension(platform,
        "cl_intel_va_api_media_sharing", "clCreateFromVA_APIMediaSurfaceINTEL")),
    .clEnqueueAcquireVA = reinterpret_cast<clEnqueueAcquireVA_fn>(load_extension(platform,
        "cl_intel_va_api_media_sharing", "clEnqueueAcquireVA_APIMediaSurfacesINTEL")),
    .clEnqueueReleaseVA = reinterpret_cast<clEnqueueReleaseVA_fn>(load_extension(platform,
        "cl_intel_va_api_media_sharing", "clEnqueueReleaseVA_APIMediaSurfacesINTEL"))
  };

  if (!extensions.clGetDeviceIDsFromVA || !extensions.clCreateFromVA
      || !extensions.clEnqueueAcquireVA || !extensions.clEnqueueReleaseVA) {
    extensions.display = nullptr;
#else
    .display = nullptr
#endif
  };
  return extensions;
}

std::vector<cl_mem>
create_va_buffers(cl_context /*ctx*/, cl_extensions /*extensions*/,
    int /*elem_size*/, int /*num_elems*/, int /*flags*/,
    const std::variant<void *, int, opencl_planes *, opencl_buffer *, VASurfaceID_proxy *> & /*ptr*/,
    int /*num_planes*/, int & /*error*/)
{
  //  VA-API surface sharing is no longer supported via this path.
  //  The vaapi field in AxVideoInterface is repurposed for opencl_planes.
  return {};
}

cl_mem
create_buffer(void *ptr, cl_context ctx, cl_extensions extensions, int size,
    int flags, int plane, bool cl_buf, int &error)
{
  if ((flags & CL_MEM_READ_ONLY) != 0) {
    flags &= ~(CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR);
    if (extensions.unified_memory) {
      flags |= CL_MEM_USE_HOST_PTR;
    } else {
      flags |= CL_MEM_COPY_HOST_PTR;
    }
  }
  if ((flags & CL_MEM_WRITE_ONLY) != 0 && cl_buf) {
    //  Output buffers are best allocated in device memory
    //  when we have an opencl buffer
    flags &= ~(CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR);
    flags |= CL_MEM_ALLOC_HOST_PTR;
    ptr = nullptr;
  }

  auto buffer = clCreateBuffer(ctx, flags, size, ptr, &error);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL buffer, error = "
                             + ax_utils::cl_error_to_string(error)
                             + ", flags = " + std::to_string(flags));
  }
  return buffer;
}

static cl_mem
make_plane_buffer(opencl_buffer *ocl, cl_context ctx, cl_extensions extensions,
    int flags, int &error)
{
  if (!ocl->buffer) {
    const int page_size = 4096;
    const auto aligned_size = (ocl->data.size() + page_size - 1) & ~(page_size - 1);
    ocl->buffer = create_buffer(
        ocl->data.data(), ctx, extensions, aligned_size, flags, 1, true, error);
    if (error != CL_SUCCESS) {
      throw std::runtime_error("Failed to create OpenCL plane buffer, error: "
                               + ax_utils::cl_error_to_string(error)
                               + ", flags = " + std::to_string(flags));
    }
  }
  clRetainMemObject(ocl->buffer);
  return ocl->buffer;
}

std::vector<cl_mem>
create_optimal_buffer(cl_context ctx, cl_extensions extensions, int elem_size,
    int num_elems, int flags,
    const std::variant<void *, int, opencl_planes *, opencl_buffer *, VASurfaceID_proxy *> &ptr,
    int plane, int &error)
{
  auto unaligned_size = elem_size * num_elems;
  auto buffers = dmabuf_import(ctx, extensions, flags, ptr, unaligned_size);
  if (!buffers.empty()) {
    return buffers;
  }
  if ((flags & CL_MEM_READ_ONLY) != 0) {
    if (auto *p = std::get_if<void *>(&ptr)) {
      auto buffer = arm_host_import(ctx, extensions, flags, *p, unaligned_size);
      if (buffer) {
        return { buffer };
      }
    }
    buffers = create_va_buffers(
        ctx, extensions, elem_size, num_elems, flags, ptr, plane, error);
    if (!buffers.empty()) {
      return buffers;
    }
  }
  if (std::holds_alternative<opencl_planes *>(ptr)) {
    //  Multi-memory plane path: lazily create cl_mem for each plane
    auto *planes = std::get<opencl_planes *>(ptr);
    std::vector<cl_mem> result;
    for (auto *plane : planes->planes) {
      result.push_back(make_plane_buffer(plane, ctx, extensions, flags, error));
    }
    return result;
  }
  if (std::holds_alternative<opencl_buffer *>(ptr)) {
    //  If we have an opencl buffer, then we can use it directly
    auto *ocl_buffer = std::get<opencl_buffer *>(ptr);
    if (!ocl_buffer->buffer) {
      const int page_size = 4096;
      const auto aligned_size = (unaligned_size + page_size - 1) & ~(page_size - 1);
      ocl_buffer->buffer = create_buffer(ocl_buffer->data.data(), ctx,
          extensions, aligned_size, flags, plane, true, error);
      //  Here we currently do not have a buffer so we create one.
      if (error != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL buffer, error: "
                                 + ax_utils::cl_error_to_string(error)
                                 + ", flags = " + std::to_string(flags));
      }
    }
    //  This ensures that the buffer lasts after its final use in this element
    clRetainMemObject(ocl_buffer->buffer);
    return { ocl_buffer->buffer };
  }
  if (std::holds_alternative<VASurfaceID_proxy *>(ptr)) {
    auto buffers = create_va_buffers(
        ctx, extensions, elem_size, num_elems, flags, ptr, plane, error);
    if (!buffers.empty()) {
      return buffers;
    }
    throw std::runtime_error("VA-API surface sharing is not yet implemented");
  }
  auto buffer = create_buffer(std::get<void *>(ptr), ctx, extensions,
      unaligned_size, flags, plane, false, error);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL buffer, error = "
                             + ax_utils::cl_error_to_string(error)
                             + ", flags = " + std::to_string(flags));
  }
  return { buffer };
}
