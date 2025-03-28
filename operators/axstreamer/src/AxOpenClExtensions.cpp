// Copyright Axelera AI, 2025
#include "AxOpenClExtensions.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

#include <iostream>
// Wayland support
//
namespace
{
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
             clGetExtensionFunctionAddress(function_name.c_str()) :
             nullptr;
}
} // namespace


cl_extensions
init_extensions(cl_platform_id platform)
{
  cl_extensions extensions{
    .clImportMemoryARM_host = reinterpret_cast<clImportMemoryARM_fn>(
        load_extension(platform, "cl_arm_import_memory_host", "clImportMemoryARM")),
    .clImportMemoryARM_dmabuf = reinterpret_cast<clImportMemoryARM_fn>(
        load_extension(platform, "cl_arm_import_memory_dma_buf", "clImportMemoryARM")),
  };

  return extensions;
}

cl_mem
create_optimal_buffer(cl_context ctx, cl_extensions extensions, int elem_size,
    int num_elems, int flags, const std::variant<void *, int> &ptr, int &error)
{
  if (std::holds_alternative<int>(ptr)) {
    auto fd = std::get<int>(ptr);
    if (extensions.clImportMemoryARM_dmabuf) {
      cl_import_properties_arm properties[]
          = { CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0 };
      auto buffer = extensions.clImportMemoryARM_dmabuf(
          ctx, flags, properties, &fd, elem_size * num_elems, &error);
      if (error != CL_SUCCESS) {
        throw std::runtime_error(
            "Failed to create buffer, error = " + std::to_string(error));
      }
      return buffer;
    } else {
      throw std::runtime_error("Import of dmabuf is not supported");
    }
  }
  if (extensions.clImportMemoryARM_host) {
    if ((flags & CL_MEM_WRITE_ONLY) == 0) {
      cl_import_properties_arm properties[] = { CL_IMPORT_TYPE_HOST_ARM, 0 };
      auto size = elem_size * num_elems;
      auto buffer = extensions.clImportMemoryARM_host(
          ctx, flags, nullptr, std::get<void *>(ptr), size, &error);
      if (error == CL_SUCCESS) {
        return buffer;
      }
    }
  }
  //  If we fail to import, then fall back to creating a buffer
  return clCreateBuffer(ctx, flags, elem_size * num_elems, std::get<void *>(ptr), &error);
}

cl_context
create_context(cl_platform_id platform, cl_device_id device, const cl_extensions &extensions)
{
  cl_int error;
  auto context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to create OpenCL context, error = " + std::to_string(error));
  }
  return context;
}
