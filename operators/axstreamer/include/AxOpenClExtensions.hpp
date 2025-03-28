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

#include "AxDataInterface.h"

#include <memory>
#include <variant>

extern "C" {
using clImportMemoryARM_fn = cl_mem (*)(cl_context context, cl_mem_flags flags,
    const cl_import_properties_arm *properties, void *memory, size_t size,
    cl_int *errorcode_ret);
}

struct cl_extensions {
  clImportMemoryARM_fn clImportMemoryARM_host;
  clImportMemoryARM_fn clImportMemoryARM_dmabuf;
};

cl_extensions init_extensions(cl_platform_id platform);

cl_mem create_optimal_buffer(cl_context ctx, const cl_extensions extensions, int elem_size,
    int num_elems, int flags, const std::variant<void *, int> &ptr, cl_int &error);

cl_context create_context(cl_platform_id platform, cl_device_id device,
    const cl_extensions &extensions);
