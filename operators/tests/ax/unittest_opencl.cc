// Copyright Axelera AI, 2025
#include "unittest_ax_common.h"

#include "AxOpenCl.hpp"

#include <vector>

static Ax::Logger defaultLogger(Ax::Severity::warning);

bool
has_opencl_platform()
{
  static bool has_opencl_platform = [] {
    cl_platform_id platformId;
    cl_uint numPlatforms;

    auto error = clGetPlatformIDs(1, &platformId, &numPlatforms);
    if (error == CL_SUCCESS) {
      cl_uint num_devices = 0;
      cl_device_id device_id;
      error = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    }
    return error == CL_SUCCESS;
  }();
  return has_opencl_platform;
}

namespace opencl_tests
{


TEST(cl_details, build_details_has_correct_version)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  auto ctx = AxAllocationContextHandle(ax_create_allocation_context("auto", nullptr));
  EXPECT_EQ(ctx->version, AX_ALLOCATION_CONTEXT_VERSION);
}

TEST(cl_details, clone_context_copies_version)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  auto ctx = AxAllocationContextHandle(ax_create_allocation_context("auto", nullptr));
  ctx->version = 999;
  auto cloned_ctx = ax_utils::clone_context(ctx.get());
  EXPECT_EQ(cloned_ctx->version, 999);
}


TEST(cl_details, cl_program_fails_with_incorrect_version)
{
  if (!has_opencl_platform()) {
    GTEST_SKIP();
  }
  auto ctx = AxAllocationContextHandle(ax_create_allocation_context("auto", nullptr));
  EXPECT_EQ(ctx->version, AX_ALLOCATION_CONTEXT_VERSION);
  ctx->version = 999; // Set to incorrect version

  EXPECT_THROW(ax_utils::CLProgram("", ctx.get(), defaultLogger), std::runtime_error);
}

TEST(cl_details, device_supports_dmabuf_import_false_without_extension)
{
  cl_platform_id platformId;
  cl_uint numPlatforms;
  if (clGetPlatformIDs(1, &platformId, &numPlatforms) != CL_SUCCESS) {
    GTEST_SKIP();
  }

  size_t extSize = 0;
  clGetPlatformInfo(platformId, CL_PLATFORM_EXTENSIONS, 0, nullptr, &extSize);
  std::vector<char> extensions(extSize);
  clGetPlatformInfo(platformId, CL_PLATFORM_EXTENSIONS, extSize, extensions.data(), nullptr);
  std::string platform_extensions(extensions.begin(), extensions.end());
  if (platform_extensions.find("cl_khr_external_memory") != std::string::npos) {
    GTEST_SKIP(); // Platform supports it; can't exercise the "absent" guard here.
  }

  // Without cl_khr_external_memory, the function must return false without ever
  // querying `device`, so an invalid device_id here must not be dereferenced.
  EXPECT_FALSE(device_supports_dmabuf_import(platformId, nullptr));
}
} // namespace opencl_tests
