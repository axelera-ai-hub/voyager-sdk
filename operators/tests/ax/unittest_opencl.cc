// Copyright Axelera AI, 2025
#include "unittest_ax_common.h"

#include "AxOpenCl.hpp"

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
} // namespace opencl_tests
