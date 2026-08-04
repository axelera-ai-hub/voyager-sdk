// Copyright Axelera AI, 2024
#pragma once

#ifdef __cplusplus
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace Ax
{
class Logger;
}

// This is an opaque handle representing an allocation context, it is used to pass
// context required for sharing some buffers (for example OpenCL) from axtransform operators
// into AxInferenceNet to allow sharing of zero copy allocation of buffers betweeen operators.
struct AxAllocationContext;

/// @brief Create a new allocation context.
/// @param which_cl The name of the OpenCL device to prefer, can be one of
///  "intel" "arm" "cpu", "gpu" "nvidea" or "auto" (default, or nullptr for default).
/// @param logger The logger to use for logging messages (or nullptr to use default logger)
extern "C" AxAllocationContext *ax_create_allocation_context(
    const char *which_cl, Ax::Logger *logger);
extern "C" void ax_free_allocation_context(AxAllocationContext *context);

struct AxAllocationContextDeleter {
  void operator()(AxAllocationContext *context) const
  {
    ax_free_allocation_context(context);
  }
};
using AxAllocationContextHandle
    = std::unique_ptr<AxAllocationContext, AxAllocationContextDeleter>;


enum class AxVideoFormat {
  UNDEFINED,
  RGB,
  RGBA,
  RGBx,
  BGR,
  BGRA,
  BGRx,
  GRAY8,
  NV12,
  NV16,
  I420,
  Y42B,
  Y444,
  YUY2,
};

inline std::string
AxToLower(std::string_view s)
{
  std::string result;
  result.reserve(s.size());
  std::transform(s.begin(), s.end(), std::back_inserter(result),
      [](unsigned char c) { return std::tolower(c); });
  return result;
}

int AxVideoFormatNumChannels(AxVideoFormat format);

AxVideoFormat AxVideoFormatFromString(const std::string &format);

/// @brief Convert AxVideoFormat to string, return "UNDEFINED" if format is not
/// a valid format.
std::string AxVideoFormatToString(AxVideoFormat format);

std::string to_string(AxVideoFormat fmt);

struct AxVideoInfo {
  int width = 0;
  int height = 0;
  int stride = 0;
  int offset = 0;
  AxVideoFormat format = AxVideoFormat::UNDEFINED;
  bool cropped = false;
  int x_offset = 0;
  int y_offset = 0;
  int actual_height = 0;
};

struct VASurfaceID_proxy {
  enum Type : int {
    vaapi_surface = 0,
    opencl = 1,
  };
  int type{ vaapi_surface };
};

struct opencl_buffer;
struct opencl_planes;

struct AxVideoInterface {
  AxVideoInfo info{};
  void *data{};
  std::vector<size_t> strides{};
  std::vector<size_t> offsets{};
  int fd = -1;
  VASurfaceID_proxy *vaapi{};
  opencl_buffer *ocl_buffer{ nullptr };
};


struct AxTensorInterface {
  std::vector<int> sizes{};
  int bytes{};
  void *data{};
  int fd{ -1 };
  opencl_buffer *ocl_buffer{ nullptr };

  size_t total() const
  {
    if (sizes.empty()) {
      return 0;
    }
    return std::accumulate(sizes.begin(), sizes.end(), size_t{ 1 }, std::multiplies<>());
  }
  size_t total_bytes() const
  {
    return total() * bytes;
  }
};

using AxTensorsInterface = std::vector<AxTensorInterface>;
using AxDataInterface = std::variant<std::monostate, AxTensorsInterface, AxVideoInterface>;

#endif
