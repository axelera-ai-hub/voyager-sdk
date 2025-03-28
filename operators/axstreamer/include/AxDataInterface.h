// Copyright Axelera AI, 2025
#pragma once

#ifdef __cplusplus
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

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
  I420,
  YUY2,
};

int AxVideoFormatNumChannels(AxVideoFormat format);

AxVideoFormat AxVideoFormatFromString(const std::string &format);

/// @brief Convert AxVideoFormat to string, return "UNDEFINED" if format is not
/// a valid format.
std::string AxVideoFormatToString(AxVideoFormat format);


struct AxVideoInfo {
  int width = 0;
  int height = 0;
  int stride = 0;
  int offset = 0;
  AxVideoFormat format = AxVideoFormat::UNDEFINED;
  bool cropped;
  int x_offset;
  int y_offset;
};

struct AxVideoInterface {
  AxVideoInfo info;
  void *data{};
  std::vector<size_t> strides;
  std::vector<size_t> offsets;
  int fd = -1;
};


struct AxTensorInterface {
  std::vector<int> sizes;
  int bytes;
  void *data;
  int fd;
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
