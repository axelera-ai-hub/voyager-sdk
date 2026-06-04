// Copyright Axelera AI, 2026
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <vector>
#include "AxDataInterface.h"

namespace Ax
{

/// @brief A buffer wrapper for contiguous planar YUV video data
/// This class manages a contiguous buffer for video frames and provides
/// conversion utilities to cv::Mat and AxVideoInterface
class VideoBuffer
{
  public:
  /// @brief Construct an empty VideoBuffer
  VideoBuffer() = default;

  /// @brief Construct a VideoBuffer with specific dimensions and format
  /// @param width Frame width in pixels
  /// @param height Frame height in pixels
  /// @param format Video format (only I420 and NV12 supported)
  VideoBuffer(int width, int height, AxVideoFormat format);

  /// @brief Move constructor
  VideoBuffer(VideoBuffer &&other) noexcept = default;

  /// @brief Move assignment
  VideoBuffer &operator=(VideoBuffer &&other) noexcept = default;

  // Disable copy construction and assignment
  VideoBuffer(const VideoBuffer &) = delete;
  VideoBuffer &operator=(const VideoBuffer &) = delete;

  /// @brief Check if buffer is valid and allocated
  /// @return true if buffer contains valid data
  bool is_valid() const
  {
    return buffer_ != nullptr && width_ > 0 && height_ > 0;
  }

  /// @brief Get the raw data pointer
  /// @return Pointer to contiguous buffer data
  uint8_t *data()
  {
    return buffer_.get();
  }

  /// @brief Get the raw data pointer (const)
  /// @return Pointer to contiguous buffer data
  const uint8_t *data() const
  {
    return buffer_.get();
  }

  /// @brief Get frame width
  int width() const
  {
    return width_;
  }

  /// @brief Get frame height
  int height() const
  {
    return height_;
  }

  /// @brief Get video format
  AxVideoFormat format() const
  {
    return format_;
  }

  /// @brief Get total buffer size in bytes
  size_t size() const
  {
    return buffer_size_;
  }

  /// @brief Check if buffer has strides (non-contiguous planes)
  bool has_strides() const
  {
    return !strides_.empty();
  }

  /// @brief Get Y plane pointer
  uint8_t *y_plane()
  {
    return buffer_.get() + (has_strides() ? offsets_[0] : 0);
  }

  /// @brief Get U plane pointer (I420) or UV plane pointer (NV12)
  uint8_t *u_plane()
  {
    if (has_strides()) {
      return buffer_.get() + offsets_[1];
    }
    return buffer_.get() + y_plane_size();
  }

  /// @brief Get V plane pointer (I420 only, nullptr for NV12)
  uint8_t *v_plane()
  {
    if (format_ != AxVideoFormat::I420) {
      return nullptr;
    }
    if (has_strides()) {
      return buffer_.get() + offsets_[2];
    }
    return buffer_.get() + y_plane_size() + uv_plane_size();
  }

  /// @brief Get stride for Y plane
  size_t y_stride() const
  {
    return has_strides() ? strides_[0] : width_;
  }

  /// @brief Get stride for U plane (I420) or UV plane (NV12)
  size_t u_stride() const
  {
    if (has_strides()) {
      return strides_[1];
    }
    return (format_ == AxVideoFormat::I420) ? width_ / 2 : width_;
  }

  /// @brief Get stride for V plane (I420 only)
  size_t v_stride() const
  {
    if (format_ != AxVideoFormat::I420) {
      return 0;
    }
    return has_strides() ? strides_[2] : width_ / 2;
  }

  /// @brief Convert to cv::Mat (single-channel, height * 1.5)
  /// The returned Mat references the internal buffer (no copy)
  /// @return cv::Mat wrapping the buffer
  cv::Mat to_cvmat();

  /// @brief Convert to AxVideoInterface
  /// The returned interface references the internal buffer (no copy)
  /// @return AxVideoInterface suitable for passing to AxInferenceNet
  AxVideoInterface to_video_interface();

  /// @brief Create VideoBuffer from existing AxVideoInterface
  /// This performs a copy if the source is not contiguous
  /// @param video Source video interface
  /// @return New VideoBuffer with copied data
  static VideoBuffer from_video_interface(const AxVideoInterface &video);

  /// @brief Wrap external buffer without taking ownership
  /// The external buffer must remain valid for the lifetime of this VideoBuffer
  /// @param data Pointer to buffer (contiguous or strided)
  /// @param width Frame width
  /// @param height Frame height
  /// @param format Video format (I420 or NV12)
  /// @param deleter Custom deleter to be called when buffer is destroyed
  /// @param strides Optional strides for each plane (empty for contiguous)
  /// @param offsets Optional offsets for each plane (empty for contiguous)
  /// @return VideoBuffer wrapping the external buffer
  static VideoBuffer wrap_external(uint8_t *data, int width, int height,
      AxVideoFormat format, std::function<void(uint8_t *)> deleter,
      const std::vector<size_t> &strides = {}, const std::vector<size_t> &offsets = {});

  // private:
  std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>> buffer_;
  size_t buffer_size_ = 0;
  int width_ = 0;
  int height_ = 0;
  AxVideoFormat format_ = AxVideoFormat::UNDEFINED;
  std::vector<size_t> strides_; // Empty for contiguous buffers
  std::vector<size_t> offsets_; // Empty for contiguous buffers

  size_t y_plane_size() const
  {
    return width_ * height_;
  }

  size_t uv_plane_size() const
  {
    if (format_ == AxVideoFormat::I420) {
      return (width_ / 2) * (height_ / 2);
    } else if (format_ == AxVideoFormat::NV12) {
      return width_ * (height_ / 2);
    }
    return 0;
  }
};

} // namespace Ax
