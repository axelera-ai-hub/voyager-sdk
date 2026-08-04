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

  /// @brief Check if buffer planes are laid out contiguously with natural strides and offsets
  bool is_contiguous() const;

  /// @brief Get Y plane pointer
  uint8_t *y_plane()
  {
    return buffer_.get() + offsets_[0];
  }

  /// @brief Get U plane pointer (I420) or UV plane pointer (NV12)
  uint8_t *u_plane()
  {
    return buffer_.get() + offsets_[1];
  }

  /// @brief Get V plane pointer (I420/Y42B/Y444 only, nullptr for NV12)
  uint8_t *v_plane()
  {
    if (format_ != AxVideoFormat::I420 && format_ != AxVideoFormat::Y42B
        && format_ != AxVideoFormat::Y444) {
      return nullptr;
    }
    return buffer_.get() + offsets_[2];
  }

  /// @brief Get stride for Y plane
  size_t y_stride() const
  {
    return strides_[0];
  }

  /// @brief Get stride for U plane (I420) or UV plane (NV12)
  size_t u_stride() const
  {
    return strides_.size() > 1 ? strides_[1] : 0;
  }

  /// @brief Get stride for V plane (I420/Y42B/Y444 only)
  size_t v_stride() const
  {
    if (format_ != AxVideoFormat::I420 && format_ != AxVideoFormat::Y42B
        && format_ != AxVideoFormat::Y444) {
      return 0;
    }
    return strides_.size() > 2 ? strides_[2] : 0;
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
  std::vector<size_t> strides_; // Always populated; natural (no padding) for owned/contiguous buffers
  std::vector<size_t> offsets_; // Always populated; plane byte offsets from buffer_.get()
  int color_range_ = 0; // GstRange: 0=unknown, 1=limited, 2=full
  int color_matrix_ = 0; // GstMatrix: 0=unknown, 3=BT601, 4=BT709, etc.

  int color_range() const
  {
    return color_range_;
  }
  void set_color_range(int range)
  {
    color_range_ = range;
  }
  int color_matrix() const
  {
    return color_matrix_;
  }
  void set_color_matrix(int matrix)
  {
    color_matrix_ = matrix;
  }

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
    } else if (format_ == AxVideoFormat::NV16) {
      return width_ * height_;
    } else if (format_ == AxVideoFormat::Y42B) {
      // 4:2:2 planar: chroma is half width, full height
      return (width_ / 2) * height_;
    } else if (format_ == AxVideoFormat::Y444) {
      return width_ * height_;
    }
    return 0;
  }
};

} // namespace Ax
