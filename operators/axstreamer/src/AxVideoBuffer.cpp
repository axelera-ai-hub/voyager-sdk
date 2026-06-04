// Copyright Axelera AI, 2026
#include "AxVideoBuffer.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace Ax
{

size_t
get_buffer_size(int width, int height, AxVideoFormat format)
{
  if (format == AxVideoFormat::I420) {
    return width * height + 2 * (width / 2) * (height / 2);
  } else if (format == AxVideoFormat::NV12) {
    return width * height + width * (height / 2);
  } else if (format == AxVideoFormat::RGB || format == AxVideoFormat::BGR) {
    return width * height * 3;
  }
  throw std::invalid_argument("Unsupported video format for buffer size calculation");
}

VideoBuffer::VideoBuffer(int width, int height, AxVideoFormat format)
    : width_(width),
      height_(height),
      format_(format)
{
  if (format != AxVideoFormat::I420 && format != AxVideoFormat::NV12
      && format != AxVideoFormat::RGB && format != AxVideoFormat::BGR) {
    throw std::invalid_argument("VideoBuffer only supports I420, NV12, RGB, and BGR formats");
  }

  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Width and height must be positive");
  }

  // Calculate buffer size for contiguous storage
  buffer_size_ = get_buffer_size(width, height, format);

  buffer_ = std::make_unique_for_overwrite<uint8_t[]>(buffer_size_);
}

cv::Mat
VideoBuffer::to_cvmat()
{
  if (!is_valid()) {
    throw std::runtime_error("Cannot convert invalid VideoBuffer to cv::Mat");
  }

  if (has_strides()) {
    throw std::runtime_error("Cannot convert strided VideoBuffer to cv::Mat directly - "
                             "use to_video_interface() instead");
  }

  if (format_ == AxVideoFormat::RGB || format_ == AxVideoFormat::BGR) {
    // For RGB/BGR, return a 3-channel Mat
    return cv::Mat(height_, width_, CV_8UC3, buffer_.get());
  }
  // For planar YUV formats, store as single-channel image with height * 1.5
  int mat_height = height_ + height_ / 2;
  return cv::Mat(mat_height, width_, CV_8UC1, buffer_.get());
}

AxVideoInterface
VideoBuffer::to_video_interface()
{
  if (!is_valid()) {
    throw std::runtime_error("Cannot convert invalid VideoBuffer to AxVideoInterface");
  }

  AxVideoInterface video;
  video.info.format = format_;
  video.info.width = width_;
  video.info.height = height_;
  video.info.actual_height = height_;
  video.data = buffer_.get();

  if (has_strides()) {
    // Use stored strides and offsets
    video.strides = strides_;
    video.offsets = offsets_;
    video.info.stride = strides_[0];
  } else {
    // Contiguous buffer - calculate strides and offsets
    size_t y_stride = width_;
    size_t y_size = y_plane_size();
    video.info.stride = y_stride;

    if (format_ == AxVideoFormat::NV12) {
      size_t uv_stride = width_;
      video.strides = { y_stride, uv_stride };
      video.offsets = { 0, y_size };
    } else if (format_ == AxVideoFormat::I420) {
      size_t u_stride = width_ / 2;
      size_t v_stride = width_ / 2;
      size_t u_size = uv_plane_size();
      video.strides = { y_stride, u_stride, v_stride };
      video.offsets = { 0, y_size, y_size + u_size };
    } else if (format_ == AxVideoFormat::RGB || format_ == AxVideoFormat::BGR) {
      // For RGB/BGR, treat as single plane with 3 channels
      video.strides = { size_t(width_) * 3 };
      video.offsets = { 0 };
    }
  }

  return video;
}

VideoBuffer
VideoBuffer::from_video_interface(const AxVideoInterface &video)
{
  if (video.info.format != AxVideoFormat::I420 && video.info.format != AxVideoFormat::NV12) {
    throw std::invalid_argument("from_video_interface only supports I420 and NV12 formats");
  }

  VideoBuffer result(video.info.width, video.info.height, video.info.format);

  // Check if source is contiguous
  bool is_contiguous = false;
  if (video.info.format == AxVideoFormat::I420) {
    is_contiguous
        = (video.strides.size() == 3) && (video.offsets.size() == 3)
          && (video.strides[0] == static_cast<size_t>(video.info.width))
          && (video.strides[1] == static_cast<size_t>(video.info.width / 2))
          && (video.strides[2] == static_cast<size_t>(video.info.width / 2))
          && (video.offsets[0] == 0)
          && (video.offsets[1] == video.info.width * video.info.height)
          && (video.offsets[2]
              == video.info.width * video.info.height
                     + (video.info.width / 2) * (video.info.height / 2));
  } else if (video.info.format == AxVideoFormat::NV12) {
    is_contiguous = (video.strides.size() == 2) && (video.offsets.size() == 2)
                    && (video.strides[0] == static_cast<size_t>(video.info.width))
                    && (video.strides[1] == static_cast<size_t>(video.info.width))
                    && (video.offsets[0] == 0)
                    && (video.offsets[1] == video.info.width * video.info.height);
  } else if (video.info.format == AxVideoFormat::RGB
             || video.info.format == AxVideoFormat::BGR) {
    is_contiguous = (video.strides.size() == 1) && (video.offsets.size() == 1)
                    && (video.strides[0] == static_cast<size_t>(video.info.width * 3))
                    && (video.offsets[0] == 0);
  }

  if (is_contiguous) {
    // Simple memcpy for contiguous data
    std::memcpy(result.data(), video.data, result.size());
  } else {
    // Copy plane by plane, handling strides
    const uint8_t *src = static_cast<const uint8_t *>(video.data);

    if (video.info.format == AxVideoFormat::I420) {
      // Copy Y plane
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.y_plane() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
      }
      // Copy U plane
      for (int y = 0; y < video.info.height / 2; y++) {
        std::memcpy(result.u_plane() + y * (video.info.width / 2),
            src + video.offsets[1] + y * video.strides[1], video.info.width / 2);
      }
      // Copy V plane
      for (int y = 0; y < video.info.height / 2; y++) {
        std::memcpy(result.v_plane() + y * (video.info.width / 2),
            src + video.offsets[2] + y * video.strides[2], video.info.width / 2);
      }
    } else if (video.info.format == AxVideoFormat::NV12) {
      // Copy Y plane
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.y_plane() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
      }
      // Copy UV plane
      for (int y = 0; y < video.info.height / 2; y++) {
        std::memcpy(result.u_plane() + y * video.info.width,
            src + video.offsets[1] + y * video.strides[1], video.info.width);
      }
    } else if (video.info.format == AxVideoFormat::RGB
               || video.info.format == AxVideoFormat::BGR) {
      // Copy RGB/BGR data
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.data() + y * video.info.width * 3,
            src + video.offsets[0] + y * video.strides[0], video.info.width * 3);
      }
    }
  }

  return result;
}

VideoBuffer
VideoBuffer::wrap_external(uint8_t *data, int width, int height,
    AxVideoFormat format, std::function<void(uint8_t *)> deleter,
    const std::vector<size_t> &strides, const std::vector<size_t> &offsets)
{
  if (format != AxVideoFormat::I420 && format != AxVideoFormat::NV12
      && format != AxVideoFormat::RGB && format != AxVideoFormat::BGR) {
    throw std::invalid_argument("VideoBuffer only supports I420, NV12, RGB, and BGR formats");
  }

  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Width and height must be positive");
  }

  if (!data) {
    throw std::invalid_argument("Data pointer cannot be null");
  }

  VideoBuffer result;
  result.width_ = width;
  result.height_ = height;
  result.format_ = format;
  result.strides_ = strides;
  result.offsets_ = offsets;

  // Calculate buffer size based on strides or contiguous layout
  if (!strides.empty() && !offsets.empty()) {
    // Strided buffer - calculate from last plane
    if (format == AxVideoFormat::I420) {
      result.buffer_size_ = offsets[2] + strides[2] * (height / 2);
    } else if (format == AxVideoFormat::NV12) {
      result.buffer_size_ = offsets[1] + strides[1] * (height / 2);
    } else if (format == AxVideoFormat::RGB || format == AxVideoFormat::BGR) {
      result.buffer_size_ = offsets[0] + strides[0] * height;
    }
  } else {
    // Contiguous buffer
    size_t y_size = width * height;
    size_t uv_size = (format == AxVideoFormat::I420) ? width * (height / 2) :
                                                       width * (height / 2);
    result.buffer_size_ = y_size + uv_size;
  }

  // Wrap external buffer with custom deleter
  result.buffer_
      = std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>>(data, deleter);

  return result;
}

} // namespace Ax
