// Copyright Axelera AI, 2026
#include "AxVideoBuffer.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace Ax
{

// All VideoBuffer-supported formats. Adding a new one requires exactly one entry here.
static const AxVideoFormat supported_formats[] = {
  AxVideoFormat::I420,
  AxVideoFormat::NV12,
  AxVideoFormat::NV16,
  AxVideoFormat::Y42B,
  AxVideoFormat::Y444,
  AxVideoFormat::RGB,
  AxVideoFormat::BGR,
  AxVideoFormat::RGBA,
  AxVideoFormat::BGRA,
  AxVideoFormat::GRAY8,
};

static bool
is_supported_format(AxVideoFormat fmt)
{
  return std::ranges::find(supported_formats, fmt) != std::end(supported_formats);
}

struct buffer_layout {
  std::vector<size_t> strides;
  std::vector<size_t> offsets;
};
static buffer_layout
natural_layout(int width, int height, AxVideoFormat format)
{
  auto w = static_cast<size_t>(width);
  auto ysize = w * static_cast<size_t>(height);
  if (format == AxVideoFormat::NV12 || format == AxVideoFormat::NV16) {
    return { { w, w }, { 0, ysize } };
  }
  if (format == AxVideoFormat::I420) {
    auto uvw = w / 2;
    return { { w, uvw, uvw }, { 0, ysize, ysize + uvw * static_cast<size_t>(height / 2) } };
  }
  if (format == AxVideoFormat::Y42B) {
    // 4:2:2 planar: chroma is half width but full height
    auto uvw = w / 2;
    auto uvsize = uvw * static_cast<size_t>(height);
    return { { w, uvw, uvw }, { 0, ysize, ysize + uvsize } };
  }
  if (format == AxVideoFormat::Y444) {
    return { { w, w, w }, { 0, ysize, ysize * 2 } };
  }
  if (format == AxVideoFormat::RGBA || format == AxVideoFormat::BGRA) {
    return { { w * 4 }, { 0 } };
  }
  if (format == AxVideoFormat::GRAY8) {
    return { { w }, { 0 } };
  }
  // RGB / BGR
  return { { w * 3 }, { 0 } };
}

size_t
get_buffer_size(int width, int height, AxVideoFormat format)
{
  if (format == AxVideoFormat::I420) {
    return width * (height + height / 2);
  } else if (format == AxVideoFormat::Y42B) {
    return width * height * 2;
  } else if (format == AxVideoFormat::Y444) {
    return width * height * 3;
  } else if (format == AxVideoFormat::NV12) {
    return width * (height + height / 2);
  } else if (format == AxVideoFormat::NV16) {
    return width * height * 2;
  } else if (format == AxVideoFormat::RGB || format == AxVideoFormat::BGR) {
    return width * height * 3;
  } else if (format == AxVideoFormat::RGBA || format == AxVideoFormat::BGRA) {
    return width * height * 4;
  } else if (format == AxVideoFormat::GRAY8) {
    return width * height;
  }
  throw std::invalid_argument("Unsupported video format for buffer size calculation");
}

VideoBuffer::VideoBuffer(int width, int height, AxVideoFormat format)
    : width_(width),
      height_(height),
      format_(format)
{
  if (!is_supported_format(format)) {
    throw std::invalid_argument(
        "VideoBuffer only supports I420, NV12, NV16, Y42B, Y444, RGB, BGR, RGBA, and BGRA formats");
  }

  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Width and height must be positive");
  }

  buffer_size_ = get_buffer_size(width, height, format);
  buffer_ = std::make_unique_for_overwrite<uint8_t[]>(buffer_size_);
  auto [s, o] = natural_layout(width, height, format);
  strides_ = std::move(s);
  offsets_ = std::move(o);
}

bool
VideoBuffer::is_contiguous() const
{
  auto [nat_strides, nat_offsets] = natural_layout(width_, height_, format_);
  return strides_ == nat_strides && offsets_ == nat_offsets;
}

cv::Mat
VideoBuffer::to_cvmat()
{
  if (!is_valid()) {
    throw std::runtime_error("Cannot convert invalid VideoBuffer to cv::Mat");
  }

  if (!is_contiguous()) {
    throw std::runtime_error("Cannot convert non-contiguous VideoBuffer to cv::Mat directly - "
                             "use to_video_interface() instead");
  }

  if (format_ == AxVideoFormat::RGBA || format_ == AxVideoFormat::BGRA) {
    return cv::Mat(height_, width_, CV_8UC4, buffer_.get());
  }
  if (format_ == AxVideoFormat::RGB || format_ == AxVideoFormat::BGR) {
    return cv::Mat(height_, width_, CV_8UC3, buffer_.get());
  }
  if (format_ == AxVideoFormat::GRAY8) {
    return cv::Mat(height_, width_, CV_8UC1, buffer_.get());
  }
  if (format_ == AxVideoFormat::Y444) {
    return cv::Mat(height_ * 3, width_, CV_8UC1, buffer_.get());
  }
  if (format_ == AxVideoFormat::Y42B || format_ == AxVideoFormat::NV16) {
    // 4:2:2 (planar Y42B or semi-planar NV16): total samples = 2 * W * H
    return cv::Mat(height_ * 2, width_, CV_8UC1, buffer_.get());
  }
  // NV12/I420: 4:2:0 subsampling gives height * 1.5
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

  video.strides = strides_;
  video.offsets = offsets_;
  video.info.stride = strides_[0];

  return video;
}

VideoBuffer
VideoBuffer::from_video_interface(const AxVideoInterface &video)
{
  if (!is_supported_format(video.info.format)) {
    throw std::invalid_argument("from_video_interface: unsupported format");
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
  } else if (video.info.format == AxVideoFormat::Y42B) {
    is_contiguous
        = (video.strides.size() == 3) && (video.offsets.size() == 3)
          && (video.strides[0] == static_cast<size_t>(video.info.width))
          && (video.strides[1] == static_cast<size_t>(video.info.width / 2))
          && (video.strides[2] == static_cast<size_t>(video.info.width / 2))
          && (video.offsets[0] == 0)
          && (video.offsets[1] == video.info.width * video.info.height)
          && (video.offsets[2]
              == video.info.width * video.info.height
                     + (video.info.width / 2) * video.info.height);
  } else if (video.info.format == AxVideoFormat::Y444) {
    is_contiguous
        = (video.strides.size() == 3) && (video.offsets.size() == 3)
          && (video.strides[0] == static_cast<size_t>(video.info.width))
          && (video.strides[1] == static_cast<size_t>(video.info.width))
          && (video.strides[2] == static_cast<size_t>(video.info.width))
          && (video.offsets[0] == 0)
          && (video.offsets[1]
              == static_cast<size_t>(video.info.width * video.info.height))
          && (video.offsets[2]
              == static_cast<size_t>(video.info.width * video.info.height * 2));
  } else if (video.info.format == AxVideoFormat::NV12
             || video.info.format == AxVideoFormat::NV16) {
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
  } else if (video.info.format == AxVideoFormat::RGBA
             || video.info.format == AxVideoFormat::BGRA) {
    is_contiguous = (video.strides.size() == 1) && (video.offsets.size() == 1)
                    && (video.strides[0] == static_cast<size_t>(video.info.width * 4))
                    && (video.offsets[0] == 0);
  } else if (video.info.format == AxVideoFormat::GRAY8) {
    is_contiguous = (video.strides.size() == 1) && (video.offsets.size() == 1)
                    && (video.strides[0] == static_cast<size_t>(video.info.width))
                    && (video.offsets[0] == 0);
  }

  if (is_contiguous) {
    std::memcpy(result.data(), video.data, result.size());
  } else {
    const uint8_t *src = static_cast<const uint8_t *>(video.data);

    if (video.info.format == AxVideoFormat::I420) {
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.y_plane() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
      }
      for (int y = 0; y < video.info.height / 2; y++) {
        std::memcpy(result.u_plane() + y * (video.info.width / 2),
            src + video.offsets[1] + y * video.strides[1], video.info.width / 2);
      }
      for (int y = 0; y < video.info.height / 2; y++) {
        std::memcpy(result.v_plane() + y * (video.info.width / 2),
            src + video.offsets[2] + y * video.strides[2], video.info.width / 2);
      }
    } else if (video.info.format == AxVideoFormat::Y42B) {
      // 4:2:2 planar: chroma planes are half width but full height
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.y_plane() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
      }
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.u_plane() + y * (video.info.width / 2),
            src + video.offsets[1] + y * video.strides[1], video.info.width / 2);
      }
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.v_plane() + y * (video.info.width / 2),
            src + video.offsets[2] + y * video.strides[2], video.info.width / 2);
      }
    } else if (video.info.format == AxVideoFormat::Y444) {
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.y_plane() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
      }
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.u_plane() + y * video.info.width,
            src + video.offsets[1] + y * video.strides[1], video.info.width);
      }
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.v_plane() + y * video.info.width,
            src + video.offsets[2] + y * video.strides[2], video.info.width);
      }
    } else if (video.info.format == AxVideoFormat::NV12) {
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.y_plane() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
      }
      for (int y = 0; y < video.info.height / 2; y++) {
        std::memcpy(result.u_plane() + y * video.info.width,
            src + video.offsets[1] + y * video.strides[1], video.info.width);
      }
    } else if (video.info.format == AxVideoFormat::NV16) {
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.y_plane() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
      }
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.u_plane() + y * video.info.width,
            src + video.offsets[1] + y * video.strides[1], video.info.width);
      }
    } else if (video.info.format == AxVideoFormat::RGB
               || video.info.format == AxVideoFormat::BGR) {
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.data() + y * video.info.width * 3,
            src + video.offsets[0] + y * video.strides[0], video.info.width * 3);
      }
    } else if (video.info.format == AxVideoFormat::RGBA
               || video.info.format == AxVideoFormat::BGRA) {
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.data() + y * video.info.width * 4,
            src + video.offsets[0] + y * video.strides[0], video.info.width * 4);
      }
    } else if (video.info.format == AxVideoFormat::GRAY8) {
      for (int y = 0; y < video.info.height; y++) {
        std::memcpy(result.data() + y * video.info.width,
            src + video.offsets[0] + y * video.strides[0], video.info.width);
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
  if (!is_supported_format(format)) {
    throw std::invalid_argument(
        "VideoBuffer only supports I420, NV12, NV16, Y42B, Y444, RGB, BGR, RGBA, and BGRA formats");
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

  if (!strides.empty() && !offsets.empty()) {
    result.strides_ = strides;
    result.offsets_ = offsets;
    // Compute buffer size from the last plane's extent
    if (format == AxVideoFormat::I420) {
      result.buffer_size_ = offsets[2] + strides[2] * (height / 2);
    } else if (format == AxVideoFormat::NV12) {
      result.buffer_size_ = offsets[1] + strides[1] * (height / 2);
    } else if (format == AxVideoFormat::NV16) {
      result.buffer_size_ = offsets[1] + strides[1] * height;
    } else if (format == AxVideoFormat::Y42B || format == AxVideoFormat::Y444) {
      result.buffer_size_ = offsets[2] + strides[2] * height;
    } else {
      result.buffer_size_ = offsets[0] + strides[0] * height;
    }
  } else {
    result.buffer_size_ = get_buffer_size(width, height, format);
    auto [s, o] = natural_layout(width, height, format);
    result.strides_ = std::move(s);
    result.offsets_ = std::move(o);
  }

  // Wrap external buffer with custom deleter
  result.buffer_
      = std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>>(data, deleter);

  return result;
}

} // namespace Ax
