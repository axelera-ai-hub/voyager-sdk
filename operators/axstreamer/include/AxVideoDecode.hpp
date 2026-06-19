// Copyright Axelera AI, 2025
// Base class for video decoding functionality in the Axelera streaming framework

#pragma once

#include <functional>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"
#include "AxVideoBuffer.hpp"

namespace Ax
{
/**
 * @class VideoDecode
 * @brief Abstract base class for video decoding operations
 *
 * This class provides a common interface for video decoding implementations
 * that can handle various video sources and formats. It uses a callback-based
 * approach to deliver decoded frames to the application.
 */
class VideoDecode
{
  public:
  /**
   * @brief Constructor for VideoDecode with VideoBuffer callback
   *
   * @param input The input source (file path, URL, device, etc.)
   * @param frame_callback Callback function to handle decoded frames (VideoBuffer)
   * @param format The video format specification
   */
  VideoDecode(const std::string &input,
      std::function<void(VideoBuffer)> frame_callback, AxVideoFormat format)
      : input(input),
        frame_callback(frame_callback),
        format(format)
  {
    if (format != AxVideoFormat::RGB && format != AxVideoFormat::BGR
        && format != AxVideoFormat::I420 && format != AxVideoFormat::NV12
        && format != AxVideoFormat::UNDEFINED) {
      throw std::invalid_argument("Unsupported video format for VideoDecode");
    }
  }

  /**
   * @brief Constructor for VideoDecode with cv::Mat callback
   *
   * This constructor accepts a callback that receives cv::Mat and internally
   * converts VideoBuffer to cv::Mat before invoking the callback.
   * For I420/NV12 formats, the Mat will be in YUV format (single-channel, height * 1.5).
   * For RGB/BGR formats, the Mat will be in the corresponding format.
   *
   * @param input The input source (file path, URL, device, etc.)
   * @param mat_callback Callback function to handle decoded frames (cv::Mat)
   * @param format The video format specification
   */
  VideoDecode(const std::string &input,
      std::function<void(cv::Mat)> mat_callback, AxVideoFormat format)
      : input(input),
        format(format)
  {
    if (format != AxVideoFormat::RGB && format != AxVideoFormat::BGR
        && format != AxVideoFormat::I420 && format != AxVideoFormat::NV12
        && format != AxVideoFormat::UNDEFINED) {
      throw std::invalid_argument("Unsupported video format for VideoDecode");
    }

    // Wrap the cv::Mat callback to convert VideoBuffer to cv::Mat
    frame_callback = [mat_callback](VideoBuffer buffer) {
      if (!buffer.is_valid()) {
        // Pass empty Mat for end of stream
        mat_callback(cv::Mat());
        return;
      }
      auto format = buffer.format();
      // Convert VideoBuffer to cv::Mat based on format
      // Use cvmat_from_buffer() to handle both contiguous and strided buffers
      if (format == AxVideoFormat::I420 || format == AxVideoFormat::NV12) {
        // For YUV formats, return the planar YUV Mat
        cv::Mat yuv_mat = cvmat_from_buffer(buffer);
        mat_callback(std::move(yuv_mat));
      } else {
        // For RGB/BGR formats, the buffer is already in the correct format
        // Create a cv::Mat view of the buffer and clone it
        int channels
            = (format == AxVideoFormat::RGB || format == AxVideoFormat::BGR) ? 3 : 1;
        int cv_type = (channels == 3) ? CV_8UC3 : CV_8UC1;
        cv::Mat frame(buffer.height(), buffer.width(), cv_type, buffer.data());
        mat_callback(frame.clone());
      }
    };
  }

  /**
   * @brief Start the video decoding process
   *
   * This method starts the video decoding in a separate thread, allowing
   * the application to continue processing without blocking.
   */
  void start_decoding()
  {
    if (!reader_thread.joinable()) {
      reader_thread
          = std::jthread([this](std::stop_token stoken) { reader_func(stoken); });
    }
  }

  /**
   * @brief Stop the video decoding process
   *
   * This method stops the decoding thread and waits for it to finish.
   * Call this method from Python with GIL released to avoid deadlock.
   */
  void stop_decoding()
  {
    if (reader_thread.joinable()) {
      reader_thread.request_stop();
      reader_thread.join();
    }
  }

  /**
   * @brief Destructor
   *
   * Derived classes should ensure proper cleanup of the reader thread.
   */
  virtual ~VideoDecode() = default;

  protected:
  /** @brief Start the reader thread
   *
   * This method is responsible for starting the video decoding process
   * in a separate thread. It should be implemented by derived classes.
   *
   * @param stoken Stop token to check for cancellation requests
   */
  virtual void reader_func(std::stop_token stoken) = 0;

  /**
   * @brief Input source specification
   *
   * This can be a file path, URL, device identifier, or any other
   * string that identifies the video source.
   */
  std::string input;

  /**
   * @brief Frame callback function
   *
   * Function that will be called for each decoded frame. The callback
   * receives a VideoBuffer object containing the decoded frame data.
   */
  std::function<void(VideoBuffer)> frame_callback;

  /**
   * @brief Video format specification
   *
   * Contains format-specific parameters and configuration for the
   * video decoding process.
   */
  AxVideoFormat format;

  /**
   * @brief Reader thread for asynchronous frame processing
   *
   * Handles the video decoding in a separate thread to avoid blocking
   * the main application thread.
   */
  std::jthread reader_thread;
};
} // namespace Ax
