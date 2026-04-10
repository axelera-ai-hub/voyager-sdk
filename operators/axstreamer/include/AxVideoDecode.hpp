// Copyright Axelera AI, 2025
// Base class for video decoding functionality in the Axelera streaming framework

#pragma once

#include <functional>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include "AxStreamerUtils.hpp"

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
   * @brief Constructor for VideoDecode
   *
   * @param input The input source (file path, URL, device, etc.)
   * @param frame_callback Callback function to handle decoded frames
   * @param format The video format specification
   */
  VideoDecode(const std::string &input,
      std::function<void(cv::Mat)> frame_callback, AxVideoFormat format)
      : input(input),
        frame_callback(frame_callback),
        format(format)
  {
    if (format != AxVideoFormat::RGB && format != AxVideoFormat::BGR
        && format != AxVideoFormat::I420 && format != AxVideoFormat::NV12) {
      throw std::invalid_argument("Unsupported video format for VideoDecode");
    }
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
      reader_thread = std::jthread(&VideoDecode::reader_func, this);
    }
  }

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of the reader thread if it's still running.
   * The destructor will request the thread to stop and wait for it to join.
   */
  virtual ~VideoDecode()
  {
    if (reader_thread.joinable()) {
      reader_thread.request_stop();
      reader_thread.join();
    }
  };

  protected:
  /** @brief Start the reader thread
   *
   * This method is responsible for starting the video decoding process
   * in a separate thread. It should be implemented by derived classes.
   */
  virtual void reader_func() = 0;

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
   * receives a reference to the decoded frame as a cv::Mat object.
   */
  std::function<void(cv::Mat)> frame_callback;

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
