// Copyright Axelera AI, 2025

#include <opencv2/imgproc.hpp>
#include "AxOpenCVVideoDecoder.hpp"

Ax::OpenCVVideoDecoder::OpenCVVideoDecoder(const std::string &input,
    std::function<void(VideoBuffer)> frame_callback, AxVideoFormat format)
    : Ax::VideoDecode(input, frame_callback, format)
{
  try {
    // Open the video capture
    cap.open(input);
    if (!cap.isOpened()) {
      throw std::runtime_error("Could not open video input: " + input);
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "Error initializing OpenCVVideoDecoder: " + std::string(e.what()));
  }
}

Ax::OpenCVVideoDecoder::OpenCVVideoDecoder(const std::string &input,
    std::function<void(cv::Mat)> frame_callback, AxVideoFormat format)
    : Ax::VideoDecode(input, frame_callback, format)
{
  try {
    // Open the video capture
    cap.open(input);
    if (!cap.isOpened()) {
      throw std::runtime_error("Could not open video input: " + input);
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "Error initializing OpenCVVideoDecoder: " + std::string(e.what()));
  }
}

void
Ax::OpenCVVideoDecoder::reader_func(std::stop_token stoken)
{
  cv::Mat frame;
  bool stopped_early = false;

  // Check stop token before each read to allow quick exit
  while (!stoken.stop_requested()) {
    // cap.read() can block, but for file-based videos it should return quickly
    if (!cap.read(frame)) {
      break; // End of video or read error
    }

    if (stoken.stop_requested()) {
      stopped_early = true;
      break; // Stop requested - exit without sending end-of-stream
    }

    if (frame.empty()) {
      break; // End of video
    }

    VideoBuffer video_buffer;
    int color_range = 2; // OpenCV always produces full-range output
    int color_matrix = 1; // RGB by default; overridden to BT.601 for YUV outputs
    if (format == AxVideoFormat::RGB) {
      // Convert BGR to RGB
      video_buffer = VideoBuffer(frame.cols, frame.rows, AxVideoFormat::RGB);
      cv::Mat rgb_frame(frame.rows, frame.cols, CV_8UC3, video_buffer.data());
      cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    } else if (format == AxVideoFormat::BGR || format == AxVideoFormat::UNDEFINED) {
      // BGR format: no conversion needed (OpenCV default)
      video_buffer = VideoBuffer(frame.cols, frame.rows, AxVideoFormat::BGR);
      std::memcpy(video_buffer.data(), frame.data, video_buffer.size());
    } else if (format == AxVideoFormat::I420) {
      // Convert BGR to I420 via cv::COLOR_BGR2YUV_I420 (full-range BT.601)
      int width = frame.cols;
      int height = frame.rows;
      video_buffer = VideoBuffer(width, height, AxVideoFormat::I420);
      cv::Mat yuv_frame(frame.rows * 3 / 2, frame.cols, CV_8UC1, video_buffer.data());
      cv::cvtColor(frame, yuv_frame, cv::COLOR_BGR2YUV_I420);
      color_matrix = 3; // BT.601
    } else if (format == AxVideoFormat::NV12) {
      // NV12 conversion: OpenCV doesn't have direct BGR2NV12
      // Convert BGR -> I420 first (full-range BT.601), then repack to NV12
      cv::Mat yuv_i420;
      cv::cvtColor(frame, yuv_i420, cv::COLOR_BGR2YUV_I420);

      int width = frame.cols;
      int height = frame.rows;

      video_buffer = VideoBuffer(width, height, AxVideoFormat::NV12);

      size_t y_size = width * height;
      size_t uv_size = (width / 2) * (height / 2);

      std::memcpy(video_buffer.y_plane(), yuv_i420.data, y_size);

      const uint8_t *u_src = yuv_i420.data + y_size;
      const uint8_t *v_src = yuv_i420.data + y_size + uv_size;
      uint8_t *uv_dst = video_buffer.u_plane();

      for (size_t i = 0; i < uv_size; i++) {
        uv_dst[i * 2] = u_src[i];
        uv_dst[i * 2 + 1] = v_src[i];
      }
      color_matrix = 3; // BT.601
    }

    video_buffer.set_color_range(color_range);
    video_buffer.set_color_matrix(color_matrix);
    frame_callback(std::move(video_buffer));
  }

  // Only send end-of-stream callback if we reached natural end (not stopped
  // early) This avoids GIL deadlock when destructor is stopping the thread
  if (!stopped_early) {
    frame_callback(VideoBuffer());
  }
}

Ax::OpenCVVideoDecoder::~OpenCVVideoDecoder()
{
  stop_decoding();
}
