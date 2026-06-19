// Copyright Axelera AI, 2026
// Unit tests for video decoder functionality

#include "unittest_ax_common.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "AxDataInterface.h"
#include "AxFFMpegVideoDecoder.hpp"
#include "AxOpenCVVideoDecoder.hpp"
#include "AxVideoBuffer.hpp"

namespace
{
// Helper class to create a test video file
class TestVideoFile
{
  public:
  TestVideoFile(int width = 640, int height = 480, int num_frames = 10, int fps = 30)
      : width_(width),
        height_(height),
        num_frames_(num_frames)
  {
    // Create temporary file path
    auto temp_path = fs::temp_directory_path()
                     / ("test_video_"
                         + std::to_string(
                             std::chrono::system_clock::now().time_since_epoch().count())
                         + ".mp4");
    video_path_ = temp_path.string();

    // Create video writer
    cv::VideoWriter writer(video_path_,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    if (!writer.isOpened()) {
      throw std::runtime_error("Failed to create test video file");
    }

    // Write frames
    for (int i = 0; i < num_frames; i++) {
      cv::Mat frame(height, width, CV_8UC3);
      // Create a frame with different color for each frame
      frame.setTo(cv::Scalar(i * 25, 128, 255 - i * 25)); // BGR
      writer.write(frame);
    }

    writer.release();
  }

  ~TestVideoFile()
  {
    // Clean up the temporary video file
    if (fs::exists(video_path_)) {
      fs::remove(video_path_);
    }
  }

  std::string path() const
  {
    return video_path_;
  }

  int width() const
  {
    return width_;
  }

  int height() const
  {
    return height_;
  }

  int num_frames() const
  {
    return num_frames_;
  }

  private:
  std::string video_path_;
  int width_;
  int height_;
  int num_frames_;
};

// Helper class to track frame callbacks
class FrameTracker
{
  public:
  FrameTracker()
      : frame_count_(0),
        end_of_stream_(false)
  {
  }

  void on_buffer(Ax::VideoBuffer &&buffer)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!buffer.is_valid()) {
      end_of_stream_ = true;
      cv_.notify_all();
    } else {
      frame_count_++;
      buffers_.push_back(std::move(buffer));
      cv_.notify_all(); // Notify on frame arrival for wait_for_first_frame
    }
  }

  bool wait_for_end(std::chrono::milliseconds timeout = std::chrono::seconds(5))
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return end_of_stream_; });
  }

  bool wait_for_first_frame(std::chrono::milliseconds timeout = std::chrono::seconds(5))
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return frame_count_ > 0; });
  }

  int frame_count() const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return frame_count_;
  }

  bool end_of_stream() const
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return end_of_stream_;
  }

  std::vector<Ax::VideoBuffer> &buffers()
  {
    return buffers_;
  }

  private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  int frame_count_;
  bool end_of_stream_;
  std::vector<Ax::VideoBuffer> buffers_;
};

} // namespace

// Test OpenCVVideoDecoder creation
TEST(VideoDecode, OpenCVDecoderCreation)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  ASSERT_NO_THROW({
    Ax::OpenCVVideoDecoder decoder(
        test_video.path(),
        [&tracker](
            Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
        AxVideoFormat::BGR);
  });
}

// Test OpenCVVideoDecoder with invalid input
TEST(VideoDecode, OpenCVDecoderInvalidInput)
{
  FrameTracker tracker;

  ASSERT_THROW(
      {
        Ax::OpenCVVideoDecoder decoder(
            "/nonexistent/video/file.mp4",
            [&tracker](Ax::VideoBuffer buffer) {
              tracker.on_buffer(std::move(buffer));
            },
            AxVideoFormat::BGR);
      },
      std::runtime_error);
}

// Test OpenCVVideoDecoder buffer callback
TEST(VideoDecode, OpenCVDecoderBufferCallback)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::OpenCVVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::BGR);

  decoder.start_decoding();

  ASSERT_TRUE(tracker.wait_for_end());
  EXPECT_GT(tracker.frame_count(), 0);
  EXPECT_TRUE(tracker.end_of_stream());

  // Check buffer properties
  auto &buffers = tracker.buffers();
  for (auto &buffer : buffers) {
    EXPECT_TRUE(buffer.is_valid());
    EXPECT_EQ(buffer.width(), test_video.width());
    EXPECT_EQ(buffer.height(), test_video.height());
    EXPECT_EQ(buffer.format(), AxVideoFormat::BGR);
  }
}

// Test OpenCVVideoDecoder with BGR format details
TEST(VideoDecode, OpenCVDecoderBGRDetails)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::OpenCVVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::BGR);

  decoder.start_decoding();

  ASSERT_TRUE(tracker.wait_for_end());
  EXPECT_GT(tracker.frame_count(), 0);
  EXPECT_TRUE(tracker.end_of_stream());

  // Check buffer properties
  auto &buffers = tracker.buffers();
  for (auto &buffer : buffers) {
    EXPECT_TRUE(buffer.is_valid());
    EXPECT_EQ(buffer.width(), test_video.width());
    EXPECT_EQ(buffer.height(), test_video.height());
    EXPECT_EQ(buffer.format(), AxVideoFormat::BGR);
    EXPECT_GT(buffer.size(), 0);

    // Verify we can convert to cv::Mat
    cv::Mat mat = buffer.to_cvmat();
    EXPECT_FALSE(mat.empty());
    EXPECT_EQ(mat.rows, test_video.height());
    EXPECT_EQ(mat.cols, test_video.width());
  }
}

// Test OpenCVVideoDecoder graceful stop
TEST(VideoDecode, OpenCVDecoderGracefulStop)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::OpenCVVideoDecoder decoder(
      test_video.path(),
      [&tracker](Ax::VideoBuffer buffer) {
        tracker.on_buffer(std::move(buffer));
        if (buffer.is_valid()) {
          // Simulate slow processing
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      },
      AxVideoFormat::BGR);

  decoder.start_decoding();

  // Wait for at least one frame to be decoded (no race condition)
  ASSERT_TRUE(tracker.wait_for_first_frame(std::chrono::seconds(5)));

  // Stop decoding - should return quickly without deadlock
  auto start_time = std::chrono::steady_clock::now();
  decoder.stop_decoding();
  auto stop_time = std::chrono::steady_clock::now();
  auto stop_duration
      = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);

  // Stop should complete within reasonable time (not hang)
  EXPECT_LT(stop_duration.count(), 2000); // 2 seconds max

  // Should have decoded at least one frame (already verified by wait_for_first_frame)
  EXPECT_GT(tracker.frame_count(), 0);
}

// Test OpenCVVideoDecoder destructor stops decoding
TEST(VideoDecode, OpenCVDecoderDestructorStops)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  {
    Ax::OpenCVVideoDecoder decoder(
        test_video.path(),
        [&tracker](Ax::VideoBuffer buffer) {
          tracker.on_buffer(std::move(buffer));
          if (buffer.is_valid()) {
            // Simulate slow processing
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
          }
        },
        AxVideoFormat::BGR);

    decoder.start_decoding();

    // Wait for at least one frame to be decoded (no race condition)
    ASSERT_TRUE(tracker.wait_for_first_frame(std::chrono::seconds(5)));

    // Destructor should stop gracefully without deadlock
  }

  // Should have decoded at least one frame (already verified by wait_for_first_frame)
  EXPECT_GT(tracker.frame_count(), 0);
}

// Test OpenCVVideoDecoder with RGB format
TEST(VideoDecode, OpenCVDecoderRGBFormat)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::OpenCVVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::RGB);

  decoder.start_decoding();

  ASSERT_TRUE(tracker.wait_for_end());
  EXPECT_GT(tracker.frame_count(), 0);

  // Check that we got RGB buffers
  auto &buffers = tracker.buffers();
  for (auto &buffer : buffers) {
    EXPECT_EQ(buffer.format(), AxVideoFormat::RGB);
  }
}

// Test FFmpegVideoDecoder creation
TEST(VideoDecode, FFmpegDecoderCreation)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  ASSERT_NO_THROW({
    Ax::FFMpegVideoDecoder decoder(
        test_video.path(),
        [&tracker](
            Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
        AxVideoFormat::BGR);
  });
}

// Test FFmpegVideoDecoder with invalid input
TEST(VideoDecode, FFmpegDecoderInvalidInput)
{
  FrameTracker tracker;

  ASSERT_THROW(
      {
        Ax::FFMpegVideoDecoder decoder(
            "/nonexistent/video/file.mp4",
            [&tracker](Ax::VideoBuffer buffer) {
              tracker.on_buffer(std::move(buffer));
            },
            AxVideoFormat::BGR);
      },
      std::runtime_error);
}

// Test FFmpegVideoDecoder buffer callback
TEST(VideoDecode, FFmpegDecoderBufferCallback)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::FFMpegVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::BGR);

  decoder.start_decoding();

  ASSERT_TRUE(tracker.wait_for_end());
  EXPECT_GT(tracker.frame_count(), 0);
  EXPECT_TRUE(tracker.end_of_stream());

  // Check buffer properties
  auto &buffers = tracker.buffers();
  for (auto &buffer : buffers) {
    EXPECT_TRUE(buffer.is_valid());
    EXPECT_EQ(buffer.width(), test_video.width());
    EXPECT_EQ(buffer.height(), test_video.height());
    EXPECT_EQ(buffer.format(), AxVideoFormat::BGR);
  }
}

// Test FFmpegVideoDecoder BGR format details
TEST(VideoDecode, FFmpegDecoderBGRDetails)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::FFMpegVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::BGR);

  decoder.start_decoding();

  ASSERT_TRUE(tracker.wait_for_end());
  EXPECT_GT(tracker.frame_count(), 0);
  EXPECT_TRUE(tracker.end_of_stream());

  // Check buffer properties
  auto &buffers = tracker.buffers();
  for (auto &buffer : buffers) {
    EXPECT_TRUE(buffer.is_valid());
    EXPECT_EQ(buffer.width(), test_video.width());
    EXPECT_EQ(buffer.height(), test_video.height());
    EXPECT_EQ(buffer.format(), AxVideoFormat::BGR);
    EXPECT_EQ(buffer.size(), test_video.width() * test_video.height() * 3);
  }
}

// Test FFmpegVideoDecoder graceful stop
TEST(VideoDecode, FFmpegDecoderGracefulStop)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::FFMpegVideoDecoder decoder(
      test_video.path(),
      [&tracker](Ax::VideoBuffer buffer) {
        tracker.on_buffer(std::move(buffer));
        if (buffer.is_valid()) {
          // Simulate slow processing
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      },
      AxVideoFormat::BGR);

  decoder.start_decoding();

  // Wait for at least one frame to be decoded (no race condition)
  ASSERT_TRUE(tracker.wait_for_first_frame(std::chrono::seconds(5)));

  // Stop decoding - should return quickly without deadlock
  auto start_time = std::chrono::steady_clock::now();
  decoder.stop_decoding();
  auto stop_time = std::chrono::steady_clock::now();
  auto stop_duration
      = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);

  // Stop should complete within reasonable time (not hang)
  EXPECT_LT(stop_duration.count(), 2000); // 2 seconds max

  // Should have decoded at least one frame (already verified by wait_for_first_frame)
  EXPECT_GT(tracker.frame_count(), 0);
}

// Test FFmpegVideoDecoder interrupt callback with file source
TEST(VideoDecode, FFmpegDecoderInterruptCallback)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::FFMpegVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::BGR);

  decoder.start_decoding();

  // Wait for at least one frame to be decoded (no race condition)
  ASSERT_TRUE(tracker.wait_for_first_frame(std::chrono::seconds(5)));

  // Stop immediately - interrupt callback should abort blocking operations
  auto start_time = std::chrono::steady_clock::now();
  decoder.stop_decoding();
  auto stop_time = std::chrono::steady_clock::now();
  auto stop_duration
      = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time);

  // Stop should complete quickly (file sources are fast, but still verify)
  EXPECT_LT(stop_duration.count(), 2000); // 2 seconds max

  // Should have decoded at least one frame (already verified by wait_for_first_frame)
  EXPECT_GT(tracker.frame_count(), 0);
}

// Test FFmpegVideoDecoder with I420 format
TEST(VideoDecode, FFmpegDecoderI420Format)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::FFMpegVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::I420);

  decoder.start_decoding();

  ASSERT_TRUE(tracker.wait_for_end());
  EXPECT_GT(tracker.frame_count(), 0);

  // Check I420 format properties
  auto &buffers = tracker.buffers();
  for (auto &buffer : buffers) {
    EXPECT_EQ(buffer.format(), AxVideoFormat::I420);
    // I420 size = width * height * 1.5 (Y plane + U/V planes)
    int expected_size = test_video.width() * test_video.height() * 3 / 2;
    EXPECT_EQ(buffer.size(), expected_size);

    // Convert to cv::Mat and check dimensions
    cv::Mat mat = buffer.to_cvmat();
    EXPECT_EQ(mat.rows, test_video.height() * 3 / 2); // height * 1.5
    EXPECT_EQ(mat.cols, test_video.width());
  }
}

// Test FFmpegVideoDecoder with NV12 format
TEST(VideoDecode, FFmpegDecoderNV12Format)
{
  TestVideoFile test_video;
  FrameTracker tracker;

  Ax::FFMpegVideoDecoder decoder(
      test_video.path(),
      [&tracker](
          Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
      AxVideoFormat::NV12);

  decoder.start_decoding();

  ASSERT_TRUE(tracker.wait_for_end());
  EXPECT_GT(tracker.frame_count(), 0);

  // Check NV12 format properties
  auto &buffers = tracker.buffers();
  for (auto &buffer : buffers) {
    EXPECT_EQ(buffer.format(), AxVideoFormat::NV12);
    // NV12 size = width * height * 1.5
    int expected_size = test_video.width() * test_video.height() * 3 / 2;
    EXPECT_EQ(buffer.size(), expected_size);

    // Convert to cv::Mat and check dimensions
    cv::Mat mat = buffer.to_cvmat();
    EXPECT_EQ(mat.rows, test_video.height() * 3 / 2); // height * 1.5
    EXPECT_EQ(mat.cols, test_video.width());
  }
}

// Test FFmpegVideoDecoder vs OpenCVVideoDecoder consistency
TEST(VideoDecode, FFmpegVsOpenCVDecoder)
{
  TestVideoFile test_video;

  // Decode with FFmpeg
  FrameTracker ffmpeg_tracker;
  {
    Ax::FFMpegVideoDecoder decoder(
        test_video.path(),
        [&ffmpeg_tracker](Ax::VideoBuffer buffer) {
          ffmpeg_tracker.on_buffer(std::move(buffer));
        },
        AxVideoFormat::BGR);
    decoder.start_decoding();
    ASSERT_TRUE(ffmpeg_tracker.wait_for_end());
  }

  // Decode with OpenCV
  FrameTracker opencv_tracker;
  {
    Ax::OpenCVVideoDecoder decoder(
        test_video.path(),
        [&opencv_tracker](Ax::VideoBuffer buffer) {
          opencv_tracker.on_buffer(std::move(buffer));
        },
        AxVideoFormat::BGR);
    decoder.start_decoding();
    ASSERT_TRUE(opencv_tracker.wait_for_end());
  }

  // Both should have decoded frames
  EXPECT_GT(ffmpeg_tracker.frame_count(), 0);
  EXPECT_GT(opencv_tracker.frame_count(), 0);

  // Frame counts should be similar (within 1 frame tolerance)
  EXPECT_NEAR(ffmpeg_tracker.frame_count(), opencv_tracker.frame_count(), 1);

  // Buffer properties should match
  auto &ffmpeg_buffers = ffmpeg_tracker.buffers();
  auto &opencv_buffers = opencv_tracker.buffers();
  if (!ffmpeg_buffers.empty() && !opencv_buffers.empty()) {
    const auto &ffmpeg_buffer = ffmpeg_buffers[0];
    const auto &opencv_buffer = opencv_buffers[0];
    EXPECT_EQ(ffmpeg_buffer.width(), opencv_buffer.width());
    EXPECT_EQ(ffmpeg_buffer.height(), opencv_buffer.height());
    EXPECT_EQ(ffmpeg_buffer.format(), opencv_buffer.format());
  }
}

// Test VideoBuffer lifecycle
TEST(VideoDecode, VideoBufferLifecycle)
{
  TestVideoFile test_video;
  std::vector<Ax::VideoBuffer> stored_buffers;

  {
    FrameTracker tracker;
    Ax::OpenCVVideoDecoder decoder(
        test_video.path(),
        [&tracker](
            Ax::VideoBuffer buffer) { tracker.on_buffer(std::move(buffer)); },
        AxVideoFormat::BGR);

    decoder.start_decoding();
    ASSERT_TRUE(tracker.wait_for_end());

    // Move buffers (they should remain valid after decoder is destroyed)
    stored_buffers = std::move(tracker.buffers());
  }

  // Decoder is now destroyed - buffers should still be valid
  ASSERT_GT(stored_buffers.size(), 0);

  for (auto &buffer : stored_buffers) {
    EXPECT_TRUE(buffer.is_valid());
    EXPECT_EQ(buffer.width(), test_video.width());
    EXPECT_EQ(buffer.height(), test_video.height());

    // Should still be able to convert to cv::Mat
    cv::Mat mat = buffer.to_cvmat();
    EXPECT_FALSE(mat.empty());
  }
}

// Test multiple decoder instances
TEST(VideoDecode, MultipleDecoders)
{
  TestVideoFile test_video;
  std::vector<std::unique_ptr<Ax::OpenCVVideoDecoder>> decoders;
  std::vector<std::shared_ptr<FrameTracker>> trackers;

  const int num_decoders = 3;

  // Create multiple decoders
  for (int i = 0; i < num_decoders; i++) {
    auto tracker = std::make_shared<FrameTracker>();

    auto decoder = std::make_unique<Ax::OpenCVVideoDecoder>(
        test_video.path(),
        [tracker](
            Ax::VideoBuffer buffer) { tracker->on_buffer(std::move(buffer)); },
        AxVideoFormat::BGR);

    decoder->start_decoding();

    decoders.push_back(std::move(decoder));
    trackers.push_back(tracker);
  }

  // Wait for all to finish
  for (auto &tracker : trackers) {
    ASSERT_TRUE(tracker->wait_for_end());
    EXPECT_GT(tracker->frame_count(), 0);
  }

  // All decoders should have decoded similar number of frames
  for (size_t i = 1; i < trackers.size(); i++) {
    EXPECT_NEAR(trackers[i]->frame_count(), trackers[0]->frame_count(), 1);
  }
}
