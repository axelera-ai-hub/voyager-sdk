// Copyright Axelera AI, 2025

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include "AxVideoBuffer.hpp"
#include "AxVideoDecode.hpp"
namespace Ax
{
class OpenCVVideoDecoder : public VideoDecode
{
  public:
  OpenCVVideoDecoder(const std::string &input,
      std::function<void(VideoBuffer)> frame_callback, AxVideoFormat format);

  OpenCVVideoDecoder(const std::string &input,
      std::function<void(cv::Mat)> frame_callback, AxVideoFormat format);

  protected:
  void reader_func() override;

  private:
  cv::VideoCapture cap;
};
} // namespace Ax
