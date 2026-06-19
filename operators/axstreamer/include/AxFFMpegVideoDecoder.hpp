// Copyright Axelera AI, 2025
#include <map>
#include <opencv2/core.hpp>
#include "AxVideoBuffer.hpp"
#include "AxVideoDecode.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace Ax
{
class FFMpegVideoDecoder : public VideoDecode
{
  public:
  FFMpegVideoDecoder(const std::string &input, std::function<void(VideoBuffer)> frame_callback,
      AxVideoFormat format = AxVideoFormat::UNDEFINED);

  FFMpegVideoDecoder(const std::string &input, std::function<void(cv::Mat)> mat_callback,
      AxVideoFormat format = AxVideoFormat::UNDEFINED);

  ~FFMpegVideoDecoder();

  protected:
  void reader_func(std::stop_token stoken) override;

  private:
  void init(AxVideoFormat format);

  AVPixelFormat get_requested_format(AxVideoFormat format);
  static int get_buffer2_callback(AVCodecContext *ctx, AVFrame *frame, int flags);
  static void free_aligned_buffer(void *opaque, uint8_t *data);
  static int interrupt_callback(void *ctx);

  AVCodecContext *codec_ctx;
  AVFormatContext *format_ctx;
  SwsContext *sws_ctx;
  int video_stream_index;
  AxVideoFormat requested_format;
  static const std::map<AxVideoFormat, AVPixelFormat> format_map;
};
} // namespace Ax
