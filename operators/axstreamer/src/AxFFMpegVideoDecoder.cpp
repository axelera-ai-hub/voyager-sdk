// Copyright Axelera AI, 2025
#include <cstdlib>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include <vector>
#include "AxFFMpegVideoDecoder.hpp"
#include "AxVideoBuffer.hpp"

void
Ax::FFMpegVideoDecoder::init(AxVideoFormat format)
{
  const AVCodec *codec = nullptr;
  sws_ctx = nullptr;
  video_stream_index = -1;
  format_ctx = nullptr;
  requested_format = format;

  // Set RTSP options
  AVDictionary *opts = nullptr;
  AVInputFormat *fmt = nullptr;
  if (input.starts_with("rtsp://")) {
    av_dict_set(&opts, "rtsp_transport", "tcp", 0); // Use TCP for RTSP
    av_dict_set(&opts, "stimeout", "5000000", 0); // Set timeout (in microseconds)
  } else if (input.starts_with("/dev/video")) {
    av_dict_set(&opts, "input_format", "mjpeg", 0); // Set MJPEG format
    // fmt = av_find_input_format("video4linux2");
  }

  // Allocate format context and set interrupt callback before opening input
  // This allows the callback to interrupt long-running operations during open
  // (e.g., unreachable RTSP hosts, slow DNS resolution)
  format_ctx = avformat_alloc_context();
  if (!format_ctx) {
    throw std::runtime_error("Could not allocate format context");
  }

  // Set interrupt callback to allow aborting blocking operations (including open)
  format_ctx->interrupt_callback.callback = interrupt_callback;
  format_ctx->interrupt_callback.opaque = this;

  // Open input file
  if (avformat_open_input(&format_ctx, input.c_str(), fmt, &opts) < 0) {
    throw std::runtime_error("Could not open input file");
  }

  // Find stream info
  if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
    throw std::runtime_error("Could not find stream information");
  }

  // Find video stream
  for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
    if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index = i;
      break;
    }
  }

  if (video_stream_index == -1) {
    throw std::runtime_error("Could not find video stream");
  }

  // Get codec
  codec = avcodec_find_decoder(format_ctx->streams[video_stream_index]->codecpar->codec_id);
  if (!codec) {
    throw std::runtime_error("Unsupported codec");
  }

  // Allocate codec context
  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    throw std::runtime_error("Could not allocate codec context");
  }

  // Fill codec context parameters
  if (avcodec_parameters_to_context(
          codec_ctx, format_ctx->streams[video_stream_index]->codecpar)
      < 0) {
    throw std::runtime_error("Could not fill codec context");
  }

  codec_ctx->get_buffer2 = get_buffer2_callback;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(60, 31, 102)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  codec_ctx->thread_safe_callbacks = 1;
#pragma GCC diagnostic pop
#endif
  if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
    throw std::runtime_error("Could not open codec");
  }
}

Ax::FFMpegVideoDecoder::FFMpegVideoDecoder(const std::string &input,
    std::function<void(VideoBuffer)> frame_callback, AxVideoFormat format)
    : Ax::VideoDecode(input, frame_callback, format)
{
  init(format);
}

Ax::FFMpegVideoDecoder::FFMpegVideoDecoder(const std::string &input,
    std::function<void(cv::Mat)> mat_callback, AxVideoFormat format)
    : Ax::VideoDecode(input, mat_callback, format)
{
  init(format);
}

Ax::FFMpegVideoDecoder::~FFMpegVideoDecoder()
{
  stop_decoding();

  if (codec_ctx) {
    avcodec_free_context(&codec_ctx);
  }
  if (format_ctx) {
    avformat_close_input(&format_ctx);
  }
}
const std::map<AxVideoFormat, AVPixelFormat> Ax::FFMpegVideoDecoder::format_map = {
  { AxVideoFormat::RGB, AV_PIX_FMT_RGB24 },
  { AxVideoFormat::BGR, AV_PIX_FMT_BGR24 },
  { AxVideoFormat::I420, AV_PIX_FMT_YUV420P },
  { AxVideoFormat::NV12, AV_PIX_FMT_NV12 },
};

AVPixelFormat
Ax::FFMpegVideoDecoder::get_requested_format(AxVideoFormat format)
{
  auto it = format_map.find(format);
  if (it != format_map.end()) {
    return it->second;
  }
  throw std::invalid_argument("Unsupported video format for FFMpegVideoDecoder");
}

void
Ax::FFMpegVideoDecoder::free_aligned_buffer(void * /*opaque*/, uint8_t *data)
{
  std::free(data);
}

int
Ax::FFMpegVideoDecoder::interrupt_callback(void *ctx)
{
  // Return non-zero to interrupt FFmpeg blocking operations
  // Check the stop token from reader_thread - this is the single source of truth
  // Note: reader_thread must be joinable (started) for stop_token to be valid
  auto *decoder = static_cast<FFMpegVideoDecoder *>(ctx);
  if (decoder->reader_thread.joinable()) {
    return decoder->reader_thread.get_stop_token().stop_requested() ? 1 : 0;
  }
  // Thread not started yet (constructor phase) - don't interrupt
  return 0;
}

int
Ax::FFMpegVideoDecoder::get_buffer2_callback(AVCodecContext *ctx, AVFrame *frame, int flags)
{
  AVPixelFormat fmt = static_cast<AVPixelFormat>(frame->format);
  const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);
  if (!desc || (desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
    return avcodec_default_get_buffer2(ctx, frame, flags);
  }

  constexpr int kRowAlignment = 64;
  int size = av_image_get_buffer_size(fmt, frame->width, frame->height, kRowAlignment);
  if (size < 0) {
    return size;
  }

  size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t aligned_size = (static_cast<size_t>(size) + page_size - 1) / page_size * page_size;
  void *buf = std::aligned_alloc(page_size, aligned_size);
  if (!buf) {
    return AVERROR(ENOMEM);
  }

  if (av_image_fill_arrays(frame->data, frame->linesize, static_cast<uint8_t *>(buf),
          fmt, frame->width, frame->height, kRowAlignment)
      < 0) {
    free(buf);
    return AVERROR(ENOMEM);
  }

  frame->buf[0] = av_buffer_create(static_cast<uint8_t *>(buf),
      static_cast<size_t>(size), free_aligned_buffer, nullptr, 0);
  if (!frame->buf[0]) {
    std::free(buf);
    return AVERROR(ENOMEM);
  }

  return 0;
}

void
Ax::FFMpegVideoDecoder::reader_func(std::stop_token stoken)
{
  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();
  bool stopped_early = false;

  while (!stoken.stop_requested() && av_read_frame(format_ctx, packet) >= 0) {
    if (stoken.stop_requested()) {
      stopped_early = true;
      break;
    }

    if (packet->stream_index == video_stream_index) {
      if (avcodec_send_packet(codec_ctx, packet) < 0) {
        break;
      }
      while (!stoken.stop_requested() && avcodec_receive_frame(codec_ctx, frame) == 0) {
        AVPixelFormat native_format = static_cast<AVPixelFormat>(frame->format);
        if (requested_format == AxVideoFormat::UNDEFINED) {
          requested_format
              = (native_format == AV_PIX_FMT_NV12)    ? AxVideoFormat::NV12 :
                (native_format == AV_PIX_FMT_YUV420P) ? AxVideoFormat::I420 :
                                                        AxVideoFormat::RGB;
        }

        // Check if we can wrap directly (I420 or NV12 format, may have stride padding)
        bool can_wrap_yuv
            = (native_format == AV_PIX_FMT_YUV420P || native_format == AV_PIX_FMT_NV12);

        VideoBuffer video_buffer;
        if (can_wrap_yuv && native_format == get_requested_format(requested_format)) {
          // True zero-copy: wrap the AVFrame buffer directly with stride
          // support Clone the frame to manage its lifetime
          AVFrame *frame_clone = av_frame_clone(frame);
          if (!frame_clone) {
            throw std::runtime_error("Failed to clone AVFrame");
          }

          AxVideoFormat buffer_format = (native_format == AV_PIX_FMT_NV12) ?
                                            AxVideoFormat::NV12 :
                                            AxVideoFormat::I420;

          // Check if contiguous
          bool is_contiguous = false;
          if (native_format == AV_PIX_FMT_YUV420P) {
            // I420 contiguity check
            is_contiguous
                = (frame->linesize[0] == frame->width)
                  && (frame->linesize[1] == frame->width / 2)
                  && (frame->linesize[2] == frame->width / 2)
                  && (frame->data[1] == frame->data[0] + frame->width * frame->height)
                  && (frame->data[2]
                      == frame->data[1] + (frame->width / 2) * (frame->height / 2));
          } else if (native_format == AV_PIX_FMT_NV12) {
            // NV12 contiguity check
            is_contiguous
                = (frame->linesize[0] == frame->width)
                  && (frame->linesize[1] == frame->width)
                  && (frame->data[1] == frame->data[0] + frame->width * frame->height);
          } else {
            is_contiguous = frame->linesize[0] == frame->width * 3
                            && frame->data[1] == nullptr && frame->data[2] == nullptr;
          }
          if (is_contiguous) {
            // Truly contiguous - wrap without strides
            video_buffer = VideoBuffer::wrap_external(frame_clone->data[0],
                frame_clone->width, frame_clone->height, buffer_format,
                [frame_clone](uint8_t *) mutable { av_frame_free(&frame_clone); });
          } else {
            // Non-contiguous with stride - wrap with stride info
            std::vector<size_t> strides;
            std::vector<size_t> offsets;

            if (native_format == AV_PIX_FMT_YUV420P) {
              // I420: 3 planes
              strides = { static_cast<size_t>(frame_clone->linesize[0]),
                static_cast<size_t>(frame_clone->linesize[1]),
                static_cast<size_t>(frame_clone->linesize[2]) };
              offsets = { 0,
                static_cast<size_t>(frame_clone->data[1] - frame_clone->data[0]),
                static_cast<size_t>(frame_clone->data[2] - frame_clone->data[0]) };
            } else {
              // NV12: 2 planes
              strides = { static_cast<size_t>(frame_clone->linesize[0]),
                static_cast<size_t>(frame_clone->linesize[1]) };
              offsets = { 0,
                static_cast<size_t>(frame_clone->data[1] - frame_clone->data[0]) };
            }
            video_buffer = VideoBuffer::wrap_external(
                frame_clone->data[0], frame_clone->width, frame_clone->height, buffer_format,
                [frame_clone](uint8_t *) mutable { av_frame_free(&frame_clone); },
                strides, offsets);
          }
        } else {
          // Different format - need to use sws_scale to convert
          video_buffer = VideoBuffer(frame->width, frame->height, requested_format);

          AVPixelFormat target_format = get_requested_format(requested_format);
          sws_ctx = sws_getCachedContext(sws_ctx, frame->width, frame->height,
              native_format, frame->width, frame->height, target_format,
              SWS_POINT, nullptr, nullptr, nullptr);
          if (!sws_ctx) {
            throw std::runtime_error("Could not create sws context");
          }

          uint8_t *dst_data[3];
          int dst_linesize[3];

          if (requested_format == AxVideoFormat::RGB
              || requested_format == AxVideoFormat::BGR) {
            // For RGB/BGR, treat as single plane with 3 channels
            dst_data[0] = video_buffer.data();
            dst_linesize[0] = frame->width * 3;
          } else if (requested_format == AxVideoFormat::I420) {
            dst_data[0] = video_buffer.y_plane();
            dst_data[1] = video_buffer.u_plane();
            dst_data[2] = video_buffer.v_plane();

            dst_linesize[0] = frame->width;
            dst_linesize[1] = frame->width / 2;
            dst_linesize[2] = frame->width / 2;
          } else if (requested_format == AxVideoFormat::NV12) {
            dst_data[0] = video_buffer.y_plane();
            dst_data[1] = video_buffer.u_plane();

            dst_linesize[0] = frame->width;
            dst_linesize[1] = frame->width;
          }
          // Convert to contiguous I420
          sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
              dst_data, dst_linesize);
        }

        frame_callback(std::move(video_buffer));
      }
    }
    av_packet_unref(packet);
  }

  // Check if we exited due to stop request
  if (stoken.stop_requested()) {
    stopped_early = true;
  }

  // Only send end-of-stream callback if we reached natural end (not stopped
  // early) This avoids GIL deadlock when destructor is stopping the thread
  if (!stopped_early) {
    frame_callback(VideoBuffer());
  }

  av_packet_free(&packet);
  av_frame_free(&frame);

  if (sws_ctx) {
    sws_freeContext(sws_ctx);
    sws_ctx = nullptr;
  }
}
