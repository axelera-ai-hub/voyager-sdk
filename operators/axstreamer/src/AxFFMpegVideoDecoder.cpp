// Copyright Axelera AI, 2025
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
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
// Single source of truth for the AxVideoFormat ↔ AVPixelFormat mapping.
// Adding a new format requires exactly one entry here.
struct FormatEntry {
  AxVideoFormat ax;
  AVPixelFormat av;
};
static const FormatEntry format_table[] = {
  { AxVideoFormat::NV12, AV_PIX_FMT_NV12 },
  { AxVideoFormat::I420, AV_PIX_FMT_YUV420P },
  { AxVideoFormat::Y42B, AV_PIX_FMT_YUV422P },
  { AxVideoFormat::Y444, AV_PIX_FMT_YUV444P },
  { AxVideoFormat::NV16, AV_PIX_FMT_NV16 },
  { AxVideoFormat::RGB, AV_PIX_FMT_RGB24 },
  { AxVideoFormat::BGR, AV_PIX_FMT_BGR24 },
  { AxVideoFormat::RGBA, AV_PIX_FMT_RGBA },
  { AxVideoFormat::BGRA, AV_PIX_FMT_BGRA },
  { AxVideoFormat::GRAY8, AV_PIX_FMT_GRAY8 },
};

static AVPixelFormat
get_av_format(AxVideoFormat fmt)
{
  auto it = std::ranges::find(format_table, fmt, &FormatEntry::ax);
  return it != std::end(format_table) ? it->av : AV_PIX_FMT_RGB24;
}

static AxVideoFormat
get_ax_format(AVPixelFormat fmt)
{
  auto it = std::ranges::find(format_table, fmt, &FormatEntry::av);
  return it != std::end(format_table) ? it->ax : AxVideoFormat::RGB;
}

// Map AVColorSpace to GstMatrix int (0=unknown,1=RGB,2=FCC,3=BT601,4=BT709,5=SMPTE240M,6=BT2020)
static int
av_colorspace_to_gst_matrix(AVColorSpace cs)
{
  switch (cs) {
    case AVCOL_SPC_RGB:
      return 1;
    case AVCOL_SPC_BT709:
      return 4;
    case AVCOL_SPC_FCC:
      return 2;
    case AVCOL_SPC_BT470BG:
      return 3;
    case AVCOL_SPC_SMPTE170M:
      return 3;
    case AVCOL_SPC_SMPTE240M:
      return 5;
    case AVCOL_SPC_BT2020_NCL:
      return 6;
    case AVCOL_SPC_BT2020_CL:
      return 6;
    default:
      return 0;
  }
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

  int linesize_align[AV_NUM_DATA_POINTERS] = {};
  int aligned_w = frame->width, aligned_h = frame->height;
  avcodec_align_dimensions2(ctx, &aligned_w, &aligned_h, linesize_align);
  int num_planes = av_pix_fmt_count_planes(fmt);
  int row_alignment = *std::max_element(linesize_align, linesize_align + num_planes);
  int size = av_image_get_buffer_size(fmt, aligned_w, aligned_h, row_alignment);
  if (size < 0) {
    return size;
  }

  size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  size_t aligned_size = (static_cast<size_t>(size) + page_size - 1) / page_size * page_size;
  using unique_aligned_ptr = std::unique_ptr<void, void (*)(void *)>;
  unique_aligned_ptr buf(std::aligned_alloc(page_size, aligned_size), std::free);
  if (!buf) {
    return AVERROR(ENOMEM);
  }
  if (av_image_fill_arrays(frame->data, frame->linesize,
          static_cast<uint8_t *>(buf.get()), fmt, aligned_w, aligned_h, row_alignment)
      < 0) {
    return AVERROR(ENOMEM);
  }

  frame->buf[0] = av_buffer_create(static_cast<uint8_t *>(buf.get()),
      static_cast<size_t>(size), free_aligned_buffer, nullptr, 0);
  if (!frame->buf[0]) {
    return AVERROR(ENOMEM);
  }
  buf.release();

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
        // Normalise deprecated full-range "J" formats to their limited-range
        // equivalents so the zero-copy path matches and swscale is never called
        // with a deprecated format. Modern FFmpeg codecs report YUV420P/YUV444P
        // with frame->color_range instead of using the "J" aliases, so this
        // case is rare in practice. Note: the normalisation re-interprets the
        // raw samples as limited-range (16-235); for truly full-range content
        // this may produce a slight level shift.
        switch (native_format) {
          case AV_PIX_FMT_YUVJ420P:
            native_format = AV_PIX_FMT_YUV420P;
            break;
          case AV_PIX_FMT_YUVJ422P:
            native_format = AV_PIX_FMT_YUV422P;
            break;
          case AV_PIX_FMT_YUVJ444P:
            native_format = AV_PIX_FMT_YUV444P;
            break;
          default:
            break;
        }
        if (requested_format == AxVideoFormat::UNDEFINED) {
          requested_format = get_ax_format(native_format);
        }

        const int color_range = static_cast<int>(frame->color_range);
        const int color_matrix = av_colorspace_to_gst_matrix(frame->colorspace);

        VideoBuffer video_buffer;
        if (native_format == get_av_format(requested_format)) {
          // Zero-copy: clone frame to manage its lifetime, wrap its buffer directly.
          // Inside this branch native_format == get_av_format(requested_format),
          // so get_ax_format(native_format) == requested_format by construction.
          AVFrame *frame_clone = av_frame_clone(frame);
          if (!frame_clone) {
            throw std::runtime_error("Failed to clone AVFrame");
          }

          const AxVideoFormat buffer_format = requested_format;

          std::vector<size_t> strides;
          std::vector<size_t> offsets;
          if (native_format == AV_PIX_FMT_YUV420P || native_format == AV_PIX_FMT_YUV422P
              || native_format == AV_PIX_FMT_YUV444P) {
            strides = { static_cast<size_t>(frame_clone->linesize[0]),
              static_cast<size_t>(frame_clone->linesize[1]),
              static_cast<size_t>(frame_clone->linesize[2]) };
            offsets = { 0,
              static_cast<size_t>(frame_clone->data[1] - frame_clone->data[0]),
              static_cast<size_t>(frame_clone->data[2] - frame_clone->data[0]) };
          } else if (native_format == AV_PIX_FMT_RGB24 || native_format == AV_PIX_FMT_BGR24
                     || native_format == AV_PIX_FMT_RGBA || native_format == AV_PIX_FMT_BGRA) {
            strides = { static_cast<size_t>(frame_clone->linesize[0]) };
            offsets = { 0 };
          } else {
            // NV12 / NV16: 2 planes
            strides = { static_cast<size_t>(frame_clone->linesize[0]),
              static_cast<size_t>(frame_clone->linesize[1]) };
            offsets = { 0,
              static_cast<size_t>(frame_clone->data[1] - frame_clone->data[0]) };
          }
          video_buffer = VideoBuffer::wrap_external(
              frame_clone->data[0], frame_clone->width, frame_clone->height, buffer_format,
              [frame_clone](uint8_t *) mutable { av_frame_free(&frame_clone); },
              strides, offsets);
        } else {
          // Different format - need to use sws_scale to convert
          video_buffer = VideoBuffer(frame->width, frame->height, requested_format);

          AVPixelFormat target_format = get_av_format(requested_format);
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
          } else if (requested_format == AxVideoFormat::NV12
                     || requested_format == AxVideoFormat::NV16) {
            dst_data[0] = video_buffer.y_plane();
            dst_data[1] = video_buffer.u_plane();

            dst_linesize[0] = frame->width;
            dst_linesize[1] = frame->width;
          } else if (requested_format == AxVideoFormat::Y42B) {
            dst_data[0] = video_buffer.y_plane();
            dst_data[1] = video_buffer.u_plane();
            dst_data[2] = video_buffer.v_plane();

            // 4:2:2 planar: chroma planes are half width, full height
            dst_linesize[0] = frame->width;
            dst_linesize[1] = frame->width / 2;
            dst_linesize[2] = frame->width / 2;
          } else if (requested_format == AxVideoFormat::Y444) {
            dst_data[0] = video_buffer.y_plane();
            dst_data[1] = video_buffer.u_plane();
            dst_data[2] = video_buffer.v_plane();

            dst_linesize[0] = frame->width;
            dst_linesize[1] = frame->width;
            dst_linesize[2] = frame->width;
          } else if (requested_format == AxVideoFormat::RGBA
                     || requested_format == AxVideoFormat::BGRA) {
            dst_data[0] = video_buffer.data();
            dst_linesize[0] = frame->width * 4;
          }
          sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
              dst_data, dst_linesize);
        }

        video_buffer.set_color_range(color_range);
        video_buffer.set_color_matrix(color_matrix);
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
