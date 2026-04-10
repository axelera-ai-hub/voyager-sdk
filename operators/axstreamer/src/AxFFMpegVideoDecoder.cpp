// Copyright Axelera AI, 2025
#include <opencv2/imgproc.hpp>
#include "AxFFMpegVideoDecoder.hpp"

Ax::FFMpegVideoDecoder::FFMpegVideoDecoder(const std::string &input,
    std::function<void(cv::Mat)> frame_callback, AxVideoFormat format)
    : Ax::VideoDecode(input, frame_callback, format)
{
  const AVCodec *codec = nullptr;
  sws_ctx = nullptr;
  video_stream_index = -1;
  format_ctx = nullptr;

  if (format != AxVideoFormat::RGB && format != AxVideoFormat::BGR
      && format != AxVideoFormat::I420 && format != AxVideoFormat::NV12) {
    throw std::invalid_argument("Unsupported video format for FFMpegVideoDecoder");
  }
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

  // Initialize codec
  if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
    throw std::runtime_error("Could not open codec");
  }
}

Ax::FFMpegVideoDecoder::~FFMpegVideoDecoder()
{
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
void
Ax::FFMpegVideoDecoder::reader_func()
{
  AVPacket *packet = av_packet_alloc();
  AVFrame *frame = av_frame_alloc();

  cv::Mat framemat;
  while (av_read_frame(format_ctx, packet) >= 0) {
    if (packet->stream_index == video_stream_index) {
      if (avcodec_send_packet(codec_ctx, packet) < 0) {
        break;
      }
      while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
        AVPixelFormat target_format = Ax::FFMpegVideoDecoder::format_map.at(format);

        // Determine cv::Mat type and size based on format
        int mat_type;
        int mat_height;
        if (format == AxVideoFormat::I420) {
          // I420: Y plane + U plane + V plane (each U/V is 1/4 size of Y)
          // Total height = height * 3/2
          mat_type = CV_8UC1;
          mat_height = frame->height + frame->height / 2;
        } else if (format == AxVideoFormat::NV12) {
          // NV12: Y plane + UV interleaved plane
          // Total height = height * 3/2
          mat_type = CV_8UC1;
          mat_height = frame->height + frame->height / 2;
        } else {
          // RGB/BGR formats
          mat_type = CV_8UC3;
          mat_height = frame->height;
        }

        framemat.create(mat_height, frame->width, mat_type);

        // Create sws context for format conversion or reuse existing context as needed
        sws_ctx = sws_getCachedContext(sws_ctx, frame->width, frame->height,
            static_cast<AVPixelFormat>(frame->format), frame->width,
            frame->height, target_format, SWS_POINT, nullptr, nullptr, nullptr);
        if (!sws_ctx) {
          throw std::runtime_error("Could not create sws context");
        }

        // For planar formats, we need to set up plane pointers
        if (format == AxVideoFormat::I420) {
          uint8_t *dst_data[3];
          int dst_linesize[3];

          dst_data[0] = framemat.data; // Y plane
          dst_data[1] = framemat.data + frame->width * frame->height; // U plane
          dst_data[2] = dst_data[1] + (frame->width / 2) * (frame->height / 2); // V plane

          dst_linesize[0] = frame->width;
          dst_linesize[1] = frame->width / 2;
          dst_linesize[2] = frame->width / 2;

          sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
              dst_data, dst_linesize);
        } else if (format == AxVideoFormat::NV12) {
          uint8_t *dst_data[2];
          int dst_linesize[2];

          dst_data[0] = framemat.data; // Y plane
          dst_data[1] = framemat.data + frame->width * frame->height; // UV plane

          dst_linesize[0] = frame->width;
          dst_linesize[1] = frame->width; // UV is interleaved, same width as Y

          sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
              dst_data, dst_linesize);
        } else {
          // RGB/BGR formats
          const int cvLinesizes[1] = { static_cast<int>(framemat.step) };
          sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
              &framemat.data, cvLinesizes);
        }
        frame_callback(std::move(framemat));
      }
    }
    av_packet_unref(packet);
  }
  frame_callback(std::move(framemat)); // Notify end of stream
  if (sws_ctx) {
    sws_freeContext(sws_ctx);
    sws_ctx = nullptr;
  }
}
