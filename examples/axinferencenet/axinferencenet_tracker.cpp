// Copyright Axelera AI, 2025
// An example program showing how to use the AxInferenceNet class to run
// inference on a video stream with multiple object tracking.
// This example demonstrates how to integrate the Deep OC-SORT tracker
// with AxInferenceNet, processing detections and re-identification embeddings.

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "AxInferenceNet.hpp"
#include "AxOpenCVRender.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"
#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"

#include "AxFFMpegVideoDecoder.hpp"
#include "AxOpenCVVideoDecoder.hpp"

#include "AxMetaClassification.hpp"
#include "TrackerFactory.h"

using namespace std::string_literals;

namespace
{
std::string
parse_args(int argc, char **argv)
{
  std::string input;
  for (auto arg = 1; arg != argc; ++arg) {
    auto s = std::string(argv[arg]);
    if (!std::filesystem::exists(s) && !s.starts_with("rtsp://")
        && !s.starts_with("/dev/video")) {
      std::cerr << "Warning: Path does not exist: " << s << std::endl;
    } else if (!input.empty()) {
      std::cerr << "Warning: Multiple input files specified: " << s << std::endl;
    } else {
      input = s;
    }
  }
  if (input.empty()) {
    std::cerr
        << "Usage: " << argv[0] << " input-source\n"
        << "  input-source: path to video source\n"
        << "\n"
        << "This example uses a cascaded AxInferenceNet to run the deep-oc-sort tracker,\n"
        << "which takes detections together with appearance embeddings as inputs. \n"
        << "The purpose of this demo is to demonstrate how to work with raw master and sub metas. \n"
        << "For each AxInferenceNet we need to load a <model>.axnet file which describes\n"
        << "the model, preprocessing, and postprocessing steps of the pipeline.  In the future\n"
        << "this will be created by deploy.py when deploying a pipeline, but for now it is\n"
        << "necessary to run the gstreamer pipeline.  The file can also be created by hand\n"
        << "or you can manually pass the parameters to AxInferenceNet.\n"

        << "\n"
        << "The first step is to compile the required models:\n"
        << "We need to run inference.py. This can be done using any media file\n"
        << "for example the fakevideo source, and we need only inference 1 frame:\n"
        << "\n"
        << "  ./inference.py yolox-deep-oc-sort-osnet fakevideo --frames=1 --no-display\n"
        << "\n"
        << "This will create yolox-pedestrian-onnx.axnet and osnet-x1-0-onnx.axnet\n"
        << "in the build/yolox-deep-oc-sort-osnet directory:\n"
        << "\n"
        << "  examples/bin/axinferencenet_tracker <your_video.mp4>" << std::endl;
    std::exit(1);
  }

  return input;
}
struct Frame {
  Ax::VideoBuffer buffer;
  Ax::MetaMap meta;
};
} // namespace

std::vector<ax::ObservedObject>
convertDetections(const AxMetaObjDetection &detections)
{
  std::vector<ax::ObservedObject> convertedDetections;
  for (auto i = size_t{}; i < detections.num_elements(); ++i) {

    const auto &[x1, y1, x2, y2] = detections.get_box_xyxy(i);
    auto id = detections.class_id(i);
    auto score = detections.score(i);

    ax::ObservedObject obs = ax::ObservedObject::FromXYXY(x1, y1, x2, y2, id, score);
    convertedDetections.push_back(obs);
  }
  return convertedDetections;
}

void
render_tracks(const std::vector<ax::TrackedObject> &tracks, cv::Mat &buffer)
{
  static constexpr std::size_t kMaxTraceLength = 25;
  static std::unordered_map<int, std::deque<cv::Point>> track_traces;

  std::unordered_set<int> active_ids;

  for (const auto &tkr : tracks) {
    const auto &bbox = tkr.GetXyxy();
    const auto x1 = static_cast<int>(std::get<0>(bbox));
    const auto y1 = static_cast<int>(std::get<1>(bbox));
    const auto x2 = static_cast<int>(std::get<2>(bbox));
    const auto y2 = static_cast<int>(std::get<3>(bbox));

    int id = tkr.track_id;
    active_ids.insert(id);
    cv::Scalar color = getColorForTracker(id);

    cv::Rect pixelBbox(cv::Point(x1, y1), cv::Point(x2, y2));
    cv::rectangle(buffer, pixelBbox, color, 2);
    cv::putText(buffer, std::to_string(id), cv::Point(pixelBbox.x, pixelBbox.y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

    cv::Point bottom_mid((x1 + x2) / 2, y2);
    auto &trace = track_traces[id];
    trace.push_back(bottom_mid);
    if (trace.size() > kMaxTraceLength) {
      trace.pop_front();
    }

    if (trace.size() >= 2) {
      std::vector<cv::Point> trace_points(trace.begin(), trace.end());
      std::vector<std::vector<cv::Point>> polyline{ trace_points };
      cv::polylines(buffer, polyline, false, color, 1, cv::LINE_AA);
    }
  }

  // Drop traces for inactive tracks to avoid unbounded growth.
  std::vector<int> stale_ids;
  stale_ids.reserve(track_traces.size());
  for (const auto &[id, _] : track_traces) {
    if (!active_ids.count(id)) {
      stale_ids.push_back(id);
    }
  }
  for (int id : stale_ids) {
    track_traces.erase(id);
  }
}

int
main(int argc, char **argv)
{
  const auto input = parse_args(argc, argv);
  Ax::Logger logger;

  const auto root = "build/yolox-deep-oc-sort-osnet"s;
  const auto props1
      = Ax::read_inferencenet_properties(root + "/osnet-x1-0-onnx.axnet", logger);
  const auto props0 = Ax::read_inferencenet_properties(
      root + "/yolox-pedestrian-onnx.axnet", logger);

  // We use BlockingQueue to communicate between the frame_completed callback of the last
  // model and the main loop, and then chain the cascaded networks in reverse order
  Ax::BlockingQueue<std::shared_ptr<Frame>> ready;
  auto net1 = Ax::create_inference_net(props1, logger, Ax::forward_to(ready));
  auto net0 = Ax::create_inference_net(props0, logger, Ax::forward_to(*net1));

  auto frame_callback = [&net0](Ax::VideoBuffer buffer) {
    if (!buffer.is_valid()) {
      net0->end_of_input();
      return;
    }
    auto frame_data = std::make_shared<Frame>();
    frame_data->buffer = std::move(buffer);
    auto video = frame_data->buffer.to_video_interface();
    net0->push_new_frame(frame_data, video, frame_data->meta);
  };

  auto video_decoder = Ax::FFMpegVideoDecoder(input, frame_callback, AxVideoFormat::I420);
  // auto video_decoder = Ax::OpenCVVideoDecoder(input, frame_callback, AxVideoFormat::RGB);
  video_decoder.start_decoding();

  TrackerParams params = CreateTrackerParams({ { "max_age", 30 },
      { "det_thresh", 0.6f }, { "iou_threshold", 0.3f }, { "max_id", 0 } });

  std::string trackerTypeStr("oc_sort");
  std::unique_ptr<ax::MultiObjTracker> tracker
      = CreateMultiObjTracker(trackerTypeStr, params);

  int frame_num = 0;
  auto display = Ax::OpenCV::create_display("AxInferenceNet Cascade Demo");

  while (1) {
    auto frame = ready.wait_one();
    if (!frame) {
      break;
    }

    std::cout << "Processing frame #" << frame_num << " " << std::endl;

    // Get the detections meta (master meta)
    auto *master_meta = frame->meta["detections"].get();
    auto &detections = dynamic_cast<AxMetaObjDetection &>(*master_meta);
    std::vector<ax::ObservedObject> convertedDetections = convertDetections(detections);

    std::vector<std::vector<float>> embeddings{};

    const auto submeta_names = master_meta->submeta_names();
    for (auto &&submeta_name : submeta_names) {
      auto submetas = master_meta->get_submetas(submeta_name);
      // Iterate through submetas
      for (auto &&sub : submetas) {
        if (!sub)
          continue;

        // Check if this is the reid submeta
        if (std::string(submeta_name) == "reid") {
          if (auto *embed_meta = dynamic_cast<AxMetaEmbeddings *>(sub)) {
            auto embs = embed_meta->get_embeddings();
            if (!embs.empty() && !embs[0].empty()) {
              embeddings.push_back(std::move(embs[0]));
            }
          }
        }
      }
    }

    if (embeddings.size() != convertedDetections.size()) {
      throw std::runtime_error("axinferencenet_tracker: embeddings size mismatch, number of embeddings "
                               + std::to_string(embeddings.size()) + " != number of detections "
                               + std::to_string(convertedDetections.size()));
    }

    const auto &activeTrackers = tracker->Update(convertedDetections, embeddings);

    // Convert from YUV to RGB for rendering
    // cvmat_from_buffer handles both contiguous and strided buffers automatically
    cv::Mat yuv_mat = Ax::cvmat_from_buffer(frame->buffer);
    cv::Mat rgb_frame;

    // Use the correct color conversion based on the buffer format
    if (frame->buffer.format() == AxVideoFormat::NV12) {
      cv::cvtColor(yuv_mat, rgb_frame, cv::COLOR_YUV2RGB_NV12);
    } else if (frame->buffer.format() == AxVideoFormat::I420) {
      cv::cvtColor(yuv_mat, rgb_frame, cv::COLOR_YUV2RGB_I420);
    } else {
      throw std::runtime_error("Unsupported video format for rendering");
    }

    render_tracks(activeTrackers, rgb_frame);

    // disable the default rendering of meta data, just show the already augmented image
    const Ax::MetaMap empty_meta;
    display->show(rgb_frame, empty_meta, AxVideoFormat::RGB, {}, 0);

    ++frame_num;
  }

  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net0->stop();
  net1->stop();
}
