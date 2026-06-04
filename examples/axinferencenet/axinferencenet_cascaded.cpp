// Copyright Axelera AI, 2025
// An example program showing how to use the AxInferenceNet class to run
// inference on a video stream With an object detection network.

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "AxFilterDetections.hpp"
#include "AxInferenceNet.hpp"
#include "AxOpenCVRender.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"
#include "axruntime/axruntime.hpp"

#include "AxFFMpegVideoDecoder.hpp"
#include "AxOpenCVVideoDecoder.hpp"

using namespace std::string_literals;
constexpr auto DEFAULT_LABELS = "ax_datasets/labels/coco.names";

namespace
{
auto
read_labels(const std::string &path)
{
  std::vector<std::string> labels;
  std::ifstream file(path);
  for (std::string line; std::getline(file, line);) {
    labels.push_back(line);
  }
  return labels;
}

std::tuple<std::vector<std::string>, std::string>
parse_args(int argc, char **argv)
{
  std::vector<std::string> labels;
  std::string input;
  for (auto arg = 1; arg != argc; ++arg) {
    auto s = std::string(argv[arg]);
    if (s.ends_with(".txt") || s.ends_with(".names")) {
      labels = read_labels(s);
    } else if (!std::filesystem::exists(s) && !s.starts_with("rtsp://")
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
        << "Usage: " << argv[0] << " [labels.txt] input-source\n"
        << "  labels.txt: path to the labels file (default: " << DEFAULT_LABELS << ")\n"
        << "  input-source: video source (e.g. file path, rtsp://, /dev/video)\n"
        << "\n"
        << "This example uses a cascaded AxInferenceNet using the fruit-demo network,\n"
        << "which consists of a pose detector, the ROIs of which are passed to a\n"
        << "segmentation network.  Finally another parallel network is used to detect\n"
        << "fruit (using a filter in the NMS).  This purpose of the demo is to show the\n"
        << "better detection that can be found by using an ROI in a cascaded model."
        << "\n"
        << "For each AxInferenceNet we need to load a <model>.axnet file which describes\n"
        << "the model, preprocessing, and postprocessing steps of the pipeline.  In the future\n"
        << "this will be created by deploy.py when deploying a pipeline, but for now it is\n"
        << "necessary to run the gstreamer pipeline.  The file can also be created by hand\n"
        << "or you can manually pass the parameters to AxInferenceNet.\n"

        << "\n"
        << "The first step is to compile or download the required models:\n"
        << "\n"
        << "  axdownloadmodel fruit-demo\n"
        << "\n"
        << "We then need to run inference.py. This can be done using any media file\n"
        << "for example the fakevideo source, and we need only inference 1 frame:\n"
        << "\n"
        << "  ./inference.py fruit-demo fakevideo --frames=1 --no-display\n"
        << "\n"
        << "This will create yolov8s-fruit.axnet, yolov8lpose-coco-onnx.axnet, and \n"
        << "yolov8sseg-coco-onnx.axnet in the build/fruit-demo directory:\n"
        << "\n"
        << "  examples/bin/axinferencenet_cascaded media/4K_fruit1.mp4" << std::endl;
    std::exit(1);
  }
  if (labels.empty()) {
    const auto root = std::getenv("AXELERA_FRAMEWORK");
    labels = read_labels(!root || root[0] == '\0' ? DEFAULT_LABELS : root + "/"s + DEFAULT_LABELS);
  }

  return { labels, input };
}

struct Frame {
  Ax::VideoBuffer buffer;
  Ax::MetaMap meta;
};

} // namespace

int
main(int argc, char **argv)
{
  const auto [labels, input] = parse_args(argc, argv);
  Ax::Logger logger;
  const auto root = "build/fruit-demo"s;
  const auto props0 = Ax::read_inferencenet_properties(
      root + "/yolov8lpose-coco-onnx.axnet", logger);
  const auto props1
      = Ax::read_inferencenet_properties(root + "/yolov8sseg-coco-onnx.axnet", logger);
  const auto props2
      = Ax::read_inferencenet_properties(root + "/yolov8s-fruit.axnet", logger);

  // We use BlockingQueue to communicate between the frame_completed callback of the last
  // model and the main loop, and then chain the cascaded networks in reverse order
  Ax::BlockingQueue<std::shared_ptr<Frame>> ready;
  auto net2 = Ax::create_inference_net(props2, logger, Ax::forward_to(ready));
  const Ax::FilterDetectionsProperties filter{
    .input_meta_key = "master_detections",
    .output_meta_key = "master_detections_adapted_as_input_for_segmentations",
    .hide_output_meta = true,
    .min_width = 50,
    .min_height = 50,
    .classes_to_keep{},
    .score = 0.5f,
    .which = Ax::Which::Center,
    .top_k = 3,
  };
  auto net1 = Ax::create_inference_net(props1, logger, Ax::forward_to(*net2));
  auto net0 = Ax::create_inference_net(
      props0, logger, Ax::filter_detections_to(*net1, filter));

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

  auto video_decoder = Ax::FFMpegVideoDecoder(input, frame_callback);
  // auto video_decoder = Ax::OpenCVVideoDecoder(input, frame_callback, AxVideoFormat::RGB);
  video_decoder.start_decoding();

  auto display = Ax::OpenCV::create_display("AxInferenceNet Cascade Demo");
  Ax::OpenCV::RenderOptions render_options;
  render_options.labels = labels;

  while (1) {
    auto frame = ready.wait_one();
    if (!frame) {
      break;
    }
    cv::Mat rgb_frame = Ax::cvmat_from_buffer(frame->buffer);

    // Use the correct color conversion based on the buffer format
    display->show(rgb_frame, frame->meta, frame->buffer.format(), render_options, 0);
  }

  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net0->stop();
  net1->stop();
  net2->stop();
}
