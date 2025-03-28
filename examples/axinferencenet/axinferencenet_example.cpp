// Copyright Axelera AI, 2025
// An example program showing how to use the AxInferenceNet class to run
// inference on a video stream With an object detection network.

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "AxInferenceNet.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxStreamerUtils.hpp"
#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"

using namespace std::string_literals;

constexpr auto DEFAULT_LABELS = "ax_datasets/labels/coco.names";

namespace
{

auto
read_model_properties(const std::string &filename)
{
  auto size = std::filesystem::file_size(filename);
  std::string content(size, '\0');
  std::ifstream in(filename);
  in.read(&content[0], size);
  return content;
}


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

std::tuple<std::string, std::vector<std::string>, std::string>
parse_args(int argc, char **argv)
{
  std::string model_properties;
  std::vector<std::string> labels;
  std::string input;
  for (auto arg = 1; arg != argc; ++arg) {
    auto s = std::string(argv[arg]);
    if (s.ends_with(".axnet")) {
      model_properties = read_model_properties(s);
    } else if (s.ends_with(".txt") || s.ends_with(".names")) {
      labels = read_labels(s);
    } else if (!std::filesystem::exists(s)) {
      std::cerr << "Warning: Path does not exist: " << s << std::endl;
    } else if (!input.empty()) {
      std::cerr << "Warning: Multiple input files specified: " << s << std::endl;
    } else {
      input = s;
    }
  }
  if (model_properties.empty() || input.empty()) {
    std::cerr
        << "Usage: " << argv[0] << " <model>.axnet [labels.txt] input-source\n"
        << "  <model>.axnet: path to the model axnet file\n"
        << "  labels.txt: path to the labels file (default: " << DEFAULT_LABELS << ")\n"
        << "  input-source: path to video source\n"
        << "\n"
        << "The <model>.axnet file is a file describing the model, preprocessing, and\n"
        << "postprocessing steps of the pipeline.  In the future this will be created\n"
        << "by deploy.py when deploying a pipeline, but for now it is necessary to run\n"
        << "the gstreamer pipeline.  The file can also be created by hand or you can\n"
        << "manually pass the parameters to AxInferenceNet.\n"
        << "\n"
        << "The first step is to compile or download a prebuilt model, here we will show\n"
        << "downloading a prebuilt model:\n"
        << "\n"
        << "  ./download_prebuilt.py yolov8s-coco-onnx\n"
        << "\n"
        << "We then need to run inference.py. This can be done using any media file\n"
        << "for example the fakevideo source, and we need only inference 1 frame:\n"
        << "\n"
        << "  ./inference.py yolov8s-coco-onnx fakevideo --frames=1 --no-display\n"
        << "\n"
        << "This will create a file yolov8s-coco-onnx.axnet in the build directory:\n"
        << "\n"
        << "  examples/bin/axinferencenet_example build/yolov8s-coco-onnx/yolov8s-coco-onnx.axnet\n"
        << std::endl;
    std::exit(1);
  }
  if (labels.empty()) {
    const auto root = std::getenv("AXELERA_FRAMEWORK");
    labels = read_labels(root[0] == '\0' ? DEFAULT_LABELS : root + "/"s + DEFAULT_LABELS);
  }

  return { model_properties, labels, input };
}

struct Frame {
  cv::Mat bgr;
  cv::Mat rgba;
  Ax::MetaMap meta;
};

void
reader_thread(cv::VideoCapture &input, Ax::InferenceNet &net)
{
  while (true) {
    auto frame = std::make_shared<Frame>();
    if (!input.read(frame->bgr)) {
      // Signal to AxInferenceNet that there is no more input and it should
      // flush any buffers still in the pipeline.
      net.end_of_input();
      break;
    }
    // Convert the frame to RGBA format, retaining the original image in the
    // frame in case we want to render it with opencv later.  Ideally this would
    // be done in the preprocessing stage of the AxInferenceNet, but at the
    // moment the rezize_cl operator does not support BGR or RGB input. This
    // will be added in a future release.
    cv::cvtColor(frame->bgr, frame->rgba, cv::COLOR_BGR2RGBA);
    AxVideoInterface video;
    video.info.width = frame->rgba.cols;
    video.info.height = frame->rgba.rows;
    video.info.format = AxVideoFormat::RGBA;
    const auto pixel_width = AxVideoFormatNumChannels(video.info.format);
    video.info.stride = frame->rgba.cols * pixel_width;
    video.data = frame->rgba.data;
    net.push_new_frame(frame, video, frame->meta);
  }
}


// This simple render function shows how to access the inference results from object detection.
// It uses opencv to draw the bounding boxes and labels on the frame
void
render(AxMetaObjDetection &detections, cv::Mat &buffer, const std::vector<std::string> &labels)
{
  for (auto i = size_t{}; i < detections.num_elements(); ++i) {
    auto box = detections.get_box_xyxy(i);
    auto id = detections.class_id(i);
    auto label = id > 0 && id < labels.size() ? labels[id] : "Unknown";
    auto msg = label + " " + std::to_string(int(detections.score(i) * 100)) + "%";
    cv::putText(buffer, msg, cv::Point(box.x1, box.y1 - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0xff, 0xff), 2);
    cv::rectangle(buffer, cv::Rect(cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2)),
        cv::Scalar(0, 0xff, 0xff), 2);
  }
}

} // namespace

int
main(int argc, char **argv)
{
  const auto [model_properties, labels, input] = parse_args(argc, argv);
  cv::VideoCapture cap(input);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video source: " << input << std::endl;
    return 1;
  }

  // We use BlockingQueue to communicate between the frame_completed callback and the main loop
  Ax::BlockingQueue<std::shared_ptr<Frame>> ready;

  auto frame_completed = [&ready](auto &completed) {
    // This function is called whenever an inference result is ready.
    // We first check to see if inference is complete and in which case stop
    // the ready queue to indicate to the main loop to exit.
    if (completed.end_of_input) {
      ready.stop();
    }
    // If we have a valid inference result, we cast the buffer handle to our own
    // Frame type and push it to the ready queue
    auto frame = std::exchange(completed.buffer_handle, {});
    ready.push(std::static_pointer_cast<Frame>(frame));
  };

  Ax::Logger logger(Ax::Severity::warning, nullptr, nullptr);
  const auto props = Ax::properties_from_string(model_properties, logger);
  auto net = Ax::create_inference_net(props, logger, frame_completed);

  // Start the reader thread which reads frames from the video source and pushes
  // them to the inference network
  std::thread reader(reader_thread, std::ref(cap), std::ref(*net));

  // Use OpenCV window to display the results
  const std::string wndname = "AxInferenceNet Demo";
  cv::namedWindow(wndname, cv::WINDOW_AUTOSIZE);
  cv::setWindowProperty(wndname, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);

  const auto start = std::chrono::high_resolution_clock::now();
  int num_frames = 0;
  while (1) {
    auto frame = ready.wait_one();
    if (!frame) {
      break;
    }
    if ((num_frames % 10) == 0) {
      // OpenCV is quite slow at rendering results, so we only render every 10th frame
      auto &detections = dynamic_cast<AxMetaObjDetection &>(*frame->meta["detections"]);
      render(detections, frame->bgr, labels);
      cv::imshow(wndname, frame->bgr);
      cv::waitKey(1);
    }
    ++num_frames;
  }
  const auto end = std::chrono::high_resolution_clock::now();

  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net->stop();
  reader.join();

  // Output some statistics
  const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  const auto taken = static_cast<float>(duration.count()) / 1e6;
  std::cout << "Executed " << num_frames << " frames in " << taken
            << "s : " << num_frames / taken << "fps" << std::endl;
  cv::destroyWindow(wndname);
}
