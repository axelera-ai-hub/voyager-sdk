---
title: "AxInferenceNet Basic Object Detection"
---
# AxInferenceNet Basic Object Detection

This example demonstrates how to use the `AxInferenceNet` class to run end-to-end object detection inference on a video stream. It loads a model from an `.axnet` file, decodes video frames, runs inference with built-in preprocessing and postprocessing, and renders bounding boxes with labels using OpenCV.

## What you'll learn

- How to load model properties from an `.axnet` file and create an `AxInferenceNet` pipeline
- How to decode video frames with `AxFFMpegVideoDecoder` and push them through the inference network
- How to retrieve object detection metadata (`AxMetaObjDetection`) from inference results
- How to render bounding boxes and class labels on frames using OpenCV

## Prerequisites

- Voyager SDK installed and activated
- C++ compiler with C++17 support
- OpenCV development libraries
- FFmpeg development libraries (or OpenCV video decoder as alternative)
- A compiled or downloaded model with its `.axnet` file (e.g., `yolov8s-coco-onnx`)

## Building

```bash
cd $AXELERA_FRAMEWORK
make examples
```

## Source

> [!TIP]
> **Download**
> This example is included in the SDK at `examples/axinferencenet/axinferencenet_example.cpp`.


```cpp
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

#include "AxInferenceNet.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxOpenCVRender.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"
#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"

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

std::tuple<std::string, std::vector<std::string>, std::string>
parse_args(int argc, char **argv)
{
  std::string model_properties;
  std::vector<std::string> labels;
  std::string input;
  for (auto arg = 1; arg != argc; ++arg) {
    auto s = std::string(argv[arg]);
    if (s.ends_with(".axnet")) {
      model_properties = s;
    } else if (s.ends_with(".txt") || s.ends_with(".names")) {
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
  if (model_properties.empty() || input.empty()) {
    std::cerr
        << "Usage: " << argv[0] << " <model>.axnet [labels.txt] input-source\n"
        << "  <model>.axnet: path to the model axnet file\n"
        << "  labels.txt: path to the labels file (default: " << DEFAULT_LABELS << ")\n"
        << "  input-source: video source (e.g. file path, rtsp://, /dev/video)\n"
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
        << "  axdownloadmodel yolov8s-coco-onnx\n"
        << "\n"
        << "We then need to run inference.py. This can be done using any media file\n"
        << "for example the fakevideo source, and we need only inference 1 frame:\n"
        << "\n"
        << "  ./inference.py yolov8s-coco-onnx fakevideo --frames=1 --no-display\n"
        << "\n"
        << "This will create a file yolov8s-coco-onnx.axnet in the build directory:\n"
        << "\n"
        << "  examples/bin/axinferencenet_example build/yolov8s-coco-onnx/yolov8s-coco-onnx.axnet media/traffic3_480p.mp4\n"
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
  cv::Mat rgb;
  Ax::MetaMap meta;
};

// This simple render function shows how to access the inference results from object detection.
// It uses opencv to draw the bounding boxes and labels on the frame
// Note that AxOpenCVRender.hpp has a more advanced set of renderers for more meta types,
// but this version is left here to show how to access the results directly
void
render(AxMetaObjDetection &detections, cv::Mat &buffer, const std::vector<std::string> &labels)
{
  for (auto i = size_t{}; i < detections.num_elements(); ++i) {
    auto box = detections.get_box_xyxy(i);
    auto id = detections.class_id(i);
    auto label = id >= 0 && id < labels.size() ? labels[id] : "Unknown";
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
  Ax::Logger logger;
  const auto [model_properties, labels, input] = parse_args(argc, argv);

  // We use BlockingQueue to communicate between the frame_completed callback and the main loop
  Ax::BlockingQueue<std::shared_ptr<Frame>> ready;
  auto props = Ax::read_inferencenet_properties(model_properties, logger);
  auto net = Ax::create_inference_net(props, logger, Ax::forward_to(ready));

  auto frame_callback = [&net](cv::Mat frame) {
    if (frame.empty()) {
      net->end_of_input();
      return;
    }
    auto frame_data = std::make_shared<Frame>();
    frame_data->rgb = std::move(frame);
    auto video = Ax::video_from_cvmat(frame_data->rgb, AxVideoFormat::RGB);
    net->push_new_frame(frame_data, video, frame_data->meta);
  };

  auto video_decoder = Ax::FFMpegVideoDecoder(input, frame_callback, AxVideoFormat::RGB);
  // auto video_decoder = Ax::OpenCVVideoDecoder(input, frame_callback, AxVideoFormat::RGB);
  video_decoder.start_decoding();

  auto display = Ax::OpenCV::create_display("AxInferenceNet Demo");
  Ax::OpenCV::RenderOptions render_options;

  while (1) {
    auto frame = ready.wait_one();
    if (!frame) {
      break;
    }
    // Ax::OpenCV::Display will render all meta, but to demonstrate how the detections can be
    // accessed we render them here manually, and disable the default renderer.
    auto &detections = dynamic_cast<AxMetaObjDetection &>(*frame->meta["detections"]);
    render(detections, frame->rgb, labels);
    const Ax::MetaMap empty_meta;
    display->show(frame->rgb, empty_meta, AxVideoFormat::RGB, render_options, 0);
  }

  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net->stop();
}
```

## Key concepts

The core pattern in this example is the **producer-consumer pipeline**. An `AxInferenceNet` is created from model properties loaded via `Ax::read_inferencenet_properties()`, and a `Ax::BlockingQueue` bridges the asynchronous inference callback to the synchronous main display loop. The `Ax::forward_to(ready)` helper wires the network's completion callback to enqueue finished frames.

Video decoding is handled by `Ax::FFMpegVideoDecoder`, which calls `frame_callback` for each decoded frame. Inside the callback, each frame is wrapped in a `Frame` struct that carries both the RGB image and a `MetaMap` for inference metadata. The frame is then pushed into the network with `push_new_frame()`.

After inference completes, detection results are available as `AxMetaObjDetection` in the frame's metadata map under the `"detections"` key. The custom `render()` function demonstrates how to iterate over detections, extract bounding box coordinates via `get_box_xyxy()`, and draw labels with confidence scores. While the SDK provides built-in renderers in `AxOpenCVRender.hpp`, this manual approach shows direct access to the detection API.

The main loop calls `ready.wait_one()` to block until a frame is available, and exits cleanly when a null frame signals end-of-stream. Finally, `net->stop()` ensures all inference threads are joined before the program exits.

---
