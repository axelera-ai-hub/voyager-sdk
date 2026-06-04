---
title: "AxInferenceNet Raw Tensor Output"
---
# AxInferenceNet Raw Tensor Output

This example demonstrates how to use `AxInferenceNet` to obtain raw tensor outputs from the neural network instead of relying on built-in postprocessing. This is useful when you need to implement custom postprocessing logic, such as a specialized NMS algorithm, application-specific filtering, or integration with your own detection pipeline. The example includes a complete YOLOv8 postprocessing implementation as a reference.

## What you'll learn

- How to configure `AxInferenceNet` to output raw tensors instead of processed metadata
- How to access raw tensor data and shape information via `AxMetaRawTensor`
- How to implement custom YOLOv8 postprocessing including letterbox correction and NMS
- How to handle coordinate transformations between model input space and original image space

## Prerequisites

- Voyager SDK installed and activated
- C++ compiler with C++17 support
- OpenCV development libraries (including `opencv2/dnn` for NMS)
- FFmpeg development libraries (or OpenCV video decoder as alternative)
- A model deployed with raw tensor output (e.g., `yolov8n-output-tensor`)

## Building

```bash
cd $AXELERA_FRAMEWORK
make examples
```

## Source

> [!TIP]
> **Download**
> This example is included in the SDK at `examples/axinferencenet/axinferencenet_tensor.cpp`.


```cpp
// Copyright Axelera AI, 2025
// An example program showing how to use the AxInferenceNet class to run
// inference on a video stream and obtain raw tensor outputs.
// This allows users to implement their own postprocessing logic
// instead of relying on built-in metadata or postprocessing.

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "AxInferenceNet.hpp"
#include "AxMetaRawTensor.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"
#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"

#include "AxFFMpegVideoDecoder.hpp"
#include "AxOpenCVRender.hpp"
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
        << "The <model>.axnet file describes the model and the pipeline stages.\n"
        << "In the axinferencenet_example.cpp, AxInferenceNet runs the full end-to-end pipeline, including preprocessing, inference, and postprocessing.\n"
        << "\n"
        << "In this example, AxInferenceNet is used to perform only preprocessing and main inference.\n"
        << "Users receive the raw tensor outputs and are responsible for implementing their own postprocessing (such as NMS and other steps).\n"
        << "This allows for custom postprocessing logic and more flexible integration.\n"
        << "\n"
        << "The first step is to deploy the model and pipeline. This can be done by running:\n"
        << "\n"
        << "  ./inference.py yolov8n-output-tensor fakevideo --frames=1 --no-display\n"
        << "\n"
        << "This will deploy the model and pipeline (may take some time on first run) and generate the .axnet file in the build folder.\n"
        << "You can use any media file as input (e.g., fakevideo or a real video file), and only need to infer 1 frame for deployment.\n"
        << "\n"
        << "To run inference, use:\n"
        << "\n"
        << "  examples/bin/axinferencenet_tensor build/yolov8n-output-tensor/yolov8n-output-tensor.axnet </path/to/the/media>\n"
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

struct Detection {
  cv::Rect bbox;
  int label;
  float score;
};

std::vector<Detection>
postprocess_yolov8(const float *data, const std::vector<int64_t> &dims,
    float prob_threshold, float nms_threshold, int top_k, int img_width,
    int img_height, int model_input_width, int model_input_height, bool letterboxed)
{
  std::vector<Detection> detections;
  if (dims.size() != 4)
    return detections;
  int batch = static_cast<int>(dims[1]);
  int num_channels = static_cast<int>(dims[2]);
  int num_anchors = static_cast<int>(dims[3]);
  int num_classes = num_channels - 4;
  if (batch != 1)
    return detections;

  cv::Mat output(num_channels, num_anchors, CV_32F, const_cast<float *>(data));
  output = output.t();

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;

  // Calculate scale and padding for letterbox
  float scale = std::min(float(model_input_width) / img_width,
      float(model_input_height) / img_height);
  float new_unpad_w = img_width * scale;
  float new_unpad_h = img_height * scale;
  float pad_w = (model_input_width - new_unpad_w) / 2.0f;
  float pad_h = (model_input_height - new_unpad_h) / 2.0f;

  for (int i = 0; i < num_anchors; ++i) {
    const float *row = output.ptr<float>(i);
    const float *bboxesPtr = row;
    const float *scoresPtr = row + 4;
    auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
    float score = *maxSPtr;
    if (score > prob_threshold) {
      float x = *bboxesPtr++;
      float y = *bboxesPtr++;
      float w = *bboxesPtr++;
      float h = *bboxesPtr;

      float x0 = x - 0.5f * w;
      float y0 = y - 0.5f * h;
      float x1 = x + 0.5f * w;
      float y1 = y + 0.5f * h;

      if (letterboxed) {
        // Undo letterbox: remove padding, then scale to original image
        x0 = (x0 - pad_w) / scale;
        y0 = (y0 - pad_h) / scale;
        x1 = (x1 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
      } else {
        // Simple resize
        x0 = x0 * img_width / model_input_width;
        y0 = y0 * img_height / model_input_height;
        x1 = x1 * img_width / model_input_width;
        y1 = y1 * img_height / model_input_height;
      }

      x0 = std::clamp(x0, 0.f, float(img_width));
      y0 = std::clamp(y0, 0.f, float(img_height));
      x1 = std::clamp(x1, 0.f, float(img_width));
      y1 = std::clamp(y1, 0.f, float(img_height));

      int label = int(maxSPtr - scoresPtr);
      cv::Rect bbox{ cv::Point(int(x0), int(y0)), cv::Point(int(x1), int(y1)) };
      bboxes.push_back(bbox);
      labels.push_back(label);
      scores.push_back(score);
    }
  }
  // NMS
  cv::dnn::NMSBoxes(bboxes, scores, prob_threshold, nms_threshold, indices, 1.f, top_k);
  for (int idx : indices) {
    Detection det;
    det.bbox = bboxes[idx];
    det.label = labels[idx];
    det.score = scores[idx];
    detections.push_back(det);
  }
  return detections;
}

void
render_detections(const std::vector<Detection> &detections, cv::Mat &buffer,
    const std::vector<std::string> &labels)
{
  for (const auto &det : detections) {
    std::string label = (det.label >= 0 && det.label < int(labels.size())) ?
                            labels[det.label] :
                            "Unknown";
    std::string msg = label + " " + std::to_string(int(det.score * 100)) + "%";
    cv::putText(buffer, msg, cv::Point(det.bbox.x, det.bbox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0xff, 0xff), 2);
    cv::rectangle(buffer, det.bbox, cv::Scalar(0, 0xff, 0xff), 2);
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

  auto display = Ax::OpenCV::create_display("AxInferenceNet Tensor Output Demo");
  int model_input_width = 640;
  int model_input_height = 640;
  bool letterboxed = true;

  while (1) {
    auto frame = ready.wait_one();
    if (!frame) {
      break;
    }

    try {
      auto &tensor_wrapper = dynamic_cast<AxMetaRawTensor &>(*frame->meta["detections"]);
      if (tensor_wrapper.get_tensor() && tensor_wrapper.get_tensor()->num_tensors() > 0) {
        std::vector<int64_t> dims = get_tensor_shape(tensor_wrapper);
        const float *data = get_tensor_data<float>(tensor_wrapper);
        auto detections = postprocess_yolov8(data, dims, 0.25f, 0.45f, 100,
            frame->rgb.cols, frame->rgb.rows, model_input_width,
            model_input_height, letterboxed);
        render_detections(detections, frame->rgb, labels);
      }
    } catch (const std::bad_cast &) {
      throw std::runtime_error(
          "Error: Expected AxMetaRawTensor for detections, got different type.");
    }
    // disable the default rendering of meta data, just show the already augmented image
    Ax::MetaMap empty_meta;
    display->show(frame->rgb, empty_meta, AxVideoFormat::RGB, {}, 0);
  }

  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net->stop();
}
```

## Key concepts

The key difference from the basic example is the use of **`AxMetaRawTensor`** instead of `AxMetaObjDetection`. When a model is deployed with raw tensor output (e.g., using the `yolov8n-output-tensor` pipeline), `AxInferenceNet` skips built-in postprocessing and delivers the network's raw output tensors. The tensor data and shape are accessed via `get_tensor_data<float>()` and `get_tensor_shape()`, giving you a pointer to the raw float buffer and its dimensions.

The `postprocess_yolov8()` function shows a complete custom postprocessing pipeline. The YOLOv8 output tensor has shape `[1, num_channels, num_anchors]` where the first 4 channels are bounding box coordinates (center x, center y, width, height) and the remaining channels are class probabilities. The function transposes this matrix, iterates over anchors, finds the highest-scoring class per anchor, and filters by a probability threshold.

A critical detail is **letterbox correction**. When the model uses letterbox preprocessing (padding to maintain aspect ratio), the raw coordinates are in the padded model-input space. The code calculates the scale factor and padding offsets, then reverses the transformation to map coordinates back to original image dimensions. The alternative simple-resize path is also provided for models that stretch the input without preserving aspect ratio. After coordinate transformation, OpenCV's `cv::dnn::NMSBoxes` performs non-maximum suppression to eliminate duplicate detections.

---
