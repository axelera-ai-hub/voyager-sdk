![](/docs/images/Ax_Page_Banner_2500x168_01.png)

_Last updated: 2025-04-15_

# AxInferenceNet C++ Integration Tutorial

## Contents

- [AxInferenceNet C++ Integration Tutorial](#axinferencenet-c-integration-tutorial)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Preparing the example](#preparing-the-example)
  - [Reading frames from the video source](#reading-frames-from-the-video-source)
  - [Rendering the results of inference](#rendering-the-results-of-inference)
  - [Setting up the inference loop](#setting-up-the-inference-loop)
  - [The main inference loop](#the-main-inference-loop)
  - [Cleanup](#cleanup)
  - [Using AxInferenceNet for Raw Tensor Output and Custom Postprocessing](#using-axinferencenet-for-raw-tensor-output-and-custom-postprocessing)
    - [How to use raw tensor output (user workflow)](#how-to-use-raw-tensor-output-user-workflow)
    - [Why and When to Use Raw Tensor Output?](#why-and-when-to-use-raw-tensor-output)
    - [Pipeline Configuration: Standard vs. Raw Tensor Output](#pipeline-configuration-standard-vs-raw-tensor-output)
    - [When to use raw tensor output](#when-to-use-raw-tensor-output)
    - [Performance note](#performance-note)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites
- Complete [Quick Start Guide](quick_start_guide.md) and [Application Integration](application.md) - understand Python API first
- C++ programming experience (C++11 or later)
- CMake 3.10+ installed
- C++ compiler (gcc/g++ 9.0+)
- OpenCV installed (for example code)
- Understanding of object detection and pipeline concepts

## Level
**Intermediate** - Requires C++ programming and understanding of inference pipelines

## Overview

**Note:** This interface is still under development, and so this example is subject to change. The core functionality will remain the same, but interfaces and type names may change.

[AxInferenceNet C++ Reference](/docs/reference/axinferencenet.md) documents the interface and provides an overview for `Ax::InferenceNet`. In this document, we work through an example program that utilizes it to perform object detection using OpenCV to decode a local video (or USB device) and for rendering the output.

To demonstrate the usage of AxInferenceNet in a real example, we are going to walk through the implementation of a simple object detection application. The example can be built using:

```bash
(venv) $ make examples
```
(This assumes that you have activated the Axelera environment).

## Preparing the example

To run this demo, you must first obtain a suitable model, for example:

```bash
(venv) $ axdownloadmodel yolov8s-coco
```

Additionally, we need to use the Axelera pipeline builder to create a description of the pipeline. This is a file used to configure AxInferenceNet for the model, and any local hardware-specific acceleration available, for example, OpenCL. In the future, this will be available without executing inference, but in this initial version, we need to just run `./inference.py` with suitable arguments.

```bash
(venv) $ ./inference.py yolov8s-coco fakevideo --frames 1 --no-display
```

`fakevideo` tells inference.py to use a fake input video source. We could also use a video file for this, but fakevideo is easiest to use as you can see.

`--frames 1` and `--no-display` are because we do not really need to execute inference, we just need to utilize the pipeline builder of `inference.py`. Nor do we need to visualize the results. We could also configure `AxInferenceNet` for different accelerated pipelines using, for example, `--disable-opencl` or `--enable-vaapi`, or `--aipu-cores 2`. Most other options will not be relevant to this example. To use multiple Metis devices, you may want to also use `--devices`. All of these options can also be modified directly, but using the pipeline builder is by far the easiest way to get started.

With that done, we should now see the following files under `build/yolov8s-coco`:

```bash
(venv) $ ls build/yolov8s-coco/
logs  yolov8s-coco  yolov8s-coco.axnet
(venv) $ cat build/yolov8s-coco/yolov8s-coco.axnet
model=build/yolov8s-coco/yolov8s-coco/1/model.json
devices=metis-0:1:0
double_buffer=True
dmabuf_inputs=True
dmabuf_outputs=True
num_children=4
preprocess0_lib=libtransform_resize_cl.so
preprocess0_options=width:640;height:640;padding:114;letterbox:1;scale_up:1;to_tensor:1;mean:0.,0.,0.;std:1.,1.,1.;quant_scale:0.003920177463442087;quant_zeropoint:-128.0
preprocess1_lib=libtransform_padding.so
preprocess1_options=padding:0,0,1,1,1,15,0,0;fill:0
preprocess1_batch=1
postprocess0_lib=libdecode_yolov8.so
postprocess0_options=meta_key:detections;classes:80;confidence_threshold:0.25;scales:0.07552404701709747,0.06546489894390106,0.07111278176307678,0.09202337265014648,0.15390163660049438,0.16983751952648163;padding:0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,48|0,0,0,0,0,0,0,48|0,0,0,0,0,0,0,48;zero_points:-65,-58,-44,146,104,110;topk:30000;multiclass:0;model_width:640;model_height:640;scale_up:1;letterbox:1
postprocess0_mode=read
postprocess1_lib=libinplace_nms.so
postprocess1_options=meta_key:detections;max_boxes:300;nms_threshold:0.45;class_agnostic:1;location:CPU
```

We can also run the example to make sure it built correctly. If we run it with no arguments, it will show usage instructions:

```bash
(venv) $ examples/bin/axinferencenet_example
Usage: examples/bin/axinferencenet_example <model>.axnet [labels.txt] input-source
  <model>.axnet: path to the model axnet file
  labels.txt: path to the labels file (default: ax_datasets/labels/coco.names)
  input-source: path to video source

The `<model>.axnet` file describes the model, preprocessing, and
postprocessing steps of the pipeline. In the future, this will be created
by deploy.py when deploying a pipeline, but for now it is necessary to run
the gstreamer pipeline. The file can also be created by hand or you can
manually pass the parameters to AxInferenceNet.

The first step is to compile or download a prebuilt model, here we will show
downloading a prebuilt model:

  axdownloadmodel yolov8s-coco-onnx

We then need to run inference.py. This can be done using any media file
for example the fakevideo source, and we need only inference 1 frame:

  ./inference.py yolov8s-coco-onnx fakevideo --frames=1 --no-display

This will create a file yolov8s-coco-onnx.axnet in the build directory:

  examples/bin/axinferencenet_example build/yolov8s-coco-onnx/yolov8s-coco-onnx.axnet
```

We can then run the demo. Note that the demo requires a display to run. This can be a local display, or if connected via SSH, ensure you use the `-X` or `-Y` option when connecting to forward X11 calls to your local X11 server.

```bash
(venv) $ examples/bin/axinferencenet_example build/yolov8s-coco/yolov8s-coco.axnet media/traffic3_480p.mp4
```

A window should display showing the object detection.

![A Screenshot of the AxInferenceNet example](/docs/images/axinferencenet_example_844x480_01.png)

We will now look at the key parts of [axinferencenet_example.cpp](/examples/axinferencenet/axinferencenet_example.cpp)

## Reading frames from the video source

First, we define a `Frame` object that we use to store information about the current frame. If OpenCV is used to read the frame then be aware that OpenCV always returns frames in BGR format, which is a bit inconvenient as we will need to convert it to RGB (see the comment below).  In this example however we will use an ffmpeg decoder (`Ax::FFMpegVideoDecoder`)
which we can request the frame in RGB format.

We also add an instance of `Ax::MetaMap` to receive the decoded inference results. This class is documented in [AxMetaMap](/docs/reference/pipeline_operators.md#axmetamap).

```cpp
struct Frame {
  cv::Mat rgb;
  Ax::MetaMap meta;
};
```

If we were to use OpenCV to read the video frames, we would here start another thread to read image data and check for end of stream. The function `Ax::video_from_cvmat` creates an `AxVideoInterface` object which contains the frame meta data such as width/height, pixel stride, and color format that notifies `AxInferenceNet` how the image data is formatted, including resolution, color format, and strides between the beginning of each row of pixel data.

This function would then push our `Frame` object, along with the `video` information structure, and a reference to the `meta`. If using multiple streams we would also include a stream_id here.

Our `std::shared_ptr<Frame>` object is implicitly converted to the opaque `std::shared_ptr<void>` which allows us to pass ownership of the `Frame` to `AxInferenceNet` without `AxInferenceNet` needing to be aware of the type.

```cpp
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
    auto video = Ax::video_from_cvmat(frame->bgr, AxVideoFormat::BGR);
    net.push_new_frame(frame, video, frame->meta);
  }
}
```

However in this example we will instead use an FFMpeg based decoder, which accepts a callback for when a frame is ready. We will define this callback later in main.

## Rendering the results of inference

Next, we define a function to render the results onto an OpenCV Mat array. This shows how to iterate over the inference results, get the bounding box, look up the class ID in the labels vector, and format the score. We then use that information to output an appropriate label and draw a rectangle.

```cpp
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
```

In the file `AxOpenCVRender.hpp` there are functions provided to render most types of Axelera Meta data, as well as an interface to display the result on the window system or terminal.

## Setting up the inference loop

Here we parse the command line arguments:

```cpp
int
main(int argc, char **argv)
{
  const auto [model_properties, labels, input] = parse_args(argc, argv);

```

Next, we use a utility from `AxStreamerUtils.hpp` called `Ax::BlockingQueue`. This is similar to the Python `queue.Queue` class and provides an easy way to communicate from one thread to another. In this case, we will use it to pass back the inference result from the `frame_completed` callback to the main loop.

We need to pass a `frame_completed` callback to AxInferenceNet. This callback can inspect the result and handle it in any way desired, but in most cases the callback needs to check for the `end_of_input` signal and stop the inference or pass the result on to another AxInferenceNet or push it to a queue.

We use the `forward_to` adapter to create a callback that pushes the result onto the `ready` queue.

We are now ready to create the `AxInferenceNet` object. We convert the `.axnet` file into an
`Ax::InferenceNetProperties` object and call `Ax::create_inference_net`, passing it the properties,
a logger, and the `frame_completed` callback.

```cpp

  // We use BlockingQueue to communicate between the frame_completed callback and the main loop
  Ax::BlockingQueue<std::shared_ptr<Frame>> ready;
  auto props = Ax::read_inferencenet_properties(model_properties, logger);
  auto net = Ax::create_inference_net(props, logger, Ax::forward_to(ready));

```

We then create a callback for the decoder to call whenever a new frame is ready.  We take the input frame, and construct our own `Frame` object, and push it to the axinferencenet work queue.  We show here how you can use the FFMpeg or OpenCV decoded classes.

```cpp
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
```

Now, create a display to show the results. 

```cpp
  auto display = Ax::OpenCV::create_display("AxInferenceNet Demo");
  Ax::OpenCV::RenderOptions render_options;
```

## The main inference loop

We handle the results of the inference in the main thread. This is not a requirement of `AxInferenceNet` but rather one of OpenCV. It is always best to interact with the OS GUI in the main thread of an application.

To obtain a frame, we call `wait_one` on the `ready` queue. This will return an empty `std::shared_ptr` if we called `stop` as a result of an `end_of_input` signal. If so, then we are all done, and we exit the inference loop.

Otherwise, we have a valid frame result. We can access the image and inference results to render the results. OpenCV rendering is easy to use, which makes it a good API for this example. But it is relatively slow so it only displays frames at 10fps. This can be configured in RenderOptions.

Note that the Display class accepts a meta map, and if we pass `frame->meta` then built-in rendering of the inference results (such as bounding boxes, keypoints, labels, and segmentations) will be performed.

```cpp
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
```

## Cleanup

In order to perform a well-behaved shutdown, we first stop `AxInferenceNet`, then join our reader thread.

```cpp
  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net->stop();
  reader.join();
```

## Using AxInferenceNet for Raw Tensor Output and Custom Postprocessing

In addition to the end-to-end pipeline example, the SDK provides `axinferencenet_tensor.cpp`, which demonstrates how to use AxInferenceNet to obtain raw tensor outputs from your model and implement your own postprocessing logic in C++, and another example `axinferencenet_cascaded.cpp` which demonstrates how to chain multiple AxInferenceNet instances together, and how to use the built-in OpenCV rendering.

### How to use raw tensor output (user workflow)

To extract the raw tensor data and shape from the meta map, use the following workflow:

```cpp
#include <AxMetaRawTensor.hpp>
// ...
auto& tensor_wrapper = dynamic_cast<AxMetaRawTensor&>(*frame->meta["detections"]);
if (tensor_wrapper.get_tensor() && tensor_wrapper.get_tensor()->num_tensors() > 0) {
    std::vector<int64_t> dims = get_tensor_shape(tensor_wrapper);
    const float* data = get_tensor_data<float>(tensor_wrapper);
    auto detections = postprocess_yolov8(
        data, dims, 0.25f, 0.45f, 100,
        frame->bgr.cols, frame->bgr.rows,
        model_input_width, model_input_height,
        letterboxed
    );
    render_detections(detections, frame->bgr, labels);
}
```

- The data pointer type (`float*`, `int8_t*`, etc.) depends on your model output. Use the correct type for your model.
- The shape is always available as a vector of `int64_t`.
- These helpers make it easy to extract the tensor data and shape for any tensor index.

### Why and When to Use Raw Tensor Output?

In the standard pipeline (e.g., yolov8-coco), postprocessing is handled by the pipeline and you receive ready-to-use detection objects. With raw tensor output, you have full control and responsibility for postprocessing, which is ideal for custom models or integration with your own analytics pipeline.

- **Support for custom models:**  If you have a model that is not supported in the Axelera model zoo, you may not have a corresponding prebuilt decoder or meta type. Using raw tensor output allows you to integrate and use your own postprocessing logic, as long as you can successfully compile your model.
- **No need to understand Axelera's metadata:**  You only need to know how to extract the tensor using `AxMetaRawTensor`. You do not need to learn or depend on our object detection meta types.
- **Easy integration:**  You can define your own detection/object structures and postprocessing, making it easier to plug AxInferenceNet into your existing pipeline or codebase.
- **Flexible and educational:**  You can experiment with different postprocessing algorithms, visualize intermediate results, and use your own object types and pipeline logic.

### Pipeline Configuration: Standard vs. Raw Tensor Output

The type of output you receive from AxInferenceNet depends on how your model YAML (and resulting `.axnet`) is configured:

| Pipeline Type      | YAML/.axnet Postprocess Steps                | Output Meta Type           | User Responsibility         |
|--------------------|----------------------------------------------|----------------------------|----------------------------|
| Standard           | `libdecode_yolov8`, `libinplace_nms`         | `AxMetaObjDetection`       | None (ready-to-use output) |
| Raw Tensor Output  | `libtransform_*`, `libdecode_to_raw_tensor`  | `AxMetaRawTensor`   | All postprocessing         |

**Standard pipeline example:**
```yaml
postprocess:
  - decodeyolo:
      max_nms_boxes: 30000
      conf_threshold: 0.25
      nms_iou_threshold: 0.45
      nms_class_agnostic: True
      nms_top_k: 300
```
Produces `.axnet` lines like:
```
postprocess0_lib=libdecode_yolov8.so
postprocess1_lib=libinplace_nms.so
```

**Raw tensor output example:**

To get raw tensor output that matches your original ONNX or PyTorch model, use this configuration:

```yaml
inference:
  handle_all: True  # Default: produces output matching your original model  
postprocess:
  - get-raw-tensor:
```

This gives you the same tensor output as your original ONNX or PyTorch model - fully dequantized, with correct dimensions and layout.

**Performance optimization (advanced):**

For improved performance, especially with smaller models, you can handle processing steps yourself:

**Option 1 - Handle nothing (maximum performance potential):**
```yaml
inference:
  handle_all: False  # You handle all processing steps yourself
postprocess:
  - get-raw-tensor:
```

**Option 2 - Fine-grained control:**
```yaml
inference:
  # Don't set handle_all, configure individual steps
  handle_dequantization_and_depadding: True
  handle_transpose: False
  handle_postamble: False
postprocess:
  - get-raw-tensor:
```

With `handle_all: False`, you take responsibility for all processing steps yourself, but this can enable additional optimizations in the pipeline.

With the standard pipeline, you receive ready-to-use detection objects in `meta["detections"]` (an `AxMetaObjDetection`). With the raw tensor output, you receive a tensor in `meta["detections"]` (an `AxMetaRawTensor`) and must implement all postprocessing (decoding, NMS, etc.) yourself.

### When to use raw tensor output

Raw tensor output is ideal for:
- Custom models not supported in the Axelera model zoo
- Experimental postprocessing algorithms
- Integration with existing analytics pipelines

For maximum performance in production applications, consider building a standard decoder as done for models in the Axelera model zoo.

### Performance note

The raw tensor output path is intended for flexibility, experimentation, and integration - not for maximum speed. This path is not as optimized as the default pipeline, especially for large batch sizes or high-throughput. If you want maximum performance, we encourage you to build a standard decoder (as done for models in the Axelera model zoo), which allows the pipeline to use fully optimized postprocessing. We plan to improve this path by optimizing how we perform postprocessing before providing the tensor output in future releases.

See the full source in [`axinferencenet_tensor.cpp`](/examples/axinferencenet/axinferencenet_tensor.cpp).

## Next Steps
- **Build the example**: Run `make examples` to compile
- **Implement cascaded models**: [axinferencenet_cascaded.cpp](../../examples/axinferencenet/axinferencenet_cascaded.cpp)
- **Access raw tensors**: [axinferencenet_tensor.cpp](../../examples/axinferencenet/axinferencenet_tensor.cpp)
- **Deploy in production**: Use C++ API for performance-critical applications


## Related Documentation
**Tutorials:**
- [Application Integration](application.md) - Python equivalent (start here if new to SDK)
- [Cascaded Models](cascaded_model.md) - Multi-model pipeline concepts
- [Video Sources](video_sources.md) - Input source configuration applies to C++ too

**References:**
- [AxInferenceNet C++ API](../reference/axinferencenet.md) - Complete class reference and method signatures
- [Pipeline Operators](../reference/pipeline_operators.md) - Understanding YAML pipelines for C++

**Examples:**
- [axinferencenet_example.cpp](../../examples/axinferencenet/axinferencenet_example.cpp) - This tutorial's example
- [axinferencenet_cascaded.cpp](../../examples/axinferencenet/axinferencenet_cascaded.cpp) - Multi-model C++ pipeline
- [axinferencenet_tensor.cpp](../../examples/axinferencenet/axinferencenet_tensor.cpp) - Raw tensor access in C++

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
