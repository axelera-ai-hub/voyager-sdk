![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# AxInferenceNet C++ Integration Tutorial

- [AxInferenceNet C++ Integration Tutorial](#axinferencenet-c-integration-tutorial)
  - [Preparing the example](#preparing-the-example)
  - [Reading frames from the video source](#reading-frames-from-the-video-source)
  - [Rendering the results of inference](#rendering-the-results-of-inference)
  - [Setting up the inference loop](#setting-up-the-inference-loop)
  - [The main inference loop](#the-main-inference-loop)
  - [Cleanup](#cleanup)

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
(venv) $ ./download_prebuilt.py yolov8s-coco
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

The <model>.axnet file is a file describing the model, preprocessing, and
postprocessing steps of the pipeline.  In the future this will be created
by deploy.py when deploying a pipeline, but for now it is necessary to run
the gstreamer pipeline.  The file can also be created by hand or you can
manually pass the parameters to AxInferenceNet.

The first step is to compile or download a prebuilt model, here we will show
downloading a prebuilt model:

  ./download_prebuilt.py yolov8s-coco-onnx

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

First, we define a `Frame` object that we use to store information about the current frame. OpenCV always returns frames in BGR format, which is a bit inconvenient as we will need to convert it to RGBA (see the comment below).

But we will need the same BGR image to perform rendering, so we add it to the `Frame` object to avoid another color conversion later in the application. We also keep the RGBA image as we need to make sure that during the pipeline execution the image data is not deallocated.

We also add an `Ax::MetaMap` class to receive the decoded inference results. This class is documented in [AxMetaMap](/docs/reference/pipeline_operators.md#axmetamap).

```cpp
struct Frame {
  cv::Mat bgr;
  cv::Mat rgba;
  Ax::MetaMap meta;
};
```

Next, we define a function that we will start in another thread to read image data, perform the BGR to RGBA color conversion, configure the `AxVideoInterface` structure that notifies `AxInferenceNet` how the image data is formatted, including resolution, color format, and strides between the beginning of each row of pixel data.

Finally, we push our `Frame` object, along with the `video` information structure, and a reference to the `meta`. If using multiple stream we would also include a stream_id here.

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
    // Convert the frame to RGBA format, retaining the original image in the
    // frame in case we want to render it with opencv later. Ideally, this would
    // be done in the preprocessing stage of the AxInferenceNet, but at the
    // moment the resize_cl operator does not support BGR or RGB input. This
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
```

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

## Setting up the inference loop

Here we parse the command line arguments and open the video file using `cv:VideoCapture`.

```cpp
int
main(int argc, char **argv)
{
  const auto [model_properties, labels, input] = parse_args(argc, argv);
  cv::VideoCapture cap(input);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video source: " << input << std::endl;
    return 1;
  }
```

Next, we use a utility from `AxStreamerUtils.hpp` called `Ax::BlockingQueue`. This is similar to the Python `queue.Queue` class and provides an easy way to communicate from one thread to another. In this case, we will use it to pass back the inference result from the `frame_completed` callback to the main loop.

We then define the `frame_completed` callback. We use a lambda, but it can be any callable in a `std::function`.

In the callback, we check for the `end_of_input` signal, and if found, we push a `stop` event onto the `Ax::BlockingQueue ready` queue. If this callback is not the result of an `end_of_input` signal, then we cast the `buffer_handle` back to a `std::shared_ptr<Frame>` and push that onto our `ready` queue so that the main loop can collect it when it is ready to.

```cpp
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
```

We are now ready to create the `AxInferenceNet` object. We convert the `.axnet` file into an
`Ax::InferenceNetProperties` object and call `Ax::create_inference_net`, passing it the properties,
a logger, and the `frame_completed` callback.

We then start the reader_thread and initialize the OpenCV window.

```cpp
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
```

## The main inference loop

We handle the results of the inference in the main thread. This is not a requirement of `AxInferenceNet` but rather one of OpenCV. It is always best to interact with the OS GUI in the main thread of an application.

To obtain a frame, we call `wait_one` on the `ready` queue. This will return an empty `std::shared_ptr` if we called `stop` as a result of an `end_of_input` signal. If so, then we are all done, and we exit the inference loop.

Otherwise, we have a valid frame result. We can access our BGR image to render the results. OpenCV rendering is easy to use, which makes it a good API for this example. But it is relatively slow, so we only render every 10 frames and show that in the OpenCV window.

```cpp
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
```

## Cleanup

In order to perform a well-behaved shutdown, we first stop `AxInferenceNet`, then join our reader thread. And finally, destroy the OpenCV window.

```cpp
  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net->stop();
  reader.join();

  cv::destroyWindow(wndname);
```
