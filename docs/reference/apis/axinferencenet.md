# AxInferenceNet C++ Reference

**Note:** This interface is still under development and is subject to change. The core functionality will remain the same, but interfaces and type names may change.

## Overview and Key Benefits

The Voyager SDK provides a rich set of APIs for C++ computer vision inference, allowing the user to choose a highly efficient inference pipeline that can handle multiple streams and multiple models. This pipeline provides the following benefits:

* A plugin architecture allowing customization of the pipeline.

* Multiple streams can be handled with ease, each frame passed in has an associated stream ID that can be used to demultiplex the output stream.

* The input streams can be of different image formats and sizes, the pipeline will handle the conversion and scaling.

* Additional preprocessing such as cropping, letterboxing, normalization, and quantization is performed efficiently.

* Plugins to support postprocess of the inference such as dequantization and Non-Maximal Suppression are provided.

* Plugins to support performing those operations efficiently on the host are provided for all model-zoo networks, and these can be used as a basis for writing your own decoding plugins.

* Chaining of inference operations such that the output of one network (say detecting objects) can be used to invoke secondary models that perform inference on the region of interest by using the original high-resolution input frame as input.

* Plugins that convert the dequantized tensor output into easy-to-use C++ objects for interpreting the inference result.

All of this is done in a pipelined manner to maximize throughput while minimizing latency.

The user is able to choose to use this pipeline with either:

* A pure C++ pipeline, using the AxInferenceNet interface where the user is responsible for forwarding video frames in a decoded form.

* A GStreamer-based pipeline using the axinferencenet GStreamer plugin element. Using GStreamer gives the user the ability to use the many plugins available in GStreamer, including source elements such as RTSP and USB sources.

The plugins that make up the pipeline are not GStreamer-specific, and the same plugins can be used in a GStreamer or C++ pipeline.

The GStreamer elements that Axelera AI provides are documented in [Axelera GStreamer Operators](../pipeline/gst-operators.md).

### class Ax::InferenceNet

AxInferenceNet is the engine of the pipeline. It is an asynchronous interface into which the user pushes input frames, and when the frame has exited the pipeline, a callback is issued with the result.

### push_new_frame
```cpp
void push_new_frame(std::shared_ptr<void> &&buffer,
      const AxVideoInterface &video, MetaMap &metamap, int stream_id = 0);
```

This enqueues a frame into the pipeline. When this frame is complete, the [InferenceDoneCallback](#inferencedonecallback) will be called.

`buffer` this is an opaque handle that is not used directly by AxInferenceNet, but ownership is retained until the frame has exited the pipeline. The user can use this to ensure that any image buffers or associated data of the frame are not deallocated prematurely. It is advised that the user creates their own type, see the worked [AxInferenceNet C++ Example](../../tutorials/examples/axinferencenet-basic.md).

`video` this indicates the image data format.

`metamap` this parameter is a reference to a mutable map to receive the results of decoding the inference output.

`stream_id` it is up to the user whether or how to use this field, it is only used internally when calculating the latency of operators in the pipeline. See [LatencyCallback](#axlatencycallback).

### end_of_input

```cpp
void end_of_input();
```

Call this to signal that the end of input has been reached, remaining frames will be flushed. After end_of_input, no new frames should be pushed (this restriction will be relaxed in the future).

### stop

```cpp
void stop();
```

Call this to stop the inference pipeline, join all threads, and release resources.

### InferenceDoneCallback

```cpp
struct CompletedFrame {
  bool end_of_input = false;
  bool inference_failed = false;
  int stream_id;
  uint64_t frame_id;
  std::shared_ptr<void> buffer_handle{};
  AxVideoInterface video = {};
  MetaMap *meta_map = nullptr;
};

using InferenceDoneCallback = std::function<void(CompletedFrame &)>;
```

This is a callback that is passed to `Ax::InferenceNet` at construction and is called whenever a frame exits the pipeline, or when [`end_of_input()`](#end_of_input) has been called and the pipeline is complete. It is required to be a valid function, and it takes a single parameter `CompletedFrame`.

The members of `CompletedFrame` are:

`end_of_input` when true, the rest of the `CompletedFrame` should be ignored. If there are further AxInferenceNets, this should be used to signal to them that they should also end by calling their [`end_of_input()`](#end_of_input). If an output queue is used from the callback, this should be used to indicate a terminating condition. The `InferenceDoneCallback` will not be called again after this special marker has been passed.

`inference_failed` when true, the inference call for this frame failed, for example with a device or queue error. `video` and the metadata are still valid, but the tensor data is stale: it is whatever a previous frame left in that pool slot, not a result for this frame. Check this before treating the frame as a normal result.

`stream_id` this is the ID passed to [`push_new_frame`](#push_new_frame).

`frame_id` this is a sequentially per-stream ID.

`buffer_handle` this is the handle passed to [`push_new_frame`](#push_new_frame). If the user wishes to gain ownership, they should move out of it thus:

```cpp
auto user_frame = std::static_pointer_cast<UserFrame>(std::move(completed.buffer_handle))
```

`meta_map` the metadata map passed to [`push_new_frame`](#push_new_frame), carrying the decoded inference results.

### Ax::LatencyCallback
```cpp
using LatencyCallback = std::function<void(const std::string &opname, uint64_t throughput_ns, uint64_t latency_ns)>;
```

This callback is passed to `Ax::InferenceNet` at construction and is called whenever an operator in the pipeline (e.g., normalization or inference) completes. It can be a default constructed.

The parameters are:

`opname` a unique name indicating the operator. For example, "inference".

`throughput_ns` the throughput for this operator in nanoseconds.

`latency_ns` the latency for this operator in nanoseconds.

## Asynchronous execution

AxInferenceNet chooses how to drive the accelerator from the model's batch size, with no property to set.

A model compiled with a batch size of 1, which is how most Axelera models perform best, runs on the asynchronous executor. That executor submits without blocking and receives completions on the runtime's own threads, so several frames are in flight on one instance at once and the pipeline is always in its low-latency mode. `max_inflight` and `max_pending` bound that concurrency, and `double_buffer` is ignored because the executor does not need it.

A batched model runs on the blocking executor, one batch per call on a dedicated worker thread per instance. `double_buffer` applies there.

Frames still reach [InferenceDoneCallback](#inferencedonecallback) in order in both cases. Because a failed frame is reported rather than dropped, a callback should check [`inference_failed`](#inferencedonecallback) before reading the tensor data.

## Ax::InferenceNetProperties

`Ax::InferenceNet` is a highly configurable pipeline, and this configuration is achieved using a properties structure that is passed at construction time:

```cpp
constexpr int Ax::MAX_OPERATORS = 8

struct Ax::OperatorProperties {
  std::string lib;
  std::string options;
  std::string mode;
  std::string batch;
};

struct Ax::InferenceNetProperties {
  std::string model;
  bool double_buffer{ false };
  bool dmabuf_inputs{ false };
  bool dmabuf_outputs{ false };
  int skip_stride{ 1 };
  int skip_count{ 0 };
  int num_children{ 0 };
  float margin{0};
  std::string options;
  std::string meta;
  std::string devices;
  std::string which_cl;
  int max_buffers{ 0 };
  int max_inflight{ 4 };
  int max_pending{ 8 };
  Ax::OperatorProperties preproc[Ax::MAX_OPERATORS];
  Ax::OperatorProperties postproc[Ax::MAX_OPERATORS];
};
```
`model` path to the Axelera model. If this is compiled using the pipeline builder it will be located at `build/networkname/modelname/1/model.json`

`double_buffer` double buffering is an optimization to increase throughput by overlaying host to device data transfer with inference execution. It typically improves throughput from 10-40% depending on the model. However, enabling it has a drawback of increasing latency. It applies to batched models only: a batch-size-1 model runs on the asynchronous executor described in [Asynchronous execution](#asynchronous-execution), which keeps several frames in flight per instance instead, and this property is ignored.

`dmabuf_inputs` enabling this causes the use of Linux DMA Buffers for transferring data to the Metis accelerator. This is ignored on systems that do not support DMA Bufs. This should always be enabled.

`dmabuf_outputs` enabling this causes the use of Linux DMA Buffers for transferring data *from* the Metis accelerator. This is ignored on systems that do not support DMA Buffers. Enabling this causes an increase in throughput but with a trade-off in latency. It should usually be enabled unless the application is very sensitive to latency.

`skip_stride` / `skip_count` at present these members are ignored.

`num_children` how many instances of the model should be instantiated per device. Most Axelera models perform best when compiled with a batch size of 1. In this case `num_children` is equal to the number of AIPU cores the model will execute on. Increasing this value up to 4 will increase throughput. But at present it will result in increased latency. In most single model applications this should be passed as 4 unless the application is very sensitive to latency. 0 is considered as equivalent to 1.

`max_inflight` / `max_pending` these tune the asynchronous executor described in [Asynchronous execution](#asynchronous-execution). `max_inflight` is how many frames may be in flight on one instance; `max_pending` is how many may be submitted but not yet completed. The defaults suit most pipelines. Both are ignored for batched models, and neither is exposed as a GStreamer element property.

`which_cl` / `max_buffers` these select the OpenCL device and cap the buffer pool. They are not currently documented and should be left at their defaults.

`options` these options are not currently documented, it should be left empty.

`meta` used in secondary models, it indicates the name of the AxMetaBase class which contains the ROI information for performing secondary (or cascaded) inference.

`device` used on hosts that have multiple Metis devices. it should be a comma separated list of device names as obtained from [`axdevice`](../tools/axdevice.md). If empty then the first available device will be used.

`preproc` / `postproc` a list of properties defining preprocessing and postprocessing operators.

**Note:** The maximum length of each list is 8. This limit is arbitrary and exists for legacy reasons, it will be relaxed in the future.

Preprocessing operators are executed on a frame *before* inference. They typically involve operations such as color conversion, scaling, normalization, padding, quantization, and sometimes some data restructuring that is more efficiently performed on the host processor than the Metis. Generally, it is best to fuse as many of these operations together as possible, for example performing normalization and quantization at the same time is faster than having two separate operators. These operators typically consist of one more AxInPlace or AxTransform operators.

Postprocessing operators are executed on a frame *after* inference. They typically involve operations such as dequantization, depadding, data restructuring that is more efficiently performed on the host processor than the Metis. Also operations such as NMS, and converting the resulting inference data into AxMetaBase objects for easy access in business logic.

There are many preprocessing and postprocessing operators provided and they are documented in [Axelera GStreamer Operators](../pipeline/gst-operators.md). Users can also write their own, using the existing operators as inspiration.

Each preproc or postproc operator accepts the following parameters:

`lib` name of the plugin, for example `"libtransform_resize.so"`. The plugin must be loadable via LD_LIBRARY_PATH.

`options` a semi-colon separated list of colon separated key:value pairs. For example `"width:640;height:640;padding:114;letterbox:1"`.

`mode` relevant only to AxInPlace plugins, and now deprecated. It is ignored.

`batch` ignored.

## read_inferencenet_properties

Determining the correct options for Ax::InferenceNet requires careful design; consideration for available accelerator hardware, such as OpenCL, as well as transferring key pieces of model information about quantization and normalization to the properties struct.

For this reason there is a utility to create them. When inference.py runs it emits a file at `build/<network>/<network>.axnet` which can be read directly by `Ax::read_inferencenet_properties`.

The format of this file for YOLOV8 might look for example:

```
model=build/ces2025-ls/yolov8s-fruit/1/model.json
devices=metis-0:3:0
double_buffer=True
dmabuf_inputs=True
dmabuf_outputs=True
num_children=0
preprocess0_lib=libtransform_resize_cl.so
preprocess0_options=width:640;height:640;padding:114;letterbox:1;scale_up:1;to_tensor:1;mean:0.,0.,0.;std:1.,1.,1.;quant_scale:0.003921568859368563;quant_zeropoint:-128.0
preprocess1_lib=libtransform_padding.so
preprocess1_options=padding:0,0,1,1,1,15,0,0;fill:0
preprocess1_batch=1
postprocess0_lib=libdecode_yolov8.so
postprocess0_options=meta_key:object_detections;classes:80;confidence_threshold:0.25;scales:0.07426197826862335,0.06717678159475327,0.06881681829690933,0.09170340746641159,0.15665404498577118,0.17088785767555237;padding:0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,48|0,0,0,0,0,0,0,48|0,0,0,0,0,0,0,48;zero_points:-68,-58,-45,146,103,110;topk:30000;multiclass:0;model_width:640;model_height:640;scale_up:1;letterbox:1;label_filter:46,47,49
postprocess0_mode=read
postprocess1_lib=libinplace_nms.so
postprocess1_options=meta_key:object_detections;max_boxes:300;nms_threshold:0.45;class_agnostic:1;location:CPU
```

Read it with either overload, then pass the result to
[`create_inference_net`](#creating-axinferencenet):

```cpp
InferenceNetProperties read_inferencenet_properties(std::istream &s, Ax::Logger &logger);
InferenceNetProperties read_inferencenet_properties(const std::string &path, Ax::Logger &logger);
```

### Creating Ax::InferenceNet

```cpp
std::unique_ptr<InferenceNet> create_inference_net(
    const InferenceNetProperties &properties, Ax::Logger &logger,
    InferenceDoneCallback done_callback);

std::unique_ptr<InferenceNet> create_inference_net(
    const InferenceNetProperties &properties, Ax::Logger &logger,
    InferenceDoneCallback done_callback, LatencyCallback latency_callback);
```

Create an instance of Ax::InferenceNet.

`properties` This is used to configure the model, pre and post proc operators,.

`logger` An Ax::Logger instance that collects and forward logging messages.

`done_callback` this is the callback which is called whenever a frame exits the pipeline, See [InferenceDoneCallback](#inferencedonecallback)

`latency_callback` this is the callback whenever an operator completes. It is used to diagnose latency problems., See [LatencyCallback](#axlatencycallback)
