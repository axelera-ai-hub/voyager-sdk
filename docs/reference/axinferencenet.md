![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# AxInferenceNet C++ Reference

- [AxInferenceNet C++ Reference](#axinferencenet-c-reference)
  - [Overview and Key Benefits](#overview-and-key-benefits)
    - [class Ax::InferenceNet](#class-axinferencenet)
    - [push\_new\_frame](#push_new_frame)
    - [end\_of\_input](#end_of_input)
    - [stop](#stop)
    - [InferenceDoneCallback](#inferencedonecallback)
    - [Ax::LatencyCallback](#axlatencycallback)
  - [Ax::InferenceNetProperties](#axinferencenetproperties)
  - [properties\_from\_string](#properties_from_string)
    - [Creating Ax::InferenceNet](#creating-axinferencenet)


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

The GStreamer elements that Axelera AI provides are documented in [Axelera GStreamer Pipeline Operators](/docs/reference/pipeline_operators.md).

![The AxInferenceNet Pipeline](/docs/images/axinferencenet_850x500_01.png)



### class Ax::InferenceNet

AxInferenceNet is the engine of the pipeline. It is an asynchronous interface into which the user pushes input frames, and when the frame has exited the pipeline, a callback is issued with the result.

### push_new_frame
```cpp
void push_new_frame(std::shared_ptr<void> &&buffer,
      const AxVideoInterface &video, MetaMap &metamap, int stream_id = 0);
```

This enqueues a frame into the pipeline. When this frame is complete, the [InferenceDoneCallback](#inferencedonecallback) will be called.

`buffer` this is an opaque handle that is not used directly by AxInferenceNet, but ownership is retained until the frame has exited the pipeline. The user can use this to ensure that any image buffers or associated data of the frame are not deallocated prematurely. It is advised that the user creates their own type, see the worked [AxInferenceNet C++ Example](/docs/tutorials/axinferencenet.md).

`video` this indicates the image data format. The class is documented in [AxDataInterface](/docs/reference/pipeline_operators.md#axdatainterface).

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
  int stream_id;
  uint64_t frame_id;
  std::shared_ptr<void> buffer_handle{};
  AxVideoInterface video = {};
};

using InferenceDoneCallback = std::function<void(CompletedFrame &)>;
```

This is a callback that is passed to `Ax::InferenceNet` at construction and is called whenever a frame exits the pipeline, or when [`end_of_input()`](#end_of_input) has been called and the pipeline is complete. It is required to be a valid function, and it takes a single parameter `CompletedFrame`.

The members of `CompletedFrame` are:

`end_of_input` when true, the rest of the `CompletedFrame` should be ignored. If there are further AxInferenceNets, this should be used to signal to them that they should also end by calling their [`end_of_input()`](#end_of_input). If an output queue is used from the callback, this should be used to indicate a terminating condition. The `InferenceDoneCallback` will not be called again after this special marker has been passed.

`stream_id` this is the ID passed to [`push_new_frame`](#push_new_frame).

`frame_id` this is a sequentially per-stream ID.

`buffer_handle` this is the handle passed to [`push_new_frame`](#push_new_frame). If the user wishes to gain ownership, they should move out of it thus:

```cpp
auto user_frame = std::static_pointer_cast<UserFrame>(std::move(completed.buffer_handle))
```

### Ax::LatencyCallback
```cpp
using LatencyCallback = std::function<void(const std::string &opname, uint64_t throughput_ns, uint64_t latency_ns)>;
```

This callback is passed to `Ax::InferenceNet` at construction and is called whenever an operator in the pipeline (e.g., normalization or inference) completes. It can be a default constructed.

The parameters are:

`opname` a unique name indicating the operator. For example, "inference".

`throughput_ns` the throughput for this operator in nanoseconds.

`latency_ns` the latency for this operator in nanoseconds.

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
  std::string options;
  std::string meta;
  std::string devices;
  Ax::OperatorProperties preproc[Ax::MAX_OPERATORS];
  Ax::OperatorProperties postproc[Ax::MAX_OPERATORS];
};
```
`model` path to the Axelera model. If this is compiled using the pipeline builder it will be located at `build/networkname/modelname/1/model.json`

`double_buffer` double buffering is an optimization to increase throughput by overlaying Host to Device data transfer with inference execution. It typically improves throughput from 10-40% depending on the model. However, enabling it has a drawback of increasing latency.

`dmabuf_inputs` enabling this causes the use of Linux DMA Buffers for transferring data to the Metis accelerator. This is ignored on systems that do not support DMA Bufs. This should always be enabled.

`dmabuf_outputs` enabling this causes the use of Linux DMA Buffers for transferring data *from* the Metis accelerator. This is ignored on systems that do not support DMA Buffers. Enabling this causes an increase in throughput but with a trade-off in latency. It should usually be enabled unless the application is very sensitive to latency.

`skip_stride` / `skip_count` at present these members are ignored.

`num_children` how many instances of the model should be instantiated per device. Most Axelera models perform best when compiled with a batch size of 1. In this case `num_children` is equal to the number of AIPU cores the model will execute on. Increasing this value up to 4 will increase throughput. But at present it will result in increased latency. In most single model applications this should be passed as 4 unless the application is very sensitive to latency. 0 is considered as equivalent to 1.

`options` these options are not currently documented, it should be left empty.

`meta` used in secondary models, it indicates the name of the AxMetaBase class which contains the ROI information for performing secondary (or cascaded) inference.

`device` used on hosts that have multiple Metis devices. it should be a comma separated list of device names as obtained from [`axdevice`](/docs/reference/axdevice.md). If empty then the first available device will be used.


`preproc` / `postproc` a list of properties defining preprocessing and postprocessing operators.

**Note:** The maximum length of each list is 8. This limit is arbitrary and exists for legacy reasons, it will be relaxed in the future.

Preprocessing operators are executed on a frame *before* inference. They typically involve operations such as color conversion, scaling, normalization, padding, quantization, and sometimes some data restructuring that is more efficiently performed on the host processor than the Metis. Generally, it is best to fuse as many of these operations together as possible, for example performing normalization and quantization at the same time is faster than having two separate operators. These operators typically consist of one more AxInPlace or AxTransform operators.

Postprocessing operators are executed on a frame *after* inference. They typically involve operations such as dequantization, depadding, data restructuring that is more efficiently performed on the host processor than the Metis. Also operations such as NMS, and converting the resulting inference data into AxMetaBase objects for easy access in business logic.

There are many preprocessing and postprocessing operators provided and they are documented in [Axelera GStreamer Pipeline Operators](/docs/reference/pipeline_operators.md). Users can also write their own, using the existing operators as inspiration.

Each preproc or postproc operator accepts the following parameters:

`lib` name of the plugin, for example `"libtransform_resize.so"`. The plugin must be loadable via LD_LIBRARY_PATH.

`options` a semi-colon separated list of colon separated key:value pairs. For example `"width:640;height:640;padding:114;letterbox:1"`.

`mode` determines whether the operator needs access, at present this parameter is ignored.


## properties_from_string

Determining the correct options for Ax::InferenceNet requires careful design; consideration for available accelerator hardware, such as OpenCL, as well as transferring key pieces of model information about quantization and normalization to the properties struct.

For this reason there is a utility to create the them. When inference.py runs it emits a file in the `build/networkname/model.axnet` directory which can be directly parsed by `Ax::properties_from_string`.

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

This can be read in and, then passed to :

```cpp
InferenceNetProperties properties_from_string(const std::string &s, Ax::Logger &logger);
```


### Creating Ax::InferenceNet

```cpp
std::unique_ptr<InferenceNet> create_inference_net(
    const InferenceNetProperties &properties, Ax::Logger &logger,
    InferenceDoneCallback done_callback, LatencyCallback latency_callback = {});
```

Create an instance of Ax::InferenceNet.

`properties` This is used to configure the model, pre and post proc operators,.

`logger` An Ax::Logger instance that collects and forward logging messages.

`done_callback` this is the callback which is called whenever a frame exits the pipeline, See [InferenceDoneCallback](#inferencedonecallback)

`latency_callback` this is the callback whenever an operator completes. It is used to diagnose latency problems., See [LatencyCallback](#axlatencycallback)
