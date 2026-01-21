![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Benchmarking and Performance Evaluation

## Contents
- [Benchmarking and Performance Evaluation](#benchmarking-and-performance-evaluation)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Compare Pipeline Accuracy](#compare-pipeline-accuracy)
  - [Optimize and Measure Pipeline Performance](#optimize-and-measure-pipeline-performance)
    - [Determining the Bottleneck](#determining-the-bottleneck)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites
- Complete [Quick Start Guide](quick_start_guide.md) and [Application Integration](application.md)
- SDK installed and virtual environment activated
- Hardware connected and working inference pipeline
- Understanding of FPS, latency, and accuracy metrics

## Level
**Intermediate** - Requires understanding of performance metrics and dataset preparation

## Overview
In this section, we'll discuss in-depth the different ways you can evaluate performance and accuracy
of each model with Metis. You can compare our performance to competitors, and see how our
state-of-the-art quantization minimizes accuracy loss when compared to the original FP32 models.

**Examples:**
- All application examples can be benchmarked using techniques from this guide

## Compare Pipeline Accuracy
The Voyager SDK offers three different modes for running each model and its associated pipeline.
This is specified with the `--pipe` argument to `inference.py`, and there are three options: `gst`,
`torch-aipu`, and `torch`. The default setting that you've used so far is `gst`, which runs
inference on the Metis AIPU and uses Axelera's GStreamer elements for the non-neural pipeline
stages.

The pipeline accuracy depends on both the accuracy of the quantized model running on the Metis
device **and** how precisely the GStreamer non-neural elements replicate the Python libraries used
to train the original FP32 model. If you run `inference.py` with the argument `--pipe=torch`, the
pipeline including the AI inference stage will be run on CPU, with the original FP32 precision.
This command configures a pipeline in which the original FP32 model and all non-neural elements are
run as Python code on the host. Therefore, to reproduce the accuracy measurements for the original
FP32 models, run the following command:

```bash
./inference.py yolov5s-v7-coco dataset --no-display --pipe=torch
```

The `--pipe=torch-aipu` argument configures all non-neural pipeline elements to execute Python code
on the host, utilizing the original PyTorch libraries used during model training. The core AI
inferencing takes place on the Metis AIPU. Therefore, to make an apples-to-apples comparison of the
accuracy of the quantized model running on the Metis device against the original precision model,
run the following command:

```bash
./inference.py yolov5s-v7-coco dataset --aipu-cores=1 --no-display --pipe=torch-aipu
```

You can then subtract the difference between the FP32 accuracy (from `--pipe=torch`) and this
measurement to obtain the loss due to model quantization. This number is typically very small.

The accuracy measurement obtained when running with `--pipe=torch-aipu` (strictly, the difference
between this and the FP32 reference) is usually the value that you should use when comparing the
accuracy of models running on Metis versus other solutions.

The end-to-end accuracy measurements (default, or with the flag `--pipe=gst`) provide insight into
the accuracy of a complete end-to-end solution comprising the combination of a specific host and
Metis device. This option uses the Axelera GStreamer pipeline, including host hardware acceleration,
and runs the quantized neural network on Metis.


## Optimize and Measure Pipeline Performance

When considering the performance of an optimized vision pipeline, there are two metrics that need
to be considered :

* Latency. This is the time it takes for a single frame to pass through the pipeline.

* Throughput. This is the number of frames that pass through the pipeline in any given second.

The pipeline created by the pipeline builder and executed in `--pipe=gst` mode for execution on
Metis consists of many stages. In brief these are :

* per input stream conversion
  * collection or buffering of compressed data from the source.
  * decoding of that compressed data into individual frames (usually this is a compressed format such as I420).
  * color conversion into the format required by the model (e.g. RGB or BGR).
  * any other image preprocessing such as barrel distortion correction.

* preparation of tensor data from that image for model input
  * image scaling and letterboxing or cropping.
  * image normalization.
  * quantization of the above data into int8 tensor data.
  * padding of the tensor data (Metis has certain alignment requirements on pixel and row data).

* execution on Metis (this may also include a batching operation if the model has been compiled to execute in batches)

* decoding of the output
  * depadding of the output tensor data.
  * dequantization of output tensor data.
  * any postprocessing in the model.
  * decoding of the output tensors into meta data objects.

NOTE: postprocessing is any additional operations on a model that are more efficiently executed
on the host. A typical example is non-maximum suppression.

To determine why a pipeline is performing as it does there are several metrics provided that can be
enabled or disabled by passing arguments to `inference.py`.

```
  --show-host-fps, --no-show-host-fps
                        show host FPS (default on)
  --show-system-fps, --no-show-system-fps
                        show system FPS (default on)
  --show-cpu-usage, --no-show-cpu-usage
                        show CPU usage (default on)
  --show-stream-timing, --no-show-stream-timing
                        show stream timing (latency and jitter) (default off)
```

**Host FPS**
This is the throughput of the inference at the point where the frame is dispatched to the Metis
accelerator. Transfers to and from the device are pipelined, so this gives an indication of the
maximum throughput of the accelerator for this model. If inference is the limiting factor in the
pipeline this will be close to the System FPS.

**System (End-to-End) FPS**
This is the most useful metric as it is the throughput of the pipeline as a whole. This will be
1 / time taken for a frame to complete the slowest pipeline element.

**CPU Usage**
This metric shows the overall CPU usage of the host. Note that this is a percentage of the total
compute available. So if all host cores are occupied all of the time then the usage will be 100%.

**Stream Timing**
This provides information about the latency, which is the time taken for an individual frame to
leave the source element until it arrives in the application code. Jitter is also measured,
this is a measure of the variation in the latency. If high levels of jitter are an issue for
the application then more buffering would be beneficial (see `--rtsp-latency`).

### Determining the Bottleneck

If the Host FPS is significantly higher than the System FPS, that suggests that another element in
the pipeline is the bottleneck. To help determine why we can use the `--show-stats` option to
`inference.py`. When doing so it is usually best to disable one of the optimizations that is
applied to the pipeline, which is double buffering of OpenCL kernels.

```bash
$ AXELERA_USE_CL_DOUBLE_BUFFER=0 ./inference.py yolov5s-v7-coco media/traffic3_720p.mp4 --show-stats --no-display
========================================================================
Element                                         Time(𝜇s)   Effective FPS
========================================================================
qtdemux0                                              14        68,221.1
h264parse0                                            22        43,575.6
capsfilter0                                            8       115,892.2
decodebin-link0                                       18        53,824.9
axtransform-colorconvert0                            464         2,154.9
inference-task0:libtransform_resize_cl_0             441         2,265.7
inference-task0:libtransform_padding_0               434         2,300.8
inference-task0:inference                          1,217           821.5
inference-task0:Inference latency                 28,718             n/a
inference-task0:libdecode_yolov5_0                   251         3,975.7
inference-task0:libinplace_nms_0                      21        45,654.3
inference-task0:Postprocessing latency               642             n/a
inference-task0:Total latency                     37,057             n/a
========================================================================
End-to-end average measurement                                     809.3
========================================================================
```

By inspecting this table, it is possible to quickly identify any performance bottlenecks in an
end-to-end application pipeline. In general, the lowest effective FPS in the table represents the
fastest frame rate at which the entire pipeline can operate, though in practice some pipeline
elements share the same hardware resources and the actual measured end-to-end performance will be
less. However we need to look into this table in some detail to fully understand the information
contained within.

The first 5 elements are GStreamer elements and the timings are measured using the GstTracer
facility, this measures the time between data arriving and leaving an element.

For the elements `qtdmux0`, `h264parse0`, `capsfilter0`, and `decodebin-link0`, these elements do
not operate exclusively with frame data, as they operate on compressed data, so the timings are not
generally very useful.

The next element `axtransform-colorconvert0` is an OpenCL kernel. This element is the reason we
need to disable OpenCL double buffering to get comprehensible numbers, as otherwise the GStreamer 
tracing facility ignores the time it takes for the transfer of the result back from the GPU. This
generally results in an underestimate of the cost of this element, and results in an overestimate
of the cost of the next element (which has to wait for the result to be ready).

With double buffering disabled we can see that the contribution to the pipeline latency is 464𝜇s, 
and this element is executing at 2154 frames per second, so it is not the bottleneck in the
pipeline. This may not be the case for larger input media sizes.

The next rows are all prefixed with `inference-task:`. These are a breakdown of the pipelined
operators implemented in an [AxInferenceNet](/docs/tutorials/axinferencenet.md) element, which includes
the pre-processing (`resize_cl` and `padding`) of the frame, the execution of the inference itself
(`inference`) and any post-processing steps such as `libdecode_yolov5` and `libinplace_nms`. You
can see that none of these are the bottleneck in this case. The performance of these will vary
depending on the complexity of decoding the output tensors, the number of detections made, and
other factors such as the confidence threshold in the NMS operator.

`Inference latency` shows that overall the contribution to the system latency from the
axinferencenet element is 28,718𝜇s.

Note that the reported accuracy of performance-optimized pipelines is usually slightly lower than
pipelines that utilize the CPU for the non-neural elements. This is because hardware accelerators
such as VA-API usually support a limited number of configuration parameters and accuracy is usually
reduced when a pipeline element does not implement precisely the same algorithm that was used
originally during training. For example, the following article explains why accuracy is lost if the
pipeline compiler is unable to match a resize algorithm used during training (as is specified in the
YAML pipeline) with a target VA-API configuration option.

In general, therefore, at the system level there is always the need to consider performance-accuracy
tradeoffs. The Voyager SDK makes it easy to measure and track performance and accuracy throughout
the full product development lifecycle.

## Next Steps
- **Monitor hardware during inference**: [AxMonitor Tutorial](axmonitor.md)
- **Optimize thermal performance**: [Thermal Guide](../reference/thermal_guide.md)
- **Deploy custom models**: [Custom Weights Tutorial](custom_weights.md)
- **Build production applications**: [Application Integration](application.md)

## Related Documentation
**Tutorials:**
- [Application Integration](application.md) - Build application before benchmarking
- [AxMonitor](axmonitor.md) - Monitor hardware utilization during benchmarks
- [Video Sources](video_sources.md) - The `dataset` source is used for accuracy evaluation
- [Custom Weights](custom_weights.md) - Optimize custom models based on benchmark results

**References:**
- [Model Zoo](../reference/model_zoo.md) - Reference accuracy metrics for pre-trained models
- [Thermal Guide](../reference/thermal_guide.md) - Manage thermal state during sustained benchmarks
- [inference.py CLI](../reference/inference.md) - Command-line options for benchmarking modes


## Further support
- For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
