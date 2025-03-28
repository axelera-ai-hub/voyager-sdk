![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Benchmarking and Performance Evaluation

- [Benchmarking and Performance Evaluation](#benchmarking-and-performance-evaluation)
  - [Compare pipeline accuracy](#compare-pipeline-accuracy)
  - [Optimize and measure pipeline performance](#optimize-and-measure-pipeline-performance)

In this section, we'll discuss in-depth the different ways you can evaluate performance and accuracy
of each model with Metis. You can compare our performance to competitors, and see how our
state-of-the-art quantization minimizes accuracy loss when compared to the original FP32 models.


## Compare pipeline accuracy

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


## Optimize and measure pipeline performance

To run an optimized version of a pipeline, for example YOLO5s, whereby execution of each non-neural
element is allocated to the accelerator most able to efficiently execute it on the Evaluation
Kit system, run the following command:

```bash
./benchmark.py yolov5s-v7-coco-onnx  media/traffic3_480p.mp4 --aipu-cores=4 \\
                                       --enable-vaapi --enable-opencl --show-host-fps
```

`benchmark.py` differs from `inference.py` in that it is designed to maximise the performance of the
end-to-end pipeline. The `--enable-` arguments causes it to offload some of the non-neural elements
to Intel hardware accelerators (via the VA-API driver) and other elements to the Intel embedded GPU
(via the OpenCL driver). This command disables the host application rendering by default (use
`--display` to see the rendering). On completion, a summary is displayed, an example of which is
shown below.

```
===========================================================================
Element                                         Latency(us)   Effective FPS
===========================================================================
qtdemux0                                                 23        41,732.2
typefind                                                 38        25,830.1
h265parse0                                               79        12,634.5
capsfilter0                                              30        33,109.4
vaapidecode0                                          1,536           651.0
capsfilter1                                              41        23,926.8
vaapipostproc0                                          138         7,231.3
axinplace-addstreamid0                                   41        24,063.7
input_tee                                                46        21,374.4
convert_in                                              144         6,919.5
axinplace0                                               27        36,330.3
axinplace1                                               33        30,060.2
axtransform-resize0                                     928         1,077.6
axinplace-normalize0                                    464         2,152.1
axtransform-padding0                                    282         3,539.5
inference-task0-yolov5s-v7-coco-onnx                  3,525           283.7
 └─ Metis                                             2,131           469.2
 └─ Host                                              3,085           324.1
decoder_task0                                         1,040           960.8
axinplace-nms0                                           69        14,310.1
===========================================================================
End-to-end average measurement                                        275.3
===========================================================================
```

The stats table contains the following columns:

*   **Element:** The name of each low-level GStreamer pipeline element.

*   **Latency (us):** The measured execution time for each pipeline element (for a single frame).

*   **Effective FPS**: The theoretical highest frame rate for each element (1/latency) assuming all
other elements could provide it with data at sufficiently high rate.


By inspecting this table, it is possible to quickly identify any performance bottlenecks in an
end-to-end application pipeline. In general, the lowest effective FPS in the table represents the
fastest frame rate at which the entire pipeline can operate, though in practice some pipeline
elements share the same hardware resources and the actual measured end-to-end performance will be
less.

The element `inference-task0-yolov5s-v7-coco-onnx` reports the time taken for Metis to perform
model inference, including the time taken by the host to transfer data to the device via PCIe. The
AIPU latency in the row below is the Metis execution time only (ignoring host transfer time). While
this latency is higher than other pipeline elements, the use of 4 AIPU cores per device would ensure
a high throughput commensurate with the preprocessing capabilities of the host.

The frame rate reported by `inference-<<model name>>`-> `(AIPU)` for a given model, is the value
that you should use to compare Metis device-level performance to other solutions.

Note that the reported accuracy of performance-optimized pipelines is usually slightly lower than
pipelines that utilize the CPU for the non-neural elements. This is because hardware accelerators
such as VA-API usually support a limited number of configuration parameters and accuracy is usually
reduced when a pipeline element does not implement precisely the same algorithm that was used
originally during training. For example, the following article explains why accuracy is lost if the
pipeline compiler is unable to match a resize algorithm used during training (as is specified in the
YAML pipeline) with a target VA-API configuration option:
[The dangers behind image resizing](https://zuru.tech/blog/the-dangers-behind-image-resizing).
In general, therefore, at the system level there is always the need to consider performance-accuracy
tradeoffs. The Voyager SDK makes it easy to measure and track performance and accuracy throughout
the full product development lifecycle.
