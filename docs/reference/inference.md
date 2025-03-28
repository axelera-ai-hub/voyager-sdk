![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Inference

- [Inference](#inference)
  - [Configuration & Advanced Usage](#configuration--advanced-usage)
    - [--pipe `<type>`](#--pipe-)
    - [--frames `<frame-count>`](#--frames-)
    - [--no-display](#--no-display)
    - [--save-output `<path>`](#--save-output-)
    - [--enable-hardware-codec](#--enable-hardware-codec)
    - [--enable/disable-vaapi, --enable/disable-opencl](#--enabledisable-vaapi---enabledisable-opencl)
    - [--enable/disable-opengl](#--enabledisable-opengl)
    - [--aipu-cores `<core-count>`](#--aipu-cores-)
    - [--show-host-fps, --show-stream-timing](#--show-host-fps---show-stream-timing)

Axelera's inference tool can be used to measure the accuracy and performance of networks
on the Axelera platform. It accepts a network or a model and input media, and will run it on
the input media, or measure its accuracy using a dataset.
We define a network to be one or more models connected together, whilst a model is a single model.

Typical usage of the inference tool is as follows:

```bash
./inference.py <network-name> <media>
```

Where `<network-name>` is the name of the network to perform inference upon, and `<media>` is the
video or image data to run the network on.

Inference will accept a network whether it is compiled and deployed or not. If all models in the network
have already been deployed for the Axelera platform, inference will run immediately. If not, on first time
running inference, the Axelera compiler and deployment tool (`deploy.py`) will be invoked automatically
to deploy the model, which may take some time. After this first usage, future inference runs will reuse
the deployed model.

Running inference in this case will run for the entire length of the provided media, and then exit.
The network output will be rendered in real time in an OpenGL window (assuming an OpenGL capable display is available).
Any network-specific rendering of the inference result will be applied, such as bounding boxes, as well as speedometers to
show CPU usage, Metis and system FPS. Progress through the media will be reported via a progress bar on the standard output,
alongside time elapsed, estimated time remaining and FPS.

Once the inference run is complete, a summary of the the end-to-end average framerate will be reported
to the standard output. In addition, the average Metis and system FPS achieved will be reported, as
will average CPU usage.

If an image is used as the input media instead, inference will not terminate automatically. Instead,
the image will persist until the window is closed, to allow for visual evaluation of the results. The
window may be closed by pressing `q`. In addition, the end-to-end summary will not be reported when
performing inference on only a single image.

Inference can therefore be run for various models, with the output performances compared and evaluated.


## Configuration & Advanced Usage

Multiple input sources may be provided when running inference, like so:

```bash
./inference.py <network-name> <media0> <media1> <media2> ...
```

In this case, the model will be run in parallel on each input stream, with the inference output being
the overall statistics from all of the media.

Various sources beyond video files may be used as input. These are:
 * A csi camera (`csi:0`)
 * A usb camera (`usb`, `usb:0`, `usb:1` etc.)
 * A video pulled from a uri (`http(s)://`, `rtsp://`)
 * A fake video can also be used to simply run the pipeline (`fakevideo:widthxheight`)

Numerous configurations can be made to the inference run by using command line arguments. All
options may be listed via the `./inference.py --help` command, but the most common ones can be found below:

### --pipe `<type>`

The pipe argument can be used to specify which pipeline type to run inference upon. The options are the
`torch` pipeline for a complete PyTorch-based pipeline that uses ONNXRuntime if the model is in ONNX format,
`gst` for a C/C++ pipeline using GStreamer with the core model on AIPU, and `torch-aipu` for a PyTorch
pipeline with the core model offloaded to AIPU. GStreamer will always use the AIPU.

The default pipeline is GStreamer (`gst`)

### --frames `<frame-count>`

The frames argument is used to restrict the number of frames from the input source(s) to process in the
inference run. By default, this is set to `0`, which will run all frames from all media. When this
argument is used with multiple input media, the frame count is the sum of frames processed across all
sources. For example, if running with four streams and `--frames 100`, this will run 25 frames across
each input source (assuming each source is processed at the same frame rate).

### --no-display

By default, inference output will be rendered to an OpenGL or OpenCV window. The `--no-display` argument
may be used to remove the display. This will, in addition, disable all rendering. As rendering can be
quite compute intensive, this can be useful to obtain more accurate CPU usage information data. In a typical
application usage, rendering would not be required to make application logic decisions. If the host is CPU
constrained (for example on an embedded system) this can sometimes improve system FPS as well, though in
general rendering is not a direct bottleneck, as rendering will be dropped if the inference
results are arriving too quickly.

### --save-output `<path>`

The save output command can be used to save the rendered output from the inference run to an mp4 file for
later review. This command can be used for either video or image inputs. When using a single input media,
a file name, such as `output.mp4` should be provided.

This option also supports saving output for multiple input media. In this case, a formatted string
identifier should be provided. For example, `output%02d.mp4`. When this is interpreted, the output streams
will be saved separately, and sequentially enumerated (two significant figures in this case). For example,
if four input streams are used in this case, the output files will be `output00.mp4`, `output01.mp4`,
`output02.mp4`, `output03.mp4`.

Note that when saving output all frames must be rendered, therefore the system FPS may be bottlenecked
by the rendering in this case.

### --enable-hardware-codec

By default, GStreamer will choose the highest scoring decode codec available. This almost
always means that it will select a hardware codec ahead of a software codec if one is available. 
Due to interactions with the rest of the pipeline this often leads to disappointing performance. For
this reason we default to using a software codec as this gives us best performance over a wide range
of platforms. 
This configuration can be used to enable GStreamer to prefer a hardware codec.

### --enable/disable-vaapi, --enable/disable-opencl

These configurations can be used to manually control which hardware accelerators are enabled during the
inference run for preprocessing. By default, whether to use VAAPI or OpenCL acceleration is detected
automatically from the availability of the VAAPI and OpenCL. However you can override this by disabling
the detection with `--disable-vaapi/opencl`. Also, you can override the automatic detection with
`--enable-vaapi/opencl`, but this is only really useful if the automatic detection gave a false negative.

Note that VAAPI is only available on Intel hosts.

### --enable/disable-opengl

OpenGL is used to render the inference annotations on the original input media. It is normally detected
automatically, and if not available then it will attempt to use OpenCV to render the annotations
(generally with reduced performance). You can force the use of OpenCV by disabling OpenGL detection
with `--disable-opengl`, or if the attempted detection of OpenGL yields a false negative you can
override this with `--enable-opengl`. If neither OpenGL nor OpenCV are available, there will be no
rendering or window, in the same way as using the `--no-display` configuration.

### --aipu-cores `<core-count>`

This configuration can be used to specify how many of the AIPU cores on the device to use for the
inference run. By default, this will be all the cores available (currently 4). This command can
be used to restrict the inference run to fewer, such as `1`, `2` or `3`.

Note this configuration can only be used if the network has a single model. If there are multiple models, the
AIPU cores to use for each model needs to be specified in the network YAML.

### --show-host-fps, --show-stream-timing

These two configurations can be used to display additional inference information. When using a host,
`--show-host-fps` is used to render a speedometer of the specific host FPS, and display the overall
FPS in the end of run summary.

`--show-stream-timing` can be used to give additional information about the performance of the
inference stream. This will be reported on the standard output throughout the inference run, with
information about latency and jitter, as well as in the end of run summary.
