![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# AxRunModel
- [AxRunModel](#axrunmodel)
  - [Configuration & Advanced Usage](#configuration--advanced-usage)
    - [-d `<devices>`, --devices `<devices>`](#-d----devices-)
    - [--seconds `<seconds>`](#--seconds-)
    - [--aipu-cores `<aipu_cores>`](#--aipu-cores-)
    - [--throttle-fps `<throttle-fps>`](#--throttle-fps-)
    - [--double-buffer, --no-double-buffer](#--double-buffer---no-double-buffer)
    - [--input-dmabuf, --no-input-dmabuf](#--input-dmabuf---no-input-dmabuf)
    - [--output-dmabuf, --no-output-dmabuf](#--output-dmabuf---no-output-dmabuf)
    - [--show-bar-chart](#--show-bar-chart)
    - [--show-histogram](#--show-histogram)

The `axrunmodel` tool is used to run a given model with real data, using available optimizations
such as DMA buffers, double buffering and running on multiple cores.

Basic usage of the `axrunmodel` tool is as follows:

```bash
axrunmodel <path-to-model>
```

The path to the model should point to the deployed `model.json` file which you wish to run. You can
obtain a model using `./download_prebuilt.py yolov8l-coco-onnx`, and the resulting model can be
found at `build/yolov8l-coco-onnx/yolov8l-coco-onnx/1/model.json`.

The `1` indicates the batch size of the model, which simply means the number of cores that a single
instance of the model will execute in parallel. `axrunmodel` will instantiate enough instances
of the model to use all requested cores.

After the run has completed, `axrunmodel` will output information about the model run, such as device
FPS, host FPS, system FPS and whether the model run was successful or not.

Device FPS and system FPS are not a measurement of throughput. They could more accurately be described
as 1/execution_duration at the device level, including data transfers at the host level. It is more
convenient to refer to it as FPS.

System FPS is a measurement of the total number of frames executed / time taken. And so it is a more
meaningful representation of the throughput.

`axrunmodel` uses the same input data for every frame, this is efficient and it means we can measure the
throughput of the model in an ideal situation where the host is able to supply sufficient input data to
maximize AIPU utilization.

## Configuration & Advanced Usage

The following arguments can be used to fine-tune the model run. The run can be throttled or truncated,
some settings may be made applied the device, optimizations can be disabled and output data
can be returned in graph form. A full list of arguments can also be obtained
from `axrunmodel --help`:

### -d `<devices>`, --devices `<devices>`

The 'devices' argument is used to select the device(s) to run the model on. The devices
must be referred to by their index, obtained by running `axdevice` without arguments. For
multiple devices the indexes should be comma separated. For example `-d0,1` will run on the 
first two enumerated devices.

If `--devices` is used, a list of devices may be provided, and the model will be run on all listed
devices.

### --seconds `<seconds>`

The 'seconds' argument can be used to run the model for the given number of seconds.

By default the model will be run for 10 seconds.

### --aipu-cores `<aipu_cores>`

The 'AIPU cores' argument can be used to configure how many AIPU cores on the device(s) to use
for running the model.

By default this is all cores, 4, but this argument can be used to restrict the model to only run
on 1, 2 or 3 core(s) of the device(s)

### --throttle-fps `<throttle-fps>`

The 'throttle FPS' argument will be used to limit the system FPS to a maximum of the
given framerate.

By default there is no throttling.

### --double-buffer, --no-double-buffer

These arguments can be used to enable/disable the double buffering optimization. By default,
double buffering is enabled for the model run, but can be disabled with the `--no-double-buffer`
argument.

### --input-dmabuf, --no-input-dmabuf

These arguments can be used to enable/disable the input DMA buffers optimization. By default,
input DMA buffers are enabled for the model run, but can be disabled with the `--no-input-dmabuf`
argument.
  
### --output-dmabuf, --no-output-dmabuf

These arguments can be used to enable/disable the output DMA buffers optimization. By default,
output DMA buffers are enabled for the model run, but can be disabled with the `--no-output-dmabuf`
argument.

### --show-bar-chart

The 'show bar chart' argument can be used to represent the device and host FPS in bar chart form.

The horizontal axis is the FPS, the vertical axis is time from beginning of the run, with the
beginning of the run at the top.

The bar chart will be displayed on the standard output, with a bar representing the FPS of
each frame. Example usage could be for comparing FPS during the run, such as detecting ramp up
or time based variability in the performance.

### --show-histogram

The 'show histogram' argument can be used to represent the device and host FPS in histogram form. The
horizontal axis is the number of items in each bin, the vertical axis is interval bins of 1 divided
by frame execution time.

The histogram will be displayed on the standard output, showing various FPS ranges using various sized
bars to show what the most common FPS ranges were during the run. This show how consistent the 
performance is.
