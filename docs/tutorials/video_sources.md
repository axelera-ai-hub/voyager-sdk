![image](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Working with different video sources

## Contents
- [Working with different video sources](#working-with-different-video-sources)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Quick notes](#quick-notes)
    - [Multiple sources](#multiple-sources)
    - [Application integration](#application-integration)
- [Sources](#sources)
  - [`dataset`](#dataset)
    - [Example:](#example)
  - [USB cameras](#usb-cameras)
    - [Example:](#example-1)
  - [RTSP cameras](#rtsp-cameras)
    - [Example:](#example-2)
  - [Local files](#local-files)
    - [Example:](#example-3)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)


## Prerequisites
- Complete [Installation Guide](install.md) - SDK must be installed
- Virtual environment activated (`source ~/voyagersdk/bin/activate`)
- Hardware connected and detected (`axdevice`)

## Level
**Beginner** - Basic configuration, no coding required


## Overview
The Voyager SDK supports a wide variety of video sources as inputs to our Pipeline Builder solution. In fact, we aim to be agnostic to the origin of incoming video frames. Our underlying video pipeline is built with GStreamer, which provides powerful capabilities for processing video media and making sure that formats and sources align. We leverage this flexibility to support a wide variety of inputs: single images, video files saved on disc, RTSP streams, and USB cameras; with a variety of resolutions and other configurations. This guide goes over the basics of how to work with these different sources with our inference scripts and application APIs.


In our APIs, incoming videos are specified as a *source*. In the Quick Start Guide, you saw
```bash
./inference.py yolov5s-v7-coco usb:0
```

In this example, `usb:0` is the source argument to the `inference.py` script, telling it to use the first enumerated USB camera.

## Quick notes

### Multiple sources

Working with multiple sources is easy: just list them. For example,
```bash
./inference.py yolov8s-coco-onnx usb:0 usb:1 usb:2
```

to work with three USB cameras (note: see caveat below). For video files saved on disk, use

```bash
./inference.py yolov8s-coco-onnx media/traffic1_1080p.mp4 media/traffic2_720p.mp4
```

### Application integration

The examples in this page all use our `inference.py` script to quickly run inference and evaluate key benchmarks. To use any of these sources with our Python APIs, just use the same string to specify the source(s) that you use with `inference.py`. See the [example application walkthrough](/docs/tutorials/application.md) for more.

# Sources

## `dataset`

The `dataset` source is a special case used for evaluation of our Model Zoo models. This argument runs the model with the validation dataset that our Model Zoo models were trained on. For example, ImageNet for the ResNet family, or COCO for the YOLO object detection family. This source is used to evaluate the accuracy of the Model Zoo models compiled for Metis. For more information, see our [benchmarking guide](/docs/tutorials/benchmarking.md).

If necessary, this setting will download the validation data and labels for each dataset.

The dataset setting loads individual images from the model’s validation set, so performance will be slightly slower than a video file.

### Example:
```bash
./inference.py yolov5s-v7-coco dataset
```

## USB cameras

On Linux with v4l2, USB cameras are enumerated in `/dev/videox`, where the `x` represents a number. As a source argument, you can either use the full `/dev/video0` path, or our shorthand `usb:0`.

Note: many USB cameras have more than one sensor that can show up under `/dev/video`. These can include infrared or other sensors. Our pipeline expects the RGB sensor as a source, this is almost always the first sensor listed for that camera. Therefore, the second camera’s RGB sensor might be at `/dev/video4` (`usb:4`) if the first camera has four sensors under `/dev/video[0-3]`.

Also, Linux might enumerate any USB cameras plugged in at boot in a different order. If you want to ensure the same order of devices at `/dev/video`, you can plug them in order after boot.

When using USB cameras, the end-to-end framerate reported by `inference.py` will typically be limited by the camera stream’s FPS (unless you use enough USB cameras to hit the pipeline FPS limit) You can specify framerate and resolution with this syntax: `(usb|/dev/videoN)(:WxH)?(@FPS)?`. For example:

`usb:0:1920x1080@30` to ask the camera to provide a 1080p 30FPS stream.

The package `v4l-utils` allows you to see the supported outputs of a USB camera. You can install it with `sudo apt-get install v4l-utils`. To see a list of supported video formats from your USB camera (at /dev/video0 in this example), run `v4l2-ctl --device /dev/video0 --list-formats-ext`.

### Example:
```bash
./inference.py yolov5s-v7-coco usb:0
```

## RTSP cameras

For RSTP network cameras, just use the URL of the RTSP camera. The general format is `rtsp://<user>:<password>@<url>`. `user` and `password` can be omitted if the stream doesn’t require authentication. The URL portion should include the host or IP address, port number, and the path to a specific stream.

For example, for a video file streaming from the same system on port 8554 and path /1:

`rtsp://127.0.0.1:8554/1`

### Example:
```bash
./inference.py yolov5s-v7-coco rtsp://127.0.0.1:8554/1
```

## Local files

We also support running inference on media already saved on your system, including video and still images. All you need to do is specify the path of the local media:

`media/traffic1_1080p.mp4`

Note that the Voyager SDK comes with a number of example videos in the directory `/media` in the SDK directory. If you don’t have these, you can deactivate the virtual environment with `deactivate` and run `./install.sh --media` to download them.

### Example:
```bash
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4
```

## Next Steps
- **Integrate into your application**: [Application Integration Tutorial](application.md)
- **Test with multiple sources**: Try running inference on multiple cameras or files simultaneously
- **Benchmark model accuracy**: [Benchmarking Tutorial](benchmarking.md) using the `dataset` source

## Related Documentation
**Tutorials:**
- [Quick Start Guide](quick_start_guide.md) - Run your first inference
- [Application Integration](application.md) - Use video sources in your Python/C++ code
- [Benchmarking](benchmarking.md) - Uses `dataset` source for model evaluation

**References:**
- [Adapters Reference](../reference/adapters.md) - Detailed adapter configuration options
- [inference.py CLI](../reference/inference.md) - Command-line inference tool documentation

**Examples:**
- [application.py](../../examples/application.py) - Basic example using USB camera
- [application_extended.py](../../examples/application_extended.py) - Shows hardware caps for camera configuration

## Further support
- For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
